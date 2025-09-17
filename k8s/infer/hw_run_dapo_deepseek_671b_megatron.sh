#!/usr/bin/env bash
set -x pipefail

# 0. download the config
# only need to download the configuration_deepseek.py and config.json
# remove the `quantization_config` in the `config.json`
# set `num_nextn_predict_layers=0` to disable MTP, which is not currently supported
#huggingface-cli download deepseek-ai/DeepSeek-V3-0324 configuration_deepseek.py config.json

project_name='DAPO'
exp_name='DAPO-DeepSeek-671b-megatron'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=0.1

loss_agg_mode="token-mean"

train_prompt_bsz=256 # must be > n_gpus. need to fix
n_resp_per_prompt=16
train_prompt_mini_bsz=32  # mini_bsz * n >= micro_bsz * pp * dp

#NNODES=${NNODES:-1}
NNODES=16

# 1. download the dist_ckpt format model from https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main
# change the MODEL_PATH and MCORE_MODEL_PATH to your own path
# Paths
MODEL_PATH="/data01/nlp/dpsk-v3-671B-BF16-dist_ckpt"
MCORE_MODEL_PATH="/data01/huawei-2025/xczhao/weights/dsv3_fp16_mcore_full_new"
RAY_DATA_HOME="/opt"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE="/data01/huawei-2025/xczhao/rl_data/dapo-math/dapo-math-17k.parquet"
TEST_FILE="/data01/huawei-2025/xczhao/rl_data/dapo-math/dapo-math-17k.parquet"

#TEST_FILE="['$aime24_test_path']"

#master ip
CURRENT_IP=$1

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True
gen_tp=4
gen_dp=16
gen_world_size=$((NNODES*8))
train_tp=1
train_ep=16
train_pp=16
enable_filter_group=True


python3 verl_offline_infer.py \
    --ray_master_ip $CURRENT_IP \
    -tp $gen_tp \
    -dp $gen_dp \
    --enable_expert_parallel \
    -n $n_resp_per_prompt \
    --gen_bs $train_prompt_bsz \
    --max_prompt_length $((2*1024)) \
    --max_response_length $((5)) \
    --max_num_batched_tokens $((5)) \
    --n_gpus_per_node 8 \
    --dataset_path $TRAIN_FILE \
    --hdfs_path $MODEL_PATH \
    --gpu_memory_utilization 0.60 \
    --nnodes $NNODES $@   2>&1 | tee /tmp/ray.output

ray_name=$(cat /tmp/ray.output | grep "submitted successfully" | awk -F "'" '{print $2}')
ray_name=${ray_name//\'}
echo "ray_name: $ray_name"
ray job logs $ray_name --follow
