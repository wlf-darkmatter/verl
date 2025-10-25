#!/usr/bin/env bash
set -x pipefail

TRAIN_FILE="/data01/huawei-2025/rl_data/dapo-math/dapo-math-17k.parquet"
MODEL_PATH="/data01/huawei-2025/xczhao/weights/pilot_112B_hf"

gen_tp=8
gen_dp=8
n_resp_per_prompt=4
train_prompt_bsz=$((16 * WORLD_SIZE / (gen_tp * gen_dp)))

RUNTIME_ENV=verl/trainer/mc2_env.yaml
cd /opt/verl
ray job submit --runtime-env="${RUNTIME_ENV}" \
    -- python3 tests/verl_offline_infer.py \
    ${kwargs[@]} \
    --ray_master_ip $MASTER_ADDR \
    --ray_master_port $ServerPort \
    --ray_debug \
    -tp $gen_tp \
    -dp $gen_dp \
    --enable_expert_parallel \
    -n $n_resp_per_prompt \
    --gen_bs $train_prompt_bsz \
    --max_prompt_length $((2*1024)) \
    --max_response_length $((1*1024)) \
    --max_num_batched_tokens $((1*1024)) \
    --n_gpus_per_node ${NPU_PER_NODE} \
    --dataset_path $TRAIN_FILE \
    --hdfs_path $MODEL_PATH \
    --gpu_memory_utilization 0.60 \
    --nnodes $NNODES $@