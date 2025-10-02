set -x

# 0. download the config
# only need to download the `configuration_deepseek.py`, `config.json`, `tokenizer_config.json`, `tokenizer.json` and `generation_config.json`
# remove the `quantization_config` in the `config.json`
# set `num_nextn_predict_layers=0` to disable MTP, which is not currently supported

# huggingface-cli download deepseek-ai/DeepSeek-V3-0324 configuration_deepseek.py config.json

# 1. download the dist_ckpt format model from https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main
# change the HF_MODEL_PATH and DIST_CKPT_PATH to your own path
DIST_CKPT_PATH="/sfs_turbo/hw/lmz/verl_ds/code/dist_ckpt"
LLM="/sfs_turbo/pretrained_models/DeepSeek-R1-BF16"
CKPT_DIR="/sfs_turbo/hw/lmz/verl_ds_908/ckpt_save_dir"


# 2. run the script
gsm8k_train_path=/sfs_turbo/hw/yzr/datasets/gsm8k_grpo/train.parquet
gsm8k_test_path=/sfs_turbo/hw/yzr/datasets/gsm8k_grpo/test.parquet
train_files=$gsm8k_train_path
test_files=$gsm8k_test_path

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
RM_PARAM_OFFLOAD=${RM_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}

NODES=16
PP=8
TP=4
EP=32
ETP=1
INFER_TP=32
INFER_DP=1
# consider TP/ETP, and enable recompute if short of memory

max_prompt_length=2048
max_response_length=4096

n_resp_per_prompt=4

    # +actor_rollout_ref.rollout.dp_model_parallel_size=$INFER_DP \
# RAY_ADDRESS='auto' ray job submit --working-dir . --
#export RAY_ADDRESS='http://127.0.0.1:8266'
RUNTIME_ENV=verl/trainer/all2allv_runtime_env.yaml
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" -- \
    python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style="constant" \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    algorithm.use_kl_in_reward=False \
    +trainer.rollout_data_dir="/sfs_turbo/hw/lmz/open_verl_ds/infer_data" \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_megatron_gsm8k_examples' \
    trainer.experiment_name='dsv3-32nodes' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=$NODES \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=7 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=6 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.total_epochs=100 $@