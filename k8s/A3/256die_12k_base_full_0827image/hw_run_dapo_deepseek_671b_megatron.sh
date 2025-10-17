set -x

echo ">>Starting script at: $(date), path = $(pwd)"

project_name='DAPO'
exp_name='DAPO-dpsk-671b-megatron-BASE-16NNODES-0827images'

adv_estimator=grpo

use_kl_in_reward=False
kl_penalty="kl"
kl_coef=0.0

use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 12))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
train_prompt_bsz=32
n_resp_per_prompt=16
train_prompt_mini_bsz=32
train_ppo_micro_batch_size_per_gpu=2
infer_ppo_micro_batch_size_per_gpu=2
# Paths
MODEL_PATH="/mnt/hpfs_test/weights/dsv3-base-fp8-zy-bf16"
MCORE_MODEL_PATH="/mnt/hpfs_test/weights/dsv3_base_bf16_mcore_zy_hs_mtp0"
CKPTS_DIR=/mnt/hpfs_test/weights/CKPT/ckpt-${exp_name}
TRAIN_FILE="/mnt/hpfs_test/data/rl_data/dapo-math-17k_dedup_r1_sys_prompt_mathdapo.parquet"
TEST_FILE="/mnt/hpfs_test/data/rl_data/dapo-math-17k_dedup_r1_sys_prompt_mathdapo.parquet"
# TEST_FILE="['$aime24_test_path']"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

optimizer_offload_fraction=1


# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=False
USE_DIST_CKPT=True


# first_layer=6
# last_layer=7
# pipeline_num_transformer_layers="[[6],[8],[8],[8],[8],[8],[8],[7]]"
first_layer=6
last_layer=7
# pipeline_num_transformer_layers="[[3],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[2]]"
offload=True
gen_tp=8
gen_dp=8

train_tp=8
train_ep=32
train_pp=8
enable_filter_group=False
train_cp=1
#    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${train_cp} \
ETP=1
#    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
#   +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
#   +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="alltoall" \
#
RUNTIME_ENV=verl/trainer/mc2_env.yaml
cd /opt/verl
ray job submit --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name="dapo_megatron_trainer" \
    actor_rollout_ref.rollout.skip.enable=True \
    actor_rollout_ref.rollout.skip.dump_dir=${JOB_LOG_DIR}/rollout_skip \
    actor_rollout_ref.rollout.skip.max_dump_step=500 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=messages \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_penalty=${kl_penalty} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    algorithm.filter_groups.enable=False \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=$first_layer \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=$last_layer \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=fused \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    +actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dp_model_parallel_size=${gen_dp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NPU_PER_NODE}" \
    trainer.balance_batch=False \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.resume_mode=auto \
    trainer.rollout_data_dir=${JOB_LOG_DIR_CURR}/rollout_data_dir \
    trainer.log_val_generations=10 \
    trainer.device="npu" $@ 2>&1 | tee /tmp/ray.output



