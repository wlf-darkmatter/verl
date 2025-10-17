    # --ray_init \
TRAIN_FILE="/mnt/hpfs_test/data/data/dapo-math-17k.parquet"
MODEL_PATH="/mnt/hpfs_test/weights/dsv3-base-fp8-wlf-bf16"

gen_tp=8
gen_dp=8
n_resp_per_prompt=4
train_prompt_bsz=$((16 * WORLD_SIZE / (gen_tp * gen_dp)))

cd /opt/verl
RUNTIME_ENV=verl/trainer/mc2_env.yaml
ray job submit --runtime-env="${RUNTIME_ENV}" \
    --working-dir /opt/verl \
    -- python3 tests/verl_offline_infer_tmp.py \
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
    --max_response_length $((12*1024)) \
    --max_num_batched_tokens $((4*1024)) \
    --n_gpus_per_node ${NPU_PER_NODE} \
    --dataset_path $TRAIN_FILE \
    --hdfs_path $MODEL_PATH \
    --gpu_memory_utilization 0.60 \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --ray_debug \
    --nnodes $NNODES $@

# ray job submit --runtime-env="${RUNTIME_ENV}" \
#     --working-dir /opt/verl \
#     -- python3 tests/mtp_offline_infer.py \
#     --model="${MODEL_PATH}" \
#     --tp-size=2 \
#     --node-size=2 \
#     --node-rank=1 \
#     --enable-expert-parallel \
#     --master-addr=10.99.48.128 \
#     --master-port=13345 
