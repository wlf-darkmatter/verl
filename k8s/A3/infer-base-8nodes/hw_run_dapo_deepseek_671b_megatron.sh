    # --ray_init \

python3 tests/verl_offline_infer.py \
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
    --max_response_length $((4*1024)) \
    --max_num_batched_tokens $((4*1024)) \
    --n_gpus_per_node ${NPU_PER_NODE} \
    --dataset_path $TRAIN_FILE \
    --hdfs_path $MODEL_PATH \
    --gpu_memory_utilization 0.60 \
    --nnodes $NNODES $@
