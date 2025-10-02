export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=1
# export HCCL_EXEC_TIMEOUT=3600
export ASCEND_GLOBAL_LOG_LEVEL=3
# CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

#######################################
#! 规避模型加载时 权重读取错误的问题
# rm -f /opt/vllm/vllm/model_executor/model_loader/base_loader.py
# cp -f /home/new_verl/k8s/patch/base_loader.py /opt/vllm/vllm/model_executor/model_loader/base_loader.py

# rm -f /opt/vllm/vllm/model_executor/models/deepseek_v2.py
# cp -f /home/new_verl/k8s/patch/deepseek_v2.py /opt/vllm/vllm/model_executor/models/deepseek_v2.py
#######################################

# mkdir -p /data01/huawei-2025/wlf/watch
# bash /data01/huawei-2025/wlf/verl/k8s/script/watch_stats.sh > /data01/huawei-2025/wlf/watch/rank${RANK}_${CURRENT_IP}.log &

# source /usr/local/Ascend/ascend-toolkit/set_env.sh;
# source /usr/local/Ascend/nnal/atb/set_env.sh;
# source /opt/pyvenv/bin/activate;

# LIB_PATH=/opt/python3.10/lib/
# export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH

unset LOCAL_WORLD_SIZE
# unset WORLD_SIZE
unset LOCAL_RANK

NPU_PER_NODE=4  # A2 NPU Number
NNODES=16         # example is 4 Nodes


#! ------------------------------------------------------------
TRAIN_FILE="/mnt/hpfs_test/data/data/dapo-math-17k.parquet"
MODEL_PATH="/mnt/hpfs_test/weights/dsv3-base-fp8-zy-bf16"

n_resp_per_prompt=4
train_prompt_bsz=16

gen_tp=8
gen_dp=8


python3 tests/verl_offline_infer.py \
    -tp $gen_tp \
    -dp $gen_dp \
    --enable_expert_parallel \
    -n $n_resp_per_prompt \
    --gen_bs $train_prompt_bsz \
    --max_prompt_length $((2*1024)) \
    --max_response_length $((4*1024)) \
    --max_num_batched_tokens $((4*1024)) \
    --n_gpus_per_node $NPU_PER_NODE \
    --dataset_path $TRAIN_FILE \
    --hdfs_path $MODEL_PATH \
    --gpu_memory_utilization 0.60 \
    --nnodes $NNODES $@