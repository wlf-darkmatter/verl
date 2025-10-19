export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=1
# export HCCL_EXEC_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"



export ASCEND_GLOBAL_LOG_LEVEL=3

#! 注意，0929加了这 1 个优化参数， libjemalloc 需要重新编译
# export LD_PRELOAD="/usr/local/lib/libjemalloc.so.2"
export TASK_QUEUE_ENABLE=2

CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

#######################################
#! 规避模型加载时 权重读取错误的问题
rm -f /opt/vllm/vllm/model_executor/model_loader/base_loader.py
cp -f /home/new_verl/k8s/patch/base_loader.py /opt/vllm/vllm/model_executor/model_loader/base_loader.py

rm -f /opt/vllm/vllm/model_executor/models/deepseek_v2.py
cp -f /home/new_verl/k8s/patch/deepseek_v2.py /opt/vllm/vllm/model_executor/models/deepseek_v2.py

rm -f /opt/vllm-ascend/vllm_ascend/ops/fused_moe.py
cp -f /home/new_verl/k8s/patch/vllm_ascend/ops/fused_moe.py /opt/vllm-ascend/vllm_ascend/ops/fused_moe.py

#######################################
# 先导模型megatron patch
rm -f /opt/Megatron-LM/megatron/core/distributed/finalize_model_grads.py
cp -f /home/new_verl/k8s/patch/megatron/finalize_model_grads.py /opt/Megatron-LM/megatron/core/distributed/finalize_model_grads.py
echo "cp -f /home/new_verl/k8s/patch/megatron/finalize_model_grads.py /opt/Megatron-LM/megatron/core/distributed/finalize_model_grads.py"
#######################################

source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;
source /opt/pyvenv/bin/activate;



LIB_PATH=/opt/python3.10/lib/
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH

unset LOCAL_WORLD_SIZE
# unset WORLD_SIZE
unset LOCAL_RANK

# export ASCEND_GLOBAL_LOG_LEVEL=1
# export ASCEND_LAUNCH_BLOCKING=1

export NPU_PER_NODE=8  # A2 NPU Number
export NNODES=2         # example is 4 Nodes

export path_log_dir=/opt/verl/logs/$MINDX_TASK_ID/trainlog  # modify according to actual situation
export ASCEND_PROCESS_LOG_PATH=/opt/verl/logs/$MINDX_TASK_ID/plog # modify according to actual situation

ray stop --force
sleep 10
rm -rf /tmp/ray
rm -rf /opt/verl
cp -r /home/new_verl /opt/verl
cd $(dirname $0)

export ServerPort=6666     # modify according to actual situation
export DashboardPort=8888  # modify according to actual situation

cd /home/new_verl
if [ "$RANK" = "0" ]; then
  # head start
  echo "This is head node"
  mkdir -p ${JOB_LOG_DIR_CURR}
  mkdir -p ${JOB_LOG_DIR_CURR}/ray_host
  echo "CURRENT_IP=$CURRENT_IP"
  ln -s ${JOB_LOG_DIR_CURR}/ray_host /tmp/ray
  #* 拷贝当前脚本文件
  mkdir -p ${JOB_LOG_DIR_CURR}/script.bak
  cp $(dirname $0)/*.sh ${JOB_LOG_DIR_CURR}/script.bak/
  cp $(dirname $0)/*.yaml ${JOB_LOG_DIR_CURR}/script.bak/

  kwargs=(--is_master --ray_dashboard_port $DashboardPort )
else
  kwargs=( )
fi

#! ------------------------------------------------------------
TRAIN_FILE="/data01/huawei-2025/rl_data/dapo-math/dapo-math-17k.parquet"
# MODEL_PATH="/data01/huawei-2025/weight/dpsk-v3-671B-BF16-dist_ckpt"
# MODEL_PATH="/data01/huawei-2025/weight/dsv3-base-hf"
MODEL_PATH="/data01/huawei-2025/xczhao/weights/4layers_pre"

n_resp_per_prompt=8
train_prompt_bsz=32

gen_tp=8
gen_dp=2


python3 /opt/verl/tests/verl_offline_infer.py \
    ${kwargs[@]} \
    --ray_init \
    --ray_master_ip $MASTER_ADDR \
    --ray_master_port $ServerPort \
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
