export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
# export HCCL_EXEC_TIMEOUT=3600
export ASCEND_GLOBAL_LOG_LEVEL=3
CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

#######################################
#! 规避模型加载时 权重读取错误的问题
#! [VLLM]
#* 规避直接读 hf 权重的报错（出现减层或者带有MTP）
rm -f /opt/vllm/vllm/model_executor/models/deepseek_v2.py
cp -f /home/code/verl/k8s/patch/0928/vllm/vllm/model_executor/models/deepseek_v2.py /opt/vllm/vllm/model_executor/models/deepseek_v2.py


#! [VLLM-ASCEND]

rm -f /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py
cp -f /home/code/verl/k8s/patch/0928/vllm-ascend/vllm_ascend/models/deepseek_v2.py /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py

#######################################



source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;
source /opt/pyvenv/bin/activate;

LIB_PATH=/opt/python3.10/lib/
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH

unset LOCAL_WORLD_SIZE
# unset WORLD_SIZE
unset LOCAL_RANK

export NPU_PER_NODE=8  # A2 NPU Number
export NNODES=8         # example is 4 Nodes

export path_log_dir=/opt/verl/logs/$MINDX_TASK_ID/trainlog  # modify according to actual situation
export ASCEND_PROCESS_LOG_PATH=/home/code/verl/plog/$(basename $(dirname $0))/${RANK}


ray stop --force
rm -rf /tmp/ray
rm -rf /opt/verl
cp -r /home/code/verl /opt/verl
cd $(dirname $0)


export ServerPort=6666     # modify according to actual situation
export DashboardPort=8888  # modify according to actual situation


cd /home/code/verl
if [[ "$RANK" = "0" ]]; then
  # head start
  echo "This is head node"
  echo "CURRENT_IP=$CURRENT_IP"

  kwargs=(--is_master --ray_dashboard_port $DashboardPort )
else
  kwargs=( )
fi

#! ------------------------------------------------------------
TRAIN_FILE="/data01/huawei-2025/rl_data/dapo-math/dapo-math-17k.parquet"
MODEL_PATH="/data01/huawei-2025/weight/dpsk-v3-671B-BF16-dist_ckpt"

n_resp_per_prompt=4
train_prompt_bsz=16

gen_tp=8
gen_dp=8


#! [VLLM]
#* 规避直接读 hf 权重的报错（出现减层或者带有MTP）
rm -f /opt/vllm/vllm/model_executor/models/deepseek_v2.py
cp -f /home/code/verl/k8s/patch/0928/vllm/vllm/model_executor/models/deepseek_v2.py /opt/vllm/vllm/model_executor/models/deepseek_v2.py


#! [VLLM-ASCEND]

rm -f /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py
cp -f /home/code/verl/k8s/patch/0928/vllm-ascend/vllm_ascend/models/deepseek_v2.py /opt/vllm-ascend/vllm_ascend/models/deepseek_v2.py



python3 tests/verl_offline_infer.py \
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
    --max_response_length $((2*1024)) \
    --max_num_batched_tokens $((4*1024)) \
    --n_gpus_per_node ${NPU_PER_NODE} \
    --dataset_path $TRAIN_FILE \
    --hdfs_path $MODEL_PATH \
    --gpu_memory_utilization 0.60 \
    --nnodes $NNODES $@
