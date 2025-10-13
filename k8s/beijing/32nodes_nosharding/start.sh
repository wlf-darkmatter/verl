export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
# export HCCL_EXEC_TIMEOUT=3600
export ASCEND_GLOBAL_LOG_LEVEL=3
export VERL_DEBUG_NOSHARDING="1"

export VERL_ENVIRONMENT_ROLLOUT_DIR=$(dirname $0)/env_rollout
mkdir -p $VERL_ENVIRONMENT_ROLLOUT_DIR

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


#! [Megatron]
rm -f /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py
cp -f /home/code/verl/k8s/patch/0928/Megatron-LM/megatron/dot_product_attention.py /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py



#######################################

mkdir -p /data01/huawei-2025/wlf/watch
bash /data01/huawei-2025/wlf/verl/k8s/script/watch_stats.sh > /data01/huawei-2025/wlf/watch/rank${RANK}_${CURRENT_IP}.log &

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
export NNODES=32         # example is 4 Nodes

export path_log_dir=/opt/verl/logs/$MINDX_TASK_ID/trainlog  # modify according to actual situation
export ASCEND_PROCESS_LOG_PATH=/home/code/verl/plog/$(basename $(dirname $0))/${RANK}


ray stop --force
rm -rf /tmp/ray
rm -rf /opt/verl
cp -r /home/code/verl /opt/verl
cd $(dirname $0)

export ServerPort=6666     # modify according to actual situation
export DashboardPort=8888  # modify according to actual situation

cnt=0
if [ "$RANK" = "0" ]; then
  # head start
  echo "This is head node"
  echo "CURRENT_IP=$CURRENT_IP"

  ray start --head --port $ServerPort --dashboard-port=$DashboardPort --node-ip-address=$CURRENT_IP --dashboard-host=$CURRENT_IP --disable-usage-stats

  while [[ $cnt -lt 100 ]]; do
    ray_status_output=$(ray status)
    npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
    npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')

    # judge npu_count_int bigger than NNODES*NPU_PER_NODE
    if [ "$npu_count_int" -ge "$((NNODES*NPU_PER_NODE))" ]; then
      echo "Ray cluster is ready with $npu_count_int npu (from $npu_count NPU resources), starting Python script."
      bash hw_run_dapo_deepseek_671b_megatron.sh
      break
    fi

    echo "Waiting for Ray to allocate $((NNODES*NPU_PER_NODE)) devices. Current device count: $npu_count_int"
    cnt=$((cnt+1))
    sleep 10
  done

else
  echo "This is worker node"
  ray start --address="$MASTER_ADDR:$ServerPort" --disable-usage-stats
fi

cnt=0
while true; do
  ray_name=$(ray job list | grep -o "raysubmit_[a-zA-Z0-9]*")
  if [[ -n $ray_name ]]; then
    echo "Job $ray_name start succeeded"
    break
  fi

  cnt=$((cnt+1))
  if [[ $cnt -gt 100 ]]; then
    echo "Job $ray_name start failed"
    ray stop --force
    rm -rf /tmp
    exit 1
  fi

  sleep 50
done

ray_name=$(ray job list | grep -o "raysubmit_[a-zA-Z0-9]*")
while true; do
  output=$(ray job status $ray_name)
  failed=$(echo $output | grep $ray_name | grep -i failed)
  succeeded=$(echo $output | grep $ray_name | grep -i succeeded)
  gcs_error=$(echo $output | grep -i 'Failed to get cluster ID from GCS server')

  if [[ -n $gcs_error ]]; then
    echo "ray cannot connect，Job $ray_name exit with exception"
    ray stop --force
   # rm -rf /tmp
    exit 1
  fi


  if [[ -n $succeeded ]]; then
    ray stop --force
 #   rm -rf /tmp
    echo "Job $ray_name exit without exception"
    exit 0
  fi

  if [[ -n $failed ]]; then
    echo "Job $ray_name exit with exception"
    ray stop --force
#    rm -rf /tmp
    exit 1
  fi

  sleep 10
done