export HCCL_SOCKET_IFNAME=bond1 # modify according to actual situation
export TP_SOCKET_IFNAME=bond1   # modify according to actual situation
export GLOO_SOCKET_IFNAME=bond1 # modify according to actual situation
# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=1
# export HCCL_EXEC_TIMEOUT=3600
# export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
unset PYTORCH_NPU_ALLOC_CONF
export ASCEND_GLOBAL_LOG_LEVEL=3


#! 注意，自定义配置
# * 确保 JOB_LOG_DIR 在共享盘下
export JOB_LOG_DIR=/home/code/logs/$(basename $(dirname $0))
export JOB_LOG_DIR_CURR=${JOB_LOG_DIR}/$(date +"%Y-%m-%d_%H")
export ASCEND_PROCESS_LOG_PATH=${JOB_LOG_DIR_CURR}/plog/${CURRENT_IP}
export VERL_MEMORY_LOG_DIR=${JOB_LOG_DIR_CURR}/memory_log
# export VERL_CUSTOM_Profiling_DIR=${JOB_LOG_DIR_CURR}/prof
# export VERL_CUSTOM_SNAPSHOT=0

export CACHE_DIR=${JOB_LOG_DIR}/CACHE; mkdir -p ${CACHE_DIR}
export ACL_OP_COMPILER_CACHE_DIR=${CACHE_DIR}/COMPILER_CACHE/${CURRENT_IP}; mkdir -p ${ACL_OP_COMPILER_CACHE_DIR}
export VERL_CUSTOM_REWARD_RULE="1"
export VERL_CUSTOM_SYNCHRONIZE="1"

#! 注意，0929加了这 1 个优化参数， libjemalloc 需要重新编译
# export LD_PRELOAD="/usr/local/lib/libjemalloc.so.2"
export TASK_QUEUE_ENABLE=2
# export TORCHELASTIC_USE_AGENT_STORE=true

#! 注意，HCCL 相关配置
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export P2P_HCCL_BUFFSIZE=30
export HCCL_BUFFSIZE=300

#! 注意，1003 加了这 几个超时配置
export RAY_DEBUG_POST_MORTEM=1
# export ASCEND_LAUNCH_BLOCKING=1

CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

#! #################  【VLLM 0.10.0 patch】  #####################
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

source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;
source /opt/pyvenv/bin/activate;

LIB_PATH=/opt/python3.10/lib/
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH

unset LOCAL_WORLD_SIZE
# unset WORLD_SIZE
unset LOCAL_RANK

export NPU_PER_NODE=16  # A2 NPU Number
export NNODES=$((WORLD_SIZE/NPU_PER_NODE))         # example is 4 Nodes


rm -rf /tmp/ray
ray stop --force
sleep 1
echo "Overwrite verl code"
#* 提速 ray 拉起速度
if [[ -f /home/code/verl/docker/pkg/rsync ]];then
   /home/code/verl/docker/pkg/rsync -az /home/code/verl/* /opt/verl/ --exclude=**/kernel_meta --exclude=plog --exclude=docker --exclude=docs
else
  unalias cp
  cp -rf /home/code/verl/* /opt/verl/
fi
echo "Overwrite verl code, done."

rm -f /opt/verl/.gitignore
cd $(dirname $0)


export ServerPort=6666     # modify according to actual situation
export DashboardPort=8888  # modify according to actual situation


echo "Manul start !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
cnt=0
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

  ray start --head --ray-debugger-external --port $ServerPort --dashboard-port=$DashboardPort --node-ip-address=$CURRENT_IP --dashboard-host=$CURRENT_IP --disable-usage-stats

  while [[ $cnt -lt 100 ]]; do
    ray_status_output=$(ray status)
    npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
    npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')

    # judge npu_count_int bigger than NNODES*NPU_PER_NODE
    if [ "$npu_count_int" -ge "$((NNODES*NPU_PER_NODE))" ]; then
      echo "Ray cluster is ready with $npu_count_int npu (from $npu_count NPU resources), starting Python script."
      bash hw_run_dapo_qwen3-30b_megatron.sh
      break
    fi

    echo "Waiting for Ray to allocate $((NNODES*NPU_PER_NODE)) devices. Current device count: $npu_count_int"
    cnt=$((cnt+1))
    sleep 10
  done

else
  echo "This is worker node"
  sleep 10
  ray start --address="$MASTER_ADDR:$ServerPort" --disable-usage-stats
fi

# start Mark 1

