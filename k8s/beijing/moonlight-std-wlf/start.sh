cwd=$(dirname $(realpath $0))
echo "dealing $cwd"
cd $cwd

export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=1
# export HCCL_EXEC_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
export ASCEND_GLOBAL_LOG_LEVEL=3
# export USE_SEED=1234
export VLLM_FIX_WEIGHT_LOADING=1


export HCCL_IF_BASE_PORT="14999"

echo "Overwrite verl code"
if [[ ! -d ../../../k8s ]];then
  echo -e "\033[1;31m路径层级不对，请确保 start.sh 在 verl/k8s/???/???/ 下\033[0m"
  exit 1
fi

#! 如果有rsync命令
if [[ -x "$(command -v rsync)" ]];then
  echo "rsync exists"
  rsync -az ../../../* /opt/verl/ --exclude=**/kernel_meta --exclude=plog --exclude=docker
else
  echo "rsync not exists"
  \cp -r ../../../* /opt/verl/
fi
rm -f /opt/verl/.gitignore

echo "Overwrite verl code, done."
#! 注意，自定义配置
# * 确保 JOB_LOG_DIR 在共享盘下
REAL_HOME_CODE=$(realpath $cwd/../../../..)
CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
export JOB_LOG_DIR=$REAL_HOME_CODE/logs/$(basename $cwd)
export JOB_LOG_DIR_CURR=${JOB_LOG_DIR}/$(date +"%Y-%m-%d_%H")
export ASCEND_PROCESS_LOG_PATH=${JOB_LOG_DIR_CURR}/plog/${CURRENT_IP}

#! 注意，0929加了这 1 个优化参数， libjemalloc 需要重新编译
# export LD_PRELOAD="/usr/local/lib/libjemalloc.so.2"
export TASK_QUEUE_ENABLE=2

#! 注意，HCCL 相关配置
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export P2P_HCCL_BUFFSIZE=30
export HCCL_BUFFSIZE=300

#! #################  【VLLM patch】  #####################
#! 规避模型加载时 权重读取错误的问题
bash /opt/verl/k8s/patch/apply_vllm-ascend.sh

#! #################  【Megatron patch】  #####################
#! [Megatron]
bash /opt/verl/k8s/patch/apply_megatron.sh

#! #################  【MindSpeed patch】  #####################
#! [MindSpeed]
bash /opt/verl/k8s/patch/apply_mindspeed.sh


#######################################

source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;
source /opt/pyvenv/bin/activate;

#! #################  【MindSpeed 预编译】  #####################
bash /opt/verl/k8s/patch/pre_mindspeed_compile.sh


LIB_PATH=/opt/python3.10/lib/
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH

pip install /data01/huawei-2025/gxj/blobfile-3.0.0-py3-none-any.whl --no-deps
pip install /data01/huawei-2025/gxj/lxml-6.0.2-cp310-cp310-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl --no-deps


unset LOCAL_WORLD_SIZE
# unset WORLD_SIZE
unset LOCAL_RANK

export NPU_PER_NODE=8
export NNODES=$((WORLD_SIZE/NPU_PER_NODE))


export ServerPort=6666     # modify according to actual situation
export DashboardPort=8888  # modify according to actual situation

cd $cwd
cnt=0
if [ "$RANK" = "0" ]; then
  # head start
  echo "This is head node"
  mkdir -p ${JOB_LOG_DIR_CURR}
  mkdir -p ${JOB_LOG_DIR_CURR}/ray_host
  echo "CURRENT_IP=$CURRENT_IP"
#   ln -s ${JOB_LOG_DIR_CURR}/ray_host /tmp/ray
  #* 拷贝当前脚本文件
  mkdir -p ${JOB_LOG_DIR_CURR}/script.bak
  cp $cwd/*.sh ${JOB_LOG_DIR_CURR}/script.bak/
  cp $cwd/*.yaml ${JOB_LOG_DIR_CURR}/script.bak/

  ray start --head --ray-debugger-external --port $ServerPort --dashboard-port=$DashboardPort --node-ip-address=$CURRENT_IP --dashboard-host=$CURRENT_IP --disable-usage-stats

  while [[ $cnt -lt 100 ]]; do
    ray_status_output=$(ray status)
    npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
    npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')

    # judge npu_count_int bigger than NNODES*NPU_PER_NODE
    if [ "$npu_count_int" -ge "$((NNODES*NPU_PER_NODE))" ]; then
      echo "Ray cluster is ready with $npu_count_int npu (from $npu_count NPU resources), starting Python script."
      bash $cwd/hw_run_dapo_deepseek_671b_megatron.sh | tee ${JOB_LOG_DIR_CURR}/ray_host/$(date +"%Y-%m-%d_%H-%M-%S")_ray.log
      break
    fi

    echo "Waiting for Ray to allocate $((NNODES*NPU_PER_NODE)) devices. Current device count: $npu_count_int"
    cnt=$((cnt+1))

  done

else
  echo "This is worker node"
  sleep 10
  ray start --address="$MASTER_ADDR:$ServerPort" --disable-usage-stats
fi

# start Mark 1

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
    sleep 10

    # rm -rf /tmp
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

#   if [[ -n $failed ]]; then
#     echo "Job $ray_name exit with exception"
#     ray stop --force
# #    rm -rf /tmp
#     exit 1
#   fi

  sleep 10
done