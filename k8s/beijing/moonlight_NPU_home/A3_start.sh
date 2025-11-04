#################### 网络配置 ###########################
#修改为对应主节点IP
# export MASTER_ADDR=90.90.122.117
export MASTER_ADDR=90.90.97.77

# 修改为当前节点的通信网卡
# SOCKET_IFNAME="enp189s0f0"
SOCKET_IFNAME="enp194s0f0"
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME   # NPU？
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME

#################### Log 目录配置 ###########################
# * 确保 JOB_LOG_DIR 在共享盘下
CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
export JOB_LOG_DIR=/data/logs
export JOB_LOG_DIR_CURR=${JOB_LOG_DIR}/$(date +"%Y%m%d_%H%M%S")
export ASCEND_PROCESS_LOG_PATH=${JOB_LOG_DIR_CURR}/plog/${CURRENT_IP}

####################   环境设置    #######################
DEFAULT_SH="/home/code/verl-gpu/k8s/beijing/moonlight_NPU_home/run.sh"

pkill -9 python

# 激活python环境
source /opt/pyvenv/bin/activate;
LIB_PATH=/opt/python3.10/lib/
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH:/home/code/verl-gpu/docker/pkg/

# Cann环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;

# ray重启
ray stop --force
rm -rf /tmp/ray

# 安装新的package
# pip install /opt/verl/recipe/moonlight_NPU/pystack-1.5.1-cp310-cp310-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl
pip install /home/code/verl-gpu/k8s/beijing/moonlight_NPU_home/blobfile-3.1.0-py3-none-any.whl --no-deps
pip install /home/code/verl-gpu/k8s/beijing/moonlight_NPU_home/lxml-6.0.2-cp310-cp310-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl --no-deps

# 机器环境变量
export NNODES=1
export NPUS_PER_NODE=8
export GPUS_PER_NODES=$NPUS_PER_NODE

# Debug相关的环境变量
ulimit -n 32768
export RAY_DEBUG=1  # 允许ray debug
export RAY_DEBUG_POST_MORTEM=1
export RAY_DEDUP_LOGS=1  # Ray 日志去重
export HYDRA_FULL_ERROR=1
export ASCEND_GLOBAL_LOG_LEVEL=3  # 3：error级？0：debug级？

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

export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"

#! #################  【VLLM patch】  #####################
#! 规避模型加载时 权重读取错误的问题
bash /home/code/verl-gpu/k8s/patch/apply_vllm-ascend.sh

#! #################  【Megatron patch】  #####################
#! [Megatron]
bash /home/code/verl-gpu/k8s/patch/apply_megatron.sh

#! #################  【MindSpeed patch】  #####################
#! [MindSpeed]
bash /home/code/verl-gpu/k8s/patch/apply_mindspeed.sh


####################   拷贝代码   ###################

sleep 1
echo "Overwrite verl code"
#* 提速 ray 拉起速度
if [[ -f /home/code/verl-gpu/docker/pkg/rsync ]];then
   /home/code/verl-gpu/docker/pkg/rsync -az /home/code/verl-gpu/* /opt/verl/ --exclude=**/kernel_meta --exclude=plog --exclude=docker --exclude=docs --exclude=model_ckpts --exclude=logs
else
  rm -rf /opt/verl/
  cp -rf /home/code/verl-gpu /opt/verl
fi
echo "Overwrite verl code, done."

rm -f /opt/verl/.gitignore
cd $(dirname $0)

#######################################
# 获取当前节点IP
echo $CURRENT_IP
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 备份脚本
  mkdir -p $JOB_LOG_DIR_CURR
  cp $(dirname $0)/A3_start.sh "${JOB_LOG_DIR_CURR}/."
  cp $(dirname $0)/A3_main.sh "${JOB_LOG_DIR_CURR}/."
  cp $(dirname $0)/run.sh "${JOB_LOG_DIR_CURR}/."

  # 主节点启动
  ray start --head --port 4918 --dashboard-host="0.0.0.0" --node-ip-address=$CURRENT_IP --dashboard-port=4919 --disable-usage-stats

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*(NPU|GPU))' | head -n 1)pwd
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -ge "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU/GPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start --address="$MASTER_ADDR:4918" --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

echo "start.sh ended on ${CURRENT_IP}"
