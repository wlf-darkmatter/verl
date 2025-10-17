set -x
export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
export HYDRA_FULL_ERROR=1

CURRENT_IP=$(ifconfig ${TP_SOCKET_IFNAME} | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

source /usr/local/Ascend/driver/bin/setenv.bash;
source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;
source /usr/local/Ascend/nnal/asdsip/set_env.sh;
source /opt/pyvenv/bin/activate;

source /etc/profile;

LIB_PATH=/opt/python3.10/lib/
export LD_LIBRARY_PATH=${LIB_PATH}:${LD_LIBRARY_PATH}


export NPU_PER_NODE=8  # A2 NPU Number
export ServerPort=6666     # modify according to actual situation
export DashboardPort=8888  # modify according to actual situation
export NNODES=$((WORLD_SIZE/NPU_PER_NODE))

unset LOCAL_WORLD_SIZE
unset WORLD_SIZE
unset LOCAL_RANK

cd /home/code/verl
if [ "$RANK" = "0" ]; then
  # head start
  echo "This is head node"
  mkdir -p ${JOB_LOG_DIR_CURR}
  mkdir -p ${JOB_LOG_DIR_CURR}/ray_host
  echo "CURRENT_IP=$CURRENT_IP"
  ln -s ${JOB_LOG_DIR_CURR}/ray_host /tmp/ray

  kwargs=(--is_master --ray_dashboard_port ${DashboardPort})
else
  kwargs=()
  sleep 30
fi

ray stop --force
sleep 10

python k8s/test_comm.py \
    --ray_init \
    --nnodes=${NNODES} \
    --ray_master_ip=${MASTER_ADDR} \
    --ray_master_port=${ServerPort} \
    --device=npu \
    ${kwargs[@]}


