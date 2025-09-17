export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
export HYDRA_FULL_ERROR=1

CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

source /usr/local/Ascend/driver/bin/setenv.bash;
source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;
source /usr/local/Ascend/nnal/asdsip/set_env.sh;
source /opt/pyvenv/bin/activate;

source /etc/profile;

LIB_PATH=/opt/python3.10/lib/
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH
export NPU_PER_NODE=8  # A2 NPU Number

export NNODES=$((WORLD_SIZE//NPU_PER_NODE))         # example is 4 Nodes

cd /home/new_verl
if [ "$RANK" = "0" ]; then
  # head start
  echo "This is head node"
  echo "CURRENT_IP=$CURRENT_IP"

  kwargs=(--is_master)

else
  kwargs=()

fi

python tests/test_comm.py ${kwargs[@]} --ray_master_ip=$MASTER_ADDR --device=npu
