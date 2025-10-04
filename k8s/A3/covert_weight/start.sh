export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;
source /opt/pyvenv/bin/activate;


# cp /data01/huawei-2025/lq/convert/4nodes/bash_dpsk_v3_4nodes.sh /home/new_verl/scripts/
# cp /data01/huawei-2025/lq/convert/4nodes/converter_hf_to_mcore_n_node.py /home/new_verl/scripts/
# cd /home/new_verl/scripts/
rm -rf /opt/verl
cp -r /home/new_verl /opt/verl
cd $(dirname $0)

#NODE_RANK=$1
export WORLD_SIZE=2
#export RANK=0
#export MASTER_ADDR=10.127.18.192
MASTER_PORT=${MASTER_PORT:-6000}
export HCCL_SOCKET_IFNAME="bond1"
export TP_SOCKET_IFNAME="bond1"
export GLOO_SOCKET_IFNAME="bond1"

export HCCL_EXEC_TIMEOUT="7200"
export HCCL_CONNECT_TIMEOUT="7200"
export HCCL_IF_BASE_PORT="23999"
export HCCL_ASYNC_ERROR_HANDLING="0"
export P2P_HCCL_BUFFSIZE="20"
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
cd /opt/verl
torchrun --nproc_per_node 16 --nnodes 2 --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    /opt/verl/scripts/converter_hf_to_mcore_n_node.py \
    --hf_model_path /mnt/hpfs_test/weights/dsv3-bf16 \
    --output_path /mnt/hpfs_test/weights/dsv3_bf16_mcore_hs \
    --pp_size 4 --ep_size 8
