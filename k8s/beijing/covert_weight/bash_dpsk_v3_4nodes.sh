#NODE_RANK=$1
export WORLD_SIZE=2
#export RANK=0
#export MASTER_ADDR=10.127.18.192
MASTER_PORT=${MASTER_PORT:-6000}
export HCCL_SOCKET_IFNAME="ens45"
export TP_SOCKET_IFNAME="ens45"
export GLOO_SOCKET_IFNAME="ens45"
export HCCL_EXEC_TIMEOUT="7200"
export HCCL_CONNECT_TIMEOUT="7200"
export HCCL_IF_BASE_PORT="23999"
export HCCL_ASYNC_ERROR_HANDLING="0"
export P2P_HCCL_BUFFSIZE="20"
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
cd /opt/verl
torchrun --nproc_per_node 8 --nnodes 4 --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT /opt/verl/scripts/converter_hf_to_mcore_n_node.py --hf_model_path /data01/huawei-2025/weight/dsv3-base-hf --output_path /data01/huawei-2025/weight/dsv3_fp16_mcore_full_base --pp_size 4 --ep_size 8
