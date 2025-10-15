export HCCL_SOCKET_IFNAME=bond1 # modify according to actual situation
export TP_SOCKET_IFNAME=bond1   # modify according to actual situation
export GLOO_SOCKET_IFNAME=bond1 # modify according to actual situation
# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
# export HCCL_EXEC_TIMEOUT=3600
export ASCEND_GLOBAL_LOG_LEVEL=3
CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

#######################################
#! 规避模型加载时 权重读取错误的问题
rm -f /opt/vllm/vllm/model_executor/model_loader/base_loader.py
cp -f /home/code/verl/k8s/patch/0827/base_loader.py /opt/vllm/vllm/model_executor/model_loader/base_loader.py

rm -f /opt/vllm/vllm/model_executor/models/deepseek_v2.py
cp -f /home/code/verl/k8s/patch/0827/deepseek_v2.py /opt/vllm/vllm/model_executor/models/deepseek_v2.py

rm -f /opt/vllm-ascend/vllm_ascend/ops/fused_moe.py
cp -f /home/code/verl/k8s/patch/0827/vllm_ascend/ops/fused_moe.py /opt/vllm-ascend/vllm_ascend/ops/fused_moe.py

#! [Megatron]
rm -f /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py
cp -f /home/code/verl/k8s/patch/0827/megatron/dot_product_attention.py /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py
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
export NNODES=2         # example is 4 Nodes
export VLLM_USE_V1=1

ray stop --force
sleep 10
rm -rf /tmp/ray
rm -rf /opt/verl
cp -r /home/code/verl /opt/verl
cd $(dirname $0)


export ServerPort=6666     # modify according to actual situation
export DashboardPort=8888  # modify according to actual situation


cd /opt/vllm
if [[ "$RANK" = "0" ]]; then
  # head start
  echo "This is head node"
  mkdir -p ${JOB_LOG_DIR_CURR}
  mkdir -p ${JOB_LOG_DIR_CURR}/ray_host
  echo "CURRENT_IP=$CURRENT_IP"
  ln -s ${JOB_LOG_DIR_CURR}/ray_host /tmp/ray

  kwargs=(--is_master --ray_dashboard_port $DashboardPort )
else
  kwargs=( )
fi

#! ------------------------------------------------------------
# TRAIN_FILE="/mnt/hpfs_test/data/rl_data/dapo-math-17k_dedup_r1_sys_prompt_mathdapo.parquet"
# MODEL_PATH="/data01/huawei-2025/weight/dpsk-v3-671B-BF16-dist_ckpt"
MODEL_PATH="/mnt/hpfs_test/weights/dsv3-base-fp8-zy-bf16"
TRAIN_FILE="/mnt/hpfs_test/data/data/dapo-math-17k.parquet"

python examples/offline_inference/data_parallel.py \
                    --model="/mnt/hpfs_test/weights/dsv3-base-fp8-wlf-bf16" \
                    --dp-size=2 \
                    --tp-size=16 \
                    --node-size=2 \
                    --node-rank=1 \
                    --master-addr=172.16.2.74 \
                    --master-port=6666