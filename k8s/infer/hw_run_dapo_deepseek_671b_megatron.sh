#!/usr/bin/env bash
set -x pipefail
export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
export HYDRA_FULL_ERROR=1
CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

bash /data01/huawei-2025/wlf/verl/k8s/script/watch_stats.sh > /data01/huawei-2025/wlf/watch/rank${RANK}_${CURRENT_IP}.log &

source /usr/local/Ascend/driver/bin/setenv.bash;
source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;
source /opt/pyvenv/bin/activate;
source /etc/profile;
LIB_PATH=/opt/python3.10/lib/
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH

unset LOCAL_WORLD_SIZE
unset WORLD_SIZE
unset LOCAL_RANK

export NPU_PER_NODE=8  # A2 NPU Number
export NNODES=8         # example is 4 Nodes

export path_log_dir=/opt/verl/logs/$MINDX_TASK_ID/trainlog  # modify according to actual situation
export ASCEND_PROCESS_LOG_PATH=/opt/verl/logs/$MINDX_TASK_ID/plog # modify according to actual situation


ray stop --force
rm -rf /tmp

export ServerPort=6666     # modify according to actual situation
export DashboardPort=8888  # modify according to actual situation

#! ------------------------------------------------------------

n_resp_per_prompt=16
#NNODES=${NNODES:-1}
NNODES=8
gen_tp=8
gen_dp=8

MODEL_PATH="/data01/nlp/dpsk-v3-671B-BF16-dist_ckpt"

TRAIN_FILE="/data01/huawei-2025/xczhao/rl_data/dapo-math/dapo-math-17k.parquet"

CURRENT_IP=$1


cd /home/new_verl
if [ "$RANK" = "0" ]; then
  # head start
  echo "This is head node"
  echo "CURRENT_IP=$CURRENT_IP"

  kwargs=(--is_master --ray_dashboard_port $DashboardPort)
else
  kwargs=()
fi

python3 tests/verl_offline_infer.py \
    --ray_master_ip $CURRENT_IP \
    --ray_master_port $ServerPort \
    -tp $gen_tp \
    -dp $gen_dp \
    --enable_expert_parallel \
    -n $n_resp_per_prompt \
    --gen_bs $train_prompt_bsz \
    --max_prompt_length $((2*1024)) \
    --max_response_length $((5)) \
    --max_num_batched_tokens $((5)) \
    --n_gpus_per_node ${NPU_PER_NODE} \
    --dataset_path $TRAIN_FILE \
    --hdfs_path $MODEL_PATH \
    --gpu_memory_utilization 0.60 \
    --nnodes $NNODES ${kwargs[@]} $@
