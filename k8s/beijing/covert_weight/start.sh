export HCCL_SOCKET_IFNAME=ens45 # modify according to actual situation
export TP_SOCKET_IFNAME=ens45   # modify according to actual situation
export GLOO_SOCKET_IFNAME=ens45 # modify according to actual situation
# export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh;
source /usr/local/Ascend/nnal/atb/set_env.sh;
source /opt/pyvenv/bin/activate;


# cp /data01/huawei-2025/lq/convert/4nodes/bash_dpsk_v3_4nodes.sh /home/code/verl/scripts/
# cp /data01/huawei-2025/lq/convert/4nodes/converter_hf_to_mcore_n_node.py /home/code/verl/scripts/
# cd /home/code/verl/scripts/
rm -rf /opt/verl
cp -r /home/code/verl /opt/verl
cd $(dirname $0)
bash bash_dpsk_v3_4nodes.sh
