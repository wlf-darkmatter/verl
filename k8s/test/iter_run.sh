cd $(dirname $0)

nnode=2

n_worker=$((nnode-1))
n_group=$((48/nnode))

path_logs=/data01/huawei-2025/wlf/test_comm_logs
mkdir -p $path_logs

yaml_prefix_name=acjob_test_tmp_
yaml_init=acjob_test_sleep.yaml
server_prefix_name=deepseek-sleep-
# server_prefix_name=deepseek-sleep-new-

# #* check ip
# for i in 11 9 22 4 1 17 7 12 5 2 14 13 20 15; do
#     echo $i
#     # kubectl delete -f ${yaml_prefix_name}${i}.yaml -n rein-learing
#     kubectl get pods -o wide -n rein-learing | grep "deepseek-sleep-0-master-0" |awk '{print $6}'
# done
list_success=()
if [[ $1 == "log" ]];then
    rm -rf ${path_logs}/*
fi

for i in $(seq 0 $((n_group-1))); do
    rm -f ${yaml_prefix_name}${i}.yaml
    cp ${yaml_init} ${yaml_prefix_name}${i}.yaml
    echo $i
    sed -i "s/name: deepseek-sleep.*/name: ${server_prefix_name}${i}/g" ${yaml_prefix_name}${i}.yaml
    sed -i "s/- deepseek-sleep.*/- ${server_prefix_name}${i}/g" ${yaml_prefix_name}${i}.yaml

    if [[ $1 == "start" ]];then
        sed -i "s/minAvailable.*/minAvailable: ${nnode}/g" ${yaml_prefix_name}${i}.yaml
        kubectl apply -f ${yaml_prefix_name}${i}.yaml -n rein-learing

    fi

    if [[ $1 == "stop" ]];then
        kubectl delete -f ${yaml_prefix_name}${i}.yaml -n rein-learing
        rm -f ${yaml_prefix_name}${i}.yaml
    fi

    if [[ $1 == "log" ]];then
        kubectl logs ${server_prefix_name}${i}-master-0 -n rein-learing > ${path_logs}/log_${i}.log
        num_ok=$(cat ${path_logs}/log_${i}.log | grep -c 建链完成)
        echo $num_ok
        if [[ $num_ok -lt 0 ]];then
            list_success+=($i)
        fi

    fi

done


echo ${list_success[@]}

