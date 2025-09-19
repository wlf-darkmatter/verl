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
list_fail=()

if [[ $1 == "log" ]];then
    rm -rf ${path_logs}/*
fi

for i in $(seq 0 $((n_group-1))); do
    rm -f ${yaml_prefix_name}${i}.yaml
    cp ${yaml_init} ${yaml_prefix_name}${i}.yaml
    echo Target-$i
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
        if [[ $num_ok -gt 0 ]];then
            echo -e "\033[32mOK\033[0m"
            list_success+=($i)
        else
            echo -e "\033[31mFAIL\033[0m"
            list_fail+=($i)
        fi

    fi

done


if [[ $1 == "log" ]];then
    #! 统计结果
    echo ${list_success[@]}
    ip_list=()
    for i in ${list_success[@]}; do
        # 获取IP
        ip_list+=($ip)
        for j in $(seq 0 $((n_worker-1))); do
            ip=$(kubectl get pods -o wide -n rein-learing | grep ${server_prefix_name}${i}-worker-${j} |awk '{print $6}')
            ip_list+=($ip)
        done

    done
    echo "【OK IP list】"
    echo ${ip_list[@]}

    ip_list=()
    for i in ${list_fail[@]}; do
        # 获取IP
        ip_list+=($ip)
        for j in $(seq 0 $((n_worker-1))); do
            ip=$(kubectl get pods -o wide -n rein-learing | grep ${server_prefix_name}${i}-worker-${j} |awk '{print $6}')
            ip_list+=($ip)
        done

    done

    echo "【Failed IP list】"
    echo ${ip_list[@]}

fi

