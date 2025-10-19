# 通用部分



watch kubectl get pods -n rein-learing

tail -f /tmp/ray/session_latest/logs/job-driver

## 监控内存打印

```bash
export VERL_MEMORY_LOG_DIR=${JOB_LOG_DIR}/memory_log
```

网页分析使用关键词：

```python
["Total:", "Allocated:", "Reserved:", "Cached:"]
```

git clone https://github.com/HuangShiqing/memory_viz_plus.git
cd memory_viz_plus
python ./_memory_viz.py trace_plot mem_snapshot.pickle -o mem_snapshot -p 1

## 观测指标

```python
["critic/rewards/mean:","actor/grad_norm:", "actor/kl_loss:", "response_length/mean:", "response_length/clip_ratio:"]
["critic/rewards/mean:","actor/grad_norm:", "actor/pg_loss:", "response_length/mean:", "response_length/clip_ratio:"]

```

# 正式执行部分

## start

kubectl apply -f verl/k8s/32nodes/acjob_deepseek671b_megatron.yaml -n rein-learing
kubectl apply -f verl/k8s/test/acjob_test_sleep.yaml -n rein-learing

## delete

kubectl delete -f verl/k8s/32nodes/acjob_deepseek671b_megatron.yaml -n rein-learing

## attach

kubectl exec -it deepseek671-32-verl-moe-verl-master-0 bash -n rein-learing

kubectl exec -it deepseek671-32-verl-moe-verl-worker-0 bash -n rein-learing

## Check status

kubectl describe pod deepseek671-32-verl-moe-verl-master-0 -n rein-learing

kubectl logs deepseek671-32-verl-moe-verl-master-0 -n rein-learing -f

# 测试部分

## start

kubectl apply -f verl/k8s/test/acjob_test_sleep.yaml -n rein-learing

## delete

kubectl delete -f verl/k8s/test/acjob_test_sleep.yaml -n rein-learing

## 其他

kubectl logs deepseek-sleep-master-0 -n rein-learing -f
kubectl logs deepseek-sleep-worker-0 -n rein-learing -f

kubectl exec -it deepseek-sleep-master-0 bash -n rein-learing
kubectl exec -it deepseek-sleep-worker-0 bash -n rein-learing


# 其他测试

## 测试连通性

```bash
export ServerPort=6666
export DashboardPort=8888
export HCCL_SOCKET_IFNAME=ens45
export TP_SOCKET_IFNAME=ens45
export GLOO_SOCKET_IFNAME=ens45

if [ "$RANK" = "0" ]; then
  kwargs=(--is_master --ray_dashboard_port ${DashboardPort})
else
  kwargs=()
fi


cd /home/code/verl; python k8s/test_comm.py --ray_init --nnodes=2 --ray_master_ip=${MASTER_ADDR} --ray_master_port=${ServerPort} --device=npu ${kwargs[@]}

```

## 共享盘上安装CANN 8.3.RC1

```bash
chmod -R 755
bash Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run -q --full --install-path=/data01/huawei-2025/CANN/8.3.RC1  ;\

source /data01/huawei-2025/CANN/8.3.RC1/ascend-toolkit/set_env.sh
bash Ascend-cann-kernels-910b_8.3.RC1_linux-aarch64.run -q --install --install-path=/data01/huawei-2025/CANN/8.3.RC1

bash Ascend-cann-nnal_8.3.RC1_linux-aarch64.run -q --install --install-path=/usr/local/Ascend ;\

```