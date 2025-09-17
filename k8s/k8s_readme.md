# 通用部分

watch kubectl get pods -n rein-learing

tail -f /tmp/ray/session_latest/logs/job-driver

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

