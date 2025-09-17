kubectl apply -f verl/k8s/32nodes/acjob_deepseek671b_megatron.yaml -n rein-learing 

kubectl get pods -n rein-learing

kubectl logs deepseek671-32-verl-moe-verl-master-0 -n rein-learing -f

kubectl delete -f verl/k8s/32nodes/acjob_deepseek671b_megatron.yaml -n rein-learing 


