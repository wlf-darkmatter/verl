import torch
model_aftersharding_rank0=torch.load('/data01/huawei-2025/wlf/verl/k8s/beijing/32nodes_0925_sleep_save/data_weight/model_aftersharding_rank0.pt')
model_nosharding_rank0=torch.load('/data01/huawei-2025/wlf/verl/k8s/beijing/32nodes_0925_sleep_save/data_weight/model_nosharding_rank0.pt')
for key,value in model_aftersharding_rank0.item():
    print(key,value.sum())
