# Node:1
# torchrun --master_addr="90.90.122.117" --master_port=20122 --nnodes=2 --node_rank=0 --nproc_per_node=8 recipe/moonlight_NPU/test_allreduce.py
# Node:2
# torchrun --master_addr="90.90.122.117" --master_port=20122 --nnodes=2 --node_rank=1 --nproc_per_node=8 recipe/moonlight_NPU/test_allreduce.py

import torch
import torch_npu
import os
device = int(os.getenv('LOCAL_RANK'))
print(device)
torch.npu.set_device(device)
# Call the hccl init process
torch.distributed.init_process_group(backend='hccl', init_method='env://')
a = torch.tensor(1).npu()
torch.distributed.all_reduce(a)
print('Hccl: ', a, device, flush=True)

