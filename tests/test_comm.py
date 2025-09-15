
import argparse
import ray
import os
import torch
import socket
from verl.single_controller.ray.base import sort_placement_group_by_node_ip
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from verl.single_controller.ray import RayResourcePool
from verl.utils.device import get_device_name, get_nccl_backend
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--nnodes", type=int, default=1)
parser.add_argument("--n_gpus_per_node", type=int, default=8)
parser.add_argument("--ray_master_ip", type=str, default=None)
parser.add_argument("--ray_master_port", type=int, default=6379)
parser.add_argument("--ray_dashboard_port", type=int, default=8265)
args = parser.parse_args()

def get_availale_curr_addr_port():
    host_ip_by_sdk = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        free_port = sock.getsockname()[1]

    return host_ip_by_sdk, free_port


def ray_init():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["VLLM_LOGGING_LEVEL"] = "WARN"
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"
    os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"

    if args.nnodes > 1:
        if args.ray_master_ip is None:
            raise RuntimeError(f"`--ray_master_ip` should be set if nnodes({args.nnodes}) > 1.")

        curr_addr, _ = get_availale_curr_addr_port()
        print(curr_addr, flush=True)

        if curr_addr == args.ray_master_ip:
            pass
            print("\033[32mMaster\033[0m", flush=True)
            ret = os.popen(f"ray start --head --port {args.ray_master_port}").read()
        else:
            print("\033[32mSlaver\033[0m", flush=True)
            ret = os.popen(f"ray start --address={args.ray_master_ip}:{args.ray_master_port}").read()
            exit(0)
        print(ret, flush=True)

    if not ray.is_initialized():
        ray.init()

class TestComm:

    def __init__(self):
        pass

    def test_allreduce(self):
        pass

def build_task(task_cls, config=None, device_name=None):
    if device_name is None:
        use_gpu = False
    else:
        use_gpu = True

    n_gpus_per_node = int(args.n_gpus_per_node)
    nnodes = int(args.nnodes)
    resource_pool = RayResourcePool(process_on_nodes=[n_gpus_per_node] * nnodes, max_colocate_count=1, use_gpu=use_gpu)

    strategy = "PACK"
    pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=device_name)
    world_size = resource_pool.world_size
    local_world_size = resource_pool.store[0]

    rank = -1
    tasks = []
    master_addr = None
    master_port = None
    for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
        for local_rank in range(local_world_size):
            rank += 1

            if rank == 0:
                master_addr, master_port = get_availale_curr_addr_port()
                info = {
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                }

            actor_name = f"vllm-{rank}"
            env_vars = {
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "WG_BACKEND": "ray",
                "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
                "RAY_LOCAL_RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "NODE_RANK": str(pg_idx),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
            }

            options = {
                "runtime_env": {"env_vars": env_vars},
                # "resources": {"CPU": 1}, #* NPU:1
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=local_rank,
                ),
                "name": actor_name,
            }
            # task = Vllm_Worker(info, config)
            # task = Vllm_Worker.remote(info, config)
            task = task_cls.options(**options).remote(info, config)
            tasks.append(task)
    return tasks
def pool_exec(list_task, func_name, *args, **kwargs):
    task_runnint_list = []
    for i, task_i in enumerate(list_task):
        task_runnint_list.append(task_i.ray_exec.remote(func_name, *args, **kwargs))
    list_output = ray.get(task_runnint_list)
    return list_output


class BasrRay:
    def __init__(self, rank_zero_info, config=None):
        pass
        self.rank_zero_info = rank_zero_info
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ["RANK"])
    def ray_exec(self, func_name, *args, **kwargs):
        return getattr(self, func_name)(*args, **kwargs)
    def get_attr(self, key):
        return getattr(self, key)
    def print_rank(self):
        print(f"Rank: {self.rank}")

@ray.remote
class TestComm(BasrRay):
    pass
    def __init__(self, rank_zero_info, config=None):
        super().__init__(rank_zero_info, config)

    def init_process_group(self):

        dist.init_process_group(
                backend=f"cpu:gloo",
                # backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=self.rank,
                world_size=self.world_size,

                                )
        print(f"建联完成")
        return dist.get_rank(), dist.get_world_size()

    def test_allreduce(self):
        tensor_size = 10000
        # 创建测试张量
        tensor = torch.ones(tensor_size, dtype=torch.float32) * self.rank
        print(f"Testing AllReduce with tensor size: {tensor_size}")
        for _ in range(2):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # 同步所有进程
        dist.barrier()


if __name__ == "__main__":
    ray_init()
    list_task = build_task(TestComm, device_name=None)

    pool_exec(list_task, "print_rank")
    pool_exec(list_task, "init_process_group")
    pool_exec(list_task, "test_allreduce")
    pass

