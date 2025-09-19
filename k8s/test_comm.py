import argparse
import os
import socket
import datetime
from functools import partial

import ray
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.base import sort_placement_group_by_node_ip
from verl.utils.device import get_device_name, get_nccl_backend


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VLLM_LOGGING_LEVEL"] = "WARN"
os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"
os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument("--nnodes", type=int, default=1)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--n_gpus_per_node", type=int, default=8)
parser.add_argument(
    "--is_master", action="store_true", help="直接设置当前机器为 master"
)
parser.add_argument(
    "--ray_master_ip",
    type=str,
    default=None,
    help="会自动判断 gloo 网卡的 ip 是否和指定的这个 ip 一致，一致则认为是 master",
)
parser.add_argument("--ray_master_port", type=int, default=6379)
parser.add_argument("--ray_dashboard_port", type=int, default=8265)
parser.add_argument("--ray_init", action="store_true")
args = parser.parse_args()


def get_availale_curr_addr_port():
    host_ip_by_sdk = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        free_port = sock.getsockname()[1]

    return host_ip_by_sdk, free_port


def ray_init():
    print(
        f"建链规模: NNODES={args.nnodes}, WORLD_SISE={args.nnodes*args.n_gpus_per_node}",
        flush=True,
    )

    if args.nnodes > 1 and args.ray_init:
        if args.ray_master_ip is None:
            raise RuntimeError(
                f"`--ray_master_ip` should be set if nnodes({args.nnodes}) > 1."
            )

        curr_addr, _ = get_availale_curr_addr_port()
        print(f"curr_addr = {curr_addr}", flush=True)

        if args.is_master or curr_addr == args.ray_master_ip:
            pass
            print("\033[32mMaster\033[0m", flush=True)
            ret = os.popen(
                f"ray start --head --port {args.ray_master_port} --dashboard-port {args.ray_dashboard_port}"
            ).read()
        else:
            print("\033[32mSlaver\033[0m", flush=True)
            ret = os.popen(
                f"ray start --address={args.ray_master_ip}:{args.ray_master_port}"
            ).read()
            exit(0)
        print(ret, flush=True)

    if not ray.is_initialized():
        ray.init()


def build_task(task_cls, config=None, device_name=None):
    if device_name is None:
        use_gpu = False
    else:
        use_gpu = True
    print(f"\033[32mStart build task\033[0m", flush=True)

    n_gpus_per_node = int(args.n_gpus_per_node)
    nnodes = int(args.nnodes)
    resource_pool = RayResourcePool(
        process_on_nodes=[n_gpus_per_node] * nnodes,
        max_colocate_count=1,
        use_gpu=use_gpu,
    )
    #* 检查当前ray的资源数是否满足


    strategy = "PACK"
    pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=device_name)
    print(f"\033[32mget_placement_groups done\033[0m", flush=True)


    world_size = resource_pool.world_size
    local_world_size = resource_pool.store[0]

    rank = -1
    tasks = []
    master_addr = None
    master_port = None
    for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
        for local_rank in range(local_world_size):
            rank += 1
            print(f"Building rank({rank}), local_rank({local_rank})", flush=True)
            if rank == 0:
                master_addr, master_port = get_availale_curr_addr_port()
                print(f"Get master_addr from ray is {master_addr}", flush=True)
                info = {
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                }

            actor_name = f"comm-{rank}"
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
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=local_rank,
                ),
                "name": actor_name,
            }
            if device_name == "npu":
                options["resources"] = {"NPU": 1}

            task = task_cls.options(**options).remote(
                info, config, device_name=device_name
            )
            tasks.append(task)

    return tasks


def pool_exec(list_task, func_name, *args, **kwargs):
    print(f"Run {func_name}")
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
        self.local_rank = int(os.environ["LOCAL_RANK"])

    def ray_exec(self, func_name, *args, **kwargs):
        return getattr(self, func_name)(*args, **kwargs)

    def get_attr(self, key):
        return getattr(self, key)

    def print_rank(self):
        print(f"Rank: {self.rank}", flush=True)


@ray.remote
class TestComm(BasrRay):
    tensor_size = (100, 100)

    def __init__(self, rank_zero_info, config=None, device_name=None, *kwargs):
        super().__init__(rank_zero_info, config)
        if device_name is None:
            self.device_name = "cpu"
        else:
            self.device_name = device_name
        print(f"Device={self.device_name}", flush=True)

    def init_process_group(self):
        print(f"\033[32m开始建链\033[0m", flush=True)
        # backend = "cpu:gloo"

        # if self.device_name == "npu":
        #     backend = backend + f",{get_device_name()}:{get_nccl_backend()}"
        backend =  f"{get_device_name()}:{get_nccl_backend()}"

        print(f"\033[33mbackend={backend}\033[0m", flush=True)
        if not torch.distributed.is_initialized():
            dist.init_process_group(get_nccl_backend())

            # dist.init_process_group(
            #     backend=backend,
            #     rank=self.rank,
            #     world_size=self.world_size,
            #     timeout=datetime.timedelta(seconds=900),  # * 默认给一个 5min 的超时时间
            # )

        torch.npu.set_device(self.local_rank)
        print(f"\033[32m建链完成\033[0m", flush=True)
        return dist.get_rank(), dist.get_world_size()

    def test_allreduce(self):

        # 创建测试张量
        dist.barrier()
        tensor = torch.ones(self.tensor_size, dtype=torch.float32, device=self.device_name) * self.rank

        ground_truth = np.sum(np.arange(self.world_size))
        print(f"Testing AllReduce with tensor size: {self.tensor_size}", flush=True)

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        assert tensor[0][0] == ground_truth
        # 同步所有进程
        dist.barrier()
        print(f"\033[32mAllReduce Done\033[0m", flush=True)

    def test_allgather(self):
        tensor = torch.ones(self.tensor_size, dtype=torch.float32, device=self.device_name) * self.rank

        gather_list = [torch.zeros_like(tensor, device=self.device_name) for _ in range(self.world_size)]

        dist.all_gather(gather_list, tensor)

        if self.rank == 0:
            gathered_data = torch.cat(gather_list)
            print([i[0][0] for i in gathered_data.chunk(self.world_size)])
        print(f"\033[32mAllGather Done\033[0m", flush=True)
        dist.barrier()

    def test_alltoall(self):
        tensor = torch.ones(self.tensor_size, dtype=torch.float32, device=self.device_name) * self.rank
        pass

    def get_tensor(self, data):
        pass


class RayTest:

    def __init__(self):
        self.list_task = build_task(TestComm, device_name=args.device)

    def test_comm(self):
        pool_exec(self.list_task, "print_rank")
        pool_exec(self.list_task, "init_process_group")
        pool_exec(self.list_task, "test_allreduce")
        pool_exec(self.list_task, "test_allgather")

    def test_host_memory(self):


        tmp_tensor = torch.zeros(
            [64, 1024, 1024, 1024], dtype=torch.float16
        )  # * 下发 1 T
        print(
            f"test_host_memory data size: {tmp_tensor.element_size() * tmp_tensor.numel()/1024**3:.2f} GB"
        )

        chunck_tmp = tmp_tensor.chunk(len(self.list_task))

        task_runnint_list = []
        for i, task_i in enumerate(self.list_task):
            tmp = chunck_tmp[i]
            task_runnint_list.append(task_i.get_tensor.remote(tmp))
            print(
                f"transfer {tmp.element_size() * tmp.numel()/1024**3:.2f} GB to rank: {i}"
            )
        list_output = ray.get(task_runnint_list)

        print(f"test_host_memory Done")


if __name__ == "__main__":
    print(f"Start Test Comm", flush=True)
    ray_init()
    ray_test = RayTest()
    # ray_test.test_host_memory()
    ray_test.test_comm()
