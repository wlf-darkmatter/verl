# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

python3 /opt/verl/test_vllm_rollout.py \
    --ray_master_ip 33.215.117.195 \
    -tp $tp \
    -dp $dp \
    -n 16 \
    --gen_bs $gbs \
    --max_prompt_length $((2*1024)) \
    --max_response_length $((5)) \
    --max_num_batched_tokens $((5)) \
    --n_gpus_per_node 16 \
    --dataset_path "/sfs_turbo/myq/dapo-math-17k-update-reasoning-shuffled.parquet" \
    --hdfs_path "/sfs_turbo/pretrained_models/Qwen3-30B-A3B" \
    --gpu_memory_utilization 0.60 \
    --nnodes $nnode \
    --ray_debug


# 如果是 MoE 模型，需要开启 --enable_expert_parallel

"""

import argparse
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import ray
import rich
import torch
import torch.distributed as dist
import torch_npu
import vllm.envs as envs
from datasets import load_dataset
from tensordict import TensorDict
from vllm import LLM, SamplingParams

from verl.protocol import DataProto
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.base import sort_placement_group_by_node_ip
from verl.utils.device import get_device_name, get_nccl_backend

# os.environ["ENABLE_MOE_ALLTOALLV"] = "1"

os.environ["VLLM_USE_V1"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument("--ray_master_ip", type=str, default=None)
parser.add_argument("--ray_master_port", type=int, default=6379)
parser.add_argument("--ray_dashboard_port", type=int, default=8265)
parser.add_argument("--is_master", action="store_true", help="直接设置当前机器为 master")

parser.add_argument("-dp", type=int, default=1)
parser.add_argument("-tp", type=int, default=1)
parser.add_argument("-n", type=int, default=4)
parser.add_argument("--gen_bs", type=int, default=128)
parser.add_argument("--n_gpus_per_node", type=int, default=8)
parser.add_argument("--nnodes", type=int, default=1)
parser.add_argument("--dataset_path", type=str, default=None, required=True)
parser.add_argument("--enable_expert_parallel", action="store_true")
parser.add_argument("--disable_chuncked_prefill", action="store_true")
parser.add_argument(
    "--load_format",
    type=str,
    default="safetensors",
    help="vllm读取权重的方式, dummy是随机初始化",
)

parser.add_argument("--hdfs_path", type=str, default="/path/to/model")
parser.add_argument("--max_num_batched_tokens", type=int, default=16 * 1024)
parser.add_argument("--max_num_seqs", type=int, default=256)
parser.add_argument("--max_prompt_length", type=int, default=1024)
parser.add_argument("--max_response_length", type=int, default=1024)
parser.add_argument("--min_response_length", type=int, default=None)

parser.add_argument("--graph", action="store_true", default=None)
parser.add_argument("--graph_batch_sizes", type=int, default=24)

parser.add_argument("--rollout_step", type=int, default=1, help="重复跑几轮")
parser.add_argument("--enable_sleep", action="store_true", help="是否使能sleep模式")

parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)

parser.add_argument("--profile", action="store_true", default=None)
default_dir = "tmp/profile"
parser.add_argument("--profile_dir", type=str, default=default_dir)
parser.add_argument("--profile_with_stack", action="store_true", default=False)
parser.add_argument("--profile_with_module", action="store_true", default=False)
parser.add_argument("--profile_with_memory", action="store_true", default=False)
parser.add_argument("--profile_level", type=int, default=0)

parser.add_argument(
    "--dev_enforce_sample_balance",
    action="store_true",
    help="是否启用repeat方式，并在实例间均匀分发",
)

parser.add_argument("--ray_debug", action="store_true", help="Enable RAY_DEBUG_POST_MORTEM")


args = parser.parse_args()

local_cache_path = "~/.cache/vllm"

WORLD_SIZE = args.nnodes * args.n_gpus_per_node
max_prompt_length = args.max_prompt_length
max_response_length = args.max_response_length
min_response_length = args.min_response_length
max_model_len = max_prompt_length + max_response_length
tp_size = int(args.tp)
dp_size = int(args.dp)
max_num_batched_tokens = args.max_num_batched_tokens  # debug

bs = args.gen_bs
# world_size = int(os.getenv("WORLD_SIZE", "-1"))

all_ranks = torch.arange(WORLD_SIZE).reshape(-1, dp_size, 1, tp_size)  # noqa
num_instance = all_ranks.shape[0]
device_per_instance = WORLD_SIZE // num_instance  # noqa
# * all_ranks.shape[0] 实例个数
torch_profiler_trace_dir = Path(args.profile_dir) / datetime.now().strftime("%Y-%m-%d__%H-%M")


def check_args():
    # * check gbs
    assert bs % (all_ranks.shape[0] * dp_size) == 0, "样本数必须是 实例数 * dp_size 的整数倍"
    assert_profile_num = 10
    if args.profile:
        if args.max_response_length >= assert_profile_num:
            raise NotImplementedError(
                f"When profile, {args.max_response_length=} >= {assert_profile_num} is too large."
            )
    # * 校验 max_num_seqs
    if args.max_num_seqs > args.max_num_batched_tokens:
        rich.print(
            f"[yellow]max_num_batched_tokens ({args.max_num_batched_tokens}) "
            + f"must be greater than or equal to max_num_seqs ({args.max_num_seqs}). "
            + f"\nChanged `max_num_seqs` to {args.max_num_batched_tokens} [/yellow]"
        )
        args.max_num_seqs = args.max_num_batched_tokens


def get_cluster_info():
    # 确保分布式环境已初始化
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized")

    world_size = dist.get_world_size()

    # 获取当前节点的IP地址
    ip_address = _get_current_node_ip()

    # 收集所有rank的IP地址
    ip_list = [None] * world_size
    dist.all_gather_object(ip_list, ip_address)

    return ip_list


def get_availale_curr_addr_port():
    host_ip_by_sdk = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        free_port = sock.getsockname()[1]

    return host_ip_by_sdk, free_port


def _get_current_node_ip() -> str:
    # 创建一个 UDP 套接字（仅用于获取接口信息）
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        # 连接到一个外部地址（无需真实通信）
        s.connect(("8.8.8.8", 80))  # Google DNS 服务器
        local_ip = s.getsockname()[0]

    return local_ip


def _init_dp_envs():
    rank = torch.distributed.get_rank()
    world_size = int(os.getenv("WORLD_SIZE", "-1"))

    for index, group_rank in enumerate(all_ranks):
        if torch.distributed.get_rank() in group_rank:
            os.environ["VLLM_INSTANCE_INDEX"] = str(index)

    group_ranks = all_ranks.transpose(1, 3).reshape(-1, dp_size).unbind(0)

    # group_ranks = [x.tolist() for x in group_ranks]
    ip_list = get_cluster_info()
    for index, group_rank in enumerate(group_ranks):
        _group_rank = group_rank.tolist()
        if torch.distributed.get_rank() in _group_rank:
            os.environ["VLLM_DP_MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT")) + 1 + index)
            os.environ["VLLM_DP_MASTER_IP"] = ip_list[_group_rank[0]]

    local_dp_rank = rank // tp_size % dp_size
    os.environ["VLLM_DP_RANK"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_PORT"] = os.environ["VLLM_DP_MASTER_PORT"]
    envs.VLLM_DP_RANK = int(os.environ["VLLM_DP_RANK"])
    envs.VLLM_DP_MASTER_IP = os.environ["VLLM_DP_MASTER_IP"]
    envs.VLLM_DP_MASTER_PORT = int(os.environ["VLLM_DP_MASTER_PORT"])

    print(f"[VLLM] using {world_size=}, TP={tp_size}, DP={dp_size}", flush=True)


@ray.remote
class Vllm_Worker(Worker):
    def __init__(self, rank_zero_info, llm_config):
        os.environ["WG_BACKEND"] = "ray"
        if rank_zero_info is None:
            raise RuntimeError()
        super().__init__()

        rank = int(os.environ.get("RANK", 0))
        if not torch.distributed.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        self.rank_zero_info = rank_zero_info

        self.llm_config = llm_config
        _init_dp_envs()
        self.dp_rank = int(os.environ.get("VLLM_DP_RANK", 0))
        self.dp_size = int(os.environ.get("VLLM_DP_SIZE", 1))
        self.vllm_instance_index = int(os.environ.get("VLLM_INSTANCE_INDEX", 1))

    def create_llm(self):
        _llm_config = self.llm_config
        self.local_model_path = Path(_llm_config.hdfs_path).__str__()

        additional_config = {}
        if args.graph:
            # * torchair Graph mode
            additional_config = {
                "torchair_graph_config": {
                    "enabled": True,
                    "use_cached_graph": False,
                    "graph_batch_sizes_init": False,
                    "graph_batch_sizes": [args.graph_batch_sizes],
                },
                "ascend_scheduler_config": {"enabled": True},
                "refresh": True,
            }
            enforce_eager = False
        else:
            enforce_eager = True

        print("\033[33m构建LLM中\033[0m")
        self.llm = LLM(
            model=self.local_model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=_llm_config.tp,
            distributed_executor_backend="external_launcher",
            dtype="bfloat16",
            enforce_eager=enforce_eager,
            gpu_memory_utilization=_llm_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=_llm_config.max_num_seqs,
            load_format=_llm_config.load_format,  #! 如果是减层
            disable_log_stats=False,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=not _llm_config.disable_chuncked_prefill,
            enable_prefix_caching=True,  # todo
            trust_remote_code=True,
            seed=0,
            enable_expert_parallel=_llm_config.enable_expert_parallel,
            additional_config=additional_config,
        )
        print("\033[33m构建LLM完毕\033[0m", flush=True)

        if args.profile:
            self.setup_profile()

    def setup_profile(self):
        level_map = {
            0: torch_npu.profiler.ProfilerLevel.Level0,
            1: torch_npu.profiler.ProfilerLevel.Level1,
            2: torch_npu.profiler.ProfilerLevel.Level2,
        }

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=level_map[args.profile_level],
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        self.profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=args.profile_with_stack,
            profile_memory=args.profile_with_memory,
            with_modules=args.profile_with_module,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_profiler_trace_dir),
        )
        print("profiler Inited!", flush=True)

    def dp_dispatch(self, prompt_batch: list):
        # *  根据实例数进行分配
        if self.dp_size == 1:
            return prompt_batch

        if isinstance(prompt_batch, list):
            ind = np.arange(len(prompt_batch)).reshape(self.dp_size, -1)[self.dp_rank]
            dispatch_prompt_batch = [prompt_batch[i] for i in ind.tolist()]
            return dispatch_prompt_batch

        if isinstance(prompt_batch, DataProto):
            return prompt_batch.chunk(self.dp_size)[self.dp_rank]

    def run_one_step(self, prompt_batch, sampling_params=None) -> DataProto:
        # breakpoint()
        prompt_batch = self.dp_dispatch(prompt_batch)
        non_tensor_batch = prompt_batch.non_tensor_batch
        vllm_inputs = [
            {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
        ]

        rich.print("[green]Rollout[/green]")
        start_time = time.time()
        if args.profile:
            self.profiler.start()
        outputs = self.llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        if args.profile:
            self.profiler.stop()
        cost_time = time.time() - start_time

        _output_dataproto = DataProto(
            TensorDict(
                {"prompt": prompt_batch, "response": outputs},
                batch_size=len(prompt_batch),
            )
        )
        _output_dataproto.meta_info = {"cost_time": cost_time}
        # * 计算吞吐
        self.calc_tps(outputs, _output_dataproto.meta_info)

        return _output_dataproto

    def offload(self):
        print("offload")
        self.llm.sleep(level=1)
        pass

    def onload(self):
        print("onload")
        self.llm.wake_up()
        pass

    def calc_tps(self, outputs: list, meta_info: dict) -> dict:
        num_prompt = 0
        num_response = 0
        for request in outputs:
            num_prompt += len(request.prompt_token_ids)
            for resp_i in request.outputs:
                num_response += len(resp_i.token_ids)

        meta_info["tokens"] = num_prompt + num_response
        meta_info["tps"] = meta_info["tokens"] / meta_info["cost_time"]
        meta_info["response"] = num_response
        meta_info["prompt"] = num_prompt

        return meta_info

    def get_attr(self, key):
        return getattr(self, key)

    def ray_exec(self, func_name, *args, **kwargs):
        return getattr(self, func_name)(*args, **kwargs)

    @property
    def _info(self):
        """
        考虑下放到 VllmRay 中，以获取其他node的信息
        """
        info = {}
        info["allocated"] = torch.npu.memory_allocated() / 1024**3
        info["cached"] = torch.npu.memory_cached() / 1024**3
        info["reserved"] = torch.npu.memory_reserved() / 1024**3
        info["dp_rank"] = self.dp_rank
        info["rank"] = self.rank

        free, _ = torch.npu.mem_get_info()
        info["free"] = free / 1024**3

        return info


class VllmRay:
    def __init__(self, config):
        self._data_index = 0
        self.config = config

        self.dataloader = self._prepare_data()
        self._build_vllm_ray()

    def _prepare_data(self):
        from torchdata.stateful_dataloader import StatefulDataLoader

        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        if args.dataset_path.endswith("parquet"):
            data = load_dataset("parquet", data_files=args.dataset_path)["train"]
        else:
            raise TypeError()

        assert len(data) >= bs

        if self.config.n == 1:
            temperature = 0.0
        else:
            temperature = 1.0

        kwargs = dict(
            n=self.config.n,
            temperature=temperature,
            top_p=1,
            top_k=-1,  # -1 for vllm rollout
            max_tokens=max_response_length,
            logprobs=0,
        )
        if self.config.min_response_length is not None:
            if self.config.min_response_length > max_response_length:
                self.config.min_response_length = max_response_length
                print(
                    f"min_tokens={self.config.min_response_length} ",
                    "must be less than or equal to max_tokens={max_response_length}. Fixed",
                )
            kwargs["min_tokens"] = self.config.min_response_length
            kwargs["ignore_eos"] = True

        self.sampling_params = SamplingParams(**kwargs)
        self.sampling_params.detokenize = False

        from verl.utils import hf_tokenizer
        from verl.utils.dataset.rl_dataset import RLHFDataset

        self.tokenizer = hf_tokenizer(self.config.hdfs_path, trust_remote_code=True)

        self.dataset = RLHFDataset(
            data_files=args.dataset_path,
            tokenizer=self.tokenizer,
            processor=None,
            config={
                "max_prompt_length": self.config.max_prompt_length,
                "filter_overlong_prompts": False,
            },
        )
        # def collate_fn(xx):
        #     return list(map(lambda x: x["prompt"][0]["content"], xx))

        dataloader = StatefulDataLoader(
            dataset=self.dataset,
            batch_size=bs,
            num_workers=0,
            drop_last=True,
            collate_fn=default_collate_fn,
            # sampler=SequentialSampler(dataset),
        )

        return dataloader

    def _build_vllm_ray(self):
        config = self.config
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        n_gpus_per_node = int(config.n_gpus_per_node)
        nnodes = int(config.nnodes)
        resource_pool = RayResourcePool(process_on_nodes=[n_gpus_per_node] * nnodes)

        strategy = "PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name="npu")
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
                    "resources": {"NPU": 1},
                    "scheduling_strategy": PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=local_rank,
                    ),
                    "name": actor_name,
                }
                # task = Vllm_Worker(info, config)
                # task = Vllm_Worker.remote(info, config)
                task = Vllm_Worker.options(**options).remote(info, config)
                tasks.append(task)

        self.vllm_tasks = tasks

        self.grouped_vllm_tasks = []

    def create_llm(self):
        self.pool_exec("create_llm")

    def generate_sequences(self, batch_dict: dict):
        # * 测试样本分配效果
        # todo 这里没有考虑到切 pp
        # breakpoint()
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        # * Tokenizer ENCODE

        self.sampling_params.n = 1
        if args.dev_enforce_sample_balance:
            # * repeat方式，实例间强制均衡分配法
            # np.arange(len(batch)).repeat(args.n).reshape(-1,len(all_ranks)).T
            # torch.arange(len(batch)).repeat_interleave(args.n).reshape(-1,len(all_ranks)).T
            #! 模拟等效
            _preencode_prompts = gen_batch.repeat(repeat_times=self.config.n, interleave=False)
            print("启用 dev_enforce_sample_balance 分配法", flush=True)

        else:
            # * Base 分配法
            _preencode_prompts = gen_batch.repeat(repeat_times=self.config.n, interleave=True)

        # torch_list_range = torch.arange(len(_preencode_prompts)).reshape(-1, len(all_ranks)).T
        # * 按实例进行分配
        task_runnint_list = []
        chunk_preencode_prompts = _preencode_prompts.chunk(num_instance)
        for rank_i, task_i in enumerate(self.vllm_tasks):
            vllm_rank, dp_rank, _, tp_rank = torch.where(all_ranks == rank_i)
            _input = chunk_preencode_prompts[vllm_rank]

            task_runnint_list.append(task_i.run_one_step.remote(_input, self.sampling_params))

        list_output = ray.get(task_runnint_list)
        return list_output

    def pool_exec(self, func_name, *args, **kwargs):
        task_runnint_list = []
        for i, task_i in enumerate(self.vllm_tasks):
            task_runnint_list.append(task_i.ray_exec.remote(func_name, *args, **kwargs))
        list_output = ray.get(task_runnint_list)
        return list_output

    def pool_getattr(self, attr_name):
        task_runnint_list = []
        for i, task_i in enumerate(self.vllm_tasks):
            task_runnint_list.append(task_i.get_attr.remote(attr_name))
        list_output = ray.get(task_runnint_list)
        return list_output


def ray_init():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["VLLM_LOGGING_LEVEL"] = "WARN"
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"
    os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"

    if args.ray_debug:
        os.environ["RAY_DEBUG_POST_MORTEM"] = "1"

    if args.nnodes > 1:
        if args.ray_master_ip is None:
            raise RuntimeError(f"`--ray_master_ip` should be set if nnodes({args.nnodes}) > 1.")

        curr_addr, _ = get_availale_curr_addr_port()
        print(curr_addr, flush=True)

        if args.is_master or curr_addr == args.ray_master_ip:
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


class Test:
    def __init__(self):
        self.Vllm = VllmRay(args)
        self._mem_info_1 = self.Vllm.pool_getattr("_info")
        self.Vllm.create_llm()
        self._mem_info_2 = self.Vllm.pool_getattr("_info")

    def test_tps(self, num_step=1):
        # * start running
        step_i = 0
        for _preencode_prompts in self.Vllm.dataloader:
            list_output = self.Vllm.generate_sequences(_preencode_prompts)

            # todo tokenizer
            # try:
            #     _output = list_output[0].batch["response"][0].outputs[0]
            #     response_text = _output.text
            #     print("===>Output===>", flush=True)
            #     if len(response_text) <= 620:
            #         print(response_text, flush=True)
            #     else:
            #         print(response_text[:300], flush=True)
            #         print("\n...\n...\n")
            #         print(response_text[-300:], flush=True)
            #     print(f"<===END, 生成结束原因: {_output.finish_reason}", flush=True)

            # except Exception as e:
            #     print(f"Print generation failed! \nreason is {e.__repr__()}")

            # * 打印 综合 TPS
            print(f"=== Step {step_i} ===", flush=True)

            tps = np.mean(list(map(lambda x: x.meta_info["tps"], list_output)))
            cost_time = np.mean(list(map(lambda x: x.meta_info["cost_time"], list_output)))
            # * 打印 综合 TPS
            max_cost_time = np.max(list(map(lambda x: x.meta_info["cost_time"], list_output)))
            sum_token = np.sum(list(map(lambda x: x.meta_info["tokens"], list_output)))

            print(f"MEAN TPS: {tps: 0.4f} tokens/s", flush=True)
            print(f"MEAN 耗时: {cost_time: 0.2f} s", flush=True)

            print(f"E2E TPS: {sum_token / max_cost_time: 0.4f} tokens/s", flush=True)
            print(f"E2E 最大耗时: {max_cost_time: 0.2f} s", flush=True)
            print(f"平均单实例 E2E TPS: {sum_token / num_instance / max_cost_time: 0.4f} tokens/s", flush=True)

            step_i += 1
            if step_i >= num_step:
                break

    def test_sleep(self):
        # * 查看当前 NPU 显存 利用率

        self.Vllm.pool_exec("offload")
        mem_info_offload = self.Vllm.pool_getattr("_info")

        self.Vllm.pool_exec("onload")
        mem_info_onload = self.Vllm.pool_getattr("_info")
        self.pprint_mem(
            mem_init=self._mem_info_1,
            mem_created=self._mem_info_2,
            mem_offload=mem_info_offload,
            mem_onload=mem_info_onload,
        )

    def pprint_mem(self, **kwargs):
        import pandas as pd
        import rich
        from rich.table import Table

        table = Table(title="Memory")

        list_pd = []
        table.add_column("name")
        for k in kwargs.keys():
            _tmp_pd = pd.DataFrame(kwargs[k])
            list_pd.append(_tmp_pd)
        for k in list_pd[0].columns:
            table.add_column(k)

        for name, values in zip(kwargs.keys(), list_pd, strict=False):
            for i, _rank in values.iterrows():
                table.add_row(name, *list(map(str, _rank.values)))
        rich.print(table)
        pass


if __name__ == "__main__":
    ray_init()
    check_args()

    test_vllm = Test()
    test_vllm.test_tps(args.rollout_step)

    ray.shutdown()
