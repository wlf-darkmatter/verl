import logging
import asyncio
import os
import numpy as np
import torch

from megatron.core import parallel_state as mpu

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.profiler import DistProfiler, GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.workers.megatron_workers import ActorRolloutRefWorker as BaseMegatronWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class ActorRolloutRefWorker(BaseMegatronWorker):
    """
    适配 ReqScheduler 的 Megatron Actor-Rollout-Ref Worker。
    """

    @register(dispatch_mode=Dispatch.REQ_DISTRIBUTION)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        rank = torch.distributed.get_rank()
        config = self.config.rollout
        
        tp_size = config.get("tensor_model_parallel_size", 1)
        my_req_idx = rank // tp_size
        is_first_tp_rank = (rank % tp_size == 0)

        # 1. 获取并移除调度信息
        reqs_idx = prompts.non_tensor_batch.pop("reqs_idx", None)
        pre_outlens = prompts.non_tensor_batch.pop("pre_outlens")

        # 2. 筛选属于当前 DP Rank 的请求
        # 这里的 reqs_idx 是基于原始 Prompt (Size B) 的分配
        my_idx = [i for i, idx in enumerate(reqs_idx) if idx == my_req_idx]
        
        if is_first_tp_rank:
            total_reqs = len(reqs_idx) if reqs_idx is not None else 0
            print(f"[ReqSched-Debug] Rank {rank} (DP-{my_req_idx}): Assigned {len(my_idx)}/{total_reqs} prompts.", flush=True)

        prompts = prompts.select_idxs(my_idx)
        
        # 统计预测信息
        if len(my_idx) > 0:
            pre_outlens = [pre_outlens[i] for i in my_idx]
            pre_longest = max(pre_outlens)
            pre_shortest = min(pre_outlens)
            pre_avg = np.mean(pre_outlens)
            
            original_prompt_ids = prompts.non_tensor_batch["raw_prompt_ids"]
            inlens = [len(i) for i in original_prompt_ids]
            predict_totallens = [i + j for i, j in zip(inlens, pre_outlens, strict=False)]
        else:
            pre_longest = pre_shortest = pre_avg = 0
            inlens = []
            predict_totallens = []

        if is_first_tp_rank and len(my_idx) > 0:
            print(
                f"[GEN-Megatron]: rank={rank}, len(my_idx)={len(my_idx)}, "
                f"pre_longest={pre_longest}, pre_avg={pre_avg:.2f}"
            )

        # 3. 扩展数据
        n_samples = self.config.rollout.get("n", 1)
        if n_samples > 1:
            prompts = prompts.repeat(repeat_times=n_samples, interleave=True)

        prompts = prompts.to(get_device_id())
        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        if self._is_offload_optimizer:
            from verl.utils.megatron_utils import offload_megatron_optimizer
            offload_megatron_optimizer(self.actor_optimizer)

        timing_generate = {}
        
        if self._is_actor:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        # 执行生成 (vLLM 将处理展开后的 B*N 个请求)
        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)
            
            # 统计实际输出
            if is_first_tp_rank and len(my_idx) > 0:
                responses = output.batch["responses"]
                pad_id = self.tokenizer.pad_token_id
                
                # 计算实际非 padding 长度
                if isinstance(responses, torch.Tensor):
                    actual_outlen = torch.sum(responses != pad_id, dim=1).tolist()
                else:
                    actual_outlen = [np.sum(np.array(r) != pad_id) for r in responses]

                actual_sum = np.sum(actual_outlen)
                actual_mean = np.mean(actual_outlen)
                actual_max = np.max(actual_outlen)
                actual_min = np.min(actual_outlen)
                
                predict_tsum = sum(predict_totallens)
                pre_osum = sum(pre_outlens)
                
                # 这里的 Predict 数据需要乘以 N 才能与 Actual (B*N) 对比
                print(
                    f"[GENTIME] {rank=}, {timing_generate['generate_sequences']:.2f}s; "
                    f"Predict(x{n_samples}): total={predict_tsum * n_samples}, out={pre_osum * n_samples}; "
                    f"ACTUAL: sum={actual_sum}, mean={actual_mean:.1f}, max={actual_max}, min={actual_min}"
                )        
        
        output = self.postprocess_data(output)

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @GPUMemoryLogger(role="megatron vllm postprocess", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        tp_size = self.config.rollout.tensor_model_parallel_size
        if tp_size == 1:
            return data
        
        rank = torch.distributed.get_rank()
        tp_rank = rank % tp_size

        if len(data) % tp_size != 0:
            chunk_size = (len(data) + tp_size - 1) // tp_size
            start = tp_rank * chunk_size
            end = min(start + chunk_size, len(data))
            if start >= len(data):
                return data[0:0] 
            return data[start:end]
        
        return data.chunk(chunks=tp_size)[tp_rank]