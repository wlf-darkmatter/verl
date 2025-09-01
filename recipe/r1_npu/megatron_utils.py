# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""Pretrain utilities."""

import gc
import os
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core import ModelParallelConfig, mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_attr_wrapped_model
from transformers import PretrainedConfig

import verl.utils.megatron.tensor_parallel as tp_utils
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fs import local_mkdir_safe
from verl.utils.model import normalize_model_name
from verl.utils.torch_dtypes import PrecisionType
from vllm.distributed.parallel_state import get_ep_group
from verl.utils.megatron_utils import broadcast_from_megatron_pp, broadcast_str_from_megatron_pp, unwrap_model


def default_tp_concat_fn(
        layer_name_mapping, 
        name, 
        train_params,
        infer_params,
        model_config,
        hf_config=None,
        convert_qkv_gate_up_by_simple_split=False
    ):
    """
    name: name of the parameter
    train_params: training parameters
    infer_params (Iterable[torch.Tensor]): a iterator towards list of parameters all-gathered from micro_dp_group
    model_config: huggingface model_config
    TODO(zhangchi.usc1992): currently, the implementation is adhoc. We can move this function to the model
    definition so that it is model-agnostic. If the model doesn't implement this function,
    we can throw an error to force user disable TP HybridEngine.
    """
    from megatron.core import mpu
    train_tp_size = mpu.get_tensor_model_parallel_world_size()

    if hasattr(model_config, 'n_routed_experts'):
        num_experts = model_config.n_routed_experts
    elif hasattr(model_config, 'num_experts'):
        num_experts = model_config.num_experts
    else:
        raise ValueError("model_config must have either 'n_routed_experts' or 'num_experts'")

    if layer_name_mapping.get("qkv_layer_name") in name and "layer_norm" not in name:
        # if the tensor is qkv, for each param on tp, split into q, k, v
        # concat q, k, v separately.
        q_lst = []
        k_lst = []
        v_lst = []
        num_attention_heads = model_config.num_attention_heads
        num_key_value_heads = model_config.num_key_value_heads
        if "vision_model" in name:
            num_attention_heads = hf_config.vision_config.num_heads
            num_key_value_heads = hf_config.vision_config.num_heads
        assert num_attention_heads % num_key_value_heads == 0
        num_q_per_kv = num_attention_heads // num_key_value_heads
        assert infer_params[0].shape[0] % (num_q_per_kv + 2) == 0, (
            f"param '{name}' shape '{infer_params[0].shape}' dim0 is not divisible by {num_q_per_kv + 2}"
        )
        kv_size_per_tp = infer_params[0].shape[0] // (num_q_per_kv + 2)
        split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
        for infer_param in infer_params:
            num_query_groups_per_partition = num_key_value_heads // train_tp_size
            for chunk in infer_param.chunk(num_query_groups_per_partition):
                split_size = [
                    kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition,
                    kv_size_per_tp // num_query_groups_per_partition,
                    kv_size_per_tp // num_query_groups_per_partition,
                ]
                q, k, v = chunk.split(split_size)
                q_lst.append(q)
                k_lst.append(k)
                v_lst.append(v)
        q = torch.cat(q_lst, dim=0)
        k = torch.cat(k_lst, dim=0)
        v = torch.cat(v_lst, dim=0)
        infer_params = torch.cat((q, k, v), dim=0) if not convert_qkv_gate_up_by_simple_split else [q, k, v]

    elif (
            layer_name_mapping.get("gate_proj_layer_name") in name
            and "layer_norm" not in name
            and "vision_model.projection" not in name
        ):
        # if the tensor is gate and proj
        gate_lst = []
        up_lst = []
        for infer_param in infer_params:
            gate, up = infer_param.chunk(2)
            gate_lst.append(gate)
            up_lst.append(up)
        gate = torch.cat(gate_lst, dim=0)
        up = torch.cat(up_lst, dim=0)
        infer_params = torch.cat((gate, up), dim=0) if not convert_qkv_gate_up_by_simple_split else [gate, up]

    elif "mlp.experts.weight1" in name:  # for moe group matmul
        gate_pp_lst = []
        up_pp_lst = []

        if os.getenv('ALL_TO_ALL_RESHARD', '0') == '0':
            for infer_param in infer_params:
                split_size = [
                    model_config.moe_intermediate_size,
                    model_config.moe_intermediate_size,
                ] * (num_experts // mpu.get_expert_tensor_and_model_parallel_world_size())
                experts_weight = infer_param.split(split_size, dim=1)
                gate_pp_lst.extend(experts_weight[::2])
                up_pp_lst.extend(experts_weight[1::2])
            infer_params = [tensor.transpose(0, 1) for pair in zip(gate_pp_lst, up_pp_lst) for tensor in pair]
        else:
            # To optimize memory, only the params for the current rank are non-empty; others are empty tensors
            infer_params = get_rollout_expert_after_resharding(infer_params, model_config, is_weight1=True)

    elif "mlp.experts.weight2" in name:  # for moe group matmul
        down_pp_lst = []
        if os.getenv('ALL_TO_ALL_RESHARD', '0') == '0':
            for infer_param in infer_params:
                split_size = [
                    model_config.moe_intermediate_size
                ] * (num_experts // mpu.get_expert_tensor_and_model_parallel_world_size())
                experts_weight = infer_param.split(split_size, dim=0)
                down_pp_lst.extend(experts_weight)
            experts_down_pp = [downs.transpose(0, 1) for downs in down_pp_lst]
            infer_params = experts_down_pp
        else:
            # To optimize memory, only the params for the current rank are non-empty; others are empty tensors
            infer_params = get_rollout_expert_after_resharding(infer_params, model_config, is_weight1=False)

    elif "mlp.experts.linear_fc2.weight" in name:  # moe
        infer_params = torch.cat(infer_params, dim=1)

    else:
        # concat tensor
        infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(train_params))
    
    return infer_params


def per_tensor_generator(
    actor_module,
    model_config,
    weight_converter,
    transformer_config,
    layer_name_mapping,
    convert_qkv_gate_up_by_simple_split=True,
):
    from megatron.core import parallel_state as mpu

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    ep_size = mpu.get_expert_model_parallel_world_size()
    etp_size = mpu.get_expert_tensor_parallel_world_size()
    ep_group = mpu.get_expert_model_parallel_group()
    etp_group = mpu.get_expert_tensor_parallel_group()
    vpp_size = len(actor_module)
    all_gather_group = mpu.get_tensor_model_parallel_group()
    all_gather_group_size = torch.distributed.get_world_size(group=all_gather_group)
    etmp_group = mpu.get_expert_tensor_and_model_parallel_group()

    def tensor_generator():
        for scan_vpp_idx in range(vpp_size):
            existing_keys = set()
            model = unwrap_model(actor_module[scan_vpp_idx])
            for name, param in model.named_parameters():
                existing_keys.add(name)
                yield name, param
            # note
            # there is a bug in megatron GPTModel
            # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in
            # state_dict(). for now we patch it by adding those keys to extra_keys.
            extra_keys = [x for x in model.state_dict().keys() if "_extra_state" not in x and x not in existing_keys]
            for name in extra_keys:
                yield name, model.state_dict()[name].to(get_device_id())

    # we need first make all rank get full model information
    meta_info = []
    for scan_vpp_idx in range(vpp_size):
        existing_keys = set()
        model = unwrap_model(actor_module[scan_vpp_idx])
        for idx, (name, _) in enumerate(model.named_parameters()):
            existing_keys.add(name)
            meta_info.append((pp_rank, scan_vpp_idx, idx, name))
        extra_keys = [x for x in model.state_dict().keys() if "_extra_state" not in x and x not in existing_keys]
        for name in extra_keys:
            meta_info.append((pp_rank, scan_vpp_idx, idx, name))

    obj_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=obj_spec_output, obj=meta_info, group=mpu.get_pipeline_model_parallel_group()
    )
    layer_list_meta = [item for sublist in obj_spec_output for item in sublist]

    gen_func = tensor_generator()

    # lazy load tensor for full model
    for cur_pp_rank, scan_vpp_idx, idx, name in layer_list_meta:
        if model_config.tie_word_embeddings and ("output_layers" in name):
            import warnings

            warnings.warn(
                "Current model sharing word and embedding weights, skip output layer conversion", stacklevel=2
            )
            continue

        if cur_pp_rank == pp_rank:
            try:
                cur_name, cur_tensor = next(gen_func)
            except StopIteration:
                cur_name, cur_tensor = None, None
            cur_name = normalize_model_name(name, cur_pp_rank, scan_vpp_idx, transformer_config)
        else:
            cur_tensor, cur_name = None, None

        # pp broadcast model tensor and name
        cur_name = broadcast_str_from_megatron_pp(cur_name)
        broad_pp_tensor = broadcast_from_megatron_pp(cur_tensor)

        # (xya): this is a hack to fix the name of the parameters
        while cur_name.startswith("module."):
            cur_name = cur_name[len("module.") :]

        # EP
        if ".mlp.experts.linear_fc" in cur_name and ep_size > 1:
            num_experts = weight_converter.mcore_config.num_moe_experts
            num_experts_per_rank = num_experts // ep_size
            infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(ep_size)]
            torch.distributed.all_gather(infer_params, broad_pp_tensor, group=ep_group)

            name_prefix, local_expert_id = cur_name.split(".weight")
            local_expert_id = int(local_expert_id)
            global_expert_ids = [num_experts_per_rank * ep_rank + local_expert_id for ep_rank in range(ep_size)]
            global_expert_names = [f"{name_prefix}.weight{expert_id}" for expert_id in global_expert_ids]

            for name, param in zip(global_expert_names, infer_params, strict=True):
                if etp_size > 1:
                    # gather etp
                    etp_params = [torch.empty_like(param) for _ in range(etp_size)]
                    torch.distributed.all_gather(etp_params, param, group=etp_group)
                    params = etp_params
                else:
                    params = [param]

                merge_params = default_tp_concat_fn(
                    layer_name_mapping,
                    name,
                    broad_pp_tensor,
                    params,
                    model_config,
                    weight_converter.hf_config,
                    convert_qkv_gate_up_by_simple_split,
                )
                if not isinstance(merge_params, list):
                    merge_params = [merge_params]
                converted_names, converted_params = weight_converter.convert_param(name, merge_params)

                yield from zip(converted_names, [param.detach() for param in converted_params], strict=True)
            continue
        
        elif ".mlp.experts.weight" in cur_name and ep_size > 1:
            if etp_size > 1:
                raise NotImplementedError("reshard for ETP params when using MoE Group Matmul not supported now")
            
            if os.getenv('ALL_TO_ALL_RESHARD', '0') == '0':
                ep_params = [torch.empty_like(broad_pp_tensor) for _ in range(ep_size)]
                torch.distributed.all_gather(ep_params, broad_pp_tensor, group=etmp_group)
            else:
                # EP param reshard method based on AllToAllV, efficient in both memory usage and performance
                ep_params = ep_param_reshard_by_alltoallv(
                    param_name=cur_name,
                    ep_param_train=broad_pp_tensor,
                    num_experts=weight_converter.mcore_config.num_moe_experts,
                    weight1_key_name="mlp.experts.weight1",
                    weight2_key_name="mlp.experts.weight2"
                )
            merge_params = default_tp_concat_fn(
                layer_name_mapping, 
                cur_name, 
                broad_pp_tensor, 
                ep_params, 
                model_config, 
                convert_qkv_gate_up_by_simple_split
            )
            converted_names, converted_params = weight_converter.convert_param(cur_name, merge_params)

            yield from zip(converted_names, converted_params)
            continue

        # tp all gather
        if tp_utils.is_tensor_parallel_param(broad_pp_tensor):
            # allocate a new tensor with proper size
            if all_gather_group_size <= 1:
                infer_params = [broad_pp_tensor]
            else:
                infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(all_gather_group_size)]
                torch.distributed.all_gather(infer_params, broad_pp_tensor, group=mpu.get_tensor_model_parallel_group())
            infer_params = default_tp_concat_fn(
                layer_name_mapping,
                cur_name,
                broad_pp_tensor,
                infer_params,
                model_config,
                weight_converter.hf_config,
                convert_qkv_gate_up_by_simple_split,
            )
        else:
            infer_params = broad_pp_tensor

        if not isinstance(infer_params, list):
            infer_params = [infer_params]
        converted_names, converted_params = weight_converter.convert_param(cur_name, infer_params)

        yield from zip(converted_names, [param.detach() for param in converted_params], strict=True)


def ep_param_reshard_by_alltoallv(
    param_name,
    ep_param_train,
    num_experts,
    weight1_key_name="mlp.experts.weight1",
    weight2_key_name="mlp.experts.weight2"
):
    """Reshard EP params by AllToAllV for better memory usage and communication performance in TP_extend_EP training

    Args:
        param_name: EP param name in the training engine
        ep_param_train: EP param shard held by this rank in the training engine
        num_experts: total number of routing experts in the complete model
        weight1_key_name: key word for the expert weight1 name in the training engine
        weight2_key_name: key word for the expert weight2 name in the training engine

    For example, Train EP4PP2 and Rollout EP8PP1, after PP allgather in veRL, the communication is like below:
    train ep ranks:     0    1    2    3  |  0    1    2    3
                        | \    \                      /    /|
                        |   \    \----\       /-----/    /  |
    rollout ep ranks:   0    1    2    3     4    5    6    7

    the send tensors for global rank 0 is: [shard_to_rank0, shard_to_rank1, empty, empty]
    the recv tensors for global rank 0 is: [shard_from_rank0, empty, empty, empty]
    the send tensors for global rank 4 is: [empty, empty, empty, empty]
    the recv tensors for global rank 4 is: [empty, empty, shard_from_rank6, empty]

    """
    ep_size_train = mpu.get_expert_tensor_and_model_parallel_world_size()
    ep_rank_train = mpu.get_expert_tensor_and_model_parallel_rank()
    ep_group_rollout = get_ep_group().device_group
    ep_size_rollout = torch.distributed.get_world_size(ep_group_rollout)
    ep_rank_rollout = torch.distributed.get_rank(group=ep_group_rollout)
    assert ep_size_rollout % ep_size_train == 0, f"EP size of rollout {ep_size_rollout} must be divisible by EP size of training {ep_size_train}"
    micro_ep_size = ep_size_rollout // ep_size_train

    assert num_experts % ep_size_train == 0 and num_experts % ep_size_rollout == 0
    num_experts_train = num_experts // ep_size_train
    num_experts_rollout = num_experts // ep_size_rollout

    if weight1_key_name in param_name:
        hidden_size = ep_param_train.shape[0]
        # The actual memory layout of weight `w13` is [num_experts_train, hidden_size, moe_intermediate_size],
        # view the tensor to a correct shape before using it.
        # Also, training phase and rollout phase expect different layouts for `w13`, with inversed dimension
        # order of `hidden_size` and `moe_intermediate_size`, necessiting the `transpose` and `contiguous` here.
        ep_param_train = ep_param_train.view(num_experts_train, hidden_size, -1).transpose(1, 2).contiguous()

        split_size = num_experts_train // micro_ep_size
        rollout_weight_shape = [split_size, ep_param_train.shape[1], hidden_size]

    elif weight2_key_name in param_name:
        hidden_size = ep_param_train.shape[1]
        # Similar to the handling of `w13`.
        ep_param_train = ep_param_train.view(num_experts_train, -1, hidden_size).transpose(1, 2).contiguous()

        split_size = num_experts_train // micro_ep_size
        rollout_weight_shape = [split_size, hidden_size, ep_param_train.shape[2]]
    else:
        raise NotImplementedError(f"Weight {param_name} not supported in EP param resharding yet!")

    # for send: get the corresponding rollout ep ranks of this training ep group
    ep_train_group_idx = (
        ep_rank_rollout // ep_size_train
    )  # train ep group idx within the larger rollout ep group of this rank
    ep_rank_range_rollout = list(
        range(ep_train_group_idx * ep_size_train, ep_train_group_idx * ep_size_train + ep_size_train, 1)
    )
    # for recv: get the src rollout ep rank of this rank
    recv_src_rank = ep_rank_rollout // micro_ep_size
    send_tensors = []   # sharded ep params to send to each rank in this training ep group by this rank
    recv_tensors = []   # recv buffers for this rank to recv sharded ep params from each rank in this training ep group
    split_start_idx = 0
    for rank_ep_train in range(ep_size_train):
        # update send_tensors
        rank_ep_rollout = ep_rank_range_rollout[rank_ep_train]
        if rank_ep_rollout // micro_ep_size == ep_rank_train:
            tensor_to_send = ep_param_train[split_start_idx:split_start_idx + split_size, ...]
            send_tensors.append(tensor_to_send)
            split_start_idx += split_size
        else:
            send_tensors.append(torch.zeros(0, dtype=ep_param_train.dtype, device=ep_param_train.device)) # placeholder

        # update recv_tensors
        if recv_src_rank == rank_ep_train:
            recv_tensors.append(
                torch.empty(rollout_weight_shape, dtype=ep_param_train.dtype, device=ep_param_train.device)
            )
        else:
            recv_tensors.append(torch.empty(0, dtype=ep_param_train.dtype, device=ep_param_train.device)) # placeholder

    torch.distributed.all_to_all(recv_tensors, send_tensors, group=mpu.get_expert_tensor_and_model_parallel_group())
    # filter out empty tensors and retain only the ep params required by this rank in rollout
    ep_params = [param for param in recv_tensors if param.numel() > 0]
    return ep_params


def get_rollout_expert_after_resharding(infer_params, model_config, is_weight1):
    """Postprocess the resharded EP parameter to return EP parameters for all ranks.
    To optimize memory, only the params for the current rank are non-empty; others are empty tensors.

    Args:
        infer_params: a tensor list but only contains one ep param of weight1 for this rank in rollout
        model_config: hugging face model config
    """
    assert len(infer_params) == 1
    rollout_ep_group = get_ep_group().device_group
    rollout_ep_size = torch.distributed.get_world_size(rollout_ep_group)
    ep_rank_rollout = torch.distributed.get_rank(group=rollout_ep_group)
    if hasattr(model_config, 'n_routed_experts'):
        num_experts = model_config.n_routed_experts
    elif hasattr(model_config, 'num_experts'):
        num_experts = model_config.num_experts
    else:
        raise ValueError("model_config must have either 'n_routed_experts' or 'num_experts'")
    num_experts_rollout = num_experts // rollout_ep_size

    # expert ids held by current rank in rollout
    local_expert_ids = list(
        range(ep_rank_rollout * num_experts_rollout, ep_rank_rollout * num_experts_rollout + num_experts_rollout)
    )

    local_expert_params = infer_params[0]
    if is_weight1:
        experts_gate_pp = [
            torch.empty(0, dtype=local_expert_params[0].dtype, device=local_expert_params[0].device)
            for idx in range(num_experts)
        ]
        experts_up_pp = [
            torch.empty(0, dtype=local_expert_params[0].dtype, device=local_expert_params[0].device)
            for idx in range(num_experts)
        ]
        for local_idx, expert_id in enumerate(local_expert_ids):
            experts_gate_pp[expert_id], experts_up_pp[expert_id] = torch.chunk(
                local_expert_params[local_idx], chunks=2, dim=0
            )
        infer_params = [tensor for pair in zip(experts_gate_pp, experts_up_pp) for tensor in pair]
        return infer_params
    else:
        experts_down_pp = [
            torch.empty(0, dtype=local_expert_params[0].dtype, device=local_expert_params[0].device)
            for idx in range(num_experts)
        ]
        for local_idx, expert_id in enumerate(local_expert_ids):
            experts_down_pp[expert_id] = local_expert_params[local_idx]
        return experts_down_pp