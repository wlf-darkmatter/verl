import torch

from mindspeed.patch_utils import MindSpeedPatchesManager
from verl.models.mcore.weight_converter import McoreToHFWeightConverterDpskv3 as McoreToHFWeightConverterDpskv3Base
from mindspeed.core.megatron_bsic.requirements_basic import dummy_compile

MindSpeedPatchesManager.patches_info['torch.compile'].remove_patch()
TRUE_COMPILE = torch.compile
DUMMY_COMPILE = dummy_compile


class McoreToHFWeightConverterDpskv3(McoreToHFWeightConverterDpskv3Base):
    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        name_map_after_layer = {
            "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
            "mlp.linear_fc2.weight": "mlp.down_proj.weight",
            "mlp.shared_experts.linear_fc2.weight": "mlp.shared_experts.down_proj.weight",
            "mlp.linear_fc1.weight": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            "mlp.shared_experts.linear_fc1.weight": [
                "mlp.shared_experts.gate_proj.weight",
                "mlp.shared_experts.up_proj.weight",
            ],
            "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
            "mlp.router.weight": "mlp.gate.weight",
            "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        }
        convert_names = []
        layer_number = name.split(".")[2]
        name_after_layer = name.split(f".{layer_number}.")[1]
        if name_after_layer in name_map_after_layer:
            mapped_name = name_map_after_layer[name_after_layer]
            if isinstance(mapped_name, list):
                assert len(params) == len(mapped_name)
                for one in mapped_name:
                    convert_names.append(f"model.layers.{layer_number}.{one}")
            else:
                assert len(params) == 1
                convert_names.append(f"model.layers.{layer_number}.{mapped_name}")
        else:
            if "mlp.experts.linear_fc1.weight" in name:
                expert_id = name.split("weight")[-1]
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
                assert len(params) == 2
            elif "mlp.experts.linear_fc2.weight" in name:
                expert_id = name.split("weight")[-1]
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
                assert len(params) == 1
            elif "mlp.experts.weight1" in name:
                num_moe_experts = int(len(params) // 2)
                for expert_id in range(num_moe_experts):
                    convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
                    convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
            elif "mlp.experts.weight2" in name:
                num_moe_experts = int(len(params) // 2)
                for expert_id in range(num_moe_experts):
                    convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
            else:
                raise NotImplementedError(f"Unsupported parameter name: {name}")

        return convert_names, params


def mcore_models_adaptation():
    from verl.models.mcore.registry import MODEL_CONFIG_CONVERTER_REGISTRY, SupportedModel

    MODEL_CONFIG_CONVERTER_REGISTRY[SupportedModel.DEEPSEEK_V3] = McoreToHFWeightConverterDpskv3

mcore_models_adaptation()
