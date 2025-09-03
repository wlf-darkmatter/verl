import torch

from mindspeed.patch_utils import MindSpeedPatchesManager
from verl.models.mcore.weight_converter import McoreToHFWeightConverterDpskv3 as McoreToHFWeightConverterDpskv3Base
from mindspeed.core.megatron_basic.requirements_basic import dummy_compile

MindSpeedPatchesManager.patches_info['torch.compile'].remove_patch()
TRUE_COMPILE = torch.compile
DUMMY_COMPILE = dummy_compile
