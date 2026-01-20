# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys
from typing import TYPE_CHECKING

from ..utils.lazy_import import _LazyModule

import_structure = {
    "configuration": [
        "ERNIE_PRETRAINED_INIT_CONFIGURATION",
        "ERNIE_PRETRAINED_RESOURCE_FILES_MAP",
        "ErnieMoEConfig",
    ],
    "modeling_moe": [
        "BaseModelOutputWithPastAndCrossAttentions",
        "CausalLMOutputWithCrossAttentions",
        "get_gate",
        "build_mpdp_group",
        "_parse_moe_group",
        "moe_ep2mp",
        "moe_statedict_cherry_pick",
        "moe_statedict_upcycle",
        "ErnieMoeMLP",
        "ErnieMoeDenseExpert",
        "BMMLinear",
        "ErnieMoeMLPFused",
        "FusedLinearAddNormFunc",
        "FusedLinearAddNorm",
        "FusedRMSLinearFunc",
        "FusedRMSLinear",
        "ErnieMoEAttention",
        "FakeMoERouterLoss",
        "ErnieDecoderLayer",
        "ErniePretrainedModel",
        "ErnieModel",
        "ErnieMoELMHead",
        "ErniePretrainingCriterion",
        "ErnieMoEForCausalLM",
    ],
    "modeling_pp": [
        "ErnieEmbeddingPipe",
        "MTPEmbeddingPipe",
        "EmptyLayer",
        "ErnieDecoderLayerPipe",
        "RMSNormPipe",
        "ErnieMoELMHeadPipe",
        "MTPLayer",
        "ErniePretrainingCriterionPipe",
        "PipelinePretrainedModel",
        "get_pp_vp_split_layers",
        "ErnieMoEForCausalLMPipe",
    ],
    "modeling": [
        "get_triangle_upper_mask",
        "gqa_qkv_split_func",
        "gqa_qkv_merge_func",
        "parallel_matmul",
        "calc_lm_head_logits",
        "finfo",
        "masked_fill",
        "mem_eff_attn",
        "inbatch_pack_offset_to_attn_mask_start_row_indices",
        "scaled_dot_product_attention",
        "_make_causal_mask",
        "_expand_mask",
        "FusedDropoutImpl",
        "RMSNorm",
        "RotaryEmbedding",
        "RopeEmbeddingLegacy",
        "ErnieMLP",
        "ErnieAttention",
        "ErnieDecoderLayer",
        "ErniePretrainedModel",
        "ErnieModel",
        "FusedHeadParallelCrossEntropy",
        "ErniePretrainingCriterion",
        "ErnieLMHead",
        "ErnieForCausalLM",
    ],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling_moe import *
    from .modeling_pp import *
    from .modeling import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
