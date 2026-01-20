# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
    "ernie.configuration": [
        "ERNIE_PRETRAINED_INIT_CONFIGURATION",
        "ERNIE_PRETRAINED_RESOURCE_FILES_MAP",
        "ErnieMoEConfig",
    ],
    "ernie.modeling_moe": [
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
    "ernie.modeling_pp": [
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
    "ernie.modeling": [
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
    "moe.token_dispatcher.fp8_utils": [
        "_get_fp8_weight_and_scale",
        "fused_stack_transpose_quant",
        "split_group_gemm",
        "has_config",
        "ExpertsGroupGemmNode",
        "ExpertsGroupGemmContiguousNode",
        "ExpertsGroupGemmWLCHNode",
    ],
    "moe.token_dispatcher.moe_utils": [
        "inplace_offload",
        "inplace_offload_if_needed",
        "topk_to_permuted_indices_single",
        "topk_to_permuted_indices",
        "permute",
        "unpermute",
        "UnZipNode",
        "ZipNode",
    ],
    "moe.moe_layer": [
        "set_grad_in_dtype_non_consistent",
        "Fp8MoeGateDispatchAndQuant",
        "recompute_fwd_gate_up_func",
        "MoEStatics",
        "GateCombine",
        "FusionFP8Expert",
        "AlltoAll",
        "AlltoAllExpertOverlap",
        "AlltoAllAsync",
        "dispatching",
        "combining_fused",
        "ReshapeKeepGradDtype",
        "MOELayer",
        "FP8FusedWLCHFunc",
        "Fp8FusedMoeFunc",
    ],
    "moe.top2_gate": [
        "cal_aux_loss_func",
        "masked_fill",
        "CalAuxLossFunctor",
        "cast_if_needed",
        "FusedGateDetachMatmul",
        "gate_detach_matmul",
        "compute_optimal_transport",
        "Top2Gate",
        "cal_orthogonal_loss_opt_each_weight_func",
        "TopKGateFused",
    ],
    "comm_utils": [
        "scatter",
        "mp_slice",
        "all_gather_varlen",
        "scatter_varlen",
        "all_gather",
        "reduce_scatter",
        "subbatch",
        "gather_varlen",
        "profile",
    ],
    "fp8_linear": [
        "fp8_gemm",
        "padding",
        "Fp8FusedMlpFunc",
        "MemEfficientFp8FusedMlpFunc",
        "Fp8FusedMlp",

    ],
    "sequence_parallel_utils": [
        "get_async_loader",
        "hack_offload_wait",
        "hack_reload_wait",
        "ScatterOp",
        "GatherOp",
        "AllGatherOp",
        "ReduceScatterOp",
        "AllGatherVarlenOp",
        "GemmReduceScatterOp",
        "AllGatherGemmOp",
        "sequence_parallel_sparse_mask_labels",
        "mark_as_sequence_parallel_parameter",
        "is_sequence_parallel_parameter",
        "create_fused_allreduce_gradient_hook",
        "create_non_fused_allreduce_gradient_hook",
        "register_sequence_parallel_allreduce_hooks",
        "is_fused_matmul_bias_supported",
        "ColumnSequenceParallelLinear",
        "MPScale",
        "RowSequenceParallelLinear"
    ],
    "utils": [
        "get_global_training_logs",
        "global_training_logs_enabled",
        "inplace_offload",
        "detach_and_requires_grad_",
        "FakeClone",
        "manual_backward",
        "FakeGather",
        "FusedUnpermutation",
    ],
}

if TYPE_CHECKING:
    from .moe import *
    from .ernie import *
    from .comm_utils import *
    from .fp8_linear import *
    from .sequence_parallel_utils import *
    from .utils import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )

#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
