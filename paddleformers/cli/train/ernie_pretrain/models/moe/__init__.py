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
    "token_dispatcher.fp8_utils": [
        "_get_fp8_weight_and_scale",
        "fused_stack_transpose_quant",
        "split_group_gemm",
        "has_config",
        "ExpertsGroupGemmNode",
        "ExpertsGroupGemmContiguousNode",
        "ExpertsGroupGemmWLCHNode",
    ],
    "token_dispatcher.moe_utils": [
        "inplace_offload",
        "inplace_offload_if_needed",
        "topk_to_permuted_indices_single",
        "topk_to_permuted_indices",
        "permute",
        "unpermute",
        "UnZipNode",
        "ZipNode",
    ],
    "moe_layer": [
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
    "top2_gate": [
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
}

if TYPE_CHECKING:
    from .token_dispatcher.fp8_utils import *
    from .token_dispatcher.moe_utils import *
    from .moe_layer import *
    from .top2_gate import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
