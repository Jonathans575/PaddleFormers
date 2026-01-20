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
    "fp8_utils": [
        "_get_fp8_weight_and_scale",
        "fused_stack_transpose_quant",
        "split_group_gemm",
        "has_config",
        "ExpertsGroupGemmNode",
        "ExpertsGroupGemmContiguousNode",
        "ExpertsGroupGemmWLCHNode",
    ],
    "moe_utils": [
        "inplace_offload",
        "inplace_offload_if_needed",
        "topk_to_permuted_indices_single",
        "topk_to_permuted_indices",
        "permute",
        "unpermute",
        "UnZipNode",
        "ZipNode",
    ],
}

if TYPE_CHECKING:
    from .fp8_utils import *
    from .moe_utils import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
