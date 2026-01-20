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
    "fp8_quant_weight_callback": [
        "FP8QuantWeightCallback",
    ],
    "gc_callback": [
        "GCCallback",
    ],
    "logging_callback": [
        "LoggingCallback",
    ],
    "moe_correction_bias_adjust_callback": [
        "MoECorrectionBiasAdjustCallback",
    ],
    "moe_logging_callback": [
        "tensor_md5",
        "GlobalRNGCallback",
        "MoeLoggingCallback",
    ],
    "ortho_loss_callback": [
        "OrthogonalCallback",
    ],
    "sp_grad_sync_callback": [
        "SPGradSyncCallback",
    ],
    "tensorboard_callback": [
        "is_tensorboard_available",
        "rewrite_logs",
        "TensorBoardCallback",
    ],
}

if TYPE_CHECKING:
    from .fp8_quant_weight_callback import *
    from .gc_callback import *
    from .logging_callback import *
    from .moe_correction_bias_adjust_callback import *
    from .moe_logging_callback import *
    from .ortho_loss_callback import *
    from .sp_grad_sync_callback import *
    from .tensorboard_callback import *
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
