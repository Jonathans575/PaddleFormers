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
    "callbacks.fp8_quant_weight_callback": [
        "FP8QuantWeightCallback",
    ],
    "callbacks.gc_callback": [
        "GCCallback",
    ],
    "callbacks.logging_callback": [
        "LoggingCallback",
    ],
    "callbacks.moe_correction_bias_adjust_callback": [
        "MoECorrectionBiasAdjustCallback",
    ],
    "callbacks.moe_logging_callback": [
        "tensor_md5",
        "GlobalRNGCallback",
        "MoeLoggingCallback",
    ],
    "callbacks.ortho_loss_callback": [
        "OrthogonalCallback",
    ],
    "callbacks.sp_grad_sync_callback": [
        "SPGradSyncCallback",
    ],
    "callbacks.tensorboard_callback": [
        "is_tensorboard_available",
        "rewrite_logs",
        "TensorBoardCallback",
    ],
    "clip.moe_clip": [
        "ClipGradForMOEByGlobalNorm",
    ],
    "lr_schedulers.cosine_lr": [
        "get_cosine_schedule_with_warmup",
    ],
    "lr_schedulers.wsd_lr": [
        "get_wsd_schedule_with_warmup",
    ],
    "tokenizers.tokenization_eb_v2": [
        "ErnieBotTokenizer",
        "add_special_tokens",
    ],
    "trainers.dygraph_optimizer.hybrid_parallel_optimizer": [
        "HybridParallelClipGrad",
        "HybridParallelOptimizer",
    ],
    "trainers.data_parallel": [
        "DataParallel",
        "sync_dp_moe_params_across_sharding",
    ],
    "trainers.pretraining_trainer": [
        "distributed_optimizer_maybe_overwrite",
        "PreTrainingArguments",
        "WeightedDistributedSampler",
        "DummySampler",
        "PretrainingTrainer",
    ],
    "utils.logging": [
        "setup_logger_output_file",
    ],
    "utils.misc": [
        "SmoothedValue",
        "TrainingLogs",
    ],
    "utils.seed_utils": [
        "set_seed",
    ],
    "utils.training_utils": [
        "reset_per_device_batch_size",
    ],
}

if TYPE_CHECKING:
    from .callbacks import *
    from .clip import *
    from .lr_schedulers import *
    from .tokenizers import *
    from .trainers import *
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
