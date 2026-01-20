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
    "dygraph_optimizer.hybrid_parallel_optimizer": [
        "HybridParallelClipGrad",
        "HybridParallelOptimizer",
    ],
    "data_parallel": [
        "DataParallel",
        "sync_dp_moe_params_across_sharding",
    ],
    "pretraining_trainer": [
        "distributed_optimizer_maybe_overwrite",
        "PreTrainingArguments",
        "WeightedDistributedSampler",
        "DummySampler",
        "PretrainingTrainer",
    ],
}

if TYPE_CHECKING:
    from .dygraph_optimizer import *
    from .data_parallel import *
    from .pretraining_trainer import *
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
