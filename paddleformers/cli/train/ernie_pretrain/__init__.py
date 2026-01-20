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

from ....utils.lazy_import import _LazyModule

import_structure = {
    "models.ernie.configuration": [
        "ERNIE_PRETRAINED_INIT_CONFIGURATION",
        "ERNIE_PRETRAINED_RESOURCE_FILES_MAP",
        "ErnieMoEConfig",
    ],
    "models.ernie.modeling_moe": [
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
    "models.ernie.modeling_pp": [
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
    "models.ernie.modeling": [
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
    "models.moe.token_dispatcher.fp8_utils": [
        "_get_fp8_weight_and_scale",
        "fused_stack_transpose_quant",
        "split_group_gemm",
        "has_config",
        "ExpertsGroupGemmNode",
        "ExpertsGroupGemmContiguousNode",
        "ExpertsGroupGemmWLCHNode",
    ],
    "models.moe.token_dispatcher.moe_utils": [
        "inplace_offload",
        "inplace_offload_if_needed",
        "topk_to_permuted_indices_single",
        "topk_to_permuted_indices",
        "permute",
        "unpermute",
        "UnZipNode",
        "ZipNode",
    ],
    "models.moe.moe_layer": [
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
    "models.moe.top2_gate": [
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
    "models.comm_utils": [
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
    "models.fp8_linear": [
        "fp8_gemm",
        "padding",
        "Fp8FusedMlpFunc",
        "MemEfficientFp8FusedMlpFunc",
        "Fp8FusedMlp",

    ],
    "models.sequence_parallel_utils": [
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
    "models.utils": [
        "get_global_training_logs",
        "global_training_logs_enabled",
        "inplace_offload",
        "detach_and_requires_grad_",
        "FakeClone",
        "manual_backward",
        "FakeGather",
        "FusedUnpermutation",
    ],
    "src.callbacks.fp8_quant_weight_callback": [
        "FP8QuantWeightCallback",
    ],
    "src.callbacks.gc_callback": [
        "GCCallback",
    ],
    "src.callbacks.logging_callback": [
        "LoggingCallback",
    ],
    "src.callbacks.moe_correction_bias_adjust_callback": [
        "MoECorrectionBiasAdjustCallback",
    ],
    "src.callbacks.moe_logging_callback": [
        "tensor_md5",
        "GlobalRNGCallback",
        "MoeLoggingCallback",
    ],
    "src.callbacks.ortho_loss_callback": [
        "OrthogonalCallback",
    ],
    "src.callbacks.sp_grad_sync_callback": [
        "SPGradSyncCallback",
    ],
    "src.callbacks.tensorboard_callback": [
        "is_tensorboard_available",
        "rewrite_logs",
        "TensorBoardCallback",
    ],
    "src.clip.moe_clip": [
        "ClipGradForMOEByGlobalNorm",
    ],
    "src.lr_schedulers.cosine_lr": [
        "get_cosine_schedule_with_warmup",
    ],
    "src.lr_schedulers.wsd_lr": [
        "get_wsd_schedule_with_warmup",
    ],
    "src.tokenizers.tokenization_eb_v2": [
        "ErnieBotTokenizer",
        "add_special_tokens",
    ],
    "src.trainers.dygraph_optimizer.hybrid_parallel_optimizer": [
        "HybridParallelClipGrad",
        "HybridParallelOptimizer",
    ],
    "src.trainers.data_parallel": [
        "DataParallel",
        "sync_dp_moe_params_across_sharding",
    ],
    "src.trainers.pretraining_trainer": [
        "distributed_optimizer_maybe_overwrite",
        "PreTrainingArguments",
        "WeightedDistributedSampler",
        "DummySampler",
        "PretrainingTrainer",
    ],
    "src.utils.logging": [
        "setup_logger_output_file",
    ],
    "src.utils.misc": [
        "SmoothedValue",
        "TrainingLogs",
    ],
    "src.utils.seed_utils": [
        "set_seed",
    ],
    "src.utils.training_utils": [
        "reset_per_device_batch_size",
    ],
    "model_config": [
        "ModelConfig",
    ],
    "workflow": [
        "load_huggingface_checkpoint",
        "get_expected_state_dict",
        "update_model_config_from_args",
        "get_tp_split_ckpt",
        "AllArguments",
        "ExpConfig",
        "create_pretrained_dataset",
        "run_ernie_pretrain",
    ],
}

if TYPE_CHECKING:
    from .src import *
    from .models import *
    from .model_config import *
    from .workflow import *
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
