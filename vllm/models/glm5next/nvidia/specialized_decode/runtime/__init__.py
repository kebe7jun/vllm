# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the measured H200/c1/DFlash K3 precision profile."""

import os
from dataclasses import dataclass

from .contracts import deployment_from_vllm, enabled, validate


@dataclass(frozen=True)
class DecodeConfig:
    query_tokens: int = 4
    max_context: int = 8192
    ar_bf16_acc: bool = True


def specialized_decode_config(config):
    if not enabled(config.additional_config or {}):
        return None
    import torch

    from vllm import envs
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    validate(
        deployment_from_vllm(
            config,
            capability.to_int() if capability is not None else 0,
            current_platform.get_device_name(),
        )
    )
    if (
        current_platform.num_compute_units(torch.accelerator.current_device_index())
        != 132
    ):
        raise ValueError("GLM specialized decode requires an unpartitioned H200")
    if envs.VLLM_BATCH_INVARIANT:
        raise ValueError("GLM specialized decode does not support batch-invariant mode")
    return DecodeConfig(max_context=config.model_config.max_model_len)


def enable_measured_profile():
    """Bind the inherited branch's hooks before weight loading and graph capture.

    This compatibility bridge keeps the already-measured generic hooks separate
    from the model-owned executor. One vLLM worker process owns one target.
    """
    from vllm import envs

    flags = {
        "VLLM_MOE_FUSED_CLAMP_ACT_QUANT": "1",
        "VLLM_FUSE_SHARED_EXPERTS": "1",
        "VLLM_MOE_SMALL_M_GEMV": "1",
        "VLLM_MOE_SMALL_M_PDL": "1",
        "VLLM_SMALL_M_WO_FP8": "1",
        "VLLM_SMALL_M_GEMV_PREFETCH": "1",
        "VLLM_MHC_FUSED_ALLREDUCE": "1",
        "VLLM_MHC_SINGLE_KERNEL": "1",
        "VLLM_KDA_FUSED_DECODE": "1",
        "VLLM_INDEXER_FUSED_PROLOGUE": "1",
        "VLLM_DFLASH2_BF16_LINEAR": "1",
        "VLLM_DFLASH2_BF16_SHARED_HEAD": "0",
        "GLM53_Q4_KDA": "1",
        "GLM53_Q6_KDA": "0",
        "GLM53_GROUPED_TC": "0",
        "GLM53_AR_BF16_ACC": "1",
    }
    conflicts = [
        name
        for name, value in flags.items()
        if name in os.environ and os.environ[name] != value
    ]
    if conflicts:
        raise ValueError(
            "GLM specialized profile conflicts with: " + ", ".join(conflicts)
        )
    os.environ.update(flags)
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()
    # These constants can have been imported before the target is constructed.
    from ..ops import small_m_moe, wo_fp8_gemv

    small_m_moe.W2_PREFETCH = True
    wo_fp8_gemv.M1_PREFETCH_MAX_KS = 1024
