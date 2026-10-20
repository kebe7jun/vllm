# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict opt-in integration for c1, fixed-width target verification."""

import os

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .kda_q4_parallel_history_fused_norm import (
    kda_q4_parallel_history_fused_norm,
)


def legacy_try_q4(layer, projected, gate_projected):
    if os.environ.get("GLM53_Q4_KDA") != "1":
        return None
    context = get_forward_context()
    if context.cudagraph_runtime_mode not in (CUDAGraphMode.FULL, CUDAGraphMode.NONE):
        return None
    metadata = context.attn_metadata
    md = metadata.get(layer.prefix) if isinstance(metadata, dict) else None
    if not isinstance(md, GDNAttentionMetadata):
        return None
    if (
        md.num_prefills
        or md.num_decodes
        or md.num_spec_decodes != 1
        or md.num_spec_decode_tokens != 4
        or md.num_actual_tokens != 4
        or layer.conv_size != 4
        or layer.head_dim != 128
        or layer.local_num_heads != 8
        or layer.q_conv1d.bias is not None
        or layer.kda_lower_bound not in (None, -5.0)
    ):
        return None
    indices = md.spec_state_indices_tensor
    accepted = md.num_accepted_tokens
    if (
        indices is None
        or accepted is None
        or indices.shape != (1, 4)
        or indices.stride(1) != 1
        or projected.dtype != torch.bfloat16
    ):
        return None
    conv, recurrent = layer.kv_cache
    if not layer._conv_state_dim_first:
        conv = conv.transpose(-1, -2)
    if (
        conv.shape[1:] != (3072, 6)
        or conv.dtype != torch.bfloat16
        or recurrent.dtype != torch.float32
        or recurrent.shape[1:] != (8, 128, 128)
    ):
        return None
    out = torch.empty((4, 1024), dtype=projected.dtype, device=projected.device)
    arrivals = getattr(layer, "_q4_history_arrivals", None)
    if arrivals is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Q4 arrivals must be allocated in eager warmup")
        arrivals = torch.zeros((2, 8), dtype=torch.int32, device=projected.device)
        layer._q4_history_arrivals = arrivals
    kda_q4_parallel_history_fused_norm(
        projected,
        gate_projected,
        layer._get_merged_conv_weight(),
        conv,
        indices,
        accepted,
        layer.A_log,
        layer.dt_bias,
        recurrent,
        layer.o_norm.weight,
        out,
        arrivals,
        eps=layer.o_norm.eps,
    )
    layer._q4_capture_calls = getattr(layer, "_q4_capture_calls", 0) + 1
    return out


def try_q4(layer, projected, gate_projected):
    prepared = getattr(layer, "_glm53_prepared_q4", None)
    if prepared is not None and prepared.owner.active:
        return prepared.run(projected, gate_projected)
    return legacy_try_q4(layer, projected, gate_projected)
