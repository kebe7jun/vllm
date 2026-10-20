# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-indexer prologue for decode steps (<= 16 tokens) in one kernel.

Replaces the fp32 head-weight GEMM (cast + cuBLAS dot + reduce), the k
LayerNorm, the Hadamard-128 + FP8 quantization of q, the weight scaling and the
kpool gate GEMM (cuBLAS split-K + reduce): eight launches per MLA layer. For
multiple tokens, one program per (token, head) owns the head's weight row,
its q row and ``head_dim / n_head`` gate rows; head 0 also normalizes k.
Single-token decode keeps its original kernel. The weight rows
are loaded before the PDL wait, so only the token inputs remain on the
critical path.
"""

import torch

from vllm.models.glm5next.nvidia.ops.kpool_compress import _hadamard128
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MAX_TOKENS = 16


@triton.jit
def _indexer_prologue_kernel(
    hid_ptr,
    stride_hid,
    q_ptr,
    stride_q,
    kw_ptr,
    stride_kw,
    wp_ptr,
    stride_wp,
    gw_ptr,
    stride_gw,
    knorm_w_ptr,
    knorm_b_ptr,
    qout_ptr,
    qs_ptr,
    w_out_ptr,
    k_out_ptr,
    g_out_ptr,
    M,
    eps,
    w_scale,
    H: tl.constexpr,
    NH: tl.constexpr,
    D: tl.constexpr,
    GR: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    j = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GR)
    # weights: independent of the previous kernels
    wp = tl.load(wp_ptr + j * stride_wp + offs_h).to(tl.float32)
    gw = tl.load(
        gw_ptr + (j * GR + offs_g)[:, None].to(tl.int64) * stride_gw + offs_h[None, :]
    ).to(tl.float32)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    for m in range(M):
        h = tl.load(hid_ptr + m * stride_hid + offs_h).to(tl.float32)
        # fp32 head weight (as hidden.float() @ wp.float().T)
        wv = tl.sum(h * wp, axis=0)
        # kpool gate rows (bf16 GEMM output)
        gv = tl.sum(gw * h[None, :], axis=1)
        tl.store(g_out_ptr + m * D + j * GR + offs_g, gv.to(g_out_ptr.dtype.element_ty))
        # Hadamard-128 + bf16 rounding + ue8m0 FP8 quant (as fwht128_quant_fp8)
        x = tl.load(q_ptr + m * stride_q + j * D + offs_d).to(tl.float32)
        x = _hadamard128(x)
        x = x.to(tl.bfloat16).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
        qsc = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
        y = tl.minimum(tl.maximum(x / qsc, -448.0), 448.0)
        tl.store(qout_ptr + (m * NH + j) * D + offs_d, y)
        tl.store(qs_ptr + m * NH + j, qsc)
        tl.store(w_out_ptr + m * NH + j, wv * qsc * w_scale)
        if j == 0:
            # k LayerNorm in fp32 (as F.layer_norm(k.float()))
            k = tl.load(kw_ptr + m * stride_kw + offs_d).to(tl.float32)
            mean = tl.sum(k, axis=0) / D
            kc = k - mean
            var = tl.sum(kc * kc, axis=0) / D
            kn = kc * (1.0 / tl.sqrt(var + eps))
            kn = kn * tl.load(knorm_w_ptr + offs_d) + tl.load(knorm_b_ptr + offs_d)
            tl.store(k_out_ptr + m * D + offs_d, kn.to(k_out_ptr.dtype.element_ty))


@triton.jit
def _indexer_row_grid_kernel(
    hid_ptr,
    stride_hid,
    q_ptr,
    stride_q,
    kw_ptr,
    stride_kw,
    wp_ptr,
    stride_wp,
    gw_ptr,
    stride_gw,
    knorm_w_ptr,
    knorm_b_ptr,
    qout_ptr,
    qs_ptr,
    w_out_ptr,
    k_out_ptr,
    g_out_ptr,
    M,
    eps,
    w_scale,
    H: tl.constexpr,
    NH: tl.constexpr,
    D: tl.constexpr,
    GR: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    j = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GR)
    wp = tl.load(wp_ptr + j * stride_wp + offs_h).to(tl.float32)
    gw = tl.load(
        gw_ptr + (j * GR + offs_g)[:, None].to(tl.int64) * stride_gw + offs_h[None, :]
    ).to(tl.float32)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    m = tl.program_id(1)
    h = tl.load(hid_ptr + m * stride_hid + offs_h).to(tl.float32)
    wv = tl.sum(h * wp, axis=0)
    gv = tl.sum(gw * h[None, :], axis=1)
    tl.store(g_out_ptr + m * D + j * GR + offs_g, gv.to(g_out_ptr.dtype.element_ty))
    x = tl.load(q_ptr + m * stride_q + j * D + offs_d).to(tl.float32)
    x = _hadamard128(x)
    x = x.to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 0.0001)
    qsc = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
    y = tl.minimum(tl.maximum(x / qsc, -448.0), 448.0)
    tl.store(qout_ptr + (m * NH + j) * D + offs_d, y)
    tl.store(qs_ptr + m * NH + j, qsc)
    tl.store(w_out_ptr + m * NH + j, wv * qsc * w_scale)
    if j == 0:
        k = tl.load(kw_ptr + m * stride_kw + offs_d).to(tl.float32)
        mean = tl.sum(k, axis=0) / D
        kc = k - mean
        var = tl.sum(kc * kc, axis=0) / D
        kn = kc * (1.0 / tl.sqrt(var + eps))
        kn = kn * tl.load(knorm_w_ptr + offs_d) + tl.load(knorm_b_ptr + offs_d)
        tl.store(k_out_ptr + m * D + offs_d, kn.to(k_out_ptr.dtype.element_ty))


NUM_WARPS = 8


def indexer_decode_prologue(
    hidden_states: torch.Tensor,
    q: torch.Tensor,
    kw: torch.Tensor,
    weights_proj: torch.Tensor,
    gate: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    weight_scale: float,
) -> tuple[torch.Tensor, ...]:
    """Returns (q_fp8 [M, NH, D], q_scale [M, NH, 1], weights [M, NH] fp32,
    k [M, D], gate_score [M, D]) for ``M <= 16`` tokens."""
    M, H = hidden_states.shape
    NH = weights_proj.shape[0]
    D = gate.shape[0]
    assert M <= MAX_TOKENS and D % NH == 0 and q.shape[-1] == NH * D
    assert hidden_states.stride(1) == 1 and q.stride(-1) == 1 and kw.stride(1) == 1
    assert weights_proj.stride(1) == 1 and gate.stride(1) == 1
    dev = hidden_states.device
    q_fp8 = torch.empty((M, NH, D), dtype=torch.float8_e4m3fn, device=dev)
    q_scale = torch.empty((M, NH, 1), dtype=torch.float32, device=dev)
    weights = torch.empty((M, NH), dtype=torch.float32, device=dev)
    k = torch.empty((M, D), dtype=kw.dtype, device=dev)
    gate_score = torch.empty((M, D), dtype=hidden_states.dtype, device=dev)
    kernel = _indexer_row_grid_kernel if M > 1 else _indexer_prologue_kernel
    grid = (NH, M) if M > 1 else (NH,)
    kernel[grid](
        hidden_states,
        hidden_states.stride(0),
        q,
        q.stride(0),
        kw,
        kw.stride(0),
        weights_proj,
        weights_proj.stride(0),
        gate,
        gate.stride(0),
        k_norm_weight,
        k_norm_bias,
        q_fp8,
        q_scale,
        weights,
        k,
        gate_score,
        M,
        k_norm_eps,
        weight_scale,
        H=H,
        NH=NH,
        D=D,
        GR=D // NH,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=NUM_WARPS,
    )
    return q_fp8, q_scale, weights, k, gate_score
