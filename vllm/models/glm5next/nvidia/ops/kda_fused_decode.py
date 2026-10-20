# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One kernel for a KDA decode step (one token per request).

Replaces, per layer, the block-diagonal f/g up-projection GEMM, the short
causal conv update, the fused recurrent KDA kernel and the sigmoid-gated
RMSNorm. One program owns one (request, head): it convolves its q/k/v
channels, advances the [V, K] recurrent state and normalizes the head output,
so no intermediate leaves the SM. Weights and both states do not depend on the
in_proj output and are loaded before the PDL wait.

Numerics follow the separate kernels: bf16 conv products accumulated in fp32,
bf16 rounding of the conv output, of the gate projections (a bf16 GEMM output
before) and of the attention output before the norm.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _kda_fused_decode_kernel(
    proj_ptr,
    stride_proj,
    convw_ptr,
    conv_state_ptr,
    stride_cs_seq,
    stride_cs_dim,
    stride_cs_tok,
    state_idx_ptr,
    fgw_ptr,
    stride_fgw,
    a_log_ptr,
    dt_bias_ptr,
    h_ptr,
    stride_h_seq,
    normw_ptr,
    o_ptr,
    stride_o,
    scale,
    eps,
    P: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BETA_OFF: tl.constexpr,
    FA_OFF: tl.constexpr,
    GA_OFF: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    GC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H
    o_d = tl.arange(0, D)

    # ---- inputs that do not depend on the in_proj output ----
    slot = tl.load(state_idx_ptr + i_b).to(tl.int64)
    if slot <= 0:
        return
    ch_q = i_h * D + o_d
    ch_k = P + i_h * D + o_d
    ch_v = 2 * P + i_h * D + o_d
    wq0 = tl.load(convw_ptr + ch_q * 4)
    wq1 = tl.load(convw_ptr + ch_q * 4 + 1)
    wq2 = tl.load(convw_ptr + ch_q * 4 + 2)
    wq3 = tl.load(convw_ptr + ch_q * 4 + 3)
    wk0 = tl.load(convw_ptr + ch_k * 4)
    wk1 = tl.load(convw_ptr + ch_k * 4 + 1)
    wk2 = tl.load(convw_ptr + ch_k * 4 + 2)
    wk3 = tl.load(convw_ptr + ch_k * 4 + 3)
    wv0 = tl.load(convw_ptr + ch_v * 4)
    wv1 = tl.load(convw_ptr + ch_v * 4 + 1)
    wv2 = tl.load(convw_ptr + ch_v * 4 + 2)
    wv3 = tl.load(convw_ptr + ch_v * 4 + 3)
    cs = conv_state_ptr + slot * stride_cs_seq
    pq = cs + ch_q * stride_cs_dim
    pk = cs + ch_k * stride_cs_dim
    pv = cs + ch_v * stride_cs_dim
    cq0 = tl.load(pq)
    cq1 = tl.load(pq + stride_cs_tok)
    cq2 = tl.load(pq + 2 * stride_cs_tok)
    ck0 = tl.load(pk)
    ck1 = tl.load(pk + stride_cs_tok)
    ck2 = tl.load(pk + 2 * stride_cs_tok)
    cv0 = tl.load(pv)
    cv1 = tl.load(pv + stride_cs_tok)
    cv2 = tl.load(pv + 2 * stride_cs_tok)
    b_a = tl.exp(tl.load(a_log_ptr + i_h).to(tl.float32))
    b_bias = tl.load(dt_bias_ptr + i_h * D + o_d).to(tl.float32)
    b_nw = tl.load(normw_ptr + o_d).to(tl.float32)
    p_h = h_ptr + slot * stride_h_seq + i_h * D * D + o_d[:, None] * D + o_d[None, :]
    b_h = tl.load(p_h).to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    row = proj_ptr + i_b * stride_proj
    xq = tl.load(row + ch_q)
    xk = tl.load(row + ch_k)
    xv = tl.load(row + ch_v)
    b_beta = tl.sigmoid(tl.load(row + BETA_OFF + i_h).to(tl.float32))

    # ---- short causal conv (width 4) + SiLU; shift the conv state ----
    acc = tl.zeros([D], dtype=tl.float32)
    acc += cq0 * wq0
    acc += cq1 * wq1
    acc += cq2 * wq2
    acc += xq * wq3
    acc = acc / (1 + tl.exp(-acc))
    b_q = acc.to(xq.dtype).to(tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    acc += ck0 * wk0
    acc += ck1 * wk1
    acc += ck2 * wk2
    acc += xk * wk3
    acc = acc / (1 + tl.exp(-acc))
    b_k = acc.to(xk.dtype).to(tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    acc += cv0 * wv0
    acc += cv1 * wv1
    acc += cv2 * wv2
    acc += xv * wv3
    acc = acc / (1 + tl.exp(-acc))
    b_v = acc.to(xv.dtype).to(tl.float32)
    tl.store(pq, cq1)
    tl.store(pq + stride_cs_tok, cq2)
    tl.store(pq + 2 * stride_cs_tok, xq)
    tl.store(pk, ck1)
    tl.store(pk + stride_cs_tok, ck2)
    tl.store(pk + 2 * stride_cs_tok, xk)
    tl.store(pv, cv1)
    tl.store(pv + stride_cs_tok, cv2)
    tl.store(pv + 2 * stride_cs_tok, xv)

    # ---- gate projections (bf16 GEMM outputs before), in column chunks ----
    o_c = tl.arange(0, GC)
    rows_f = fgw_ptr + (i_h * D + o_d)[:, None] * stride_fgw
    rows_g = fgw_ptr + (P + i_h * D + o_d)[:, None] * stride_fgw + D
    g1 = tl.zeros([D], dtype=tl.float32)
    g2 = tl.zeros([D], dtype=tl.float32)
    for c in tl.static_range(D // GC):
        fa_c = tl.load(row + FA_OFF + c * GC + o_c).to(tl.float32)
        ga_c = tl.load(row + GA_OFF + c * GC + o_c).to(tl.float32)
        wf_c = tl.load(rows_f + c * GC + o_c[None, :]).to(tl.float32)
        wg_c = tl.load(rows_g + c * GC + o_c[None, :]).to(tl.float32)
        g1 += tl.sum(wf_c * fa_c[None, :], axis=1)
        g2 += tl.sum(wg_c * ga_c[None, :], axis=1)
    g1 = g1.to(xq.dtype).to(tl.float32)
    g2 = g2.to(xq.dtype).to(tl.float32)

    # ---- recurrent KDA step (as fused_recurrent_kda, one token) ----
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale
    b_gk = g1 + b_bias
    b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_gk)))
    b_h *= tl.exp(b_gk[None, :])
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_h, b_h.to(p_h.dtype.element_ty))

    # ---- sigmoid-gated RMSNorm over the head ----
    b_o = b_o.to(xq.dtype).to(tl.float32)
    var = tl.sum(b_o * b_o) / D
    rstd = 1 / tl.sqrt(var + eps)
    y = b_o * rstd * b_nw
    y = y * tl.sigmoid(g2)
    tl.store(o_ptr + i_b * stride_o + i_h * D + o_d, y.to(o_ptr.dtype.element_ty))


NUM_WARPS = 4
GATE_CHUNK = 32


def kda_fused_decode(
    proj: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    fg_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    head_dim: int,
    scale: float,
    eps: float,
    lower_bound: float,
) -> None:
    """Decode step for ``state_indices.numel()`` requests of one token each.

    ``proj`` is the in_proj output ``[B, 3P + H + 2D]`` (q | k | v | beta |
    f_a | g_a), ``conv_state`` is viewed as ``(slot, channel, tap)`` (any
    strides), ``recurrent_state`` is ``[slot, H, V, K]`` and ``out`` receives
    the gated-normed attention output ``[B, H * D]``.
    """
    B = state_indices.numel()
    H, D = num_heads, head_dim
    P = H * D
    assert proj.stride(-1) == 1 and proj.shape[-1] == 3 * P + H + 2 * D
    assert conv_weight.shape == (3 * P, 4) and conv_weight.is_contiguous()
    assert conv_state.shape[1] == 3 * P and conv_state.shape[2] >= 3
    assert fg_weight.shape == (2 * P, 2 * D) and fg_weight.stride(-1) == 1
    assert recurrent_state.shape[1:] == (H, D, D)
    assert recurrent_state[0].is_contiguous()
    _kda_fused_decode_kernel[(B * H,)](
        proj,
        proj.stride(0),
        conv_weight,
        conv_state,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        state_indices,
        fg_weight,
        fg_weight.stride(0),
        a_log,
        dt_bias,
        recurrent_state,
        recurrent_state.stride(0),
        norm_weight,
        out,
        out.stride(0),
        scale,
        eps,
        P=P,
        H=H,
        D=D,
        BETA_OFF=3 * P,
        FA_OFF=3 * P + H,
        GA_OFF=3 * P + H + D,
        LOWER_BOUND=lower_bound,
        GC=GATE_CHUNK,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=NUM_WARPS,
    )
