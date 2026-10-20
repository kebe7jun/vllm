"""Optional c1 q4 full fusion: BV8 recurrence and last-arrival gated RMSNorm.

Caller owns persistent int32[2,8] counters. No overlapping invocations or PDL.
The accepted recurrent source must be active; this variant returns final output.
This file is isolated and has not been compiled or executed on a GPU.
"""

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton


@triton.jit
def _conv(x, c0, c1, c2, w0, w1, w2, w3):
    value = tl.full(x.shape, 0, tl.float32)
    value += c0 * w0
    value += c1 * w1
    value += c2 * w2
    value += x * w3
    return (value / (1 + tl.exp(-value))).to(x.dtype).to(tl.float32)


@triton.jit
def _q4_parallel_history_fused_norm(
    proj,
    fg,
    convw,
    cs,
    indices,
    accepted,
    alog,
    bias,
    states,
    output,
    arrivals,
    norm_weight,
    eps,
    sp: tl.constexpr,
    sf: tl.constexpr,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    sw0: tl.constexpr,
    sw1: tl.constexpr,
    sh: tl.constexpr,
    so: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    Q: tl.constexpr,
    BV: tl.constexpr,
):
    h, part = tl.program_id(0), tl.program_id(1)
    P: tl.constexpr = H * D
    kd = tl.arange(0, D)
    vd = part * BV + tl.arange(0, BV)
    ac = tl.load(accepted).to(tl.int64) - 1
    slot = tl.load(indices).to(tl.int64)
    src = tl.load(indices + ac).to(tl.int64)
    if slot <= 0:
        return
    cq, ck, cv = h * D + kd, P + h * D + kd, 2 * P + h * D + vd
    pq = cs + slot * cs0 + cq * cs1
    pk = cs + slot * cs0 + ck * cs1
    pv = cs + slot * cs0 + cv * cs1
    q0 = tl.load(pq + ac * cs2)
    q1 = tl.load(pq + (ac + 1) * cs2)
    q2 = tl.load(pq + (ac + 2) * cs2)
    k0 = tl.load(pk + ac * cs2)
    k1 = tl.load(pk + (ac + 1) * cs2)
    k2 = tl.load(pk + (ac + 2) * cs2)
    v0 = tl.load(pv + ac * cs2)
    v1 = tl.load(pv + (ac + 1) * cs2)
    v2 = tl.load(pv + (ac + 2) * cs2)

    # All CTA lanes finish every history read before its single arrival.
    tl.debug_barrier()
    arrival = tl.atomic_add(arrivals + h, 1, sem="acq_rel", scope="gpu")
    # Transfer the elected atomic lane's acquire to every publishing lane.
    tl.debug_barrier()
    if arrival == D // BV - 1:
        tl.store(pq, q1)
        tl.store(pq + cs2, q2)
        tl.store(pk, k1)
        tl.store(pk + cs2, k2)
        for t in tl.static_range(Q):
            tl.store(pq + (t + 2) * cs2, tl.load(proj + t * sp + cq))
            tl.store(pk + (t + 2) * cs2, tl.load(proj + t * sp + ck))
        tl.debug_barrier()
        tl.atomic_xchg(arrivals + h, 0, sem="release", scope="gpu")

    # V history is disjoint across partitions and never needs an arrival.
    tl.store(pv, v1)
    tl.store(pv + cs2, v2)
    for t in tl.static_range(Q):
        tl.store(pv + (t + 2) * cs2, tl.load(proj + t * sp + cv))
    if src <= 0:
        return

    wq0 = tl.load(convw + cq * sw0)
    wq1 = tl.load(convw + cq * sw0 + sw1)
    wq2 = tl.load(convw + cq * sw0 + 2 * sw1)
    wq3 = tl.load(convw + cq * sw0 + 3 * sw1)
    wk0 = tl.load(convw + ck * sw0)
    wk1 = tl.load(convw + ck * sw0 + sw1)
    wk2 = tl.load(convw + ck * sw0 + 2 * sw1)
    wk3 = tl.load(convw + ck * sw0 + 3 * sw1)
    wv0 = tl.load(convw + cv * sw0)
    wv1 = tl.load(convw + cv * sw0 + sw1)
    wv2 = tl.load(convw + cv * sw0 + 2 * sw1)
    wv3 = tl.load(convw + cv * sw0 + 3 * sw1)
    ah = tl.exp(tl.load(alog + h).to(tl.float32))
    gb = tl.load(bias + h * D + kd).to(tl.float32)
    offset = h * D * D + vd[:, None] * D + kd[None, :]
    # Same [8, 128] FP32 source tile as baseline, including its initial +0.
    state = tl.zeros([BV, D], tl.float32)
    state += tl.load(states + src * sh + offset).to(tl.float32)
    for t in range(Q):
        row = proj + t * sp
        xq = tl.load(row + cq)
        xk = tl.load(row + ck)
        xv = tl.load(row + cv)
        qq = _conv(xq, q0, q1, q2, wq0, wq1, wq2, wq3)
        kk = _conv(xk, k0, k1, k2, wk0, wk1, wk2, wk3)
        vv = _conv(xv, v0, v1, v2, wv0, wv1, wv2, wv3)
        q0, q1, q2 = q1, q2, xq
        k0, k1, k2 = k1, k2, xk
        v0, v1, v2 = v1, v2, xv
        qq = qq / tl.sqrt(tl.sum(qq * qq) + 1e-6)
        kk = kk / tl.sqrt(tl.sum(kk * kk) + 1e-6)
        qq = qq * D**-0.5
        gate = tl.load(fg + t * sf + h * D + kd).to(tl.float32)
        gate += gb
        gate = -5.0 / (1.0 + tl.exp(-(ah * gate)))
        state *= exp(gate[None, :])
        vv -= tl.sum(state * kk[None, :], 1)
        beta = tl.sigmoid(tl.load(row + 3 * P + h).to(tl.float32))
        vv *= beta
        state += vv[:, None] * kk[None, :]
        value = tl.sum(state * qq[None, :], 1)
        tl.store(
            output + t * so + h * D + vd,
            value.to(output.dtype.element_ty),
            cache_modifier=".cg",
        )
        target = tl.load(indices + t).to(tl.int64)
        if target > 0:
            tl.store(states + target * sh + offset, state)

    # Every partition publishes all four BF16 rows before its norm arrival.
    tl.debug_barrier()
    norm_arrival = tl.atomic_add(arrivals + H + h, 1, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    if norm_arrival == D // BV - 1:
        ts = tl.arange(0, Q)
        ds = tl.arange(0, D)
        raw_ptr = output + ts[:, None] * so + h * D + ds[None, :]
        raw = tl.load(raw_ptr, cache_modifier=".cg").to(tl.float32)
        # Same FP32 formula/order as the frozen sigmoid RMSNorm, eps=1e-5.
        variance = tl.sum(raw * raw, axis=1) / D
        rstd = 1 / tl.sqrt(variance + eps)
        normalized = raw * rstd[:, None]
        nw = tl.load(norm_weight + ds).to(tl.float32)
        value = normalized * nw[None, :]
        gate2 = tl.load(fg + ts[:, None] * sf + P + h * D + ds[None, :]).to(tl.float32)
        value = value * tl.sigmoid(gate2)
        tl.store(raw_ptr, value.to(output.dtype.element_ty), cache_modifier=".cg")
        tl.debug_barrier()
        tl.atomic_xchg(arrivals + H + h, 0, sem="release", scope="gpu")


def kda_q4_parallel_history_fused_norm(
    proj,
    fg,
    conv_weight,
    conv_state,
    indices,
    accepted,
    a_log,
    g_bias,
    recurrent_state,
    norm_weight,
    out,
    arrivals,
    *,
    eps=1e-5,
):
    """Advance active c1 q4 and normalize output in place; return compiled code.

    The caller allocates zeroed int32[2,8] arrivals before capture and retains
    scratch/output/state for graph lifetime. Row zero counts history readers;
    row one counts all-four-token output producers. Full previous-kernel
    completion is required before reuse. Counters return to zero without a
    reset launch. Norm is affine BF16 sigmoid RMSNorm, no residual/bias/prenorm,
    exactly eps=1e-5. Accepted source and convolution slot must be positive;
    invalid source/padded invocations skip output and must use another path.
    """
    import torch

    h, d, q, p = 8, 128, 4, 1024
    assert proj.shape == (q, 3 * p + h + 2 * d)
    assert fg.shape == (q, 2 * p) and fg.stride(1) == proj.stride(1) == 1
    assert conv_weight.shape == (3 * p, 4)
    assert conv_state.shape[1:] == (3 * p, q + 2)
    assert recurrent_state.shape[1:] == (h, d, d)
    assert recurrent_state.stride()[1:] == (d * d, d, 1)
    assert out.shape == (q, p) and out.stride(1) == 1
    assert indices.shape == (1, q) and indices.stride(1) == 1
    assert accepted.shape == (1,) and accepted.is_contiguous()
    assert a_log.numel() == h and a_log.is_contiguous()
    assert g_bias.numel() == p and g_bias.is_contiguous()
    assert arrivals.shape == (2, h) and arrivals.is_contiguous()
    assert norm_weight.shape == (d,) and norm_weight.is_contiguous()
    assert norm_weight.dtype == torch.bfloat16 and eps == 1e-5
    assert arrivals.dtype == torch.int32
    assert indices.dtype == accepted.dtype == torch.int32
    assert proj.dtype == fg.dtype == conv_state.dtype == out.dtype == torch.bfloat16
    assert recurrent_state.dtype == torch.float32
    assert a_log.dtype == g_bias.dtype == torch.float32
    assert conv_weight.dtype in (torch.float32, torch.bfloat16)
    tensors = (
        fg,
        conv_weight,
        conv_state,
        indices,
        accepted,
        a_log,
        g_bias,
        recurrent_state,
        out,
        arrivals,
        norm_weight,
    )
    assert proj.is_cuda and all(x.device == proj.device for x in tensors)
    # No PDL: stream/graph kernel completion is the generation boundary.
    return _q4_parallel_history_fused_norm[(h, d // 8)](
        proj,
        fg,
        conv_weight,
        conv_state,
        indices,
        accepted,
        a_log,
        g_bias,
        recurrent_state,
        out,
        arrivals,
        norm_weight,
        eps,
        proj.stride(0),
        fg.stride(0),
        *conv_state.stride(),
        *conv_weight.stride(),
        recurrent_state.stride(0),
        out.stride(0),
        H=h,
        D=d,
        Q=q,
        BV=8,
        num_warps=1,
        num_stages=3,
    )
