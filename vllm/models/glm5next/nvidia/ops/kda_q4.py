"""Isolated fixed-q4 KDA prototype; gate projections remain unchanged."""

from vllm.platforms import current_platform
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
def _q4_kernel(
    proj, fg, convw, cs, indices, accepted, alog, bias, states, norm, output, eps,
    sp: tl.constexpr, sf: tl.constexpr,
    cs0: tl.constexpr, cs1: tl.constexpr, cs2: tl.constexpr,
    si: tl.constexpr, sh: tl.constexpr, so: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, Q: tl.constexpr, BV: tl.constexpr,
    FUSE_NORM: tl.constexpr, launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    nh, part = tl.program_id(0), tl.program_id(1)
    b, h = nh // H, nh % H
    P: tl.constexpr = H * D
    kd = tl.arange(0, D)
    vd = part * BV + tl.arange(0, BV)
    ac = tl.load(accepted + b).to(tl.int64) - 1
    src = tl.load(indices + b * si + ac).to(tl.int64)
    conv_slot = tl.load(indices + b * si).to(tl.int64)
    if src <= 0 or conv_slot <= 0:
        return
    cq = h * D + kd
    ck = P + h * D + kd
    cv = 2 * P + h * D + vd
    pq = cs + conv_slot * cs0 + cq * cs1
    pk = cs + conv_slot * cs0 + ck * cs1
    pv = cs + conv_slot * cs0 + cv * cs1
    q0 = tl.load(pq + ac * cs2)
    q1 = tl.load(pq + (ac + 1) * cs2)
    q2 = tl.load(pq + (ac + 2) * cs2)
    k0 = tl.load(pk + ac * cs2)
    k1 = tl.load(pk + (ac + 1) * cs2)
    k2 = tl.load(pk + (ac + 2) * cs2)
    v0 = tl.load(pv + ac * cs2)
    v1 = tl.load(pv + (ac + 1) * cs2)
    v2 = tl.load(pv + (ac + 2) * cs2)
    wq0 = tl.load(convw + cq * 4)
    wq1 = tl.load(convw + cq * 4 + 1)
    wq2 = tl.load(convw + cq * 4 + 2)
    wq3 = tl.load(convw + cq * 4 + 3)
    wk0 = tl.load(convw + ck * 4)
    wk1 = tl.load(convw + ck * 4 + 1)
    wk2 = tl.load(convw + ck * 4 + 2)
    wk3 = tl.load(convw + ck * 4 + 3)
    wv0 = tl.load(convw + cv * 4)
    wv1 = tl.load(convw + cv * 4 + 1)
    wv2 = tl.load(convw + cv * 4 + 2)
    wv3 = tl.load(convw + cv * 4 + 3)
    ah = tl.exp(tl.load(alog + h).to(tl.float32))
    gb = tl.load(bias + h * D + kd).to(tl.float32)
    offset = h * D * D + vd[:, None] * D + kd[None, :]
    state = tl.load(states + src * sh + offset).to(tl.float32)
    # Partitioned heads share Q/K history. Publish their conv history in a
    # separate ordered kernel; a single full-head owner can publish in place.
    if BV == D:
        tl.store(pq, q1)
        tl.store(pq + cs2, q2)
        tl.store(pk, k1)
        tl.store(pk + cs2, k2)
        tl.store(pv, v1)
        tl.store(pv + cs2, v2)
    for t in range(Q):
        row = proj + (b * Q + t) * sp
        xq = tl.load(row + cq)
        xk = tl.load(row + ck)
        xv = tl.load(row + cv)
        qq = _conv(xq, q0, q1, q2, wq0, wq1, wq2, wq3)
        kk = _conv(xk, k0, k1, k2, wk0, wk1, wk2, wk3)
        vv = _conv(xv, v0, v1, v2, wv0, wv1, wv2, wv3)
        q0, q1, q2 = q1, q2, xq
        k0, k1, k2 = k1, k2, xk
        v0, v1, v2 = v1, v2, xv
        if BV == D:
            tl.store(pq + (t + 2) * cs2, xq)
            tl.store(pk + (t + 2) * cs2, xk)
            tl.store(pv + (t + 2) * cs2, xv)
        qq = qq / tl.sqrt(tl.sum(qq * qq) + 1e-6)
        kk = kk / tl.sqrt(tl.sum(kk * kk) + 1e-6)
        qq *= D ** -0.5
        gate = tl.load(fg + (b * Q + t) * sf + h * D + kd).to(tl.float32)
        gate = -5.0 / (1.0 + tl.exp(-(ah * (gate + gb))))
        state *= exp(gate[None, :])
        vv -= tl.sum(state * kk[None, :], 1)
        beta = tl.sigmoid(tl.load(row + 3 * P + h).to(tl.float32))
        vv *= beta
        state += vv[:, None] * kk[None, :]
        value = tl.sum(state * qq[None, :], 1)
        target = tl.load(indices + b * si + t).to(tl.int64)
        if target > 0:
            tl.store(states + target * sh + offset, state)
        value = value.to(output.dtype.element_ty)
        if FUSE_NORM:
            value = value.to(tl.float32)
            rstd = 1 / tl.sqrt(tl.sum(value * value) / D + eps)
            nw = tl.load(norm + vd).to(tl.float32)
            g2 = tl.load(fg + (b * Q + t) * sf + P + h * D + vd).to(tl.float32)
            value = value * rstd * nw * tl.sigmoid(g2)
        tl.store(output + (b * Q + t) * so + h * D + vd, value)


@triton.jit
def _commit_conv(proj, cs, indices, accepted,
                 sp: tl.constexpr, cs0: tl.constexpr, cs1: tl.constexpr,
                 cs2: tl.constexpr, si: tl.constexpr,
                 P: tl.constexpr, Q: tl.constexpr, launch_pdl: tl.constexpr):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    b = tl.program_id(1)
    ch = tl.program_id(0) * 128 + tl.arange(0, 128)
    ac = tl.load(accepted + b).to(tl.int64) - 1
    slot = tl.load(indices + b * si).to(tl.int64)
    src = tl.load(indices + b * si + ac)
    if slot <= 0 or src <= 0:
        return
    dst = cs + slot * cs0 + ch * cs1
    one = tl.load(dst + (ac + 1) * cs2, ch < 3 * P, 0)
    two = tl.load(dst + (ac + 2) * cs2, ch < 3 * P, 0)
    tl.debug_barrier()
    tl.store(dst, one, ch < 3 * P)
    tl.store(dst + cs2, two, ch < 3 * P)
    for t in tl.static_range(Q):
        x = tl.load(proj + (b * Q + t) * sp + ch, ch < 3 * P, 0)
        tl.store(dst + (t + 2) * cs2, x, ch < 3 * P)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


def kda_q4(proj, fg, conv_weight, conv_state, indices, accepted, a_log,
           g_bias, recurrent_state, norm_weight, out, *, bv=32, fuse_norm=False, num_ctas=1, eps=1e-5):
    """Advance four verification tokens, restoring the previous accepted prefix."""
    h, d, q = a_log.numel(), 128, 4
    p = h * d
    assert indices.ndim == 2 and indices.shape[1] == q and indices.stride(1) == 1
    assert proj.shape == (indices.shape[0] * q, 3 * p + h + 2 * d)
    assert proj.stride(1) == fg.stride(1) == 1 and fg.shape == (proj.shape[0], 2*p)
    assert conv_state.shape[1:] == (3 * p, q + 2)
    assert recurrent_state.shape[1:] == (h, d, d)
    assert bv in (8, 16, 32, 64, 128) and (not fuse_norm or bv == d)
    code = _q4_kernel[(indices.shape[0] * h, d // bv)](
        proj, fg, conv_weight, conv_state, indices, accepted, a_log, g_bias,
        recurrent_state, norm_weight, out, eps,
        proj.stride(0), fg.stride(0), *conv_state.stride(), indices.stride(0),
        recurrent_state.stride(0), out.stride(0),
        H=h, D=d, Q=q, BV=bv, FUSE_NORM=fuse_norm,
        launch_pdl=current_platform.is_arch_support_pdl(), num_warps=4, num_ctas=num_ctas,
    )
    if bv != d:
        _commit_conv[(triton.cdiv(3 * p, 128), indices.shape[0])](
            proj, conv_state, indices, accepted, proj.stride(0),
            *conv_state.stride(), indices.stride(0), P=p, Q=q,
            launch_pdl=current_platform.is_arch_support_pdl(), num_warps=4,
        )
    return code
