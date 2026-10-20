# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K grouped GEMV path for FP8 block-quantized MoE at tiny batch.

At decode batch sizes (M <= 16) the per-expert GEMMs are GEMVs: each
``fused_moe_kernel`` block streams a whole K row range on its own, so a layer
reads only ~50% of HBM bandwidth and the activation, quantization and
``moe_sum`` steps add launches. Here every (token, slot) pair runs as a
split-K GEMV over many programs, and the last program to finish a tile
(found with an atomic counter) does the tail in registers:

* ``w13``: reduce the split-K partials, clamped SiLU-and-mul, per-128-group
  FP8 quantization of the intermediate.
* ``w2``: reduce the partials over splits and top-k slots with the routing
  weights and write the bf16 output row.

No expert alignment/sorting is needed. Numerics follow the Triton path
(bf16 rounding of the GEMM outputs, fp8 intermediate with fp32 scales).
"""

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .fused_allreduce import (
    push_tile,
)

try:
    from triton.language.extra.cuda import libdevice
except ImportError:  # ROCm / older Triton
    from triton.language.extra import libdevice

MAX_TOKENS = 16
GROUP = 128

_counters: dict[tuple, torch.Tensor] = {}


def _counter(key: str, numel: int, device: torch.device) -> torch.Tensor:
    """Zeroed int32 counters; every kernel resets the ones it uses."""
    buf = _counters.get((key, device))
    if buf is None or buf.numel() < numel:
        buf = torch.zeros(max(numel, 4096), dtype=torch.int32, device=device)
        _counters[(key, device)] = buf
    return buf


@triton.jit
def _w13_splitk_kernel(
    xq_ptr,
    xs_ptr,
    stride_xs_m,
    stride_xs_g,
    w1_ptr,
    w1s_ptr,
    topk_ids_ptr,
    part_ptr,
    cnt_ptr,
    inter_ptr,
    inters_ptr,
    limit,
    eps,
    TOPK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    CLAMP: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    UE8M0: tl.constexpr,
    USE_DOT: tl.constexpr,
    X_BF16: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    tl.static_assert(BN == 128, "one quant group per tile")
    tl.static_assert(BK % 128 == 0, "weight scale blocks are 128 wide")
    tl.static_assert(K % (SPLITS * BK) == 0)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    pid_mj = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    KS: tl.constexpr = K // SPLITS
    NT: tl.constexpr = N // BN
    KG: tl.constexpr = K // 128
    m = pid_mj // TOPK
    e = tl.load(topk_ids_ptr + pid_mj).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    w_rows_g = w1_ptr + e * (2 * N * K) + offs_n[:, None] * K
    w_rows_u = w_rows_g + N * K
    ws_g = w1s_ptr + e * (2 * NT * KG) + pid_n * KG
    ws_u = ws_g + NT * KG
    x_row = xq_ptr + m.to(tl.int64) * K
    if USE_DOT:
        # tensor cores: the token row is broadcast to a 16-row A tile
        rows = tl.arange(0, 16)
        acc_g2 = tl.zeros((16, BN), dtype=tl.float32)
        acc_u2 = tl.zeros((16, BN), dtype=tl.float32)
        for kb in range(KS // 128):
            k0 = pid_s * KS + kb * 128
            x16 = tl.load(x_row + k0 + offs_k[None, :] + rows[:, None] * 0)
            sx = tl.load(xs_ptr + m * stride_xs_m + (k0 // 128) * stride_xs_g)
            wg = tl.load(w_rows_g + k0 + offs_k[None, :])
            wu = tl.load(w_rows_u + k0 + offs_k[None, :])
            sg = tl.load(ws_g + k0 // 128)
            su = tl.load(ws_u + k0 // 128)
            acc_g2 += tl.dot(x16, tl.trans(wg), out_dtype=tl.float32) * (sx * sg)
            acc_u2 += tl.dot(x16, tl.trans(wu), out_dtype=tl.float32) * (sx * su)
        acc_g = tl.sum(tl.where(rows[:, None] == 0, acc_g2, 0.0), axis=0)
        acc_u = tl.sum(tl.where(rows[:, None] == 0, acc_u2, 0.0), axis=0)
    else:
        acc_g = tl.zeros((BN,), dtype=tl.float32)
        acc_u = tl.zeros((BN,), dtype=tl.float32)
        for kb in range(KS // 128):
            k0 = pid_s * KS + kb * 128
            if X_BF16:
                # fused per-128-group activation quantization (same math as
                # per_token_group_quant_fp8, rounded through fp8)
                xf = tl.load(x_row + k0 + offs_k).to(tl.float32)
                sx = tl.maximum(tl.max(tl.abs(xf), axis=0), eps) / FP8_MAX
                if UE8M0:
                    sx = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(sx), 1e-10))))
                x = (
                    tl.clamp(xf / sx, FP8_MIN, FP8_MAX)
                    .to(inter_ptr.dtype.element_ty)
                    .to(tl.float32)
                )
            else:
                x = tl.load(x_row + k0 + offs_k).to(tl.float32)
                sx = tl.load(xs_ptr + m * stride_xs_m + (k0 // 128) * stride_xs_g)
            wg = tl.load(w_rows_g + k0 + offs_k[None, :]).to(tl.float32)
            wu = tl.load(w_rows_u + k0 + offs_k[None, :]).to(tl.float32)
            sg = tl.load(ws_g + k0 // 128)
            su = tl.load(ws_u + k0 // 128)
            acc_g += tl.sum(wg * x[None, :], axis=1) * (sx * sg)
            acc_u += tl.sum(wu * x[None, :], axis=1) * (sx * su)
    # dependents may launch now: the streaming is done and only the tail is left
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    pbase = part_ptr + (pid_mj * SPLITS + pid_s).to(tl.int64) * (2 * N)
    tl.store(pbase + offs_n, acc_g)
    tl.store(pbase + N + offs_n, acc_u)

    tl.debug_barrier()
    prev = tl.atomic_add(cnt_ptr + pid_mj * NT + pid_n, 1, sem="acq_rel", scope="gpu")
    if prev == SPLITS - 1:
        tl.store(cnt_ptr + pid_mj * NT + pid_n, 0)
        sp = tl.arange(0, SPLITS)
        rbase = part_ptr + (pid_mj * SPLITS + sp[:, None]).to(tl.int64) * (2 * N)
        g = tl.sum(tl.load(rbase + offs_n[None, :]), axis=0)
        u = tl.sum(tl.load(rbase + N + offs_n[None, :]), axis=0)
        # the GEMM writes bf16 before the activation in the unfused path
        g = g.to(tl.bfloat16).to(tl.float32)
        u = u.to(tl.bfloat16).to(tl.float32)
        if CLAMP:
            g = tl.minimum(g, limit)
            u = tl.clamp(u, -limit, limit)
        y = g / (1.0 + libdevice.exp(-g)) * u
        y = y.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(y), axis=0), eps)
        scale = amax / FP8_MAX
        if UE8M0:
            scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1e-10))))
        q = tl.clamp(y / scale, FP8_MIN, FP8_MAX)
        tl.store(
            inter_ptr + pid_mj.to(tl.int64) * N + offs_n,
            q.to(inter_ptr.dtype.element_ty),
        )
        tl.store(inters_ptr + pid_mj * NT + pid_n, scale)


@triton.jit
def _w2_splitk_kernel(
    iq_ptr,
    is_ptr,
    w2_ptr,
    w2s_ptr,
    topk_ids_ptr,
    topk_w_ptr,
    part_ptr,
    cnt_ptr,
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    seq_ptr,
    my_rank,
    M: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    NJ: tl.constexpr,
    USE_DOT: tl.constexpr,
    PUSH: tl.constexpr,
    WORLD: tl.constexpr,
    launch_pdl: tl.constexpr,
    PREFETCH: tl.constexpr = False,
):
    tl.static_assert(BK == 128)
    tl.static_assert(NJ >= TOPK * SPLITS)
    tl.static_assert(N % (SPLITS * BK) == 0)
    tl.static_assert(BH % 128 == 0)
    pid_mj = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    KS: tl.constexpr = N // SPLITS
    HT: tl.constexpr = H // BH
    NG: tl.constexpr = N // BK
    m = pid_mj // TOPK
    offs_h = pid_h * BH + tl.arange(0, BH)
    offs_k = tl.arange(0, BK)
    if PREFETCH:
        # The routing comes from the top-k kernel, which finished before the w13
        # kernel (our PDL primary) got past its own wait: the expert's weight
        # tile can stream in before ours.
        tl.static_assert(KS == BK and not USE_DOT)
        e = tl.load(topk_ids_ptr + pid_mj, cache_modifier=".cg").to(tl.int64)
        w_rows = w2_ptr + e * (H * N) + offs_h[:, None] * N
        ws = w2s_ptr + e * ((H // 128) * NG) + (offs_h // 128) * NG
        k0 = pid_s * KS
        w_pf = tl.load(w_rows + k0 + offs_k[None, :])
        sw_pf = tl.load(ws + k0 // BK)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        # the dependent (all-reduce) has few CTAs: let its prologue overlap
        tl.extra.cuda.gdc_launch_dependents()
    if not PREFETCH:
        e = tl.load(topk_ids_ptr + pid_mj).to(tl.int64)
        w_rows = w2_ptr + e * (H * N) + offs_h[:, None] * N
        ws = w2s_ptr + e * ((H // 128) * NG) + (offs_h // 128) * NG
    wgt = tl.load(topk_w_ptr + pid_mj).to(tl.float32)
    x_row = iq_ptr + pid_mj.to(tl.int64) * N
    if PREFETCH:
        x = tl.load(x_row + k0 + offs_k).to(tl.float32)
        sx = tl.load(is_ptr + pid_mj * NG + k0 // BK)
        acc = tl.sum(w_pf.to(tl.float32) * x[None, :], axis=1) * (sx * sw_pf)
    elif USE_DOT:
        rows = tl.arange(0, 16)
        acc2 = tl.zeros((16, BH), dtype=tl.float32)
        for kb in range(KS // BK):
            k0 = pid_s * KS + kb * BK
            x16 = tl.load(x_row + k0 + offs_k[None, :] + rows[:, None] * 0)
            sx = tl.load(is_ptr + pid_mj * NG + k0 // BK)
            w = tl.load(w_rows + k0 + offs_k[None, :])
            sw = tl.load(ws + k0 // BK)
            acc2 += tl.dot(x16, tl.trans(w), out_dtype=tl.float32) * (sx * sw)[None, :]
        acc = tl.sum(tl.where(rows[:, None] == 0, acc2, 0.0), axis=0)
    else:
        acc = tl.zeros((BH,), dtype=tl.float32)
        for kb in range(KS // BK):
            k0 = pid_s * KS + kb * BK
            x = tl.load(x_row + k0 + offs_k).to(tl.float32)
            sx = tl.load(is_ptr + pid_mj * NG + k0 // BK)
            w = tl.load(w_rows + k0 + offs_k[None, :]).to(tl.float32)
            sw = tl.load(ws + k0 // BK)
            acc += tl.sum(w * x[None, :], axis=1) * (sx * sw)
    # bf16 rounding of the per-slot GEMM output, then the routing weight
    acc = acc.to(tl.bfloat16).to(tl.float32) * wgt
    tl.store(part_ptr + (pid_mj * SPLITS + pid_s).to(tl.int64) * H + offs_h, acc)

    tl.debug_barrier()
    prev = tl.atomic_add(cnt_ptr + m * HT + pid_h, 1, sem="acq_rel", scope="gpu")
    if prev == TOPK * SPLITS - 1:
        tl.store(cnt_ptr + m * HT + pid_h, 0)
        js = tl.arange(0, NJ)
        rbase = part_ptr + (m * TOPK * SPLITS + js[:, None]).to(tl.int64) * H
        y = tl.sum(
            tl.load(
                rbase + offs_h[None, :], mask=(js < TOPK * SPLITS)[:, None], other=0.0
            ),
            axis=0,
        )
        tl.store(out_ptr + m.to(tl.int64) * H + offs_h, y.to(out_ptr.dtype.element_ty))
        if PUSH:
            push_tile(
                y,
                m * H + offs_h,
                offs_h < H,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                seq_ptr,
                my_rank,
                M * H,
                WORLD,
            )


W13_SPLITS = 16
W13_BK = 128
W2_SPLITS = 4
W2_BH = 256
NUM_WARPS = 4
W2_WARPS = 4
# w2 loads its expert tile before the PDL wait (single K block per program)
W2_PREFETCH = envs.VLLM_SMALL_M_GEMV_PREFETCH
USE_DOT = False


def small_m_moe(
    out: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    clamp_limit: float | None,
    use_ue8m0: bool,
    eps: float = 1e-10,
    push: dict | None = None,
) -> torch.Tensor:
    """out[M, H] = sum_j w[m, j] * MLP_{e[m, j]}(x[m]); all shapes TP-local.

    xq: [M, K] fp8 with per-128-group scales xs [M, K // 128] (any strides).
    w1: [E, 2N, K] fp8, w1_scale [E, 2N // 128, K // 128]; w2: [E, H, N] fp8,
    w2_scale [E, H // 128, N // 128]. topk_ids / topk_weights: [M, TOPK].
    """
    M, K = xq.shape
    x_bf16 = xq.dtype == torch.bfloat16
    if x_bf16:
        assert xs is None and xq.is_contiguous()
        xs = torch.empty((1, 1), dtype=torch.float32, device=xq.device)  # unused
    E, N2, K1 = w1.shape
    N = N2 // 2
    H = w2.shape[1]
    TOPK = topk_ids.shape[1]
    assert K == K1 and out.shape[1] == H and w2.shape[2] == N
    assert M <= MAX_TOKENS and N % 128 == 0 and H % W2_BH == 0
    assert w1.is_contiguous() and w2.is_contiguous()
    assert w1_scale.is_contiguous() and w2_scale.is_contiguous()
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
    assert topk_ids.shape == (M, TOPK)
    dev = xq.device
    fp8 = current_platform.fp8_dtype()
    finfo = torch.finfo(fp8)
    pdl = current_platform.is_arch_support_pdl() and envs.VLLM_MOE_SMALL_M_PDL
    tsplits = min(W13_SPLITS, K // W13_BK)
    # GLM TP8 with the shared expert appended: H200 cold-L2 sweeps favor
    # fewer splits for verification and more for the single-token draft.
    if W13_SPLITS == 16 and W13_BK == 128 and (K, H, N, TOPK) == (4096, 4096, 256, 9):
        if M == 1:
            tsplits = 32
        elif M in (3, 4):
            tsplits = 8
    # largest split count <= W2_SPLITS that divides w2's K blocks
    ssplits = next(s for s in range(W2_SPLITS, 0, -1) if (N // 128) % s == 0)

    inter = torch.empty((M * TOPK, N), dtype=fp8, device=dev)
    inter_s = torch.empty((M * TOPK, N // 128), dtype=torch.float32, device=dev)
    part1 = torch.empty((M * TOPK * tsplits, 2 * N), dtype=torch.float32, device=dev)
    cnt1 = _counter("w13", MAX_TOKENS * TOPK * (N // 128), dev)
    _w13_splitk_kernel[(M * TOPK, N // 128, tsplits)](
        xq,
        xs,
        xs.stride(0),
        xs.stride(1),
        w1,
        w1_scale,
        topk_ids,
        part1,
        cnt1,
        inter,
        inter_s,
        float(clamp_limit) if clamp_limit is not None else 0.0,
        eps,
        TOPK=TOPK,
        N=N,
        K=K,
        BN=128,
        BK=W13_BK,
        SPLITS=tsplits,
        CLAMP=clamp_limit is not None,
        FP8_MIN=finfo.min,
        FP8_MAX=finfo.max,
        UE8M0=use_ue8m0,
        USE_DOT=USE_DOT,
        X_BF16=x_bf16,
        launch_pdl=pdl,
        num_warps=NUM_WARPS,
    )
    part2 = torch.empty((M * TOPK * ssplits, H), dtype=torch.float32, device=dev)
    cnt2 = _counter("w2", MAX_TOKENS * (H // W2_BH), dev)
    if push is None:
        peers, seq, my_rank, world = [out] * 8, cnt2, 0, 1
    else:
        peers, seq, my_rank, world = (
            push["peers"],
            push["seq"],
            push["my_rank"],
            push["world"],
        )
    _w2_splitk_kernel[(M * TOPK, H // W2_BH, ssplits)](
        inter,
        inter_s,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        part2,
        cnt2,
        out,
        *peers,
        seq,
        my_rank,
        M=M,
        TOPK=TOPK,
        H=H,
        N=N,
        BH=W2_BH,
        BK=128,
        SPLITS=ssplits,
        NJ=triton.next_power_of_2(TOPK * ssplits),
        USE_DOT=USE_DOT,
        PUSH=push is not None,
        WORLD=world,
        launch_pdl=pdl,
        PREFETCH=pdl and W2_PREFETCH and not USE_DOT and ssplits * 128 == N,
        num_warps=W2_WARPS,
    )
    return out
