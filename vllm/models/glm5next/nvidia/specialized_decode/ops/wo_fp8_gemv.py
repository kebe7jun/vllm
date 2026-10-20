# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only FP8 GEMV for M <= 16 rows: bf16 activations, fp8 weights with
[128, 128] block scales, fp32 accumulation, bf16 output.

Decode projections are HBM-bound; storing an FP8 copy of a bf16 weight halves
the bytes streamed per token. Used only for tiny M, so prefill keeps the bf16
weights and numerics.
"""

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .fused_allreduce import (
    get_mhc_fused_allreduce,
    push_tile,
)

MAX_TOKENS = 16
BLOCK = 128


_counters: dict[torch.device, torch.Tensor] = {}


def _counter(numel: int, device: torch.device) -> torch.Tensor:
    buf = _counters.get(device)
    if buf is None or buf.numel() < numel:
        buf = torch.zeros(max(numel, 8192), dtype=torch.int32, device=device)
        _counters[device] = buf
    return buf


@triton.jit
def _wo_fp8_gemv_kernel(
    x_ptr,
    stride_xm,
    w_ptr,
    stride_wn,
    s_ptr,
    stride_sn,
    part_ptr,
    cnt_ptr,
    out_ptr,
    stride_om,
    M,
    N,
    K,
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
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    PUSH: tl.constexpr,
    WORLD: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Program = (N tile, K split). x rides in a 16-row MMA tile; the fp8 weight
    tile is converted to bf16 in registers and scaled per 128-wide k block."""
    tl.static_assert(BN <= 128 and 128 % BN == 0)
    tl.static_assert(BK % 128 == 0)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    KS = tl.cdiv(K, SPLITS * BK) * BK
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    rows = tl.arange(0, 16)
    n_ok = offs_n < N
    m_ok = rows < M
    w_rows = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn
    x_rows = x_ptr + rows[:, None].to(tl.int64) * stride_xm
    s_row = s_ptr + (pid_n * BN // 128) * stride_sn
    acc = tl.zeros((16, BN), dtype=tl.float32)
    for k0 in range(pid_s * KS, tl.minimum((pid_s + 1) * KS, K), BK):
        for kb in tl.static_range(BK // 128):
            kk = k0 + kb * 128
            k_ok = (kk + offs_k) < K
            x = tl.load(
                x_rows + kk + offs_k[None, :],
                mask=m_ok[:, None] & k_ok[None, :],
                other=0.0,
            )
            w = tl.load(
                w_rows + kk + offs_k[None, :],
                mask=n_ok[:, None] & k_ok[None, :],
                other=0.0,
            )
            sc = tl.load(s_row + kk // 128, mask=kk < K, other=0.0)
            acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32) * sc
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    if SPLITS == 1:
        tl.store(
            out_ptr + rows[:, None].to(tl.int64) * stride_om + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=m_ok[:, None] & n_ok[None, :],
        )
        if PUSH:
            push_tile(
                acc,
                rows[:, None] * N + offs_n[None, :],
                m_ok[:, None] & n_ok[None, :],
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
                M * N,
                WORLD,
            )
        return
    pbase = part_ptr + (rows[:, None] * SPLITS + pid_s).to(tl.int64) * N
    tl.store(pbase + offs_n[None, :], acc, mask=m_ok[:, None] & n_ok[None, :])
    tl.debug_barrier()
    prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if prev == SPLITS - 1:
        tl.store(cnt_ptr + pid_n, 0)
        for m in range(M):
            sp = tl.arange(0, SPLITS)
            rbase = part_ptr + (m * SPLITS + sp[:, None]).to(tl.int64) * N
            y = tl.sum(
                tl.load(rbase + offs_n[None, :], mask=n_ok[None, :], other=0.0), axis=0
            )
            tl.store(
                out_ptr + m * stride_om + offs_n,
                y.to(out_ptr.dtype.element_ty),
                mask=n_ok,
            )
            if PUSH:
                push_tile(
                    y,
                    m * N + offs_n,
                    n_ok,
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
                    M * N,
                    WORLD,
                )


@triton.jit
def _wo_fp8_gemv_m1_kernel(
    x_ptr,
    w_ptr,
    stride_wn,
    s_ptr,
    stride_sn,
    part_ptr,
    cnt_ptr,
    out_ptr,
    N,
    K,
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
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    PUSH: tl.constexpr,
    WORLD: tl.constexpr,
    launch_pdl: tl.constexpr,
    PREFETCH_KS: tl.constexpr = 0,
):
    """Single-row GEMV: elementwise FMA over a [BN, 128] fp8 tile per k block
    (no MMA tile padding), split-K with a last-program reduction.

    With ``PREFETCH_KS`` (= K // SPLITS) the program's whole weight slice and
    its scales are loaded before the PDL wait: they do not depend on the
    previous kernel, so their latency hides behind its tail."""
    tl.static_assert(BN <= 128 and 128 % BN == 0)
    tl.static_assert(BK % 128 == 0)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    n_ok = offs_n < N
    w_rows = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn
    s_row = s_ptr + (pid_n * BN // 128) * stride_sn
    if PREFETCH_KS > 0:
        KG: tl.constexpr = PREFETCH_KS // 128
        offs_ks = tl.arange(0, PREFETCH_KS)
        k_base = pid_s * PREFETCH_KS
        w = tl.load(w_rows + k_base + offs_ks[None, :], mask=n_ok[:, None], other=0.0)
        sc = tl.load(s_row + k_base // 128 + tl.arange(0, KG))
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        x = tl.load(x_ptr + k_base + offs_ks).to(tl.float32)
        part = tl.sum(tl.reshape(w.to(tl.float32) * x[None, :], (BN, KG, 128)), axis=2)
        acc = tl.sum(part * sc[None, :], axis=1)
    else:
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
        KS = tl.cdiv(K, SPLITS * BK) * BK
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(pid_s * KS, tl.minimum((pid_s + 1) * KS, K), BK):
            for kb in tl.static_range(BK // 128):
                kk = k0 + kb * 128
                k_ok = (kk + offs_k) < K
                x = tl.load(x_ptr + kk + offs_k, mask=k_ok, other=0.0).to(tl.float32)
                w = tl.load(
                    w_rows + kk + offs_k[None, :],
                    mask=n_ok[:, None] & k_ok[None, :],
                    other=0.0,
                ).to(tl.float32)
                sc = tl.load(s_row + kk // 128, mask=kk < K, other=0.0)
                acc += tl.sum(w * x[None, :], axis=1) * sc
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
    if SPLITS == 1:
        tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=n_ok)
        if PUSH:
            push_tile(
                acc,
                offs_n,
                n_ok,
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
                N,
                WORLD,
            )
        return
    tl.store(part_ptr + pid_s.to(tl.int64) * N + offs_n, acc, mask=n_ok)
    tl.debug_barrier()
    prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if prev == SPLITS - 1:
        tl.store(cnt_ptr + pid_n, 0)
        sp = tl.arange(0, SPLITS)
        y = tl.sum(
            tl.load(
                part_ptr + sp[:, None].to(tl.int64) * N + offs_n[None, :],
                mask=n_ok[None, :],
                other=0.0,
            ),
            axis=0,
        )
        tl.store(out_ptr + offs_n, y.to(out_ptr.dtype.element_ty), mask=n_ok)
        if PUSH:
            push_tile(
                y,
                offs_n,
                n_ok,
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
                N,
                WORLD,
            )


def quantize_weight_fp8_block(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (fp8 [N, K], fp32 scales [ceil(N/128), ceil(K/128)])."""
    N, K = weight.shape
    fp8 = current_platform.fp8_dtype()
    fmax = torch.finfo(fp8).max
    nb, kb = -(-N // BLOCK), -(-K // BLOCK)
    w = torch.zeros((nb * BLOCK, kb * BLOCK), dtype=torch.float32, device=weight.device)
    w[:N, :K] = weight.float()
    wb = w.view(nb, BLOCK, kb, BLOCK)
    amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / fmax
    q = (
        (wb / scale)
        .clamp(-fmax, fmax)
        .to(fp8)
        .view(nb * BLOCK, kb * BLOCK)[:N, :K]
        .contiguous()
    )
    return q, scale.view(nb, kb).contiguous()


BN = 32
BK = 256
SPLITS = 0  # 0: per-shape table
NUM_WARPS = 4
NUM_STAGES = 3
USE_M1 = True

# (N, K) -> (BN, BK, SPLITS, num_warps, num_stages): graph-replay sweeps on H100
# at M=1 with the weights rotated out of L2.
_CONFIGS = {
    (6416, 4096): (32, 128, 8, 2, 2),
    (4096, 2048): (32, 128, 8, 4, 2),
    (2048, 4096): (64, 256, 16, 4, 3),
    (4096, 4096): (32, 256, 8, 2, 2),
    (4096, 1536): (32, 512, 1, 4, 3),
    (8192, 512): (32, 128, 1, 4, 3),
    (160, 4096): (32, 128, 16, 4, 3),
    (38720, 4096): (32, 256, 1, 2, 3),
}


def _config(N: int, K: int) -> tuple[int, int, int, int, int]:
    cfg = _CONFIGS.get((N, K))
    if cfg is not None:
        return cfg
    if N >= 16384:
        return 32, 256, 1, 2, 3
    return (32, 128, 8, 4, 2) if K >= 2048 else (32, 128, 1, 4, 3)


# M == 1 elementwise kernel, (N, K) -> (BN, BK, SPLITS, num_warps, num_stages);
# HBM-rotated graph-replay sweeps on H100.
M1_CONFIGS: dict[tuple[int, int], tuple[int, int, int, int, int]] = {
    (6416, 4096): (32, 128, 8, 4, 2),
    (4096, 4096): (32, 128, 8, 8, 2),
    (4096, 2048): (32, 128, 8, 8, 2),
    (2048, 4096): (32, 128, 16, 8, 2),
    (4096, 1536): (32, 128, 8, 8, 2),
    (8192, 512): (32, 512, 1, 8, 2),
    (160, 4096): (32, 128, 16, 8, 2),
    (38720, 4096): (32, 512, 1, 2, 3),
}
M1_BN = 0  # sweep overrides: 0 -> table/default
M1_BK = 256
M1_SPLITS = 8
M1_WARPS = 4
M1_STAGES = 3


# Largest per-program K slice (elements) the M=1 GEMV prefetches into
# registers before the PDL wait; 0 disables the prefetch.
M1_PREFETCH_MAX_KS = 1024 if envs.VLLM_SMALL_M_GEMV_PREFETCH else 0


def _prefetch_ks(K: int, splits: int) -> int:
    ks = K // splits
    if (
        M1_PREFETCH_MAX_KS <= 0
        or K % splits
        or ks % 128
        or ks > M1_PREFETCH_MAX_KS
        or ks & (ks - 1)
    ):
        return 0
    return ks


def _m1_config(N: int, K: int) -> tuple[int, int, int, int, int]:
    if M1_BN > 0:
        return M1_BN, M1_BK, M1_SPLITS, M1_WARPS, M1_STAGES
    cfg = M1_CONFIGS.get((N, K))
    if cfg is not None:
        return cfg
    if N >= 16384:
        return 32, 512, 1, 2, 3
    return (32, 128, 8, 8, 2) if K >= 2048 else (32, 128, 1, 8, 2)


def wo_fp8_gemv(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    scale: torch.Tensor,
    push_for_reduce: bool = False,
) -> torch.Tensor:
    """Weight-only FP8 GEMV: out = x @ w^T for M <= 16 rows. With
    ``push_for_reduce`` (M <= 4) the output is also pushed to the TP peers for the
    fused mHC all-reduce when that path is active."""
    M, K = x.shape
    N = w_fp8.shape[0]
    assert (
        M <= MAX_TOKENS
        and w_fp8.shape[1] == K
        and w_fp8.stride(1) == 1
        and x.stride(1) == 1
    )
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    push = get_mhc_fused_allreduce().producer_args(out) if push_for_reduce else None
    if M == 1 and USE_M1:
        bn, bk, splits, nw, ns = _m1_config(N, K)
        splits = min(splits, triton.cdiv(K, bk))
        nt = triton.cdiv(N, bn)
        part = (
            torch.empty((splits, N), dtype=torch.float32, device=x.device)
            if splits > 1
            else out
        )
        cnt = _counter(nt, x.device)
        if push is None:
            peers, seq, my_rank, world = [out] * 8, cnt, 0, 1
        else:
            peers, seq, my_rank, world = (
                push["peers"],
                push["seq"],
                push["my_rank"],
                push["world"],
            )
        _wo_fp8_gemv_m1_kernel[(nt, splits)](
            x,
            w_fp8,
            w_fp8.stride(0),
            scale,
            scale.stride(0),
            part,
            cnt,
            out,
            N,
            K,
            *peers,
            seq,
            my_rank,
            BN=bn,
            BK=bk,
            SPLITS=splits,
            PUSH=push is not None,
            WORLD=world,
            launch_pdl=current_platform.is_arch_support_pdl(),
            PREFETCH_KS=_prefetch_ks(K, splits),
            num_warps=nw,
            num_stages=ns,
        )
        return out
    bn, bk, splits, nw, ns = (
        (BN, BK, SPLITS, NUM_WARPS, NUM_STAGES) if SPLITS > 0 else _config(N, K)
    )
    splits = min(splits, triton.cdiv(K, bk))
    nt = triton.cdiv(N, bn)
    part = (
        torch.empty((16 * splits, N), dtype=torch.float32, device=x.device)
        if splits > 1
        else out
    )
    cnt = _counter(nt, x.device)
    if push is None:
        peers, seq, my_rank, world = [out] * 8, cnt, 0, 1
    else:
        peers, seq, my_rank, world = (
            push["peers"],
            push["seq"],
            push["my_rank"],
            push["world"],
        )
    _wo_fp8_gemv_kernel[(nt, splits)](
        x,
        x.stride(0),
        w_fp8,
        w_fp8.stride(0),
        scale,
        scale.stride(0),
        part,
        cnt,
        out,
        out.stride(0),
        M,
        N,
        K,
        *peers,
        seq,
        my_rank,
        BN=bn,
        BK=bk,
        SPLITS=splits,
        PUSH=push is not None,
        WORLD=world,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=nw,
        num_stages=ns,
    )
    return out


def maybe_attach_fp8_copy(layer: torch.nn.Module) -> None:
    """Store an FP8 block-scaled copy of a bf16 [N, K] weight for the decode GEMV."""
    w = getattr(layer, "weight", None)
    if (
        w is None
        or not w.is_cuda
        or w.dim() != 2
        or w.dtype != torch.bfloat16
        or w.shape[1] % BLOCK != 0
        or w.shape[1] < 512
    ):
        return
    q, scale = quantize_weight_fp8_block(w.data)
    layer._wo_fp8_weight = q
    layer._wo_fp8_scale = scale


def wo_fp8_apply(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None):
    """The decode path if this layer has an FP8 copy and x is small; else None."""
    q = getattr(layer, "_wo_fp8_weight", None)
    if (
        q is None
        or x.dim() != 2
        or x.shape[0] > MAX_TOKENS
        or x.dtype != torch.bfloat16
    ):
        return None
    # a row-parallel layer whose output gets all-reduced: candidate for the
    # fused mHC all-reduce (the push is a no-op unless that path is active)
    reduces = (
        getattr(layer, "input_is_parallel", None) is not None
        and getattr(layer, "reduce_results", False)
        and getattr(layer, "tp_size", 1) > 1
        and bias is None
    )
    out = wo_fp8_gemv(x, q, layer._wo_fp8_scale, push_for_reduce=reduces)
    if bias is not None:
        out = out + bias
    return out
