# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row softmax for a handful of very wide rows (decode sampling over the vocab).

``torch.softmax`` runs one CTA per row, so a single 150k-wide row takes ~40 us
on H100. This splits each row over many programs: pass 1 writes per-block
(max, sum) statistics, pass 2 normalizes.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MAX_TOKENS = 16
BLOCK = 4096
NUM_WARPS = 4


@triton.jit
def _softmax_stats_kernel(
    x_ptr, stride_xm, stats_ptr, N, NB, BLOCK: tl.constexpr, launch_pdl: tl.constexpr
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    m = tl.program_id(0)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(
        x_ptr + m.to(tl.int64) * stride_xm + offs, mask=offs < N, other=float("-inf")
    ).to(tl.float32)
    lmax = tl.max(x, axis=0)
    # a fully masked block (e.g. by a grammar bitmask) contributes nothing
    lsum = tl.where(lmax == float("-inf"), 0.0, tl.sum(tl.exp(x - lmax), axis=0))
    tl.store(stats_ptr + (m * NB + b) * 2, lmax)
    tl.store(stats_ptr + (m * NB + b) * 2 + 1, lsum)


@triton.jit
def _softmax_norm_kernel(
    x_ptr,
    stride_xm,
    stats_ptr,
    out_ptr,
    stride_om,
    N,
    NB,
    BLOCK: tl.constexpr,
    NBP: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    m = tl.program_id(0)
    b = tl.program_id(1)
    bs = tl.arange(0, NBP)
    ok = bs < NB
    lmax = tl.load(stats_ptr + (m * NB + bs) * 2, mask=ok, other=float("-inf"))
    lsum = tl.load(stats_ptr + (m * NB + bs) * 2 + 1, mask=ok, other=0.0)
    gmax = tl.max(lmax, axis=0)
    gsum = tl.sum(lsum * tl.exp(lmax - gmax), axis=0)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(
        x_ptr + m.to(tl.int64) * stride_xm + offs, mask=offs < N, other=float("-inf")
    ).to(tl.float32)
    tl.store(
        out_ptr + m.to(tl.int64) * stride_om + offs,
        tl.exp(x - gmax) / gsum,
        mask=offs < N,
    )


def small_m_softmax(x: torch.Tensor) -> torch.Tensor:
    """softmax(x, dim=-1) in fp32 for x[M, N] with M <= 16."""
    M, N = x.shape
    assert M <= MAX_TOKENS and x.stride(1) == 1
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    nb = triton.cdiv(N, BLOCK)
    stats = torch.empty((M * nb * 2,), dtype=torch.float32, device=x.device)
    pdl = current_platform.is_arch_support_pdl()
    _softmax_stats_kernel[(M, nb)](
        x, x.stride(0), stats, N, nb, BLOCK=BLOCK, launch_pdl=pdl, num_warps=NUM_WARPS
    )
    _softmax_norm_kernel[(M, nb)](
        x,
        x.stride(0),
        stats,
        out,
        out.stride(0),
        N,
        nb,
        BLOCK=BLOCK,
        NBP=triton.next_power_of_2(nb),
        launch_pdl=pdl,
        num_warps=NUM_WARPS,
    )
    return out
