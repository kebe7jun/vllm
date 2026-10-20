# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clamped SiLU-and-mul fused with per-token-group FP8 quantization.

Equivalent to ``silu_and_mul_with_clamp`` (bf16 output) followed by
``per_token_group_quant_fp8`` with row-major scales, in one launch. Used by the
Triton MoE path for models with a SwiGLU clamp (``swiglu_limit``), which the
CUDA ``silu_and_mul_per_block_quant`` kernel does not take.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

try:
    from triton.language.extra.cuda import libdevice
except ImportError:  # ROCm / older Triton
    from triton.language.extra import libdevice


@triton.jit
def _silu_mul_clamp_block_quant_kernel(
    in_ptr,  # [T, 2N] (gate || up), bf16
    out_ptr,  # [T, N] fp8
    scale_ptr,  # [T, N // GROUP] f32, row-major
    T,
    limit,
    eps,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    UE8M0: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    tl.static_assert(N % GROUP == 0)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    NG: tl.constexpr = N // GROUP
    row = tl.program_id(0)
    if row < T:
        cols = tl.arange(0, N)
        base = row.to(tl.int64) * (2 * N)
        gate = tl.load(in_ptr + base + cols).to(tl.float32)
        up = tl.load(in_ptr + base + N + cols).to(tl.float32)
        gate = tl.minimum(gate, limit)
        up = tl.clamp(up, -limit, limit)
        y = gate / (1.0 + libdevice.exp(-gate)) * up
        # the unfused path stores the activation as bf16 before quantizing
        y = y.to(tl.bfloat16).to(tl.float32)
        yg = tl.reshape(y, (NG, GROUP))
        amax = tl.maximum(tl.max(tl.abs(yg), axis=1), eps)
        scale = amax / FP8_MAX
        if UE8M0:
            scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1e-10))))
            q = yg * (1.0 / scale)[:, None]
        else:
            q = yg / scale[:, None]
        q = tl.clamp(q, FP8_MIN, FP8_MAX)
        tl.store(
            out_ptr + row.to(tl.int64) * N + cols,
            tl.reshape(q, (N,)).to(out_ptr.dtype.element_ty),
        )
        tl.store(scale_ptr + row.to(tl.int64) * NG + tl.arange(0, NG), scale)


def silu_mul_clamp_block_quant_supported(n: int, group_size: int) -> bool:
    """One program holds a whole output row of ``n`` columns, so ``n`` must be
    a power of two of at least one quant group."""
    return n >= group_size and n & (n - 1) == 0


def silu_mul_clamp_block_quant(
    x: torch.Tensor,
    limit: float,
    group_size: int,
    use_ue8m0: bool,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [T, 2N] gate||up. Returns (fp8 [T, N], fp32 scales [T, N // group_size])."""
    assert x.ndim == 2 and x.shape[1] % (2 * group_size) == 0
    T, N2 = x.shape
    N = N2 // 2
    fp8_dtype = current_platform.fp8_dtype()
    finfo = torch.finfo(fp8_dtype)
    out = torch.empty((T, N), dtype=fp8_dtype, device=x.device)
    scales = torch.empty((T, N // group_size), dtype=torch.float32, device=x.device)
    if T > 0:
        _silu_mul_clamp_block_quant_kernel[(T,)](
            x,
            out,
            scales,
            T,
            float(limit),
            eps,
            N=N,
            GROUP=group_size,
            FP8_MIN=finfo.min,
            FP8_MAX=finfo.max,
            UE8M0=use_ue8m0,
            launch_pdl=current_platform.is_arch_support_pdl(),
            num_warps=4 if N <= 1024 else 8,
        )
    return out, scales
