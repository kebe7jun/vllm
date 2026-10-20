# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sigmoid top-k routing with correction bias for <= 16 tokens in one launch.

Matches ``topk_sigmoid`` (unbiased sigmoid scores as weights, biased scores for
the choice, optional renormalization, routed scaling factor) and appends the
fused shared-expert slots, replacing the CUDA top-k plus two ``torch.cat``.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MAX_TOKENS = 16
# one warp: the 8 argmax rounds reduce with shuffles only, no block barriers
NUM_WARPS = 1


@triton.jit
def _topk_sigmoid_kernel(
    logits_ptr,
    stride_lm,
    bias_ptr,
    ids_ptr,
    w_ptr,
    E,
    scale,
    shared_w,
    EP: tl.constexpr,
    TOPK: tl.constexpr,
    NSLOT: tl.constexpr,
    NSP: tl.constexpr,
    RENORM: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    m = tl.program_id(0)
    offs = tl.arange(0, EP)
    ok = offs < E
    x = tl.load(logits_ptr + m.to(tl.int64) * stride_lm + offs, mask=ok, other=0.0).to(
        tl.float32
    )
    s = 1.0 / (1.0 + tl.exp(-x))
    s = tl.where((s != s) | (tl.abs(s) == float("inf")), 0.0, s)
    bias = tl.load(bias_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    choice = tl.where(ok, s + bias, float("-inf"))
    slots = tl.arange(0, NSP)
    ids = tl.zeros((NSP,), dtype=tl.int32)
    for i in tl.static_range(TOPK):
        idx = tl.argmax(choice, axis=0)
        ids = tl.where(slots == i, idx.to(tl.int32), ids)
        choice = tl.where(offs == idx, float("-inf"), choice)
    # weights: the unbiased scores of the picks, re-read instead of reduced
    routed = slots < TOPK
    xw = tl.load(
        logits_ptr + m.to(tl.int64) * stride_lm + ids, mask=routed, other=0.0
    ).to(tl.float32)
    ws = 1.0 / (1.0 + tl.exp(-xw))
    ws = tl.where(routed & ((ws == ws) & (tl.abs(ws) != float("inf"))), ws, 0.0)
    if RENORM:
        ws = ws / tl.sum(ws, axis=0)
    ws = ws * scale
    ids = tl.where(routed, ids, E + (slots - TOPK))
    ws = tl.where(routed, ws, shared_w)
    tl.store(
        ids_ptr + m * NSLOT + slots,
        ids.to(ids_ptr.dtype.element_ty),
        mask=slots < NSLOT,
    )
    tl.store(w_ptr + m * NSLOT + slots, ws, mask=slots < NSLOT)


def small_m_topk_sigmoid(
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    num_fused_shared_experts: int = 0,
    shared_expert_weight: float = 1.0,
    indices_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (topk_weights f32 [M, topk + nfse], topk_ids int32 [M, topk + nfse])."""
    M, E = router_logits.shape
    assert M <= MAX_TOKENS and router_logits.stride(1) == 1
    nslot = topk + num_fused_shared_experts
    ids = torch.empty(
        (M, nslot), dtype=indices_type or torch.int32, device=router_logits.device
    )
    ws = torch.empty((M, nslot), dtype=torch.float32, device=router_logits.device)
    _topk_sigmoid_kernel[(M,)](
        router_logits,
        router_logits.stride(0),
        e_score_correction_bias,
        ids,
        ws,
        E,
        float(routed_scaling_factor),
        float(shared_expert_weight),
        EP=triton.next_power_of_2(E),
        TOPK=topk,
        NSLOT=nslot,
        NSP=triton.next_power_of_2(nslot),
        RENORM=renormalize,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=NUM_WARPS,
    )
    return ws, ids
