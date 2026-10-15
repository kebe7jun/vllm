# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel all-reduce folded into the single-kernel mHC post+pre step
(decode, one token), Lamport style.

Producers (the decode GEMV kernels whose last program owns a finished output
tile) store their bf16 partial row straight into a symmetric-memory slot on
every TP peer. No flags, counters or fences: a slot starts out filled with
-0.0 (the sentinel; producers map a real -0.0 to +0.0) and the consumer, the
single mHC kernel, polls its slice until no element is the sentinel, sums the
``world`` partials and, once all its blocks have read them, empties the slot
again. Parity double-buffering keeps a peer that is a step ahead from writing
into a slot that is still being read. The standalone all-reduce kernel of that
site is skipped.

Layout: ``data[site][parity][rank][H]`` (bf16) in symmetric memory on every
rank; ``seq`` (local) is bumped once per active decode step and its low bit is
the parity producers write and consumers read. Sites restart every step and are
consumed within the step.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton

NUM_SITES = 64
MAX_WORLD = 4
LAMPORT_SENTINEL = -2147450880  # 0x80008000: a pair of bf16 -0.0


@triton.jit
def push_tile(
    acc_row, offs_n, n_ok, p0, p1, p2, p3, seq_ptr, my_rank, H, WORLD: tl.constexpr
):
    """Store one finished bf16 output tile into this step's parity half of the
    site slot on every peer (-0.0 is the empty marker, so it becomes +0.0)."""
    par = tl.load(seq_ptr) & 1
    off = (par * WORLD + my_rank) * H + offs_n
    v = acc_row.to(tl.bfloat16)
    v = tl.where(v == 0, tl.zeros_like(v), v)
    tl.store(p0 + off, v, mask=n_ok)
    if WORLD > 1:
        tl.store(p1 + off, v, mask=n_ok)
    if WORLD > 2:
        tl.store(p2 + off, v, mask=n_ok)
    if WORLD > 3:
        tl.store(p3 + off, v, mask=n_ok)


@triton.jit
def _bump_seq_kernel(seq_ptr):
    tl.store(seq_ptr, tl.load(seq_ptr) + 1)


class _State:
    def __init__(self, group, rank: int, world: int, hidden: int, device: torch.device):
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.enable_symm_mem_for_group(group.group_name)
        self.data = symm_mem.empty(
            (NUM_SITES, 2, world, hidden), dtype=torch.bfloat16, device=device
        )
        self.data.view(torch.int32).fill_(LAMPORT_SENTINEL)
        hd = symm_mem.rendezvous(self.data, group.group_name)
        self.peer_data = [
            hd.get_buffer(r, self.data.shape, self.data.dtype) for r in range(world)
        ]
        self.seq = torch.zeros(1, dtype=torch.int32, device=device)
        self.rank, self.world, self.hidden = rank, world, hidden
        torch.accelerator.synchronize()
        torch.distributed.barrier(group)


class MhcFusedAllReduce:
    def __init__(self) -> None:
        self.state: _State | None = None
        self.active = False
        self.site = 0
        self.pending: dict[int, int] = {}  # data_ptr of a pushed output -> site
        self.skipped: dict[
            int, int
        ] = {}  # data_ptr whose all-reduce was skipped -> site

    @staticmethod
    def enabled() -> bool:
        return (
            envs.VLLM_MHC_FUSED_ALLREDUCE
            and envs.VLLM_MHC_SINGLE_KERNEL
            and torch.cuda.is_available()
        )

    def begin_step(self, num_tokens: int, hidden: int, device: torch.device) -> None:
        """Called once per forward (first mHC layer). Activates the fused path
        for single-token decode steps and flips the slot parity."""
        if self.skipped:
            raise RuntimeError(
                "mHC fused all-reduce: a skipped all-reduce was never consumed"
            )
        self.pending.clear()
        self.active = False
        self.site = 0
        if not self.enabled() or num_tokens != 1:
            return
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        if tp.world_size < 2 or tp.world_size > MAX_WORLD:
            return
        if self.state is None:
            if torch.cuda.is_current_stream_capturing():
                return  # cannot allocate while capturing; eager warmup creates it
            self.state = _State(
                tp.device_group, tp.rank_in_group, tp.world_size, hidden, device
            )
        if self.state.hidden != hidden:
            return
        self.active = True
        _bump_seq_kernel[(1,)](self.state.seq)

    def producer_args(self, out: torch.Tensor) -> dict | None:
        """Reserve a site for ``out`` ([1, H] partial); returns the push kernel args."""
        st = self.state
        if (
            not self.active
            or st is None
            or out.shape[0] != 1
            or out.shape[-1] != st.hidden
        ):
            return None
        site = self.site
        if site >= NUM_SITES:
            return None  # more all-reduces per step than sites: fall back
        self.site = site + 1
        self.pending[out.data_ptr()] = site
        pd = [st.peer_data[r][site] for r in range(st.world)]
        while len(pd) < MAX_WORLD:
            pd.append(pd[0])
        return dict(peers=pd, seq=st.seq, my_rank=st.rank, world=st.world)

    def maybe_skip_all_reduce(self, t: torch.Tensor) -> bool:
        site = self.pending.pop(t.data_ptr(), None)
        if site is None:
            return False
        self.skipped[t.data_ptr()] = site
        return True

    def consumer_kwargs(self, x: torch.Tensor) -> dict:
        """Kwargs for the single mHC kernel if ``x`` is a pushed, un-reduced partial."""
        site = self.skipped.pop(x.data_ptr(), None)
        if site is None:
            return {}
        st = self.state
        assert st is not None
        return dict(peer_x=st.data[site].view(torch.int32), seq=st.seq, world=st.world)


_FAR = MhcFusedAllReduce()


def get_mhc_fused_allreduce() -> MhcFusedAllReduce:
    return _FAR
