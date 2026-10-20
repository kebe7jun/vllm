# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured single-kernel mHC post/pre with optional fused TP reduction."""

from vllm.platforms import current_platform
from vllm.tilelang_utils import T, tilelang_jit

ENABLE_PDL = current_platform.is_arch_support_pdl() and current_platform.is_cuda()
_MIX_ON_TRAILING_THREADS = current_platform.is_rocm()
LAMPORT_SENTINEL = -2147450880


def _is_mix_thread():
    if _MIX_ON_TRAILING_THREADS:
        return T.get_thread_binding() >= 64
    return T.get_thread_binding() < 32


def _is_collapse_thread():
    if _MIX_ON_TRAILING_THREADS:
        return T.get_thread_binding() < 64
    return T.get_thread_binding() >= 32


@tilelang_jit
def mhc_single_post_pre_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    hc_scale,
    hc_base,
    norm_weight,
    yp_out,
    rp_out,
    residual_out,
    post_mix_out,
    comb_mix_out,
    layer_input,
    bar,
    peer_x,
    seq,
    hc: int,
    hidden: int,
    n_out: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_thr: int = 128,
    tile_n: int = 2,
    split_k: int = 8,
    world: int = 0,
    ar_bf16_acc: bool = False,
):
    """Single-kernel mHC post + pre-norm for a few independent tokens.

    Phase 1 (all blocks): new residual from the layer output ``x`` and the
    previous residual, plus split-k partials of the pre-mix GEMM. A grid
    barrier (atomic arrive, self-resetting ``bar``) then lets one block compute
    the layer input (pre-mix collapse + RMSNorm) from the bf16 residual while a
    second block runs the Sinkhorn on the 4x4 comb mix for the next layer.
    With ``world > 0`` the layer output is the Lamport-style fused TP
    all-reduce: ``peer_x[parity, rank, token]`` holds bf16 partials as int32
    pairs, polled until no half equals the -0.0 sentinel; once every block has
    read them (grid barrier) the consumed parity is emptied again.
    """
    m = T.dynamic("num_tokens")
    h = hidden
    h_per_split = h // split_k
    n_tiles = n_out // tile_n
    vec = h_per_split // n_thr
    hw = h // n_thr  # collapse: elements per thread over the whole hidden
    hwp = hw // 2
    n_blocks = n_tiles * split_k
    num_warps = n_thr // 32
    hp = h // 2
    vp = vec // 2
    wmax = max(world, 1)

    comb_mix: T.Tensor((m, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    weight_t: T.Tensor((n_out, hc, h), T.float32)  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor((3,), T.float32)  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor((n_out,), T.float32)  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor((h,), T.bfloat16)  # type: ignore[no-redef, valid-type]
    yp_out: T.Tensor((split_k, m, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    rp_out: T.Tensor((split_k, m), T.float32)  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post_mix_out: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
    comb_mix_out: T.Tensor((m, hc * hc), T.float32)  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    bar: T.Tensor((m, 3), T.int32)  # type: ignore[no-redef, valid-type]
    peer_x: T.Tensor((2, wmax, m, hp), T.int32)  # type: ignore[no-redef, valid-type]
    seq: T.Tensor((1,), T.int32)  # type: ignore[no-redef, valid-type]

    with T.Kernel(m, n_tiles, split_k, threads=n_thr) as (i_m, i_nt, i_ks):
        tid = T.get_thread_binding()
        warp_id = tid // 32
        lane = tid % 32
        h0 = i_ks * h_per_split + tid * vec
        p0 = h0 // 2

        s_warp = T.alloc_shared((num_warps, tile_n + 1), T.float32)
        s_mix = T.alloc_shared((n_out,), T.float32)
        s_pre = T.alloc_shared((hc,), T.float32)

        pm = T.alloc_local((hc,), T.float32)
        cm = T.alloc_local((hc, hc), T.float32)
        acc = T.alloc_local((tile_n,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        xr = T.alloc_local((vec,), T.float32)
        rf = T.alloc_local((hc, vec), T.bfloat16)
        rb = T.alloc_local((hc, vec), T.bfloat16)
        wf = T.alloc_local((tile_n, hc, vec), T.float32)
        new_r = T.alloc_local((hc, vec), T.float32)
        wv = T.alloc_local((wmax, vp), T.int32)
        rfull = T.alloc_local((hc, hw), T.bfloat16)
        nw = T.alloc_local((hw,), T.bfloat16)
        ok = T.alloc_var(T.int32, init=0)
        par = T.alloc_var(T.int32, init=0)
        xv = T.alloc_var(T.float32, init=0.0)

        T.clear(acc)
        T.clear(sqr)

        # Everything but the layer output x comes from the previous mHC step
        # or the weights, several kernels back: load it before the PDL wait.
        for j in T.unroll(hc):
            pm[j] = post_mix[i_m, j]
        for j in T.unroll(hc):
            for k in T.unroll(hc):
                cm[k, j] = comb_mix[i_m, k, j]
        for k in T.unroll(hc):
            for v in T.vectorized(vec):
                rf[k, v] = residual_in[i_m, k, h0 + v]
        for n in T.unroll(tile_n):
            for j in T.unroll(hc):
                for v in T.vectorized(vec):
                    wf[n, j, v] = weight_t[i_nt * tile_n + n, j, h0 + v]

        if ENABLE_PDL:
            T.pdl_sync()
            # dependents (the next projection) may start loading weights now
            T.pdl_trigger()

        # ---- layer output x (own partial + peers, or plain) ----
        if world > 0:
            par = seq[0] & 1
            ok = 0
            while ok == 0:
                ok = 1
                for r in T.unroll(world):
                    for v in T.unroll(vp):
                        wv[r, v] = T.atomic_load(peer_x[par, r, i_m, p0 + v], "relaxed")
                        if (wv[r, v] & 0xFFFF) == 0x8000 or (
                            (wv[r, v] >> 16) & 0xFFFF
                        ) == 0x8000:
                            ok = 0
            for v in T.unroll(vp):
                xv = T.reinterpret(wv[0, v] << 16, T.float32)
                for r in T.unroll(world - 1):
                    xv += T.reinterpret(wv[r + 1, v] << 16, T.float32)
                    if ar_bf16_acc:
                        xv = T.cast(T.cast(xv, T.bfloat16), T.float32)
                xr[2 * v] = T.cast(T.cast(xv, T.bfloat16), T.float32)
                xv = T.reinterpret(wv[0, v] & -65536, T.float32)
                for r in T.unroll(world - 1):
                    xv += T.reinterpret(wv[r + 1, v] & -65536, T.float32)
                    if ar_bf16_acc:
                        xv = T.cast(T.cast(xv, T.bfloat16), T.float32)
                xr[2 * v + 1] = T.cast(T.cast(xv, T.bfloat16), T.float32)
        else:
            xb = T.alloc_local((vec,), T.bfloat16)
            for v in T.vectorized(vec):
                xb[v] = x_in[i_m, h0 + v]
            for v in T.unroll(vec):
                xr[v] = T.cast(xb[v], T.float32)

        for v in T.unroll(vec):
            for j in T.unroll(hc):
                new_r[j, v] = pm[j] * xr[v]
                for k in T.unroll(hc):
                    new_r[j, v] += cm[k, j] * T.cast(rf[k, v], T.float32)

        if i_nt == 0:
            for j in T.unroll(hc):
                for v in T.unroll(vec):
                    rb[j, v] = T.cast(new_r[j, v], T.bfloat16)
                    sqr[0] += new_r[j, v] * new_r[j, v]
            for j in T.unroll(hc):
                for v in T.vectorized(vec):
                    residual_out[i_m, j, h0 + v] = rb[j, v]

        for n in T.unroll(tile_n):
            for j in T.unroll(hc):
                for v in T.unroll(vec):
                    acc[n] += wf[n, j, v] * new_r[j, v]

        for n in T.unroll(tile_n):
            acc[n] = T.warp_reduce_sum(acc[n])
        if i_nt == 0:
            sqr[0] = T.warp_reduce_sum(sqr[0])
        if lane == 0:
            for n in T.unroll(tile_n):
                s_warp[warp_id, n] = acc[n]
            if i_nt == 0:
                s_warp[warp_id, tile_n] = sqr[0]
        T.sync_threads()
        if warp_id == 0:
            if lane < tile_n:
                v1 = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v1 += s_warp[w, lane]
                yp_out[i_ks, i_m, i_nt * tile_n + lane] = v1
            if i_nt == 0 and lane == 0:
                v2 = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v2 += s_warp[w, tile_n]
                rp_out[i_ks, i_m] = v2
        T.sync_threads()

        # ---- grid barrier 1: every block's partials are out ----
        is_collapse = (i_nt == 0) and (i_ks == 0)
        is_mix = (i_nt == 1) and (i_ks == 0)
        if tid == 0:
            # return_prev selects TileLang's atomic_ref path, which honours the
            # memory order (the plain int AtomicAdd ignores it)
            T.atomic_add(bar[i_m, 0], 1, memory_order="acq_rel", return_prev=True)
        if is_collapse or is_mix:
            if tid == 0:
                while T.atomic_load(bar[i_m, 0], "acquire") < n_blocks:
                    pass
            T.sync_threads()
            if tid < n_out:
                vals = T.alloc_local((split_k,), T.float32)
                rr = T.alloc_local((split_k,), T.float32)
                for s in T.unroll(split_k):
                    vals[s] = T.atomic_load(yp_out[s, i_m, tid], "relaxed")
                    rr[s] = T.atomic_load(rp_out[s, i_m], "relaxed")
                rms = T.alloc_var(T.float32, init=0.0)
                mv = T.alloc_var(T.float32, init=0.0)
                for s in T.unroll(split_k):
                    rms += rr[s]
                for s in T.unroll(split_k):
                    mv += vals[s]
                rms = T.rsqrt(rms / (hc * h) + rms_eps)
                s_mix[tid] = mv * rms
        if is_collapse:
            # the whole new residual (bf16, as the reference re-reads it)
            for k in T.unroll(hc):
                for v in T.vectorized(hw):
                    rfull[k, v] = residual_out[i_m, k, tid * hw + v]
            for v in T.vectorized(hw):
                nw[v] = norm_weight[tid * hw + v]
            if world > 0:
                # every block has read this step's partials (they arrived at the
                # barrier after phase 1): empty the consumed parity for the step
                # after next, which is the earliest a peer can write it again
                for r in T.unroll(world):
                    for v in T.vectorized(hwp):
                        peer_x[par, r, i_m, tid * hwp + v] = LAMPORT_SENTINEL
        if is_collapse or is_mix:
            T.sync_threads()
        if is_mix:
            # warp 0, lane j < hc owns row j of the 4x4 comb mix: row sums stay in
            # registers, column sums cross the 4 lanes with two xor-shuffles.
            if warp_id == 0:
                if lane < hc:
                    post_mix_out[i_m, lane] = (
                        T.sigmoid(s_mix[lane + hc] * hc_scale[1] + hc_base[lane + hc])
                        * hc_post_mult_value
                    )
                jr = T.min(lane, hc - 1)
                row = T.alloc_local((hc,), T.float32)
                col = T.alloc_local((hc,), T.float32)
                rs = T.alloc_var(T.float32, init=0.0)
                rm = T.alloc_var(T.float32, init=0.0)
                for k in T.unroll(hc):
                    row[k] = (
                        s_mix[jr * hc + k + 2 * hc] * hc_scale[2]
                        + hc_base[jr * hc + k + 2 * hc]
                    )
                rm = T.max(T.max(row[0], row[1]), T.max(row[2], row[3]))
                for k in T.unroll(hc):
                    row[k] = T.exp(row[k] - rm)
                rs = T.call_extern(
                    T.float32, "__fdividef", 1.0, (row[0] + row[1]) + (row[2] + row[3])
                )
                for k in T.unroll(hc):
                    row[k] = row[k] * rs + hc_sinkhorn_eps
                for k in T.unroll(hc):
                    col[k] = row[k] + T.shfl_xor(row[k], 1)
                for k in T.unroll(hc):
                    col[k] = col[k] + T.shfl_xor(col[k], 2)
                for k in T.unroll(hc):
                    row[k] = row[k] * T.call_extern(
                        T.float32, "__fdividef", 1.0, col[k] + hc_sinkhorn_eps
                    )
                for _ in T.serial(sinkhorn_repeat - 1):
                    rs = T.call_extern(
                        T.float32,
                        "__fdividef",
                        1.0,
                        ((row[0] + row[1]) + (row[2] + row[3])) + hc_sinkhorn_eps,
                    )
                    for k in T.unroll(hc):
                        row[k] = row[k] * rs
                    for k in T.unroll(hc):
                        col[k] = row[k] + T.shfl_xor(row[k], 1)
                    for k in T.unroll(hc):
                        col[k] = col[k] + T.shfl_xor(col[k], 2)
                    for k in T.unroll(hc):
                        row[k] = row[k] * T.call_extern(
                            T.float32, "__fdividef", 1.0, col[k] + hc_sinkhorn_eps
                        )
                if lane < hc:
                    for k in T.unroll(hc):
                        comb_mix_out[i_m, lane * hc + k] = row[k]
            if tid == 0:
                dep = T.alloc_var(T.int32, init=0)
                dep = T.atomic_add(
                    bar[i_m, 2], 1, memory_order="acq_rel", return_prev=True
                )
                if dep == 1:
                    bar[i_m, 0] = 0
                    bar[i_m, 2] = 0
        if is_collapse:
            if tid < hc:
                s_pre[tid] = (
                    T.sigmoid(s_mix[tid] * hc_scale[0] + hc_base[tid]) + hc_pre_eps
                )
            T.sync_threads()
            ol = T.alloc_local((hw,), T.float32)
            ssq = T.alloc_local((1,), T.float32)
            T.clear(ol)
            T.clear(ssq)
            for j in T.serial(hc):
                for v in T.unroll(hw):
                    ol[v] += s_pre[j] * T.cast(rfull[j, v], T.float32)
            for v in T.unroll(hw):
                ssq[0] += ol[v] * ol[v]
            ssq[0] = T.warp_reduce_sum(ssq[0])
            if lane == 0:
                s_warp[warp_id, 0] = ssq[0]
            T.sync_threads()
            tot = T.alloc_var(T.float32, init=0.0)
            for w in T.unroll(num_warps):
                tot += s_warp[w, 0]
            rn = T.alloc_var(T.float32, init=0.0)
            rn = T.rsqrt(tot / h + norm_eps)
            ob = T.alloc_local((hw,), T.bfloat16)
            for v in T.unroll(hw):
                ob[v] = T.cast(
                    T.cast(T.cast(ol[v], T.bfloat16), T.float32)
                    * rn
                    * T.cast(nw[v], T.float32),
                    T.bfloat16,
                )
            for v in T.vectorized(hw):
                layer_input[i_m, tid * hw + v] = ob[v]
            if tid == 0:
                dep2 = T.alloc_var(T.int32, init=0)
                dep2 = T.atomic_add(
                    bar[i_m, 2], 1, memory_order="acq_rel", return_prev=True
                )
                if dep2 == 1:
                    bar[i_m, 0] = 0
                    bar[i_m, 2] = 0
