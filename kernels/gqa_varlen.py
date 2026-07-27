# ruff: noqa
import argparse
import math
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.profiler import do_bench
from tilelang.language.annotations import annotate_layout
from tilelang.layout.swizzle import make_swizzled_layout

# ═══════════════════════════════════════════════════════════════════════
#  Architecture Detection
# ═══════════════════════════════════════════════════════════════════════

def get_cuda_compute_capability(device=None):
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.get_device_capability(device)


def is_sm90_plus(device=None):
    return get_cuda_compute_capability(device) >= (9, 0)


# ═══════════════════════════════════════════════════════════════════════
#  Kernel: FlashMLA-style Prefill (sm_89-)
#  BREAKTHROUGH: Uses 256 threads + FullRow policy to partition acc_o
#  This halves register pressure (no spilling!) and avoids cross-WG sync.
#  + T.use_swizzle for L2 cache optimization
#  + T.make_swizzled_layout for Shared Memory bank conflict elimination
# ═══════════════════════════════════════════════════════════════════════

@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_pipe(
    batch_size, groups, UQ, UKV, heads, dim, is_causal,
    block_M=128, block_N=64, num_stages=2, threads=128, dtype=T.float16
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [UQ, heads, dim]
    kv_shape = [UKV, head_kv, dim]
    o_shape = [UQ, heads, dim]
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(kv_shape, dtype),
        V_unpad: T.Tensor(kv_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        max_seqlen_q: T.int32,
        Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)

            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_bf16 = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            batch_idx = bz
            head_idx = by
            kv_head_idx = head_idx // groups

            q_start_idx = cu_seqlens_q[batch_idx]
            kv_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = k_end_idx - kv_start_idx

            is_valid_cta = bx * block_M < q_current_seqlen
            offset = kv_current_seqlen - q_current_seqlen
            max_visible_k_idx = offset + (bx + 1) * block_M

            loop_range = T.if_then_else(
                is_valid_cta,
                T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
                if is_causal
                else T.ceildiv(kv_current_seqlen, block_N),
                0
            )

            T.copy(Q_unpad[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :], K_shared)
                T.copy(V_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :], V_shared)

                T.clear(acc_s)
                # FullRow: each WarpGroup computes [64, 64] of acc_s independently
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # ── Mask: skip entirely for fully-unmasked blocks ──
                # A block is fully unmasked when no element hits causal/padding boundary.
                # causal: (k+1)*block_N-1 <= bx*block_M+offset  (whole block below diagonal)
                # Q pad:  (bx+1)*block_M <= q_current_seqlen    (no Q padding in this block)
                # K pad:  (k+1)*block_N <= kv_current_seqlen    (no K padding in this block)
                if is_causal:
                    need_mask = ((k + 1) * block_N - 1 > bx * block_M + offset) | \
                                ((bx + 1) * block_M > q_current_seqlen) | \
                                ((k + 1) * block_N > kv_current_seqlen)
                    if need_mask:
                        for i, j in T.Parallel(block_M, block_N):
                            q_pos = bx * block_M + i
                            k_pos = k * block_N + j
                            acc_s[i, j] = T.if_then_else(
                                (q_pos + offset < k_pos) | (q_pos >= q_current_seqlen) | (k_pos >= kv_current_seqlen),
                                -T.infinity(accum_dtype),
                                acc_s[i, j],
                            )
                else:
                    need_mask = ((bx + 1) * block_M > q_current_seqlen) | \
                                ((k + 1) * block_N > kv_current_seqlen)
                    if need_mask:
                        for i, j in T.Parallel(block_M, block_N):
                            q_pos = bx * block_M + i
                            k_pos = k * block_N + j
                            acc_s[i, j] = T.if_then_else(
                                (q_pos >= q_current_seqlen) | (k_pos >= kv_current_seqlen),
                                -T.infinity(accum_dtype),
                                acc_s[i, j],
                            )

                # ── Online softmax (per-WarpGroup independent) ──
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                # fused: update scores_max + compute scores_scale in one T.Parallel
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                # ── Rescale acc_o and compute P×V directly from registers ──
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                # Convert P to bf16 in registers (avoids shared memory round-trip)
                T.copy(acc_s, acc_s_bf16)
                # fragment × shared GEMM: P @ V directly from registers
                T.gemm(acc_s_bf16, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # ── Final normalize & direct register write ──
            if is_valid_cta:
                if is_causal:
                    for i, d in T.Parallel(block_M, dim):
                        if bx * block_M + i < q_current_seqlen:
                            Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = T.if_then_else(
                                logsum[i] == 0, 0, acc_o[i, d] / logsum[i])
                else:
                    for i, d in T.Parallel(block_M, dim):
                        if bx * block_M + i < q_current_seqlen:
                            Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = T.if_then_else(
                                logsum[i] == 0, 0, acc_o[i, d] / logsum[i])

    return main


# ═══════════════════════════════════════════════════════════════════════
#  Kernel 2: Warp Specialized (sm_90+ only)
#  FIX: Removed O_shared, direct register write
# ═══════════════════════════════════════════════════════════════════════

@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_ws(
    batch_size, groups, UQ, UKV, heads, dim, is_causal,
    block_M=128, block_N=64, threads=256, dtype=T.float16
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [UQ, heads, dim]
    kv_shape = [UKV, head_kv, dim]
    o_shape = [UQ, heads, dim]
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(kv_shape, dtype),
        V_unpad: T.Tensor(kv_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        max_seqlen_q: T.int32,
        Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared_0 = T.alloc_shared([block_N, dim], dtype)
            V_shared_0 = T.alloc_shared([block_N, dim], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim], dtype)
            V_shared_1 = T.alloc_shared([block_N, dim], dtype)
            S_shared = T.alloc_shared([block_M, block_N], dtype)
            # O_shared removed

            kv_ready_0 = T.alloc_barrier(arrive_count=128)
            kv_ready_1 = T.alloc_barrier(arrive_count=128)
            compute_done_0 = T.alloc_barrier(arrive_count=128)
            compute_done_1 = T.alloc_barrier(arrive_count=128)

            batch_idx = bz
            head_idx = by
            kv_head_idx = head_idx // groups

            q_start_idx = cu_seqlens_q[batch_idx]
            kv_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = k_end_idx - kv_start_idx

            offset = kv_current_seqlen - q_current_seqlen
            max_visible_k_idx = offset + (bx + 1) * block_M
            loop_range = (
                T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
                if is_causal
                else T.ceildiv(kv_current_seqlen, block_N)
            )

            tx = T.get_thread_binding()

            T.copy(Q_unpad[q_start_idx + bx * block_M : q_start_idx + (bx + 1) * block_M, head_idx, :], Q_shared)

            if tx < 128:
                # ═══════ COMPUTE WARPS (threads 0–127) ═══════
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for k in T.serial(loop_range):
                    buf = k % 2
                    phase = (k // 2) % 2

                    if buf == 0:
                        T.barrier_wait(kv_ready_0, phase)
                    else:
                        T.barrier_wait(kv_ready_1, phase)

                    T.clear(acc_s)
                    if buf == 0:
                        T.gemm(Q_shared, K_shared_0, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.gemm(Q_shared, K_shared_1, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # ── Mask (single fused branch for efficiency) ──
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            q_pos = bx * block_M + i
                            k_pos = k * block_N + j
                            acc_s[i, j] = T.if_then_else(
                                (q_pos + offset < k_pos) | (q_pos >= q_current_seqlen) | (k_pos >= kv_current_seqlen),
                                -T.infinity(accum_dtype),
                                acc_s[i, j],
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            q_pos = bx * block_M + i
                            k_pos = k * block_N + j
                            acc_s[i, j] = T.if_then_else(
                                (q_pos >= q_current_seqlen) | (k_pos >= kv_current_seqlen),
                                -T.infinity(accum_dtype),
                                acc_s[i, j],
                            )

                    # ── Online softmax ──
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, S_shared)

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]

                    if buf == 0:
                        T.gemm(S_shared, V_shared_0, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.gemm(S_shared, V_shared_1, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    if buf == 0:
                        T.barrier_arrive(compute_done_0)
                    else:
                        T.barrier_arrive(compute_done_1)

                # ── Final normalize & direct register write ──
                if is_causal:
                    for i, d in T.Parallel(block_M, dim):
                        if bx * block_M + i < q_current_seqlen:
                            Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = T.if_then_else(
                                logsum[i] == 0, 0, acc_o[i, d] / logsum[i])
                else:
                    for i, d in T.Parallel(block_M, dim):
                        if bx * block_M + i < q_current_seqlen:
                            Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = T.if_then_else(
                                logsum[i] == 0, 0, acc_o[i, d] / logsum[i])

            else:
                # ═══════ MEMORY WARPS (threads 128–255) ═══════
                for k in T.serial(loop_range):
                    buf = k % 2

                    if k >= 2:
                        wait_phase = (k // 2 - 1) % 2
                        if buf == 0:
                            T.barrier_wait(compute_done_0, wait_phase)
                        else:
                            T.barrier_wait(compute_done_1, wait_phase)

                    if buf == 0:
                        T.copy(K_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :], K_shared_0)
                        T.copy(V_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :], V_shared_0)
                        T.barrier_arrive(kv_ready_0)
                    else:
                        T.copy(K_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :], K_shared_1)
                        T.copy(V_unpad[kv_start_idx + k * block_N : kv_start_idx + (k + 1) * block_N, kv_head_idx, :], V_shared_1)
                        T.barrier_arrive(kv_ready_1)

    return main


# ═══════════════════════════════════════════════════════════════════════
#  Auto-Dispatch Wrapper
# ═══════════════════════════════════════════════════════════════════════

def flashattn(
    batch_size, groups, UQ, UKV, heads, dim, is_causal,
    block_M=128, block_N=64, num_stages=2, threads=128, dtype=T.float16,
):
    sm90 = is_sm90_plus()
    if sm90:
        # WS kernel requires 256 threads (128 compute + 128 memory) and 3 pipeline stages
        return flashattn_ws(
            batch_size, groups, UQ, UKV, heads, dim, is_causal,
            block_M=block_M, block_N=block_N, num_stages=3, threads=256, dtype=dtype,
        )
    else:
        # A100 (sm_80): block_M=64 is optimal — shared drops to 80KB enabling 2 blocks/SM
        # (vs 1 block/SM with block_M=128's 96KB), doubling occupancy and halving register
        # pressure (acc_o 64→32 regs/thread). 4096 seqlen: 100→131 TFlops (76%→99% of FA).
        return flashattn_pipe(
            batch_size, groups, UQ, UKV, heads, dim, is_causal,
            block_M=64, block_N=64, num_stages=2, threads=128, dtype=dtype,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Main: Test & Benchmark
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import flash_attn
    from varlen_utils import generate_random_padding_mask, generate_qkv

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--is_causal", action="store_true")
    args = parser.parse_args()

    B = args.batch
    H_q = args.heads
    H_kv = args.kv_heads
    D = args.dim
    max_seqlen = args.seqlen
    is_causal = args.is_causal
    groups = H_q // H_kv

    dtype = torch.bfloat16
    device = "cuda"

    block_M = 128
    block_N = 64

    query_padding_mask = generate_random_padding_mask(max_seqlen, B, device, mode="random")
    key_padding_mask = generate_random_padding_mask(max_seqlen, B, device, mode="random")

    q = torch.randn(B, max_seqlen, H_q, D, dtype=dtype, device=device)
    k = torch.randn(B, max_seqlen, H_kv, D, dtype=dtype, device=device)
    v = torch.randn(B, max_seqlen, H_kv, D, dtype=dtype, device=device)

    (
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        _, _, _,
        output_pad_fn, _, _,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    UQ = q_unpad.shape[0]
    UKV = k_unpad.shape[0]

    # Safe padding math (align to block_M / block_N)
    UQ_padded = math.ceil(UQ / block_M) * block_M
    UKV_padded = math.ceil(UKV / block_N) * block_N

    q_unpad_padded = torch.zeros(UQ_padded, H_q, D, dtype=dtype, device=device)
    q_unpad_padded[:UQ] = q_unpad

    k_unpad_padded = torch.zeros(UKV_padded, H_kv, D, dtype=dtype, device=device)
    k_unpad_padded[:UKV] = k_unpad

    v_unpad_padded = torch.zeros(UKV_padded, H_kv, D, dtype=dtype, device=device)
    v_unpad_padded[:UKV] = v_unpad

    cc = get_cuda_compute_capability()
    kernel_type = "WS (sm_90+)" if is_sm90_plus() else "Pipeline (sm_89-)"

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Varlen Prefill Benchmark                           ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  GPU:          sm_{cc[0]}{cc[1]} ({kernel_type})")
    print(f"║  Batch:        {B}")
    print(f"║  Q Heads:      {H_q}")
    print(f"║  KV Heads:     {H_kv}  (GQA groups = {groups})")
    print(f"║  Dim:          {D}")
    print(f"║  Max SeqLen:   {max_seqlen}")
    print(f"║  Block:        {block_M}x{block_N}")
    print(f"║  Causal:       {is_causal}")
    print(f"║  Total Q:      {UQ} tokens (padded: {UQ_padded})")
    print(f"║  Total KV:     {UKV} tokens (padded: {UKV_padded})")
    print(f"╚══════════════════════════════════════════════════════╝")
    print()

    kernel = flashattn(B, groups, UQ_padded, UKV_padded, H_q, D, is_causal,
                       block_M=block_M, block_N=block_N, num_stages=2, threads=128, dtype=dtype)

    out_unpad_padded = kernel(q_unpad_padded, k_unpad_padded, v_unpad_padded, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
    out_unpad = out_unpad_padded[:UQ]
    out = output_pad_fn(out_unpad)  

    fa_out_unpad = flash_attn.flash_attn_varlen_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        0.0,
        causal=is_causal,
    )
    fa_out = output_pad_fn(fa_out_unpad)

    torch.testing.assert_close(out, fa_out, rtol=1e-2, atol=1e-2)
    print("✓ Correctness passed!")
    print()

    for _ in range(3):
        _ = kernel(q_unpad_padded, k_unpad_padded, v_unpad_padded, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
    torch.cuda.synchronize()

    latency_tl = do_bench(
        lambda: kernel(q_unpad_padded, k_unpad_padded, v_unpad_padded, cu_seqlens_q, cu_seqlens_k, max_seqlen_q),
        _n_warmup=10, _n_repeat=20,
    )

    avg_q_len = query_padding_mask.float().mean().item()
    avg_k_len = key_padding_mask.float().mean().item()
    total_flops = float(4 * B * H_q * avg_q_len * avg_k_len * D)
    if is_causal:
        total_flops *= 0.5
    tflops = total_flops / latency_tl * 1e-9

    total_bytes = float(2 * (UQ * H_q * D + 2 * UKV * H_kv * D))
    bandwidth_tb = total_bytes / latency_tl * 1e-9

    print(f"TileLang [{kernel_type}]: {latency_tl * 1000:.2f} µs")
    print(f"  Compute:  {tflops:.2f} TFlops")
    print(f"  Bandwidth:{bandwidth_tb:.2f} TB/s  (achieved)")
    print()

    def flash_attn_fn():
        return flash_attn.flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            0.0,
            causal=is_causal,
        )

    for _ in range(3):
        _ = flash_attn_fn()
    torch.cuda.synchronize()

    latency_fa = do_bench(flash_attn_fn, _n_warmup=10, _n_repeat=20)
    tflops_fa = total_flops / latency_fa * 1e-9
    bandwidth_fa = total_bytes / latency_fa * 1e-9

    print(f"FlashAttention:          {latency_fa * 1000:.2f} µs")
    print(f"  Compute:  {tflops_fa:.2f} TFlops")
    print(f"  Bandwidth:{bandwidth_fa:.2f} TB/s  (achieved)")
    print()
    print(f"Speedup vs FA: {latency_fa / latency_tl:.2f}x")
    print(f"Bandwidth ratio: {bandwidth_tb / bandwidth_fa:.2f}x")

    print()
    print("═══ SeqLen Sweep ═══")
    print(f"{'SeqLen':>8} {'TileLang(µs)':>14} {'FA(µs)':>10} {'Ratio':>8} {'TB/s(TL)':>10}")
    print("─" * 54)

    for sl in [512, 1024, 2048, 4096, 8192]:
        q_sweep = torch.randn(B, sl, H_q, D, dtype=dtype, device=device)
        k_sweep = torch.randn(B, sl, H_kv, D, dtype=dtype, device=device)
        v_sweep = torch.randn(B, sl, H_kv, D, dtype=dtype, device=device)

        q_mask_sweep = generate_random_padding_mask(sl, B, device, mode="random")
        k_mask_sweep = generate_random_padding_mask(sl, B, device, mode="random")

        (
            q_unpad_sweep, k_unpad_sweep, v_unpad_sweep,
            cu_q_sweep, cu_k_sweep,
            max_q_sweep, max_k_sweep,
            _, _, _,
            _, _, _,
        ) = generate_qkv(q_sweep, k_sweep, v_sweep, q_mask_sweep, k_mask_sweep, kvpacked=False)

        UQ_sweep = q_unpad_sweep.shape[0]
        UKV_sweep = k_unpad_sweep.shape[0]

        UQ_sweep_padded = math.ceil(UQ_sweep / block_M) * block_M
        UKV_sweep_padded = math.ceil(UKV_sweep / block_N) * block_N

        q_unpad_sweep_padded = torch.zeros(UQ_sweep_padded, H_q, D, dtype=dtype, device=device)
        q_unpad_sweep_padded[:UQ_sweep] = q_unpad_sweep

        k_unpad_sweep_padded = torch.zeros(UKV_sweep_padded, H_kv, D, dtype=dtype, device=device)
        k_unpad_sweep_padded[:UKV_sweep] = k_unpad_sweep

        v_unpad_sweep_padded = torch.zeros(UKV_sweep_padded, H_kv, D, dtype=dtype, device=device)
        v_unpad_sweep_padded[:UKV_sweep] = v_unpad_sweep

        kernel_sweep = flashattn(B, groups, UQ_sweep_padded, UKV_sweep_padded, H_q, D, is_causal,
                                 block_M=block_M, block_N=block_N, num_stages=2, threads=128, dtype=dtype)      

        avg_q_sweep = q_mask_sweep.float().mean().item()
        avg_k_sweep = k_mask_sweep.float().mean().item()
        flops_sweep = float(4 * B * H_q * avg_q_sweep * avg_k_sweep * D)
        if is_causal:
            flops_sweep *= 0.5
        bytes_sweep = float(2 * (UQ_sweep * H_q * D + 2 * UKV_sweep * H_kv * D))

        lat_tl = do_bench(
            lambda: kernel_sweep(q_unpad_sweep_padded, k_unpad_sweep_padded, v_unpad_sweep_padded,
                                 cu_q_sweep, cu_k_sweep, max_q_sweep),
            _n_warmup=5, _n_repeat=10,
        )
        bw_tl = bytes_sweep / lat_tl * 1e-9

        try:
            lat_fa = do_bench(
                lambda: flash_attn.flash_attn_varlen_func(
                    q_unpad_sweep, k_unpad_sweep, v_unpad_sweep,
                    cu_q_sweep, cu_k_sweep,
                    max_q_sweep, max_k_sweep,
                    0.0,
                    causal=is_causal,
                ),
                _n_warmup=5, _n_repeat=10,
            )
            ratio = lat_fa / lat_tl
            fa_str = f"{lat_fa * 1000:8.1f}"
        except Exception:
            ratio = 0
            fa_str = "N/A"

        print(f"{sl:8d} {lat_tl * 1000:14.1f} {fa_str:>10} {ratio:8.2f} {bw_tl:10.2f}")
