"""
TileLang GQA Decode with Paged KV Cache — Auto Architecture Detection

Structure based on reference code:
  T.Pipelined + separate K_shared / V_shared
  
Fix for Layout infer conflict:
  Replace acc_s_cast (Fragment) with S_shared (2KB SharedMem)
  Fragment→SharedMem has no layout constraints
  2KB write is negligible for memory-bound decode
"""

import torch
import tilelang
import tilelang.language as T
from functools import lru_cache


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
#  Kernel 1: Pipeline (all architectures)
# ═══════════════════════════════════════════════════════════════════════

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
    compile_flags=["-O3", "-DENABLE_BF16"],
)
def gqa_decode_paged_pipe(
    batch: int, heads: int, groups: int, max_seqlen_kv: int,
    dim: int, page_size: int,
    block_H: int = 16, block_N: int = 64,
    num_stages: int = 2, threads: int = 128, dtype=T.bfloat16,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    accum_dtype = T.float32
    num_pages = max_seqlen_kv // page_size
    kv_group_num = heads // groups
    valid_block_H = min(block_H, kv_group_num)

    # Page mapping
    if page_size >= block_N:
        num_blockn_in_page = page_size // block_N
        pages_per_block = 1
    else:
        num_blockn_in_page = 1
        pages_per_block = block_N // page_size

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, dim], dtype),
        K_cache: T.Tensor([max_seqlen_kv, groups, dim], dtype),
        V_cache: T.Tensor([max_seqlen_kv, groups, dim], dtype),
        block_table: T.Tensor([batch, num_pages], T.int32),
        seqlen_kv: T.Tensor([batch], T.int32),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, 1, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            # ── FIX: S_shared replaces acc_s_cast ──
            # block_H=16 → S_shared is only 16*64*2 = 2KB
            # Fragment→SharedMem has no layout constraints → No Pipeline infer conflict
            # Separate K_shared/V_shared → No overlapping buffer regions error
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], dtype)

            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group_num // valid_block_H)
            seqlen_kv_b = seqlen_kv[bid]
            loop_range = T.ceildiv(seqlen_kv_b, block_N)

            T.copy(Q[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :],
                   Q_shared[:valid_block_H, :])
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # ── Load K ──
                if pages_per_block == 1:
                    page_idx = k // num_blockn_in_page
                    block_in_page = k % num_blockn_in_page
                    physical_page = block_table[bid, page_idx]
                    offset = physical_page * page_size + block_in_page * block_N
                    T.copy(K_cache[offset:offset + block_N, cur_kv_head, :], K_shared)
                else:
                    for sub in T.serial(pages_per_block):
                        page_idx = k * pages_per_block + sub
                        physical_page = block_table[bid, page_idx]
                        T.copy(K_cache[physical_page * page_size:(physical_page + 1) * page_size,
                                       cur_kv_head, :],
                               K_shared[sub * page_size:(sub + 1) * page_size, :])

                # ── Q @ K^T ──
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)

                # ── Mask ──
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        (i < valid_block_H) and (k * block_N + j < seqlen_kv_b),
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )

                # ── Online softmax ──
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                # ── FIX: Store S to shared memory (fp32→bf16 cast) ──
                T.copy(acc_s, S_shared)

                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]

                # ── Load V ──
                if pages_per_block == 1:
                    T.copy(V_cache[offset:offset + block_N, cur_kv_head, :], V_shared)
                else:
                    for sub in T.serial(pages_per_block):
                        page_idx = k * pages_per_block + sub
                        physical_page = block_table[bid, page_idx]
                        T.copy(V_cache[physical_page * page_size:(physical_page + 1) * page_size,
                                       cur_kv_head, :],
                               V_shared[sub * page_size:(sub + 1) * page_size, :])

                # ── S @ V ──
                T.gemm(S_shared, V_shared, acc_o,
                       policy=T.GemmWarpPolicy.FullRow)

            # ── Final normalize & store ──
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] = T.if_then_else(
                    (i < valid_block_H) and (logsum[i] > 0),
                    acc_o[i, j] / logsum[i], 0)
            T.copy(acc_o[:valid_block_H, :], O_shared)
            T.copy(O_shared,
                   Output[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :])

    return main


# ═══════════════════════════════════════════════════════════════════════
#  Kernel 2: Warp Specialized (sm_90+ only)
# ═══════════════════════════════════════════════════════════════════════

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
    compile_flags=["-O3", "-DENABLE_BF16"],
)
def gqa_decode_paged_ws(
    batch: int, heads: int, groups: int, max_seqlen_kv: int,
    dim: int, page_size: int,
    block_H: int = 16, block_N: int = 64,
    threads: int = 256, dtype=T.bfloat16,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    accum_dtype = T.float32
    num_pages = max_seqlen_kv // page_size
    kv_group_num = heads // groups
    valid_block_H = min(block_H, kv_group_num)

    if page_size >= block_N:
        num_blockn_in_page = page_size // block_N
        pages_per_block = 1
    else:
        num_blockn_in_page = 1
        pages_per_block = block_N // page_size

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, dim], dtype),
        K_cache: T.Tensor([max_seqlen_kv, groups, dim], dtype),
        V_cache: T.Tensor([max_seqlen_kv, groups, dim], dtype),
        block_table: T.Tensor([batch, num_pages], T.int32),
        seqlen_kv: T.Tensor([batch], T.int32),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, 1, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared_0 = T.alloc_shared([block_N, dim], dtype)
            V_shared_0 = T.alloc_shared([block_N, dim], dtype)
            K_shared_1 = T.alloc_shared([block_N, dim], dtype)
            V_shared_1 = T.alloc_shared([block_N, dim], dtype)
            # Also use S_shared in WS for consistency
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], dtype)

            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            kv_ready_0 = T.alloc_barrier(arrive_count=128)
            kv_ready_1 = T.alloc_barrier(arrive_count=128)
            compute_done_0 = T.alloc_barrier(arrive_count=128)
            compute_done_1 = T.alloc_barrier(arrive_count=128)

            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group_num // valid_block_H)
            seqlen_kv_b = seqlen_kv[bid]
            loop_range = T.ceildiv(seqlen_kv_b, block_N)

            tx = T.get_thread_binding()

            T.copy(Q[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :],
                   Q_shared[:valid_block_H, :])

            if tx < 128:
                # ═══════ COMPUTE WARPS (0–127) ═══════
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
                        T.gemm(Q_shared, K_shared_0, acc_s, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.gemm(Q_shared, K_shared_1, acc_s, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (i < valid_block_H) and (k * block_N + j < seqlen_kv_b),
                            acc_s[i, j], -T.infinity(accum_dtype))

                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, S_shared)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]

                    if buf == 0:
                        T.gemm(S_shared, V_shared_0, acc_o,
                               policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.gemm(S_shared, V_shared_1, acc_o,
                               policy=T.GemmWarpPolicy.FullRow)

                    if buf == 0:
                        T.barrier_arrive(compute_done_0)
                    else:
                        T.barrier_arrive(compute_done_1)

                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] = T.if_then_else(
                        (i < valid_block_H) and (logsum[i] > 0),
                        acc_o[i, j] / logsum[i], 0)
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared,
                       Output[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :])

            else:
                # ═══════ MEMORY WARPS (128–255) ═══════
                for k in T.serial(loop_range):
                    buf = k % 2

                    if k >= 2:
                        wait_phase = (k // 2 - 1) % 2
                        if buf == 0:
                            T.barrier_wait(compute_done_0, wait_phase)
                        else:
                            T.barrier_wait(compute_done_1, wait_phase)

                    if pages_per_block == 1:
                        page_idx = k // num_blockn_in_page
                        block_in_page = k % num_blockn_in_page
                        physical_page = block_table[bid, page_idx]
                        offset = physical_page * page_size + block_in_page * block_N

                        if buf == 0:
                            T.copy(K_cache[offset:offset + block_N, cur_kv_head, :], K_shared_0)
                            T.copy(V_cache[offset:offset + block_N, cur_kv_head, :], V_shared_0)
                            T.barrier_arrive(kv_ready_0)
                        else:
                            T.copy(K_cache[offset:offset + block_N, cur_kv_head, :], K_shared_1)
                            T.copy(V_cache[offset:offset + block_N, cur_kv_head, :], V_shared_1)
                            T.barrier_arrive(kv_ready_1)
                    else:
                        if buf == 0:
                            for sub in T.serial(pages_per_block):
                                page_idx = k * pages_per_block + sub
                                physical_page = block_table[bid, page_idx]
                                T.copy(K_cache[physical_page * page_size:(physical_page + 1) * page_size,
                                               cur_kv_head, :],
                                       K_shared_0[sub * page_size:(sub + 1) * page_size, :])
                                T.copy(V_cache[physical_page * page_size:(physical_page + 1) * page_size,
                                               cur_kv_head, :],
                                       V_shared_0[sub * page_size:(sub + 1) * page_size, :])
                            T.barrier_arrive(kv_ready_0)
                        else:
                            for sub in T.serial(pages_per_block):
                                page_idx = k * pages_per_block + sub
                                physical_page = block_table[bid, page_idx]
                                T.copy(K_cache[physical_page * page_size:(physical_page + 1) * page_size,
                                               cur_kv_head, :],
                                       K_shared_1[sub * page_size:(sub + 1) * page_size, :])
                                T.copy(V_cache[physical_page * page_size:(physical_page + 1) * page_size,
                                               cur_kv_head, :],
                                       V_shared_1[sub * page_size:(sub + 1) * page_size, :])
                            T.barrier_arrive(kv_ready_1)

    return main


# ═══════════════════════════════════════════════════════════════════════
#  Kernel Cache & Auto-Dispatch
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def _get_kernel(bool_sm90, batch, heads, groups, max_seqlen_kv, dim, page_size,
                block_H, block_N, num_stages, threads, dtype_key):
    dtype_map = {"bfloat16": T.bfloat16, "float16": T.float16}
    if bool_sm90:
        return gqa_decode_paged_ws(
            batch, heads, groups, max_seqlen_kv, dim, page_size,
            block_H=block_H, block_N=block_N, threads=threads,
            dtype=dtype_map[dtype_key],
        )
    else:
        return gqa_decode_paged_pipe(
            batch, heads, groups, max_seqlen_kv, dim, page_size,
            block_H=block_H, block_N=block_N, num_stages=num_stages, threads=threads,
            dtype=dtype_map[dtype_key],
        )


def gqa_decode_paged_fn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen_kv: torch.Tensor,
    page_size: int = 16,
) -> torch.Tensor:
    B, H_q, D = q.shape
    H_kv = k_cache.shape[2]
    groups = H_kv
    max_seqlen_kv = k_cache.shape[0] * page_size

    k_cache_flat = k_cache.reshape(-1, H_kv, D).contiguous()
    v_cache_flat = v_cache.reshape(-1, H_kv, D).contiguous()

    block_N = max(64, page_size)
    block_H = 16
    dk = "bfloat16" if q.dtype == torch.bfloat16 else "float16"

    sm90 = is_sm90_plus()

    if sm90:
        threads = 256
        num_stages = 0  # WS kernel 使用 T.serial，不需要 num_stages
    else:
        threads = 256
        num_stages = 1

    # Pad block_table to match the kernel's compile-time shape [batch, num_pages].
    # TileLang uses the static shape for stride computation; a narrower runtime
    # tensor would cause incorrect indexing (stride mismatch).
    num_pages = max_seqlen_kv // page_size
    if block_table.shape[1] < num_pages:
        padded = torch.full(
            (B, num_pages), -1, dtype=torch.int32, device=block_table.device,
        )
        padded[:, :block_table.shape[1]] = block_table
        block_table = padded

    kernel = _get_kernel(
        sm90, B, H_q, groups, max_seqlen_kv, D, page_size,
        block_H, block_N, num_stages, threads, dk,
    )

    return kernel(q, k_cache_flat, v_cache_flat, block_table, seqlen_kv)


# ═══════════════════════════════════════════════════════════════════════
#  Reference
# ═══════════════════════════════════════════════════════════════════════

def ref_gqa_decode_paged(q, k_cache, v_cache, block_table, seqlen_kv, page_size=16):
    B, H_q, D = q.shape
    H_kv = k_cache.shape[2]
    groups = H_q // H_kv
    output = torch.zeros(B, H_q, D, dtype=q.dtype, device=q.device)
    scale = D ** -0.5
    for b in range(B):
        seq_len = seqlen_kv[b].item()
        num_used_pages = (seq_len + page_size - 1) // page_size
        pages = block_table[b, :num_used_pages]
        k_full = k_cache[pages].reshape(-1, H_kv, D)[:seq_len]
        v_full = v_cache[pages].reshape(-1, H_kv, D)[:seq_len]
        for kv_h in range(H_kv):
            q_h = q[b, kv_h * groups:(kv_h + 1) * groups, :]
            scores = torch.matmul(q_h, k_full[:, kv_h, :].T) * scale
            weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            output[b, kv_h * groups:(kv_h + 1) * groups, :] = torch.matmul(weights, v_full[:, kv_h, :])
    return output


# ═══════════════════════════════════════════════════════════════════════
#  Test & Benchmark
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from tilelang.profiler import do_bench

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--page_size", type=int, default=16)
    args = parser.parse_args()

    B = args.batch
    H_q = args.heads
    H_kv = args.kv_heads
    D = args.dim
    page_size = args.page_size
    max_seqlen = args.seqlen

    # Add 10% headroom for seqlen variation
    max_pages = int(max_seqlen * 1.1) // page_size
    total_kv_tokens = max_pages * page_size

    dtype = torch.bfloat16
    device = "cuda"

    # Generate realistic variable-length sequences
    # Most sequences near max_seqlen, a few shorter
    seqlen_kv = torch.full((B,), max_seqlen, dtype=torch.int32, device=device)
    # Randomly make 20% of batches shorter (simulating real inference)
    num_short = max(1, B // 5)
    short_lengths = torch.randint(max_seqlen // 4, max_seqlen, (num_short,),
                                  dtype=torch.int32, device=device)
    seqlen_kv[:num_short] = short_lengths

    q = torch.randn(B, H_q, D, dtype=dtype, device=device)
    k_cache = torch.randn(max_pages, page_size, H_kv, D, dtype=dtype, device=device)
    v_cache = torch.randn(max_pages, page_size, H_kv, D, dtype=dtype, device=device)

    # Sequential page tables (no swapping for benchmark)
    block_table = (
        torch.arange(max_pages, dtype=torch.int32, device=device)
        .unsqueeze(0).expand(B, -1).contiguous()
    )

    cc = get_cuda_compute_capability()
    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  GQA Decode Paged KV Benchmark                      ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  GPU:          sm_{cc[0]}{cc[1]} (mbarrier: {is_sm90_plus()})")
    print(f"║  Batch:        {B}")
    print(f"║  Q Heads:      {H_q}")
    print(f"║  KV Heads:     {H_kv}  (GQA groups = {H_q // H_kv})")
    print(f"║  Dim:          {D}")
    print(f"║  Max SeqLen:   {max_seqlen}")
    print(f"║  Page Size:    {page_size}")
    print(f"║  Total KV:     {total_kv_tokens} tokens")
    print(f"║  KV Cache:     {2 * total_kv_tokens * H_kv * D / 1024**3:.2f} GB")
    print(f"╚══════════════════════════════════════════════════════╝")
    print()

    # ── Correctness ──
    out_tl = gqa_decode_paged_fn(q, k_cache, v_cache, block_table, seqlen_kv,
                                  page_size=page_size)
    out_ref = ref_gqa_decode_paged(q, k_cache, v_cache, block_table, seqlen_kv,
                                    page_size=page_size)
    torch.testing.assert_close(out_tl, out_ref, rtol=1e-2, atol=1e-2)
    print("✓ Correctness passed!")
    print()

    # ── Benchmark ──
    kernel_type = "WS (sm_90+)" if is_sm90_plus() else "Pipeline-ns1 (sm_89-)"

    # Warmup
    for _ in range(3):
        _ = gqa_decode_paged_fn(q, k_cache, v_cache, block_table, seqlen_kv,
                                 page_size=page_size)
    torch.cuda.synchronize()

    latency_tl = do_bench(
        lambda: gqa_decode_paged_fn(q, k_cache, v_cache, block_table, seqlen_kv,
                                     page_size=page_size),
        _n_warmup=10, _n_repeat=20,
    )

    # Theoretical FLOPs: 2 matmuls per head
    # Q@K^T: 2 * B * H_q * D * avg_seqlen
    # S@V:   2 * B * H_q * D * avg_seqlen
    avg_seqlen = seqlen_kv.float().mean().item()
    total_flops = 4 * B * H_q * D * avg_seqlen
    tflops = total_flops / latency_tl * 1e-9

    # Memory bytes read: Q + K + V per batch
    # Q: B * H_q * D * 2 bytes
    # K: B * avg_seqlen * H_kv * D * 2 bytes
    # V: B * avg_seqlen * H_kv * D * 2 bytes
    total_bytes = 2 * B * (H_q * D + 2 * avg_seqlen * H_kv * D)
    bandwidth_tb = total_bytes / latency_tl * 1e-9

    print(f"TileLang [{kernel_type}]: {latency_tl * 1000:.2f} µs")
    print(f"  Compute:  {tflops:.2f} TFlops")
    print(f"  Bandwidth:{bandwidth_tb:.2f} TB/s  (achieved)")
    print()

    # ── FlashAttention comparison ──
    try:
        import flash_attn

        # Prepare paged KV for flash_attn
        k_cache_flat = k_cache.reshape(-1, H_kv, D).contiguous()
        v_cache_flat = v_cache.reshape(-1, H_kv, D).contiguous()

        def flash_attn_fn():
            return flash_attn.flash_attn_with_kvcache(
                q.unsqueeze(1),  # [B, 1, H_q, D]
                k_cache_flat.unsqueeze(0).expand(B, -1, -1, -1),
                v_cache_flat.unsqueeze(0).expand(B, -1, -1, -1),
                cache_seqlens=seqlen_kv,
            )

        # Warmup
        for _ in range(3):
            _ = flash_attn_fn()
        torch.cuda.synchronize()

        out_fa = flash_attn_fn().squeeze(1)
        torch.testing.assert_close(out_tl, out_fa, rtol=1e-2, atol=1e-2)
        print("✓ flash_attn correctness passed!")

        latency_fa = do_bench(flash_attn_fn, _n_warmup=10, _n_repeat=20)
        tflops_fa = total_flops / latency_fa * 1e-9
        bandwidth_fa = total_bytes / latency_fa * 1e-9

        print(f"FlashAttention:          {latency_fa * 1000:.2f} µs")
        print(f"  Compute:  {tflops_fa:.2f} TFlops")
        print(f"  Bandwidth:{bandwidth_fa:.2f} TB/s  (achieved)")
        print()
        print(f"Speedup vs FA: {latency_fa / latency_tl:.2f}x")
        print(f"Bandwidth ratio: {bandwidth_tb / bandwidth_fa:.2f}x")

    except ImportError:
        print("flash_attn not available")
    except Exception as e:
        print(f"flash_attn failed: {e}")

    # ── Vary seqlen sweep ──
    print()
    print("═══ SeqLen Sweep ═══")
    print(f"{'SeqLen':>8} {'TileLang(µs)':>14} {'FA(µs)':>10} {'Ratio':>8} {'TB/s(TL)':>10}")
    print("─" * 54)

    for sl in [1024, 2048, 4096, 8192, 16384]:
        if sl > total_kv_tokens:
            continue

        seq_lens = torch.full((B,), sl, dtype=torch.int32, device=device)
        seq_lens[:B//5] = torch.randint(sl // 4, sl, (B // 5,),
                                         dtype=torch.int32, device=device)

        avg_sl = seq_lens.float().mean().item()
        flops_sl = 4 * B * H_q * D * avg_sl
        bytes_sl = 2 * B * (H_q * D + 2 * avg_sl * H_kv * D)

        lat_tl = do_bench(
            lambda: gqa_decode_paged_fn(q, k_cache, v_cache, block_table,
                                         seq_lens, page_size=page_size),
            _n_warmup=5, _n_repeat=10,
        )
        bw_tl = bytes_sl / lat_tl * 1e-9

        try:
            import flash_attn
            lat_fa = do_bench(
                lambda: flash_attn.flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache_flat.unsqueeze(0).expand(B, -1, -1, -1),
                    v_cache_flat.unsqueeze(0).expand(B, -1, -1, -1),
                    cache_seqlens=seq_lens,
                ),
                _n_warmup=5, _n_repeat=10,
            )
            ratio = lat_fa / lat_tl
            fa_str = f"{lat_fa * 1000:8.1f}"
        except:
            ratio = 0
            fa_str = "N/A"

        print(f"{sl:8d} {lat_tl * 1000:14.1f} {fa_str:>10} {ratio:8.2f} {bw_tl:10.2f}")
