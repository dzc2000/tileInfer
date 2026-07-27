import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench
from tvm import DataType
@tilelang.jit(out_idx=[-2,-1])            # 自动推导输出索引
def gemv_perblock_kernel(
    N, K, num_blocks_n,
    BLOCK_N=8, reduce_threads=32, dtype="bfloat16"
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
        x: T.Tensor((K,), dtype),
        weight: T.Tensor((N, K), dtype),
        partial_max: T.Tensor((num_blocks_n,), "float32"),
        partial_idx: T.Tensor((num_blocks_n,), "int64"),
    ):
        with T.Kernel(num_blocks_n, threads=(reduce_threads, BLOCK_N)) as bn:
            tk = T.get_thread_binding(0)
            tn = T.get_thread_binding(1)

            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_accum = T.alloc_local((1,), "float32")
            T.clear(C_accum)

            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    k_idx = bk * BLOCK_K + tk * TILE_K + k
                    A_local[k] = T.if_then_else(k_idx < K, x[k_idx], T.cast(0, dtype))
                    B_local[k] = T.if_then_else(
                        k_idx < K,
                        weight[bn * BLOCK_N + tn, k_idx],
                        T.cast(0, dtype),
                    )
                for k in T.serial(TILE_K):
                    C_accum[0] += (
                        A_local[k].astype("float32") * B_local[k].astype("float32")
                    )

            C_reduced = T.alloc_local((1,), "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint64(1),
                        C_accum[0],
                        True,
                        C_reduced[0],
                        tk,
                        dtype="handle",
                    )
                )

            block_max = T.alloc_local((1,), "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: T.max(x, y), [-T.infinity("float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(1), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint64(1), C_reduced[0], True, block_max[0], tn, dtype="handle",
                    )
                )

            candidate_tn = T.alloc_local((1,), "float32")
            candidate_tn[0] = T.if_then_else(
                C_reduced[0] == block_max[0],
                T.cast(tn, "float32"),
                T.cast(BLOCK_N, "float32"),
            )
            min_tn_f = T.alloc_local((1,), "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: T.min(x, y), [T.infinity("float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(2), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), candidate_tn[0], True,
                        min_tn_f[0], tn, dtype="handle",
                    )
                )
            if tn == 0:
                partial_max[bn] = block_max[0]
                partial_idx[bn] = T.cast(bn * BLOCK_N + T.cast(min_tn_f[0], "int32"), "int64")

    return main
 
@tilelang.jit(out_idx=[-1])
def argmax_reduce_kernel(N: int, block_size: int, dtype: str = "float32"):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    VEC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    TILE_SIZE = block_size * VEC

    @T.prim_func
    def main(
        partial_max: T.Tensor((N,), dtype),
        partial_idx: T.Tensor((N,), "int64"),
        output: T.Tensor((1,), "int64"),
    ):
        with T.Kernel(1, threads=block_size) as bx:
            tk = T.get_thread_binding(0)

            pmax_s = T.alloc_shared((TILE_SIZE,), dtype)
            pidx_s = T.alloc_shared((TILE_SIZE,), "int64")

            local_max = T.alloc_local((1,), "float32")
            local_idx_f = T.alloc_local((1,), "float32")
            local_max[0] = -T.infinity("float32")
            local_idx_f[0] = T.infinity("float32")

            for ko in T.serial(T.ceildiv(N, TILE_SIZE)):
                T.copy(partial_max[ko * TILE_SIZE], pmax_s)
                T.copy(partial_idx[ko * TILE_SIZE], pidx_s)
                for j in T.serial(VEC):
                    elem_idx = j * block_size + tk
                    actual_idx = ko * TILE_SIZE + elem_idx
                    if actual_idx < N:
                        val = T.cast(pmax_s[elem_idx], "float32")
                        idx_f = T.cast(pidx_s[elem_idx], "float32")
                        if val > local_max[0]:
                            local_max[0] = val
                            local_idx_f[0] = idx_f
                        elif val == local_max[0]:
                            if idx_f < local_idx_f[0]:
                                local_idx_f[0] = idx_f

            global_max = T.alloc_local((1,), "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: T.max(x, y), [-T.infinity("float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), local_max[0], True,
                        global_max[0], tk, dtype="handle",
                    )
                )

            if local_max[0] != global_max[0]:
                local_idx_f[0] = T.infinity("float32")

            global_min_idx_f = T.alloc_local((1,), "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: T.min(x, y), [T.infinity("float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(1), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint64(1), local_idx_f[0], True,
                        global_min_idx_f[0], tk, dtype="handle",
                    )
                )

            if tk == 0:
                output[0] = T.cast(global_min_idx_f[0], "int64")

    return main

def align_up(x, alignment=8):
    return (x + alignment - 1) // alignment * alignment

@tilelang.jit(out_idx=[-2, -1])
def fused_gemm_partial_argmax(
    M, N, K, block_M, block_N, block_K, num_stages, thread_num, enable_rasteration, dtype="float16", accum_dtype="float32"
):
    num_N_blocks = T.ceildiv(N, block_N)
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        # 注意：Shape 转置为 (num_N_blocks, M)，确保 M 维度连续，方便 T.copy 写回
        partial_max: T.Tensor((num_N_blocks, M), "float32"),
        partial_idx: T.Tensor((num_N_blocks, M), "int64"),
    ):
        with T.Kernel(num_N_blocks, T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # 核心：分配 C_shared，将 GEMM 结果转移到标准 Shared Memory 再做 Argmax
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)

            # ================= Step 1: 标准 GEMM =================
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            # ================= Step 2: 在 Shared Memory 上做局部 Argmax =================
            T.copy(C_local, C_shared)

            row_extreme = T.alloc_fragment((block_M,), "float32")
            out_idx = T.alloc_fragment((block_M,), "int64")

            T.fill(row_extreme, -T.infinity("float32"))
            T.reduce_max(C_shared, row_extreme, dim=1, clear=False)

            # 并行构建候选索引：匹配 max 处 = j，否则 = block_N（哨兵），
            # 再 reduce_min 沿 dim=1 取首个匹配 j，替代低效的串行 scan+loop_break。
            cand = T.alloc_fragment((block_M, block_N), "int32")
            for i, j in T.Parallel(block_M, block_N):
                if C_shared[i, j] == row_extreme[i]:
                    cand[i, j] = j
                else:
                    cand[i, j] = block_N
            first_idx = T.alloc_fragment((block_M,), "int32")
            T.reduce_min(cand, first_idx, dim=1)
            for i in T.Parallel(block_M):
                out_idx[i] = T.cast(bx * block_N + first_idx[i], "int64")

            # ================= Step 3: 连续写回 =================
            # 因为转置了 Shape，这里变成了 1D Fragment 到 1D Global 的连续拷贝，彻底避免 Data Race
            T.copy(row_extreme, partial_max[bx, by * block_M])
            T.copy(out_idx, partial_idx[bx, by * block_M])

    return main

@tilelang.jit(out_idx=[-1])
def argmax_reduce_2d_kernel(M: int, num_N_blocks: int, block_m: int, threads: int):
    VEC = 4
    TILE_SIZE = threads * VEC

    @T.prim_func
    def main(
        # 对应 Kernel 1 的输出 Shape
        partial_max: T.Tensor((num_N_blocks, M), "float32"),
        partial_idx: T.Tensor((num_N_blocks, M), "int64"),
        out: T.Tensor((M,), "int64"),
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            tk = T.get_thread_binding(0)

            pmax_s = T.alloc_shared((TILE_SIZE, block_m), "float32")
            pidx_s = T.alloc_shared((TILE_SIZE, block_m), "int64")

            local_max = T.alloc_local((block_m,), "float32")
            local_idx_f = T.alloc_local((block_m,), "float32")
            
            for i in T.Serial(block_m):
                local_max[i] = -T.infinity("float32")
                local_idx_f[i] = T.infinity("float32")

            # 分块加载，每块加载 (TILE_SIZE, block_m)
            for ko in T.serial(T.ceildiv(num_N_blocks, TILE_SIZE)):
                T.copy(partial_max[ko * TILE_SIZE, pid_m * block_m], pmax_s)
                T.copy(partial_idx[ko * TILE_SIZE, pid_m * block_m], pidx_s)

                for j in T.Serial(VEC):
                    elem_idx = j * threads + tk
                    actual_idx = ko * TILE_SIZE + elem_idx

                    if actual_idx < num_N_blocks:
                        for i in T.Serial(block_m):
                            val = pmax_s[elem_idx, i]
                            idx_f = T.cast(pidx_s[elem_idx, i], "float32")
                            if val > local_max[i]:
                                local_max[i] = val
                                local_idx_f[i] = idx_f
                            elif val == local_max[i]:
                                if idx_f < local_idx_f[i]:
                                    local_idx_f[i] = idx_f

            # 对每一行进行两次 T.tvm_thread_allreduce
            for i in T.Serial(block_m):
                global_max = T.alloc_local((1,), "float32")
                with T.attr(
                    T.comm_reducer(lambda x, y: T.max(x, y), [-T.infinity("float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1), local_max[i], True,
                            global_max[0], tk, dtype="handle",
                        )
                    )

                if local_max[i] != global_max[0]:
                    local_idx_f[i] = T.infinity("float32")

                global_min_idx_f = T.alloc_local((1,), "float32")
                with T.attr(
                    T.comm_reducer(lambda x, y: T.min(x, y), [T.infinity("float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(1), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1), local_idx_f[i], True,
                            global_min_idx_f[0], tk, dtype="handle",
                        )
                    )

                if tk == 0:
                    out[pid_m * block_m + i] = T.cast(T.cast(global_min_idx_f[0], "int32"), "int64")

    return main

 
@tilelang.jit(out_idx=[-2, -1])
def argmax_reduce_kernel_with_max(N: int, block_size: int):
    VEC = 4
    TILE_SIZE = block_size * VEC

    @T.prim_func
    def main(
        partial_max: T.Tensor((N,), "float32"),
        partial_idx: T.Tensor((N,), "int64"),
        output_idx: T.Tensor((1,), "int64"),
        output_max: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1, threads=block_size) as bx:
            tk = T.get_thread_binding(0)

            pmax_s = T.alloc_shared((TILE_SIZE,), "float32")
            pidx_s = T.alloc_shared((TILE_SIZE,), "int64")

            local_max = T.alloc_local((1,), "float32")
            local_idx_f = T.alloc_local((1,), "float32")
            local_max[0] = -T.infinity("float32")
            local_idx_f[0] = T.infinity("float32")

            for ko in T.serial(T.ceildiv(N, TILE_SIZE)):
                T.copy(partial_max[ko * TILE_SIZE], pmax_s)
                T.copy(partial_idx[ko * TILE_SIZE], pidx_s)
                for j in T.serial(VEC):
                    elem_idx = j * block_size + tk
                    actual_idx = ko * TILE_SIZE + elem_idx
                    if actual_idx < N:
                        val = pmax_s[elem_idx]
                        idx_f = T.cast(pidx_s[elem_idx], "float32")
                        if val > local_max[0]:
                            local_max[0] = val
                            local_idx_f[0] = idx_f
                        elif val == local_max[0]:
                            if idx_f < local_idx_f[0]:
                                local_idx_f[0] = idx_f

            global_max = T.alloc_local((1,), "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: T.max(x, y), [-T.infinity("float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), local_max[0], True,
                        global_max[0], tk, dtype="handle",
                    )
                )

            if local_max[0] != global_max[0]:
                local_idx_f[0] = T.infinity("float32")

            global_min_idx_f = T.alloc_local((1,), "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: T.min(x, y), [T.infinity("float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(1), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint64(1), local_idx_f[0], True,
                        global_min_idx_f[0], tk, dtype="handle",
                    )
                )

            if tk == 0:
                output_idx[0] = T.cast(global_min_idx_f[0], "int64")
                output_max[0] = global_max[0]

    return main


@tilelang.jit(out_idx=[-2, -1])
def argmax_reduce_2d_kernel_with_max(M: int, num_N_blocks: int, block_m: int, threads: int):
    VEC = 4
    TILE_SIZE = threads * VEC

    @T.prim_func
    def main(
        partial_max: T.Tensor((num_N_blocks, M), "float32"),
        partial_idx: T.Tensor((num_N_blocks, M), "int64"),
        out_idx: T.Tensor((M,), "int64"),
        out_max: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            tk = T.get_thread_binding(0)

            pmax_s = T.alloc_shared((TILE_SIZE, block_m), "float32")
            pidx_s = T.alloc_shared((TILE_SIZE, block_m), "int64")

            local_max = T.alloc_local((block_m,), "float32")
            local_idx_f = T.alloc_local((block_m,), "float32")

            for i in T.Serial(block_m):
                local_max[i] = -T.infinity("float32")
                local_idx_f[i] = T.infinity("float32")

            for ko in T.serial(T.ceildiv(num_N_blocks, TILE_SIZE)):
                T.copy(partial_max[ko * TILE_SIZE, pid_m * block_m], pmax_s)
                T.copy(partial_idx[ko * TILE_SIZE, pid_m * block_m], pidx_s)

                for j in T.Serial(VEC):
                    elem_idx = j * threads + tk
                    actual_idx = ko * TILE_SIZE + elem_idx

                    if actual_idx < num_N_blocks:
                        for i in T.Serial(block_m):
                            val = pmax_s[elem_idx, i]
                            idx_f = T.cast(pidx_s[elem_idx, i], "float32")
                            if val > local_max[i]:
                                local_max[i] = val
                                local_idx_f[i] = idx_f
                            elif val == local_max[i]:
                                if idx_f < local_idx_f[i]:
                                    local_idx_f[i] = idx_f

            for i in T.Serial(block_m):
                global_max = T.alloc_local((1,), "float32")
                with T.attr(
                    T.comm_reducer(lambda x, y: T.max(x, y), [-T.infinity("float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1), local_max[i], True,
                            global_max[0], tk, dtype="handle",
                        )
                    )

                if local_max[i] != global_max[0]:
                    local_idx_f[i] = T.infinity("float32")

                global_min_idx_f = T.alloc_local((1,), "float32")
                with T.attr(
                    T.comm_reducer(lambda x, y: T.min(x, y), [T.infinity("float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(1), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1), local_idx_f[i], True,
                            global_min_idx_f[0], tk, dtype="handle",
                        )
                    )

                if tk == 0:
                    out_idx[pid_m * block_m + i] = T.cast(T.cast(global_min_idx_f[0], "int32"), "int64")
                    out_max[pid_m * block_m + i] = global_max[0]

    return main


def fused_lm_head_argmax(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    支持 1D (单Token) 和 2D (多Token/Batch) 的融合 LM_Head + Argmax
    """
    N, K = weight.shape
    dtype = str(x.dtype).replace("torch.", "")

    if x.dim() == 1:
        if x.shape[0] != K:
            raise ValueError(f"1D x shape {x.shape} doesn't match weight K {K}")

        BLOCK_N = 8
        reduce_threads = 32
        num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N

        kernel1 = gemv_perblock_kernel(N, K, num_blocks_n, BLOCK_N, reduce_threads, dtype)
        partial_max, partial_idx = kernel1(x, weight)

        kernel2 = argmax_reduce_kernel(num_blocks_n, block_size=256, dtype="float32")
        out = kernel2(partial_max, partial_idx)

        return out

    elif x.dim() == 2:
        M = x.shape[0]
        if x.shape[1] != K:
            raise ValueError(f"2D x shape {x.shape} doesn't match weight K {K}")

        block_M = 64
        block_N = 128
        block_K = 64
        num_stages = 3
        thread_num = 256
        enable_rasteration = True
        accum_dtype = "float32"

        num_N_blocks = (N + block_N - 1) // block_N

        kernel1 = fused_gemm_partial_argmax(
            M, N, K, block_M, block_N, block_K,
            num_stages, thread_num, enable_rasteration,
            dtype, accum_dtype
        )
        partial_max, partial_idx = kernel1(x, weight)

        block_m_reduce = 4
        threads_reduce = 128
        kernel2 = argmax_reduce_2d_kernel(M, num_N_blocks, block_m_reduce, threads_reduce)
        out = kernel2(partial_max, partial_idx)

        return out

    else:
        raise ValueError(f"Unsupported input dimensions: {x.dim()}. Expected 1D or 2D.")


def fused_lm_head_argmax_with_max(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    融合 LM_Head + Argmax，同时返回 token_ids 和对应的 max logit 值。
    用于 TP 分布式 argmax：各 rank 得到局部 (max_val, local_idx)，
    再通过 all_gather 确定全局 argmax，避免 materialize 完整 logits。
    """
    N, K = weight.shape
    dtype = str(x.dtype).replace("torch.", "")

    if x.dim() == 1:
        if x.shape[0] != K:
            raise ValueError(f"1D x shape {x.shape} doesn't match weight K {K}")

        BLOCK_N = 8
        reduce_threads = 32
        num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N

        kernel1 = gemv_perblock_kernel(N, K, num_blocks_n, BLOCK_N, reduce_threads, dtype)
        partial_max, partial_idx = kernel1(x, weight)

        kernel2 = argmax_reduce_kernel_with_max(num_blocks_n, block_size=256)
        out_idx, out_max = kernel2(partial_max, partial_idx)

        return out_idx, out_max

    elif x.dim() == 2:
        M = x.shape[0]
        if x.shape[1] != K:
            raise ValueError(f"2D x shape {x.shape} doesn't match weight K {K}")

        block_M = 64
        block_N = 128
        block_K = 64
        num_stages = 3
        thread_num = 256
        enable_rasteration = True
        accum_dtype = "float32"

        num_N_blocks = (N + block_N - 1) // block_N

        kernel1 = fused_gemm_partial_argmax(
            M, N, K, block_M, block_N, block_K,
            num_stages, thread_num, enable_rasteration,
            dtype, accum_dtype
        )
        partial_max, partial_idx = kernel1(x, weight)

        block_m_reduce = 4
        threads_reduce = 128
        kernel2 = argmax_reduce_2d_kernel_with_max(M, num_N_blocks, block_m_reduce, threads_reduce)
        out_idx, out_max = kernel2(partial_max, partial_idx)

        return out_idx, out_max

    else:
        raise ValueError(f"Unsupported input dimensions: {x.dim()}. Expected 1D or 2D.")

def ref_program(x, weight):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    logits = torch.nn.functional.linear(x, weight)

    return logits.argmax(dim=-1)


if __name__ == "__main__":
    # 小型正确性测试
    M, N, K = 16, 512, 256
    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn(M, K, dtype=dtype, device=device)
    weight = torch.randn(N, K, dtype=dtype, device=device)

    out_tl = fused_lm_head_argmax(x, weight)
    out_ref = ref_program(x, weight)
    torch.testing.assert_close(out_tl, out_ref, rtol=1e-2, atol=1e-2)
    print("✓ 正确性通过")

    M_real, N_real, K_real = 64, 24832, 5120
    print(f"\n性能 M={M_real} N={N_real} K={K_real}")
    x_real = torch.randn(M_real, K_real, dtype=dtype, device=device)
    w_real = torch.randn(N_real, K_real, dtype=dtype, device=device)

    ms_fused = do_bench(lambda: fused_lm_head_argmax(x_real, w_real), backend="event")
    ms_ref   = do_bench(lambda: ref_program(x_real, w_real), backend="event")
    print(f"TileLang fused: {ms_fused:.4f} ms")
    print(f"Torch ref     : {ms_ref:.4f} ms")