import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench
from tvm import DataType

@tilelang.jit(out_idx=[-1])
def splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "bfloat16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K
 
    @T.prim_func
    def main(
            A: T.Tensor((K,), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(reduce_threads, BLOCK_N)) as bn:
            tk = T.get_thread_binding(0)  # threadIdx.x
            tn = T.get_thread_binding(1)  # threadIdx.y
            
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
 
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                
                for k in T.vectorized(TILE_K):
                    k_idx = bk * BLOCK_K + tk * TILE_K + k
                    A_local[k] = T.if_then_else(k_idx < K, A[k_idx], T.cast(0, dtype))
                    B_local[k] = T.if_then_else(
                        k_idx < K, B[bn * BLOCK_N + tn, k_idx], T.cast(0, dtype))
                
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            
            C_reduced = T.alloc_local((1,), accum_dtype)
            with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                # Warp 级 Shuffle 规约
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        C_accum[0],
                        True,
                        C_reduced[0],
                        tk,
                        dtype="handle",
                    ))
 
            C[bn * BLOCK_N + tn] = C_reduced[0]
 
    return main

@tilelang.jit(out_idx=[-1])
def gemv_alloc_reducer(
    M, N, block_M=64, block_N=128, num_stages=2, threads=128, dtype: str = "bfloat16", accum_dtype: str = "float"
):
    @T.prim_func
    def main(
            x: T.Tensor((N,), dtype),
            a: T.Tensor((M, N), dtype),
            o: T.Tensor((M,), dtype),
    ):

        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as i0_m:
            o_reducer = T.alloc_reducer(block_M, accum_dtype, replication="all")
            T.clear(o_reducer)
            for i0_n in T.Pipelined(T.ceildiv(N, block_N), num_stages=num_stages):
                a_smem = T.alloc_shared((block_M, block_N), dtype)
                T.copy(a[i0_m * block_M, i0_n * block_N], a_smem)
                a_frag = T.alloc_fragment((block_M, block_N), dtype)
                T.copy(a_smem, a_frag)
                x_frag = T.alloc_fragment(block_N, dtype)
                T.copy(x[i0_n * block_N], x_frag)
                for i1_m, i1_n in T.Parallel(block_M, block_N):
                    o_reducer[i1_m] += T.cast(a_frag[i1_m, i1_n], accum_dtype) * T.cast(x_frag[i1_n], accum_dtype)
            T.finalize_reducer(o_reducer)
            T.copy(o_reducer, o[i0_m * block_M])

    return main


@tilelang.jit(out_idx=[-1])
def gemv_int8_alloc_reducer(
    M, N, block_M=64, block_N=128, num_stages=2, threads=128,
    dtype: str = "bfloat16", accum_dtype: str = "float",
):
    @T.prim_func
    def main(
            x: T.Tensor((N,), dtype),
            w: T.Tensor((M, N), "int8"),
            scale: T.Tensor((M,), "float32"),
            o: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as i0_m:
            o_reducer = T.alloc_reducer(block_M, accum_dtype, replication="all")
            T.clear(o_reducer)
            for i0_n in T.Pipelined(T.ceildiv(N, block_N), num_stages=num_stages):
                w_smem = T.alloc_shared((block_M, block_N), "int8")
                T.copy(w[i0_m * block_M, i0_n * block_N], w_smem)
                a_frag = T.alloc_fragment((block_M, block_N), dtype)
                for i, j in T.Parallel(block_M, block_N):
                    a_frag[i, j] = T.cast(w_smem[i, j], dtype)
                x_frag = T.alloc_fragment(block_N, dtype)
                T.copy(x[i0_n * block_N], x_frag)
                for i1_m, i1_n in T.Parallel(block_M, block_N):
                    o_reducer[i1_m] += T.cast(a_frag[i1_m, i1_n], accum_dtype) * T.cast(x_frag[i1_n], accum_dtype)
            T.finalize_reducer(o_reducer)
            s_frag = T.alloc_fragment(block_M, "float32")
            T.copy(scale[i0_m * block_M], s_frag)
            for i in T.Parallel(block_M):
                o_reducer[i] = o_reducer[i] * s_frag[i]
            T.copy(o_reducer, o[i0_m * block_M])

    return main


@tilelang.jit(out_idx=[-1])
def dequantize_gemv_kernel(
    N: int,
    K: int,
    n_partition: int = 4,
    reduce_thread: int = 32,
    in_dtype: str = "bfloat16",
    out_dtype: str = "bfloat16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    block_K = reduce_thread * micro_size_k

    @T.prim_func
    def main(
        A: T.Tensor((K,), in_dtype),
        B: T.Tensor((N, K), "int8"),
        Scale: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, n_partition),
            threads=(reduce_thread, n_partition),
        ) as bx:
            A_local = T.alloc_local((micro_size_k,), in_dtype)
            B_local = T.alloc_local((micro_size_k,), "int8")
            B_dequant_local = T.alloc_local((micro_size_k,), in_dtype)
            accum_res = T.alloc_local((1,), accum_dtype)
            reduced_accum_res = T.alloc_local((1,), accum_dtype)

            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

            T.clear(accum_res)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for v in T.vectorized(micro_size_k):
                    A_local[v] = A[ko * block_K + kr * micro_size_k + v]

                for v in T.vectorized(micro_size_k):
                    B_local[v] = B[
                        bx * n_partition + ni,
                        ko * block_K + kr * micro_size_k + v,
                    ]

                for ki in T.serial(micro_size_k):
                    B_dequant_local[ki] = T.cast(B_local[ki], in_dtype)

                for ki in T.serial(micro_size_k):
                    accum_res[0] += T.cast(A_local[ki], accum_dtype) * T.cast(B_dequant_local[ki], accum_dtype)

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[0],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    )
                )
            if kr == 0:
                out_idx = bx * n_partition + ni
                C[out_idx] = T.cast(reduced_accum_res[0] * Scale[out_idx], out_dtype)

    return main

def gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dtype = str(x.dtype).replace("torch.", "")
    N, K = weight.shape
    block_M = 64
    block_N = 128 if K >= 128 else K
    if N >= block_M and N % block_M == 0 and K % block_N == 0:
        kernel = gemv_alloc_reducer(N, K, block_M=block_M, block_N=block_N,
                                     num_stages=3, threads=128, dtype=dtype)
        out = kernel(x, weight)
    else:
        BLOCK_N = min(8, N)
        reduce_threads = 32
        kernel = splitk_gemv(N, K, BLOCK_N, reduce_threads, dtype=dtype)
        out = kernel(x, weight)
    return out


def gemv_int8(x: torch.Tensor, weight_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    N, K = weight_int8.shape
    n_partition = 8
    reduce_thread = 32
    kernel = dequantize_gemv_kernel(
        N, K, n_partition=n_partition, reduce_thread=reduce_thread,
        in_dtype="bfloat16", out_dtype="bfloat16", accum_dtype="float",
    )
    out = kernel(x, weight_int8, scale)
    return out


def bf16_linear_forward(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor = None) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]

    if M == 1:
        x_flat = x.squeeze(0)
        if scale is not None:
            out = gemv_int8(x_flat, weight, scale)
        else:
            out = gemv(x_flat, weight)
        return out.unsqueeze(0)
    else:
        if scale is not None:
            w_bf16 = weight.to(x.dtype) * scale.unsqueeze(1).to(x.dtype)
            return torch.nn.functional.linear(x, w_bf16)
        return torch.nn.functional.linear(x, weight)

def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.dim() == 1:
        x = A.unsqueeze(0)          # (1, K)
    else:
        x = A                        # 假设已经是 (1, K)
    
    logits = torch.nn.functional.linear(x, B)          # (1, N)
    
    # 去掉 batch 维度得到 (N,)
    C = logits.squeeze(0)
    return C

if __name__ == "__main__":
    N, K = 512, 256
    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn(K, dtype=dtype, device=device)
    weight = torch.randn(N, K, dtype=dtype, device=device)

    out_tl = gemv(x, weight)
    out_ref = ref_program(x, weight)
    torch.testing.assert_close(out_tl, out_ref, rtol=1e-2, atol=1e-2)
    print("✓ 正确性通过")

    # 性能测试（1/10 规模）
    N_real, K_real = 24832, 5120
    print(f"\n性能 N={N_real} K={K_real}")
    x_real = torch.randn(K_real, dtype=dtype, device=device)
    w_real = torch.randn(N_real, K_real, dtype=dtype, device=device)

    ms_fused = do_bench(lambda: gemv(x_real, w_real), backend="event")
    ms_ref   = do_bench(lambda: ref_program(x_real, w_real), backend="event")
    print(f"TileLang fused: {ms_fused:.4f} ms")
    print(f"Torch ref     : {ms_ref:.4f} ms")