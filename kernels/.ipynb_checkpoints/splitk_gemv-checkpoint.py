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


def bf16_linear_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]

    if M == 1:
        x_flat = x.squeeze(0)
        out = gemv(x_flat, weight)
        return out.unsqueeze(0)
    else:
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