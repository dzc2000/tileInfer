# kernels/act_mul.py
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def silu_mul(N, NUM_ELE_PER_THREAD = 8, threads = 256, dtype=T.bfloat16):

    @T.prim_func
    def main(A: T.Tensor((N), dtype),  B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
        with T.Kernel(T.ceildiv(N, threads * NUM_ELE_PER_THREAD), threads = threads) as (b_x):
            A_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)
            B_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)
            C_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)

            s_start = b_x * threads * NUM_ELE_PER_THREAD
            s_end = (b_x + 1) * threads * NUM_ELE_PER_THREAD

            T.copy(
                A[s_start:s_end],
                A_register,
            )

            T.copy(
                B[s_start:s_end],
                B_register,
            )

            for tid, i in T.Parallel(threads, NUM_ELE_PER_THREAD):
                C_register[tid * NUM_ELE_PER_THREAD + i] = (
                    A_register[tid * NUM_ELE_PER_THREAD + i] *
                    T.sigmoid(A_register[tid * NUM_ELE_PER_THREAD + i]) *
                    B_register[tid * NUM_ELE_PER_THREAD + i])

            T.copy(
                C_register,
                C[s_start:s_end],
            )
    
    return main


def fused_silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(a) * b: a * sigmoid(a) * b."""
    N = a.numel()
    dtype = a.dtype
    assert dtype in (torch.bfloat16, torch.float16), \
        f"fused_silu_mul only supports bfloat16/float16, got {dtype}"
    tl_dtype = T.bfloat16 if dtype == torch.bfloat16 else T.float16
    BLOCK = 256 * 8
    N_padded = ((N + BLOCK - 1) // BLOCK) * BLOCK
    kernel = silu_mul(N_padded, dtype=tl_dtype)
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if N_padded != N:
        a_flat = torch.nn.functional.pad(a_flat, (0, N_padded - N))
        b_flat = torch.nn.functional.pad(b_flat, (0, N_padded - N))
    out = kernel(a_flat, b_flat)
    return out[:N].reshape(a.shape)


def fused_sigmoid_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused a * sigmoid(b)."""
    N = a.numel()
    dtype = a.dtype
    assert dtype in (torch.bfloat16, torch.float16), \
        f"fused_sigmoid_mul only supports bfloat16/float16, got {dtype}"
    tl_dtype = T.bfloat16 if dtype == torch.bfloat16 else T.float16
    BLOCK = 256 * 8
    N_padded = ((N + BLOCK - 1) // BLOCK) * BLOCK
    kernel = sigmoid_mul(N_padded, dtype=tl_dtype)
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if N_padded != N:
        a_flat = torch.nn.functional.pad(a_flat, (0, N_padded - N))
        b_flat = torch.nn.functional.pad(b_flat, (0, N_padded - N))
    out = kernel(a_flat, b_flat)
    return out[:N].reshape(a.shape)


@tilelang.jit(out_idx=[-1])
def sigmoid_mul(N, NUM_ELE_PER_THREAD=8, threads = 256, dtype=T.bfloat16):

    @T.prim_func
    def main(A: T.Tensor((N), dtype),  B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
        with T.Kernel(T.ceildiv(N, threads * NUM_ELE_PER_THREAD), threads = threads) as (b_x):
            A_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)
            B_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)
            C_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)

            s_start = b_x * threads * NUM_ELE_PER_THREAD
            s_end = (b_x + 1) * threads * NUM_ELE_PER_THREAD

            T.copy(
                A[s_start:s_end],
                A_register,
            )

            T.copy(
                B[s_start:s_end],
                B_register,
            )

            for tid, i in T.Parallel(threads, NUM_ELE_PER_THREAD):
                C_register[tid * NUM_ELE_PER_THREAD + i] = (
                    A_register[tid * NUM_ELE_PER_THREAD + i] *
                    T.sigmoid(B_register[tid * NUM_ELE_PER_THREAD + i])
                )

            T.copy(
                C_register,
                C[s_start:s_end],
            )
    
    return main