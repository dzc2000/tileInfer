import tilelang
import tilelang.testing
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1])
def rms_norm(M, N, blk_m, eps, add_one, dtype):

    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), W: T.Tensor((N, ), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            X_shared = T.alloc_shared((blk_m, N), dtype)
            X_pow_local = T.alloc_fragment((blk_m, N), T.float)
            X_local = T.alloc_fragment((blk_m, N), dtype)
            X_powsum = T.alloc_fragment((blk_m,), T.float)

            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], X_shared)
            T.copy(X_shared, X_local)
            for i, j in T.Parallel(blk_m, N):
                X_pow_local[i, j] = X_local[i, j] * X_local[i, j]
            T.reduce_sum(X_pow_local, X_powsum, dim=1)
            for i in T.Parallel(blk_m):
                X_powsum[i] = T.rsqrt(X_powsum[i] / N + eps)

            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] *= X_powsum[i] * (W[j] + float(add_one))
            T.copy(X_local, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, add_one_to_weight: bool = True) -> torch.Tensor:
    """High-level RMSNorm wrapper matching qwen_36.py interface."""
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    M = x_2d.shape[0]
    blk_m = 1
    kernel = rms_norm(M, D, blk_m, eps, add_one_to_weight, x_2d.dtype)
    out = kernel(x_2d, weight)
    return out.reshape(orig_shape)

def ref_program(x, w, eps=1e-5, add_one=True):
    w = w + (1.0 if add_one else 0.0)
    x = x * w / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x

if __name__ == "__main__":
    M = 32
    N = 5120
    eps = 1e-5
    add_one = True
    kernel = rms_norm(M, N, 1, eps, add_one, T.bfloat16)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, dtype=torch.bfloat16, device="cuda")

    
    out = kernel(x, w)
    torch.testing.assert_close(out, ref_program(x, w, eps, add_one), rtol=0.01, atol=0.01)
    print("All checks pass.")

    from tilelang.profiler import do_bench
    latency = do_bench(lambda: ref_program(x, w, eps, add_one), warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = do_bench(lambda: kernel(x, w), warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))