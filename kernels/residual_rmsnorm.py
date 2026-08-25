import tilelang
import tilelang.testing
import tilelang.language as T
import torch
from tilelang.profiler import do_bench

@tilelang.jit(out_idx=[-2, -1])
def residual_rmsnorm_kernel(M, N, blk_m=16, eps=1e-6, add_one=False, dtype=T.bfloat16):

    @T.prim_func
    def main(R: T.Tensor((M, N), dtype), X: T.Tensor((M, N), dtype), W: T.Tensor((N, ), dtype), NR: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            X_shared = T.alloc_shared((blk_m, N), dtype)
            R_shared = T.alloc_shared((blk_m, N), dtype)
            pow_local = T.alloc_fragment((blk_m, N), T.float)
            X_local = T.alloc_fragment((blk_m, N), dtype)
            R_local = T.alloc_fragment((blk_m, N), dtype)
            pow_sum = T.alloc_fragment((blk_m,), T.float)

            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], X_shared)
            T.copy(X_shared, X_local)
            T.copy(R[bx * blk_m : (bx + 1) * blk_m, :], R_shared)
            T.copy(R_shared, R_local)

            for i, j in T.Parallel(blk_m, N):
                x_f = T.cast(X_local[i, j], T.float32) + T.cast(R_local[i, j], T.float32)
                X_local[i, j] = T.cast(x_f, dtype)
                pow_local[i, j] = x_f * x_f
            
            T.copy(X_local, NR[bx * blk_m : (bx + 1) * blk_m, :])
            T.reduce_sum(pow_local, pow_sum, dim=1)
            for i in T.Parallel(blk_m):
                pow_sum[i] = T.rsqrt(pow_sum[i] / N + eps)

            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] *= pow_sum[i] * (W[j] + float(add_one))
            T.copy(X_local, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main

def residual_rmsnorm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_one_to_weight: bool = True,
) -> tuple:
    """Fused residual-add + RMSNorm in a single Tilelang kernel.

    Computes: new_residual = residual + x
              normed = rmsnorm(new_residual, weight, eps)

    Args:
        residual: [..., D] tensor
        x: [..., D] tensor to add
        weight: [D] norm weight
        eps: epsilon
        add_one_to_weight: True for Qwen3.5's (1+weight) scaling

    Returns:
        (new_residual, normed)  - both [..., D] in BF16
    """
    orig_shape = residual.shape
    D = orig_shape[-1]
    r_2d = residual.reshape(-1, D)
    x_2d = x.reshape(-1, D)
    M = r_2d.shape[0]

    # 选择块大小（可根据 M 调整）
    blk_m = 1  # 避免共享内存溢出：blk_m * N * 2bytes * 2 buffers 需在 GPU shared memory 限制内

    # 生成并调用 kernel
    kernel = residual_rmsnorm_kernel(M, D, blk_m, eps, add_one_to_weight, r_2d.dtype)
    new_r, normed = kernel(r_2d, x_2d, weight)

    # 恢复原始形状
    return new_r.reshape(orig_shape), normed.reshape(orig_shape)

# Alias matching qwen_36.py interface
fused_residual_rmsnorm = residual_rmsnorm

def ref_program(residual, x, w, eps=1e-6, add_one=True):
    """参考实现：残差加和 + RMSNorm"""
    w = w + (1.0 if add_one else 0.0)
    h = residual + x
    norm = h * w / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return h, norm

if __name__ == "__main__":
    # 测试参数：三维形状 B, S, D = 4, 1, 5120
    B, S, D = 4, 1, 5120
    eps = 1e-6
    add_one = True                     # 与 Qwen3.5 保持一致
    dtype = torch.bfloat16
    device = "cuda"

    # 生成随机输入，均需要梯度（用于反向测试，此处仅前向，但保留）
    residual = torch.randn(B, S, D, dtype=dtype, device=device, requires_grad=True)
    x = torch.randn(B, S, D, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, dtype=dtype, device=device, requires_grad=True)

    # 调用我们的融合算子
    new_residual, normed = residual_rmsnorm(residual, x, weight, eps, add_one_to_weight=add_one)

    # 调用参考实现
    new_residual_ref, normed_ref = ref_program(residual, x, weight, eps, add_one=add_one)

    # 前向结果对比（相对/绝对容差根据 bfloat16 精度设定）
    torch.testing.assert_close(new_residual, new_residual_ref, rtol=1e-1, atol=1e-2)
    torch.testing.assert_close(normed, normed_ref, rtol=1e-1, atol=1e-2)
    print("前向计算通过 ✓")

    # 前向性能对比（只测 normed 的计算，因为 new_residual 只是加法）
    # 为了避免梯度影响，使用 detach 后的张量
    residual_perf = residual.detach()
    x_perf = x.detach()
    weight_perf = weight.detach()
    
    ms_fwd = do_bench(
        lambda: residual_rmsnorm(residual_perf, x_perf, weight_perf, eps, add_one_to_weight=add_one),
        backend="event"
    )
    ms_fwd_ref = do_bench(
        lambda: ref_program(residual_perf, x_perf, weight_perf, eps, add_one=add_one),
        backend="event"
    )
    print(f"前向  tilelang: {ms_fwd:.4f} ms   ref: {ms_fwd_ref:.4f} ms")
