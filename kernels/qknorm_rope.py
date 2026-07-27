import tilelang
import tilelang.testing
import tilelang.language as T
import torch
from tilelang.profiler import do_bench

@tilelang.jit(out_idx=[-1])
def qknorm_rope_kernel(B, num_heads, HEAD_DIM, ROTARY_DIM, threads, eps=1e-5, dtype=T.bfloat16):
    HALF_ROT = ROTARY_DIM // 2
    @T.prim_func
    def main( 
        X: T.Tensor((B, num_heads, 1, HEAD_DIM), dtype),
        W: T.Tensor((HEAD_DIM, ), dtype),
        COS: T.Tensor((ROTARY_DIM, ), dtype),
        SIN: T.Tensor((ROTARY_DIM, ), dtype),
        Y: T.Tensor((B, num_heads, 1, HEAD_DIM), dtype),
    ):
        with T.Kernel(B, num_heads, threads=threads) as (batch_id, head_id):
            x_shared = T.alloc_shared((HEAD_DIM, ), dtype)
            w_shared = T.alloc_shared((HEAD_DIM, ), dtype)
            # Optimization: only allocate and load first HALF_ROT length
            cos_shared = T.alloc_shared((HALF_ROT, ), dtype)
            sin_shared = T.alloc_shared((HALF_ROT, ), dtype)
            T.copy(X[batch_id, head_id, 0, :], x_shared)
            T.copy(W, w_shared)
            T.copy(COS[0:HALF_ROT], cos_shared)
            T.copy(SIN[0:HALF_ROT], sin_shared)
            # ============ 1. RMSNorm ============
            x_local = T.alloc_fragment((HEAD_DIM, ), T.float32)
            pow_local = T.alloc_fragment((HEAD_DIM, ), T.float32)
            T.copy(x_shared, x_local)
            for i in T.Parallel(HEAD_DIM):
                pow_local[i] = x_local[i] * x_local[i]
            pow_sum = T.alloc_fragment((1, ), T.float32)
            T.reduce_sum(pow_local, pow_sum, dim=0)
            pow_sum[0] = T.rsqrt(pow_sum[0] / HEAD_DIM + eps)
            for i in T.Parallel(HEAD_DIM):
                x_local[i] = x_local[i] * pow_sum[0] * (1.0 + T.cast(w_shared[i], T.float32))
                x_shared[i] = T.cast(x_local[i], dtype)
            T.sync_threads()
            # ============ 2. RoPE ============
            for i in T.serial(HALF_ROT):
                c = T.cast(cos_shared[i], T.float32)
                s = T.cast(sin_shared[i], T.float32)
                val_i = T.cast(x_shared[i], T.float32)
                val_paired = T.cast(x_shared[i + HALF_ROT], T.float32)
                x_shared[i] = T.cast(val_i * c - val_paired * s, dtype)
                x_shared[i + HALF_ROT] = T.cast(val_paired * c + val_i * s, dtype)
            T.copy(x_shared, Y[batch_id, head_id, 0, :])
    return main

def fused_qknorm_rope(x, weight, cos_flat, sin_flat, eps=1e-5):
    B, num_heads, S, head_dim = x.shape
    assert S == 1
    rotary_dim = cos_flat.shape[0]
    kernel = qknorm_rope_kernel(B, num_heads, head_dim, rotary_dim, threads=128, eps=eps, dtype=x.dtype)
    return kernel(x, weight, cos_flat, sin_flat)

def ref_fused_qknorm_rope(
    x: torch.Tensor,          # [B, num_heads, 1, head_dim]
    weight: torch.Tensor,     # [head_dim]
    cos: torch.Tensor,        # [B, 1, rotary_dim]  注意：算子实际只消费前 rotary_dim // 2
    sin: torch.Tensor,        # [B, 1, rotary_dim]
    eps: float = 1e-5,
) -> torch.Tensor:
    B, H, S, D = x.shape
    rd = cos.shape[-1]        # rotary_dim
    assert rd % 2 == 0
    half = rd // 2
    x_float = x.float()
    # RMSNorm: variance mean over last dim
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(var + eps)
    x_normed = (1.0 + weight.float()) * x_normed   # weight 形状 [D] 会自动广播
    # 分离旋转部分和直通部分
    x_rot = x_normed[..., :rd]      # [B, H, 1, rd]
    x_pass = x_normed[..., rd:]     # [B, H, 1, D-rd]
    # 分离旋转部分的前半段和后半段
    rot1 = x_rot[..., :half]        # [B, H, 1, half]
    rot2 = x_rot[..., half:]        # [B, H, 1, half]

    cos_f = cos[..., :half].unsqueeze(2).float()   # [B, 1, 1, half]
    sin_f = sin[..., :half].unsqueeze(2).float()   # [B, 1, 1, half]

    out_rot1 = rot1 * cos_f - rot2 * sin_f
    out_rot2 = rot2 * cos_f + rot1 * sin_f
    x_rot_out = torch.cat([out_rot1, out_rot2], dim=-1) # [B, H, 1, rd]
    out = torch.cat([x_rot_out, x_pass], dim=-1)
    return out.to(torch.bfloat16)

def test_fused_qknorm_rope():
    B, num_heads = 4, 8
    head_dim = 128
    rotary_dim = 64
    eps = 1e-5
    dtype = torch.bfloat16
    device = "cuda"

    # 1. 生成随机输入
    x = torch.randn(B, num_heads, 1, head_dim, dtype=dtype, device=device)
    weight = torch.randn(head_dim, dtype=dtype, device=device)
    
    # 2. 生成 cos/sin：kernel 需要 [rotary_dim] 扁平，ref 需要 [B, 1, rotary_dim]
    cos_flat = torch.randn(rotary_dim, dtype=dtype, device=device)
    sin_flat = torch.randn(rotary_dim, dtype=dtype, device=device)
    # ref 版本：取相同值，扩展成 [B, 1, rd]
    cos_ref = cos_flat.unsqueeze(0).unsqueeze(0).expand(B, 1, rotary_dim).contiguous()
    sin_ref = sin_flat.unsqueeze(0).unsqueeze(0).expand(B, 1, rotary_dim).contiguous()
    
    # 3. 前向对比
    y_fused = fused_qknorm_rope(x, weight, cos_flat, sin_flat, eps)
    y_ref = ref_fused_qknorm_rope(x, weight, cos_ref, sin_ref, eps)
    
    torch.testing.assert_close(y_fused, y_ref, rtol=1e-2, atol=1e-2)
    print("前向验证通过 ✓")
    
    # 4. 性能测试（detach）
    x_perf = x.detach()
    w_perf = weight.detach()
    cos_perf = cos_flat.detach()
    sin_perf = sin_flat.detach()
    cos_ref_perf = cos_ref.detach()
    sin_ref_perf = sin_ref.detach()
    
    ms_fused = do_bench(
        lambda: fused_qknorm_rope(x_perf, w_perf, cos_perf, sin_perf, eps),
        backend="event"
    )
    ms_ref = do_bench(
        lambda: ref_fused_qknorm_rope(x_perf, w_perf, cos_ref_perf, sin_ref_perf, eps),
        backend="event"
    )
    print(f"TileLang fused: {ms_fused:.4f} ms")
    print(f"PyTorch ref   : {ms_ref:.4f} ms")
    print(f"Speedup: {ms_ref / ms_fused:.2f}x")

if __name__ == "__main__":
    test_fused_qknorm_rope()
