import tilelang
import tilelang.language as T
import torch
from tilelang.profiler import do_bench

_GEMM_M = 16     
DV_TILE = 16          
THREADS = 128          

@tilelang.jit(out_idx=[-2, -1])
def fused_postproj_recurrent_kernel_manual(B, num_k_heads, num_v_heads, Dk, Dv, BLOCK_DK, FUSE_NORM, dtype):
    V_PER_K = num_v_heads // num_k_heads
    INV_SQRT_DK = Dk ** -0.5
    KEY_DIM = num_k_heads * Dk
    CONV_DIM = KEY_DIM * 2 + num_v_heads * Dv

    @T.prim_func
    def kernel(
        QKV: T.Tensor((B, CONV_DIM), dtype),
        Alpha: T.Tensor((B, num_v_heads), dtype),
        BetaRaw: T.Tensor((B, num_v_heads), dtype),
        NegAExp: T.Tensor((num_v_heads,), dtype),
        DtBias: T.Tensor((num_v_heads,), dtype),
        State: T.Tensor((B, num_v_heads, Dk, Dv), dtype),
        NormW: T.Tensor((Dv,), dtype),
        Z: T.Tensor((B, num_v_heads, Dv), dtype),
        NewState: T.Tensor((B, num_v_heads, Dk, Dv), dtype),
        Output: T.Tensor((B, num_v_heads, Dv), dtype),
    ):
        with T.Kernel(num_v_heads, B, threads=THREADS) as (h, bid):
            k_head = h // V_PER_K

            q_full = T.alloc_shared((Dk,), dtype)
            k_full = T.alloc_shared((Dk,), dtype)
            v = T.alloc_shared((Dv,), dtype)

            T.copy(QKV[bid, k_head * Dk], q_full)
            T.copy(QKV[bid, KEY_DIM + k_head * Dk], k_full)
            T.copy(QKV[bid, KEY_DIM * 2 + h * Dv], v)

            pow_frag = T.alloc_fragment((Dk,), T.float32)
            pow_sum_frag = T.alloc_fragment((1,), T.float32)

            for i in T.Parallel(Dk):
                pow_frag[i] = T.cast(q_full[i], T.float32) * T.cast(q_full[i], T.float32)
            T.reduce_sum(pow_frag, pow_sum_frag, dim=-1)
            q_norm = T.max(T.sqrt(pow_sum_frag[0]), T.float32(1e-6))
            for i in T.Parallel(Dk):
                q_full[i] = T.cast((T.cast(q_full[i], T.float32) / q_norm) * INV_SQRT_DK, dtype)

            for i in T.Parallel(Dk):
                pow_frag[i] = T.cast(k_full[i], T.float32) * T.cast(k_full[i], T.float32)
            T.reduce_sum(pow_frag, pow_sum_frag, dim=-1)
            k_norm = T.max(T.sqrt(pow_sum_frag[0]), T.float32(1e-6))
            for i in T.Parallel(Dk):
                k_full[i] = T.cast(T.cast(k_full[i], T.float32) / k_norm, dtype)

            alpha_val = T.cast(Alpha[bid, h], T.float32)
            neg_a_exp_val = T.cast(NegAExp[h], T.float32)
            dt_bias_val = T.cast(DtBias[h], T.float32)
            sp_input = alpha_val + dt_bias_val
            sp = T.if_then_else(sp_input > 20.0, sp_input, T.log(1.0 + T.exp(sp_input)))
            gate = neg_a_exp_val * sp
            decay = T.exp(gate)

            beta = T.sigmoid(T.cast(BetaRaw[bid, h], T.float32))

            num_tiles = Dk // BLOCK_DK

            S_read = T.alloc_shared((BLOCK_DK, Dv), dtype)
            S_decayed = T.alloc_shared((BLOCK_DK, Dv), T.float32)
            k_tile = T.alloc_shared((BLOCK_DK,), dtype)

            partial_accum = T.alloc_shared((num_tiles, Dv), T.float32)

            for r_start in T.Pipelined(num_tiles, num_stages=3):
                T.copy(State[bid, h, r_start * BLOCK_DK, 0], S_read)
                T.copy(k_full[r_start * BLOCK_DK], k_tile)

                for i, j in T.Parallel(BLOCK_DK, Dv):
                    S_decayed[i, j] = T.cast(S_read[i, j], T.float32) * decay

                for i, j in T.Parallel(BLOCK_DK, Dv):
                    NewState[bid, h, r_start * BLOCK_DK + i, j] = T.cast(S_decayed[i, j], dtype)

                for j in T.Parallel(Dv):
                    dot_k = T.alloc_local([1], T.float32)
                    dot_k[0] = T.float32(0.0)
                    for i in T.serial(BLOCK_DK):
                        dot_k[0] = dot_k[0] + S_decayed[i, j] * T.cast(k_tile[i], T.float32)
                    partial_accum[r_start, j] = dot_k[0]

            accumulated = T.alloc_shared((Dv,), T.float32)
            T.clear(accumulated)
            for t in T.serial(num_tiles):
                for j in T.Parallel(Dv):
                    accumulated[j] = accumulated[j] + partial_accum[t, j]

            delta = T.alloc_shared((Dv,), T.float32)
            for j in T.Parallel(Dv):
                delta[j] = beta * (T.cast(v[j], T.float32) - accumulated[j])

            S_read2 = T.alloc_shared((BLOCK_DK, Dv), dtype)
            S_updated = T.alloc_shared((BLOCK_DK, Dv), T.float32)
            q_tile = T.alloc_shared((BLOCK_DK,), dtype)

            partial_output = T.alloc_shared((num_tiles, Dv), T.float32)

            for r_start in T.Pipelined(num_tiles, num_stages=3):
                T.copy(NewState[bid, h, r_start * BLOCK_DK, 0], S_read2)
                T.copy(k_full[r_start * BLOCK_DK], k_tile)
                T.copy(q_full[r_start * BLOCK_DK], q_tile)

                for i, j in T.Parallel(BLOCK_DK, Dv):
                    S_updated[i, j] = T.cast(S_read2[i, j], T.float32) + T.cast(k_tile[i], T.float32) * delta[j]

                for i, j in T.Parallel(BLOCK_DK, Dv):
                    NewState[bid, h, r_start * BLOCK_DK + i, j] = T.cast(S_updated[i, j], dtype)

                for j in T.Parallel(Dv):
                    dot_q = T.alloc_local([1], T.float32)
                    dot_q[0] = T.float32(0.0)
                    for i in T.serial(BLOCK_DK):
                        dot_q[0] = dot_q[0] + S_updated[i, j] * T.cast(q_tile[i], T.float32)
                    partial_output[r_start, j] = dot_q[0]

            output = T.alloc_shared((Dv,), T.float32)
            T.clear(output)
            for t in T.serial(num_tiles):
                for j in T.Parallel(Dv):
                    output[j] = output[j] + partial_output[t, j]

            if FUSE_NORM:
                out_sq = T.alloc_shared((Dv,), T.float32)
                for j in T.Parallel(Dv):
                    out_sq[j] = output[j] * output[j]
                var_buf = T.alloc_shared((1,), T.float32)
                T.reduce_sum(out_sq, var_buf, dim=-1)
                var_val = var_buf[0] / T.float32(Dv)
                rrms = T.rsqrt(var_val + T.float32(1e-6))

                norm_w = T.alloc_shared((Dv,), T.float32)
                T.copy(NormW[0], norm_w)

                o_normed = T.alloc_shared((Dv,), T.float32)
                for j in T.Parallel(Dv):
                    o_normed[j] = output[j] * rrms * norm_w[j]

                z_val = T.alloc_shared((Dv,), T.float32)
                T.copy(Z[bid, h, 0], z_val)

                final_output = T.alloc_shared((Dv,), dtype)
                for j in T.Parallel(Dv):
                    sig_z = T.sigmoid(z_val[j])
                    final_output[j] = T.cast(o_normed[j] * sig_z * z_val[j], dtype)

                T.copy(final_output, Output[bid, h, 0])
            else:
                final_output = T.alloc_shared((Dv,), dtype)
                for j in T.Parallel(Dv):
                    final_output[j] = T.cast(output[j], dtype)
                T.copy(final_output, Output[bid, h, 0])

    return kernel       

@tilelang.jit(out_idx=[-2, -1])
def fused_postproj_recurrent_kernel_gemm(B, num_k_heads, num_v_heads, Dk, Dv, BLOCK_DK, FUSE_NORM, dtype):
    V_PER_K = num_v_heads // num_k_heads
    INV_SQRT_DK = Dk ** -0.5
    KEY_DIM = num_k_heads * Dk
    CONV_DIM = KEY_DIM * 2 + num_v_heads * Dv

    @T.prim_func
    def kernel(
        QKV: T.Tensor((B, CONV_DIM), dtype),
        Alpha: T.Tensor((B, num_v_heads), dtype),
        BetaRaw: T.Tensor((B, num_v_heads), dtype),
        NegAExp: T.Tensor((num_v_heads,), dtype),
        DtBias: T.Tensor((num_v_heads,), dtype),
        State: T.Tensor((B, num_v_heads, Dk, Dv), dtype),
        NormW: T.Tensor((Dv,), dtype),
        Z: T.Tensor((B, num_v_heads, Dv), dtype),
        NewState: T.Tensor((B, num_v_heads, Dk, Dv), dtype),
        Output: T.Tensor((B, num_v_heads, Dv), dtype),
    ):
        with T.Kernel(num_v_heads, B, threads=THREADS) as (h, bid):
            k_head = h // V_PER_K

            q_full = T.alloc_shared((Dk,), dtype)
            k_full = T.alloc_shared((Dk,), dtype)
            v = T.alloc_shared((Dv,), dtype)

            T.copy(QKV[bid, k_head * Dk], q_full)
            T.copy(QKV[bid, KEY_DIM + k_head * Dk], k_full)
            T.copy(QKV[bid, KEY_DIM * 2 + h * Dv], v)

            pow_frag = T.alloc_fragment((Dk,), T.float32)
            pow_sum_frag = T.alloc_fragment((1,), T.float32)

            for i in T.Parallel(Dk):
                pow_frag[i] = T.cast(q_full[i], T.float32) * T.cast(q_full[i], T.float32)
            T.reduce_sum(pow_frag, pow_sum_frag, dim=-1)
            q_norm = T.max(T.sqrt(pow_sum_frag[0]), T.float32(1e-6))
            for i in T.Parallel(Dk):
                q_full[i] = T.cast((T.cast(q_full[i], T.float32) / q_norm) * INV_SQRT_DK, dtype)

            for i in T.Parallel(Dk):
                pow_frag[i] = T.cast(k_full[i], T.float32) * T.cast(k_full[i], T.float32)
            T.reduce_sum(pow_frag, pow_sum_frag, dim=-1)
            k_norm = T.max(T.sqrt(pow_sum_frag[0]), T.float32(1e-6))
            for i in T.Parallel(Dk):
                k_full[i] = T.cast(T.cast(k_full[i], T.float32) / k_norm, dtype)

            alpha_val = T.cast(Alpha[bid, h], T.float32)
            neg_a_exp_val = T.cast(NegAExp[h], T.float32)
            dt_bias_val = T.cast(DtBias[h], T.float32)
            sp_input = alpha_val + dt_bias_val
            sp = T.if_then_else(sp_input > 20.0, sp_input, T.log(1.0 + T.exp(sp_input)))
            gate = neg_a_exp_val * sp
            decay = T.exp(gate)

            beta = T.sigmoid(T.cast(BetaRaw[bid, h], T.float32))

            num_tiles = Dk // BLOCK_DK
            num_dv_tiles = Dv // DV_TILE

            S_read = T.alloc_shared((BLOCK_DK, Dv), dtype)
            S_decayed = T.alloc_shared((BLOCK_DK, Dv), T.float32)
            S_updated = T.alloc_shared((BLOCK_DK, Dv), T.float32)

            S_sub = T.alloc_shared((BLOCK_DK, DV_TILE), dtype)
            k_padded = T.alloc_shared((_GEMM_M, BLOCK_DK), dtype)
            q_padded = T.alloc_shared((_GEMM_M, BLOCK_DK), dtype)

            gemm_acc = T.alloc_fragment((_GEMM_M, DV_TILE), T.float32)
            gemm_res = T.alloc_shared((_GEMM_M, DV_TILE), T.float32)

            partial_accum = T.alloc_shared((num_tiles, Dv), T.float32)
            partial_output = T.alloc_shared((num_tiles, Dv), T.float32)

            for r_start in T.Pipelined(num_tiles, num_stages=2):
                T.copy(State[bid, h, r_start * BLOCK_DK, 0], S_read)

                for i, j in T.Parallel(BLOCK_DK, Dv):
                    S_decayed[i, j] = T.cast(S_read[i, j], T.float32) * decay

                for i, j in T.Parallel(BLOCK_DK, Dv):
                    NewState[bid, h, r_start * BLOCK_DK + i, j] = T.cast(S_decayed[i, j], dtype)

                for i in T.Parallel(BLOCK_DK):
                    k_padded[0, i] = k_full[r_start * BLOCK_DK + i]
                for i, j in T.Parallel(_GEMM_M - 1, BLOCK_DK):
                    k_padded[i + 1, j] = T.cast(T.float32(0.0), dtype)

                for dv_start in T.serial(num_dv_tiles):
                    for i, j in T.Parallel(BLOCK_DK, DV_TILE):
                        S_sub[i, j] = T.cast(S_decayed[i, dv_start * DV_TILE + j], dtype)

                    T.gemm(k_padded, S_sub, gemm_acc, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(gemm_acc, gemm_res)

                    for j in T.Parallel(DV_TILE):
                        partial_accum[r_start, dv_start * DV_TILE + j] = gemm_res[0, j]

            accumulated = T.alloc_shared((Dv,), T.float32)
            T.clear(accumulated)
            for t in T.serial(num_tiles):
                for j in T.Parallel(Dv):
                    accumulated[j] = accumulated[j] + partial_accum[t, j]

            delta = T.alloc_shared((Dv,), T.float32)
            for j in T.Parallel(Dv):
                delta[j] = beta * (T.cast(v[j], T.float32) - accumulated[j])

            for r_start in T.Pipelined(num_tiles, num_stages=3):
                T.copy(NewState[bid, h, r_start * BLOCK_DK, 0], S_read)

                for i, j in T.Parallel(BLOCK_DK, Dv):
                    S_updated[i, j] = T.cast(S_read[i, j], T.float32) + T.cast(k_full[r_start * BLOCK_DK + i], T.float32) * delta[j]

                for i, j in T.Parallel(BLOCK_DK, Dv):
                    NewState[bid, h, r_start * BLOCK_DK + i, j] = T.cast(S_updated[i, j], dtype)

                for i in T.Parallel(BLOCK_DK):
                    q_padded[0, i] = q_full[r_start * BLOCK_DK + i]
                for i, j in T.Parallel(_GEMM_M - 1, BLOCK_DK):
                    q_padded[i + 1, j] = T.cast(T.float32(0.0), dtype)

                for dv_start in T.serial(num_dv_tiles):
                    for i, j in T.Parallel(BLOCK_DK, DV_TILE):
                        S_sub[i, j] = T.cast(S_updated[i, dv_start * DV_TILE + j], dtype)

                    T.gemm(q_padded, S_sub, gemm_acc, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(gemm_acc, gemm_res)

                    for j in T.Parallel(DV_TILE):
                        partial_output[r_start, dv_start * DV_TILE + j] = gemm_res[0, j]

            output = T.alloc_shared((Dv,), T.float32)
            T.clear(output)
            for t in T.serial(num_tiles):
                for j in T.Parallel(Dv):
                    output[j] = output[j] + partial_output[t, j]

            if FUSE_NORM:
                out_sq = T.alloc_shared((Dv,), T.float32)
                for j in T.Parallel(Dv):
                    out_sq[j] = output[j] * output[j]
                var_buf = T.alloc_shared((1,), T.float32)
                T.reduce_sum(out_sq, var_buf, dim=-1)
                var_val = var_buf[0] / T.float32(Dv)
                rrms = T.rsqrt(var_val + T.float32(1e-6))

                norm_w = T.alloc_shared((Dv,), T.float32)
                T.copy(NormW[0], norm_w)

                o_normed = T.alloc_shared((Dv,), T.float32)
                for j in T.Parallel(Dv):
                    o_normed[j] = output[j] * rrms * norm_w[j]

                z_val = T.alloc_shared((Dv,), T.float32)
                T.copy(Z[bid, h, 0], z_val)

                final_output = T.alloc_shared((Dv,), dtype)
                for j in T.Parallel(Dv):
                    sig_z = T.sigmoid(z_val[j])
                    final_output[j] = T.cast(o_normed[j] * sig_z * z_val[j], dtype)

                T.copy(final_output, Output[bid, h, 0])
            else:
                final_output = T.alloc_shared((Dv,), dtype)
                for j in T.Parallel(Dv):
                    final_output[j] = T.cast(output[j], dtype)
                T.copy(final_output, Output[bid, h, 0])

    return kernel

def dispatch(device):
    props = torch.cuda.get_device_properties(device)
    sm_version = props.major * 10 + props.minor
    if sm_version >= 90:
        return 'gemm'
    return 'manual'

def fused_postproj_recurrent(
    qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias_f, state,
    num_k_heads=16, num_v_heads=48, head_dim=128,
    norm_weight=None, z=None, norm_eps=1e-6, BLOCK_DK=16, dtype=T.bfloat16
):
    B = qkv_conv.shape[0]
    Dk = Dv = head_dim
    assert Dk % BLOCK_DK == 0, f"Dk={Dk} must be divisible by BLOCK_DK={BLOCK_DK}"
    assert Dv % DV_TILE == 0, f"Dv={Dv} must be divisible by DV_TILE={DV_TILE}"
    fuse_norm = (norm_weight is not None) and (z is not None)

    if not fuse_norm:
        norm_weight = torch.empty(Dv, device=qkv_conv.device, dtype=torch.bfloat16)
        z = torch.empty(B, num_v_heads, Dv, device=qkv_conv.device, dtype=torch.bfloat16)

    style = dispatch(qkv_conv.device)
    if style == 'gemm':
        kernel_func = fused_postproj_recurrent_kernel_gemm
    else:
        kernel_func = fused_postproj_recurrent_kernel_manual

    kernel = kernel_func(B, num_k_heads, num_v_heads, Dk, Dv, BLOCK_DK, fuse_norm, dtype)

    new_state, out = kernel(qkv_conv.contiguous(), alpha.contiguous(), beta_raw.contiguous(),
                            neg_A_exp, dt_bias_f, state.contiguous(),
                            norm_weight, z.contiguous())
    return out, new_state

def ref_fused_postproj_recurrent(
    qkv_conv: torch.Tensor,
    alpha: torch.Tensor,
    beta_raw: torch.Tensor,
    neg_A_exp: torch.Tensor,
    dt_bias_f: torch.Tensor,
    state: torch.Tensor,
    num_k_heads=16, num_v_heads=48, head_dim=128,
    norm_weight: torch.Tensor = None,
    z: torch.Tensor = None,
    norm_eps: float = 1e-6,
) -> tuple:
    B = qkv_conv.shape[0]
    H = num_v_heads
    Dk = Dv = head_dim
    V_PER_K = num_v_heads // num_k_heads
    KEY_DIM = num_k_heads * Dk
    INV_SQRT_DK = Dk ** -0.5

    q_raw = qkv_conv[:, :KEY_DIM].view(B, num_k_heads, Dk)
    k_raw = qkv_conv[:, KEY_DIM:KEY_DIM*2].view(B, num_k_heads, Dk)
    v = qkv_conv[:, KEY_DIM*2:].view(B, num_v_heads, Dv)

    q_raw_exp = q_raw.repeat_interleave(V_PER_K, dim=1)
    k_raw_exp = k_raw.repeat_interleave(V_PER_K, dim=1)

    q_raw_f = q_raw_exp.float()
    k_raw_f = k_raw_exp.float()
    v_f = v.float()

    q_norm = torch.sqrt(torch.sum(q_raw_f * q_raw_f, dim=-1, keepdim=True)).clamp(min=1e-6)
    q = (q_raw_f / q_norm) * INV_SQRT_DK

    k_norm = torch.sqrt(torch.sum(k_raw_f * k_raw_f, dim=-1, keepdim=True)).clamp(min=1e-6)
    k = k_raw_f / k_norm

    alpha_f = alpha.float()
    neg_a_exp_f = neg_A_exp.unsqueeze(0).expand(B, -1)
    dt_bias_f_exp = dt_bias_f.unsqueeze(0).expand(B, -1)

    sp_input = alpha_f + dt_bias_f_exp
    sp = torch.where(sp_input > 20.0, sp_input, torch.log(1.0 + torch.exp(sp_input)))
    gate = neg_a_exp_f * sp
    decay = torch.exp(gate).unsqueeze(-1)

    beta = torch.sigmoid(beta_raw.float()).unsqueeze(-1)

    state_f = state.float()

    k_scaled = k * decay
    q_scaled = q * decay

    accumulated_Sk = torch.einsum('bhdv,bhd->bhv', state_f, k_scaled)
    accumulated_Sq = torch.einsum('bhdv,bhd->bhv', state_f, q_scaled)

    qk_dot_val = torch.sum(q * k, dim=-1, keepdim=True)
    delta = beta * (v_f - accumulated_Sk)
    output = accumulated_Sq + qk_dot_val * delta

    new_state = state_f * decay.unsqueeze(-1) + torch.einsum('bhd,bhv->bhdv', k, delta)

    if norm_weight is not None and z is not None:
        var = torch.mean(output * output, dim=-1, keepdim=True)
        rrms = torch.rsqrt(var + norm_eps)
        norm_w = norm_weight.float().unsqueeze(0).unsqueeze(0)
        o_normed = output * rrms * norm_w

        z_val = z.float()
        z_silu = z_val * torch.sigmoid(z_val)
        output = o_normed * z_silu

    return output.to(torch.bfloat16), new_state.to(torch.bfloat16)

def test_fused_postproj_recurrent():
    B = 4
    num_k_heads = 16
    num_v_heads = 48
    head_dim = 128
    KEY_DIM = num_k_heads * head_dim
    CONV_DIM = KEY_DIM * 2 + num_v_heads * head_dim
    Dk = Dv = head_dim
    norm_eps = 1e-6
    dtype = torch.bfloat16
    device = "cuda"
    BLOCK_DK = 16

    qkv_conv = torch.randn(B, CONV_DIM, dtype=dtype, device=device)
    alpha = torch.randn(B, num_v_heads, dtype=dtype, device=device)
    beta_raw = torch.randn(B, num_v_heads, dtype=dtype, device=device)
    neg_A_exp = -torch.exp(torch.randn(num_v_heads, dtype=dtype, device=device))
    dt_bias_f = torch.randn(num_v_heads, dtype=dtype, device=device)
    state = torch.randn(B, num_v_heads, Dk, Dv, dtype=dtype, device=device)

    norm_weight = torch.randn(Dv, dtype=dtype, device=device)
    z = torch.randn(B, num_v_heads, Dv, dtype=dtype, device=device)

    out_ref, new_state_ref = ref_fused_postproj_recurrent(
        qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias_f,
        state.clone(), num_k_heads, num_v_heads, head_dim,
        norm_weight, z, norm_eps
    )

    out_fused, new_state_fused = fused_postproj_recurrent(
        qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias_f,
        state.clone(), num_k_heads, num_v_heads, head_dim,
        norm_weight, z, norm_eps, BLOCK_DK=BLOCK_DK
    )

    torch.testing.assert_close(out_fused, out_ref, rtol=1e-1, atol=1e-2)
    torch.testing.assert_close(new_state_fused, new_state_ref, rtol=1e-1, atol=1e-2)
    print("✅ 前向验证通过！")

    qkv_perf = qkv_conv.detach()
    alpha_perf = alpha.detach()
    beta_perf = beta_raw.detach()
    negA_perf = neg_A_exp.detach()
    dtb_perf = dt_bias_f.detach()
    normw_perf = norm_weight.detach()
    z_perf = z.detach()

    ms_fused = do_bench(
        lambda: fused_postproj_recurrent(
            qkv_perf, alpha_perf, beta_perf, negA_perf, dtb_perf,
            state.clone(), num_k_heads, num_v_heads, head_dim,
            normw_perf, z_perf, norm_eps, BLOCK_DK=BLOCK_DK
        )
    )
    ms_ref = do_bench(
        lambda: ref_fused_postproj_recurrent(
            qkv_perf, alpha_perf, beta_perf, negA_perf, dtb_perf,
            state.clone(), num_k_heads, num_v_heads, head_dim,
            normw_perf, z_perf, norm_eps
        )
    )

    print(f"TileLang fused : {ms_fused:.4f} ms")
    print(f"PyTorch ref    : {ms_ref:.4f} ms")
    print(f"🚀 Speedup     : {ms_ref / ms_fused:.2f}x")

if __name__ == "__main__":
    test_fused_postproj_recurrent()