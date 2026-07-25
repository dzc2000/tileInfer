"""
TileLang Gated Delta Rule Prefill Kernel (sm80+).

Chunked implementation of the gated delta rule recurrence for DeltaNet prefill.
Replaces FlashQLA (sm90-only) on A100 and other pre-Hopper GPUs.

Algorithm per chunk (CHUNK tokens):
  1. O_state[i] = exp(G_i) * Q[i] @ S          (state contribution)
  2. A[i,j] = exp(G_i - G_j) * beta[j] * (Q[i] · K[j])   (intra-chunk causal attention with decay)
  3. O[i] = O_state[i] + A[i,:] @ V[:]          (total output)
  4. S = exp(G_chunk) * S + K^T @ diag(w) @ V   (state update, w[j] = exp(G_chunk-G_j)*beta[j])

Shared memory budget (CHUNK=32, Dk=Dv=128):
  S[128,128] f32 = 64KB, Q/K[32,128] bf16 = 8KB each, V[32,128] bf16 = 8KB,
  A[32,32] f32 = 4KB, O[32,128] f32 = 16KB, misc ~1KB  →  ~109KB  (A100: 163KB)
"""
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

CHUNK = 32
THREADS = 256

_kernel_cache: dict[tuple, object] = {}


@tilelang.jit(out_idx=[-2, -1])
def _gated_delta_rule_prefill_kernel(
    B, seq_len, num_k_heads, num_v_heads, Dk, Dv, dtype,
):
    V_PER_K = num_v_heads // num_k_heads

    @T.prim_func
    def kernel(
        Q: T.Tensor((B, seq_len, num_k_heads, Dk), dtype),
        K: T.Tensor((B, seq_len, num_k_heads, Dk), dtype),
        V: T.Tensor((B, seq_len, num_v_heads, Dv), dtype),
        G: T.Tensor((B, seq_len, num_v_heads), "float32"),
        Beta: T.Tensor((B, seq_len, num_v_heads), "float32"),
        InitState: T.Tensor((B, num_v_heads, Dk, Dv), "float32"),
        Output: T.Tensor((B, seq_len, num_v_heads, Dv), dtype),
        FinalState: T.Tensor((B, num_v_heads, Dk, Dv), "float32"),
    ):
        with T.Kernel(num_v_heads, B, threads=THREADS) as (h, bid):
            k_head = h // V_PER_K

            S = T.alloc_shared((Dk, Dv), "float32")

            for d, e in T.Parallel(Dk, Dv):
                S[d, e] = InitState[bid, h, d, e]

            Q_buf = T.alloc_shared((CHUNK, Dk), dtype)
            K_buf = T.alloc_shared((CHUNK, Dk), dtype)
            V_buf = T.alloc_shared((CHUNK, Dv), dtype)
            A = T.alloc_shared((CHUNK, CHUNK), "float32")
            O_buf = T.alloc_shared((CHUNK, Dv), "float32")
            cum_g = T.alloc_shared((CHUNK,), "float32")
            beta_buf = T.alloc_shared((CHUNK,), "float32")

            num_chunks = seq_len // CHUNK

            for chunk_idx in T.serial(num_chunks):
                cs = chunk_idx * CHUNK

                for i, d in T.Parallel(CHUNK, Dk):
                    Q_buf[i, d] = Q[bid, cs + i, k_head, d]
                    K_buf[i, d] = K[bid, cs + i, k_head, d]
                for i, e in T.Parallel(CHUNK, Dv):
                    V_buf[i, e] = V[bid, cs + i, h, e]
                for i in T.Parallel(CHUNK):
                    beta_buf[i] = Beta[bid, cs + i, h]

                cum_g[0] = G[bid, cs, h]
                for i in T.serial(CHUNK - 1):
                    cum_g[i + 1] = cum_g[i] + G[bid, cs + i + 1, h]

                for i, e in T.Parallel(CHUNK, Dv):
                    acc = T.alloc_local([1], "float32")
                    acc[0] = T.float32(0.0)
                    for d in T.serial(Dk):
                        acc[0] = acc[0] + T.cast(Q_buf[i, d], "float32") * S[d, e]
                    O_buf[i, e] = T.exp(cum_g[i]) * acc[0]

                for i, j in T.Parallel(CHUNK, CHUNK):
                    acc = T.alloc_local([1], "float32")
                    acc[0] = T.float32(0.0)
                    for d in T.serial(Dk):
                        acc[0] = acc[0] + T.cast(Q_buf[i, d], "float32") * T.cast(K_buf[j, d], "float32")
                    A[i, j] = T.if_then_else(
                        j <= i,
                        T.exp(cum_g[i] - cum_g[j]) * beta_buf[j] * acc[0],
                        T.float32(0.0),
                    )

                for i, e in T.Parallel(CHUNK, Dv):
                    acc = T.alloc_local([1], "float32")
                    acc[0] = T.float32(0.0)
                    for j in T.serial(CHUNK):
                        acc[0] = acc[0] + A[i, j] * T.cast(V_buf[j, e], "float32")
                    O_buf[i, e] = O_buf[i, e] + acc[0]

                for i, e in T.Parallel(CHUNK, Dv):
                    Output[bid, cs + i, h, e] = T.cast(O_buf[i, e], dtype)

                chunk_g = cum_g[CHUNK - 1]
                chunk_decay = T.exp(chunk_g)

                for d, e in T.Parallel(Dk, Dv):
                    acc = T.alloc_local([1], "float32")
                    acc[0] = T.float32(0.0)
                    for j in T.serial(CHUNK):
                        w = T.exp(chunk_g - cum_g[j]) * beta_buf[j]
                        acc[0] = acc[0] + T.cast(K_buf[j, d], "float32") * w * T.cast(V_buf[j, e], "float32")
                    S[d, e] = S[d, e] * chunk_decay + acc[0]

            for d, e in T.Parallel(Dk, Dv):
                FinalState[bid, h, d, e] = S[d, e]

    return kernel


def _get_kernel(B, seq_len, num_k_heads, num_v_heads, Dk, Dv, dtype):
    dk = "bfloat16" if dtype == torch.bfloat16 else "float16"
    key = (B, seq_len, num_k_heads, num_v_heads, Dk, Dv, dk)
    if key not in _kernel_cache:
        _kernel_cache[key] = _gated_delta_rule_prefill_kernel(
            B, seq_len, num_k_heads, num_v_heads, Dk, Dv, dk,
        )
    return _kernel_cache[key]


def chunk_gated_delta_rule_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in replacement for FlashQLA chunk_gated_delta_rule on sm80+.

    Args:
        q: [B, T, num_k_heads, Dk]
        k: [B, T, num_k_heads, Dk]
        v: [B, T, num_v_heads, Dv]
        g: [B, T, num_v_heads]  (log-space decay, float32)
        beta: [B, T, num_v_heads]  (float32)
        scale: float  (applied to output)
        initial_state: [B, num_v_heads, Dk, Dv] or None
        output_final_state: bool

    Returns:
        output: [B, T, num_v_heads, Dv]
        final_state: [B, num_v_heads, Dk, Dv] or None
    """
    B, T_len, num_k_heads, Dk = q.shape
    _, _, num_v_heads, Dv = v.shape

    pad_len = (CHUNK - T_len % CHUNK) % CHUNK
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    T_padded = T_len + pad_len

    if initial_state is None:
        init_state = torch.zeros(B, num_v_heads, Dk, Dv, device=q.device, dtype=torch.float32)
    else:
        init_state = initial_state.float().contiguous()

    kernel = _get_kernel(B, T_padded, num_k_heads, num_v_heads, Dk, Dv, q.dtype)

    output, final_state = kernel(
        q.contiguous(), k.contiguous(), v.contiguous(),
        g.float().contiguous(), beta.float().contiguous(),
        init_state,
    )

    output = output[:, :T_len] * scale

    if output_final_state:
        return output, final_state
    return output, None
