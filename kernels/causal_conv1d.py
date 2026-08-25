import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

@tilelang.jit(out_idx=[-1])
def causal_conv1d_update(B, D, K, bias=False, apply_silu=False, channels_per_block=128, dtype=T.bfloat16):

    @T.prim_func
    def main(
        X: T.Tensor((B, D), dtype),
        S: T.Tensor((B, D, K), dtype),
        W: T.Tensor((D, K), dtype),
        Bias: T.Tensor((D,), dtype),
        Y: T.Tensor((B, D), dtype),
    ):
        HAS_BIAS = 1 if bias else 0
        APPLY_SILU = 1 if apply_silu else 0
        CH_PER_BLOCK = channels_per_block
        KERNEL_SIZE = K

        grid_channels = T.ceildiv(D, CH_PER_BLOCK)

        with T.Kernel(B, grid_channels, threads=CH_PER_BLOCK) as (batch_id, block_id):
            ch_start = block_id * CH_PER_BLOCK
            ch_end = (block_id + 1) * CH_PER_BLOCK
            x_local = T.alloc_fragment((CH_PER_BLOCK, ), T.float)
            T.copy(X[batch_id, ch_start:ch_end], x_local)

            acc = T.alloc_fragment((CH_PER_BLOCK, ), T.float)
            T.clear(acc)
            for i in T.serial(KERNEL_SIZE):
                s_local = T.alloc_fragment((CH_PER_BLOCK,), dtype)
                if i < KERNEL_SIZE - 1:
                    T.copy(S[batch_id, ch_start:ch_end, i + 1], s_local)
                else:
                    T.copy(x_local, s_local)

                T.copy(s_local, S[batch_id, ch_start:ch_end, i])

                w_local = T.alloc_fragment((CH_PER_BLOCK, ), T.float)
                T.copy(W[ch_start:ch_end, i], w_local)

                for j in T.Parallel(CH_PER_BLOCK):
                    acc[j] += s_local[j] * w_local[j]

            if HAS_BIAS:
                for j in T.Parallel(CH_PER_BLOCK):
                    acc[j] += Bias[ch_start + j]

            if APPLY_SILU:
                for j in T.Parallel(CH_PER_BLOCK):
                    acc[j] *= T.sigmoid(acc[j])

            T.copy(acc, Y[batch_id, ch_start:ch_end])

    return main


@tilelang.jit(out_idx=[-1])
def causal_conv1d_prefill_kernel(B, SeqLen, D, K, bias=False, apply_silu=False,
                                  channels_per_block=128, dtype=T.bfloat16):

    @T.prim_func
    def main(
        X: T.Tensor((B, SeqLen, D), dtype),
        W: T.Tensor((D, K), dtype),
        Bias: T.Tensor((D,), dtype),
        Y: T.Tensor((B, SeqLen, D), dtype),
    ):
        HAS_BIAS = 1 if bias else 0
        APPLY_SILU = 1 if apply_silu else 0
        CH_PER_BLOCK = channels_per_block
        KERNEL_SIZE = K

        grid_channels = T.ceildiv(D, CH_PER_BLOCK)

        with T.Kernel(B, SeqLen, grid_channels, threads=CH_PER_BLOCK) as (batch_id, t_idx, block_id):
            ch_start = block_id * CH_PER_BLOCK
            ch_end = (block_id + 1) * CH_PER_BLOCK

            acc = T.alloc_fragment((CH_PER_BLOCK,), T.float32)
            T.clear(acc)

            for k in T.serial(KERNEL_SIZE):
                x_val = T.alloc_fragment((CH_PER_BLOCK,), dtype)
                w_val = T.alloc_fragment((CH_PER_BLOCK,), dtype)

                t_read = t_idx - k
                t_safe = T.max(t_read, 0)
                is_valid = t_read >= 0

                T.copy(X[batch_id, t_safe, ch_start:ch_end], x_val)

                T.copy(W[ch_start:ch_end, KERNEL_SIZE - 1 - k], w_val)

                for j in T.Parallel(CH_PER_BLOCK):
                    x_f = T.if_then_else(
                        is_valid,
                        T.cast(x_val[j], T.float32),
                        T.float32(0.0)
                    )
                    acc[j] += x_f * T.cast(w_val[j], T.float32)

            if HAS_BIAS:
                for j in T.Parallel(CH_PER_BLOCK):
                    acc[j] += T.cast(Bias[ch_start + j], T.float32)

            if APPLY_SILU:
                for j in T.Parallel(CH_PER_BLOCK):
                    acc[j] *= T.sigmoid(acc[j])

            T.copy(acc, Y[batch_id, t_idx, ch_start:ch_end])

    return main


def causal_conv1d_update_fn(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor = None,
    apply_silu: bool = True,
) -> torch.Tensor:
    """High-level wrapper matching qwen_36.py interface.

    Args:
        x: [B, D] input
        conv_state: [B, D, K] conv state (updated in-place)
        conv_w: [D, K] conv weights
        conv_b: [D] optional bias
        apply_silu: whether to apply SiLU activation

    Returns:
        y: [B, D] conv output
    """
    B, D = x.shape
    K = conv_w.shape[1]
    has_bias = conv_b is not None
    tl_dtype = T.bfloat16 if x.dtype == torch.bfloat16 else T.float16

    if not has_bias:
        conv_b = torch.zeros(D, device=x.device, dtype=x.dtype)

    kernel = causal_conv1d_update(
        B, D, K, bias=has_bias, apply_silu=apply_silu,
        channels_per_block=128, dtype=tl_dtype,
    )
    y = kernel(x, conv_state, conv_w, conv_b)
    return y


def causal_conv1d_prefill(
    x: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor = None,
    apply_silu: bool = True,
) -> torch.Tensor:
    """Parallel causal conv1d for prefill (processes entire sequence at once).

    Args:
        x: [B, T, D] input sequence
        conv_w: [D, K] conv weights (depthwise)
        conv_b: [D] optional bias
        apply_silu: whether to apply SiLU activation

    Returns:
        y: [B, T, D] conv output
    """
    B, SeqLen, D = x.shape
    K = conv_w.shape[1]
    has_bias = conv_b is not None
    tl_dtype = T.bfloat16 if x.dtype == torch.bfloat16 else T.float16

    if not has_bias:
        conv_b = torch.zeros(D, device=x.device, dtype=x.dtype)

    kernel = causal_conv1d_prefill_kernel(
        B, SeqLen, D, K, bias=has_bias, apply_silu=apply_silu,
        channels_per_block=128, dtype=tl_dtype,
    )
    y = kernel(x, conv_w, conv_b)
    return y


def ref_causal_conv1d_update(
    X: torch.Tensor,
    S: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor = None,
    apply_silu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_S = torch.cat([S[:, :, 1:], X.unsqueeze(-1)], dim=2)
    y = (new_S.float() * W.float().unsqueeze(0)).sum(dim=-1)
    if bias is not None:
        y = y + bias.float()
    if apply_silu:
        y = y * torch.sigmoid(y)
    return y.to(X.dtype)


def ref_causal_conv1d_prefill(
    X: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor = None,
    apply_silu: bool = True,
) -> torch.Tensor:
    """Reference implementation for causal conv1d prefill."""
    B, SeqLen, D = X.shape
    K = W.shape[1]
    Y = torch.zeros(B, SeqLen, D, dtype=X.dtype, device=X.device)

    for t in range(SeqLen):
        for k in range(min(K, t + 1)):
            Y[:, t, :] += W[:, K - 1 - k] * X[:, t - k, :]

    Y = Y.float()
    if bias is not None:
        Y = Y + bias.float()
    if apply_silu:
        Y = Y * torch.sigmoid(Y)
    return Y.to(X.dtype)


if __name__ == "__main__":
    B, SeqLen, D, K = 2, 32, 10240, 4
    channels_per_block = 128
    bias = True
    apply_silu = True
    dtype = torch.bfloat16
    device = "cuda"

    X = torch.randn(B, SeqLen, D, dtype=dtype, device=device)
    W = torch.randn(D, K, dtype=dtype, device=device)
    Bias = torch.randn(D, dtype=dtype, device=device) if bias else None

    Y_ref = ref_causal_conv1d_prefill(X, W, Bias, apply_silu)

    kernel = causal_conv1d_prefill_kernel(
        B, SeqLen, D, K, bias=bias, apply_silu=apply_silu,
        channels_per_block=channels_per_block, dtype=T.bfloat16
    )
    Y_out = kernel(X, W, Bias)

    torch.testing.assert_close(Y_out, Y_ref, rtol=1e-2, atol=1e-2)
    print("✓ causal_conv1d_prefill 正确性验证通过")

    X_perf = torch.randn(B, SeqLen, D, dtype=dtype, device=device)
    W_perf = torch.randn(D, K, dtype=dtype, device=device)
    Bias_perf = torch.randn(D, dtype=dtype, device=device) if bias else None

    ms_kernel = do_bench(
        lambda: causal_conv1d_prefill(X_perf, W_perf, Bias_perf, apply_silu),
        backend="event"
    )
    ms_ref = do_bench(
        lambda: ref_causal_conv1d_prefill(X_perf, W_perf, Bias_perf, apply_silu),
        backend="event"
    )
    print(f"TileLang kernel: {ms_kernel:.4f} ms")
    print(f"Reference: {ms_ref:.4f} ms")
    print(f"Speedup: {ms_ref / ms_kernel:.2f}x")