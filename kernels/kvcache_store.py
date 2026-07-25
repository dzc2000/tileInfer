"""
TileLang kernel for paged KV cache store operations.

store_kvcache: writes K/V tensors into paged KV cache using slot_mapping
  - slot_mapping[i] = physical slot index in the flat KV cache
  - slot_mapping[i] = -1 means skip (padding)
"""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit
def _store_kvcache_tilelang(N, num_slots, D, block_N, dtype=T.bfloat16):

    @T.prim_func
    def main(
        Key: T.Tensor((N, D), dtype),
        Value: T.Tensor((N, D), dtype),
        Slot_Mapping: T.Tensor((N,), T.int32),
        K_Cache: T.Tensor((num_slots, D), dtype),
        V_Cache: T.Tensor((num_slots, D), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
            K_shared = T.alloc_shared((block_N, D), dtype)
            V_shared = T.alloc_shared((block_N, D), dtype)
            slot_shared = T.alloc_shared((block_N,), T.int32)

            T.copy(Key[bx * block_N : (bx + 1) * block_N, :], K_shared)
            T.copy(Value[bx * block_N : (bx + 1) * block_N, :], V_shared)
            T.copy(Slot_Mapping[bx * block_N : (bx + 1) * block_N], slot_shared)

            for i, j in T.Parallel(block_N, D):
                slot = slot_shared[i]
                if slot >= 0:
                    K_Cache[slot, j] = K_shared[i, j]
                    V_Cache[slot, j] = V_shared[i, j]

    return main


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Store K/V into paged KV cache (in-place scatter write).

    Args:
        key: [N, num_heads, head_dim]
        value: [N, num_heads, head_dim]
        k_cache: [num_slots, num_heads * head_dim] flat, contiguous (updated in-place)
        v_cache: [num_slots, num_heads * head_dim] flat, contiguous (updated in-place)
        slot_mapping: [N] int32, -1 means skip
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    key_flat = key.reshape(N, D).contiguous()
    value_flat = value.reshape(N, D).contiguous()

    num_slots = k_cache.shape[0]

    block_N = 64 if N >= 64 else (32 if N >= 32 else 16)

    tl_dtype = T.bfloat16 if key.dtype == torch.bfloat16 else T.float16
    kernel = _store_kvcache_tilelang(N, num_slots, D, block_N, dtype=tl_dtype)
    kernel(key_flat, value_flat, slot_mapping, k_cache, v_cache)


def _ref_store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """Reference implementation for correctness check."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    key_flat = key.reshape(N, D)
    value_flat = value.reshape(N, D)
    for i in range(N):
        slot = slot_mapping[i].item()
        if slot >= 0:
            k_cache[slot] = key_flat[i]
            v_cache[slot] = value_flat[i]


if __name__ == "__main__":
    N = 8
    num_heads = 4
    head_dim = 128
    num_slots = 1024
    D = num_heads * head_dim
    block_N = 32

    torch.manual_seed(42)
    key = torch.randn(N, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn(N, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    slot_mapping = torch.randint(0, num_slots, (N,), dtype=torch.int32, device="cuda")
    slot_mapping[::10] = -1

    key_flat = key.reshape(N, D)
    value_flat = value.reshape(N, D)

    kernel = _store_kvcache_tilelang(N, num_slots, D, block_N, dtype=T.bfloat16)
    k_cache_out = torch.zeros(num_slots, D, dtype=torch.bfloat16, device="cuda")
    v_cache_out = torch.zeros(num_slots, D, dtype=torch.bfloat16, device="cuda")
    kernel(key_flat, value_flat, slot_mapping, k_cache_out, v_cache_out)

    k_cache_ref = torch.zeros(num_slots, D, dtype=torch.bfloat16, device="cuda")
    v_cache_ref = torch.zeros(num_slots, D, dtype=torch.bfloat16, device="cuda")
    _ref_store_kvcache(key, value, k_cache_ref, v_cache_ref, slot_mapping)

    torch.testing.assert_close(k_cache_out, k_cache_ref, rtol=0.01, atol=0.01)
    torch.testing.assert_close(v_cache_out, v_cache_ref, rtol=0.01, atol=0.01)
    print("store_kvcache correctness check passed!")

    def run_tilelang():
        k = torch.zeros(num_slots, D, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(num_slots, D, dtype=torch.bfloat16, device="cuda")
        kernel(key_flat, value_flat, slot_mapping, k, v)

    ms_tl = do_bench(run_tilelang, warmup=100)
    print(f"Tilelang store_kvcache: {ms_tl:.3f} ms")
