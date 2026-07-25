# ruff: noqa
"""Benchmark + correctness for gqa_varlen PIPE branch (A100 / sm_80)."""
import math
import torch
import tilelang
from tilelang.profiler import do_bench
from varlen_utils import generate_random_padding_mask, generate_qkv
from gqa_varlen import flashattn, flashattn_pipe, is_sm90_plus, get_cuda_compute_capability

B = 8
H_q = 32
H_kv = 8
D = 128
dtype = torch.bfloat16
device = "cuda"
is_causal = True
groups = H_q // H_kv

cc = get_cuda_compute_capability()
print(f"GPU cc: {cc}  sm90+: {is_sm90_plus()}  branch={'WS' if is_sm90_plus() else 'PIPE'}")

block_M = 128
block_N = 64
PEAK_TFLOPS = 312.0  # A100 bf16

CONFIGS = [
    (2, 128, 128, 64, "bM=128 (orig)"),
    (2, 128, 64, 64, "bM=64 (opt)"),
]


def make_data(sl, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, sl, H_q, D, dtype=dtype, device=device, generator=g)
    k = torch.randn(B, sl, H_kv, D, dtype=dtype, device=device, generator=g)
    v = torch.randn(B, sl, H_kv, D, dtype=dtype, device=device, generator=g)
    # share mask between q and k => self-attention, offset=0, causal diagonal aligned
    qm = generate_random_padding_mask(sl, B, device, mode="random")
    km = qm
    (q_unpad, k_unpad, v_unpad, cu_q, cu_k, max_q, max_k, _, _, _, opad, _, _) = generate_qkv(
        q, k, v, qm, km, kvpacked=False
    )
    UQ = q_unpad.shape[0]
    UKV = k_unpad.shape[0]
    UQp = math.ceil(UQ / block_M) * block_M
    UKVp = math.ceil(UKV / block_N) * block_N
    qp = torch.zeros(UQp, H_q, D, dtype=dtype, device=device); qp[:UQ] = q_unpad
    kp = torch.zeros(UKVp, H_kv, D, dtype=dtype, device=device); kp[:UKV] = k_unpad
    vp = torch.zeros(UKVp, H_kv, D, dtype=dtype, device=device); vp[:UKV] = v_unpad
    avg_q = qm.float().mean().item() * sl
    avg_k = km.float().mean().item() * sl
    flops = float(4 * B * H_q * avg_q * avg_k * D)
    if is_causal:
        flops *= 0.5
    return dict(qp=qp, kp=kp, vp=vp, cu_q=cu_q, cu_k=cu_k, max_q=max_q,
                UQ=UQ, UKV=UKV, opad=opad, q=q, k=k, v=v, qm=qm, km=km,
                flops=flops, avg_q=avg_q, avg_k=avg_k)


def sdpa_ref(q, k, v, qm, km, is_causal):
    """torch SDPA reference (padded). Expand KV heads for GQA.
    Build full additive mask manually (causal + padding) to avoid
    SDPA's is_causal+attn_mask interaction issues."""
    B_, Sq, Hq, D_ = q.shape
    _, Sk, Hkv, _ = k.shape
    g_ = Hq // Hkv
    qh = q.transpose(1, 2)  # (B, H, S, D)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    kh = kh.repeat_interleave(g_, dim=1)
    vh = vh.repeat_interleave(g_, dim=1)
    # additive mask: 0 = keep, -inf = drop
    attn = torch.zeros(B_, 1, Sq, Sk, dtype=torch.float32, device=q.device)
    # k padding
    kpad = ~km.bool().view(B_, 1, 1, Sk)
    attn = attn.masked_fill(kpad, float("-inf"))
    # q padding (whole row dropped -> output meaningless)
    qpad = ~qm.bool().view(B_, 1, Sq, 1)
    attn = attn.masked_fill(qpad, float("-inf"))
    # causal: drop where j > i
    if is_causal:
        causal = torch.triu(torch.ones(Sq, Sk, dtype=torch.bool, device=q.device), diagonal=1)
        attn = attn.masked_fill(causal.view(1, 1, Sq, Sk), float("-inf"))
    out = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, attn_mask=attn.to(dtype)
    )
    out = out.transpose(1, 2)  # (B, S, H, D)
    # zero out padded q positions (SDPA may produce NaN there)
    out = out * qm.bool().view(B_, Sq, 1, 1).to(out.dtype)
    return out


def tl_padded_out(ker, d):
    out_pad = ker(d["qp"], d["kp"], d["vp"], d["cu_q"], d["cu_k"], d["max_q"])
    out_unpad = out_pad[:d["UQ"]]
    return d["opad"](out_unpad)


def correctness_check(ker_fn, d, label=""):
    """Compare TL output vs SDPA reference."""
    with torch.no_grad():
        tl_out = tl_padded_out(ker_fn, d)
        ref = sdpa_ref(d["q"], d["k"], d["v"], d["qm"], d["km"], is_causal)
    # only compare valid (non-padded q) positions
    valid = d["qm"].bool()
    diff = (tl_out - ref).abs()
    diff[~valid] = 0
    max_diff = diff.max().item()
    mean_diff = diff[valid].mean().item()
    # relative error on large-magnitude positions
    ref_abs = ref.abs()
    rel = diff / (ref_abs + 1e-3)
    rel[~valid] = 0
    max_rel = rel.max().item()
    ok = max_diff < 0.5 and max_rel < 0.1  # bf16 tolerance
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}: max_abs={max_diff:.4f} mean_abs={mean_diff:.4f} max_rel={max_rel:.4f}")
    return ok


def bench_ker(ker, d, n_warmup=10, n_repeat=20):
    for _ in range(3):
        ker(d["qp"], d["kp"], d["vp"], d["cu_q"], d["cu_k"], d["max_q"])
    torch.cuda.synchronize()
    lat = do_bench(
        lambda: ker(d["qp"], d["kp"], d["vp"], d["cu_q"], d["cu_k"], d["max_q"]),
        _n_warmup=n_warmup, _n_repeat=n_repeat,
    )
    return lat  # milliseconds


# ---------- Correctness ----------
print("\n=== Correctness check (vs torch SDPA) ===")
d_test = make_data(1024, seed=42)
for (stg, th, bM, bN, label) in CONFIGS:
    try:
        ker_t = flashattn_pipe(B, groups, d_test["qp"].shape[0], d_test["kp"].shape[0], H_q, D, is_causal,
                                block_M=bM, block_N=bN, num_stages=stg, threads=th, dtype=dtype)
        correctness_check(ker_t, d_test, f"{label} (stg={stg} th={th} bM={bM} bN={bN})")
    except Exception as e:
        print(f"  [SKIP] {label}: {type(e).__name__}")

# ---------- Benchmark sweep ----------
print("\n=== Performance sweep ===")
print(f"{'SeqLen':>8} {'us':>10} {'TFlops':>9} {'%peak':>7}  cfg")
print("-" * 60)

for sl in [512, 1024, 2048, 4096]:
    d = make_data(sl, seed=1)
    UQp = d["qp"].shape[0]; UKVp = d["kp"].shape[0]
    for (stg, th, bM, bN, label) in CONFIGS:
        ker = flashattn_pipe(B, groups, UQp, UKVp, H_q, D, is_causal,
                             block_M=bM, block_N=bN, num_stages=stg, threads=th, dtype=dtype)
        lat = bench_ker(ker, d)
        us = lat * 1e3
        tflops = d["flops"] / (lat * 1e-3) * 1e-12
        pct = tflops / PEAK_TFLOPS * 100
        print(f"{sl:8d} {us:10.1f} {tflops:9.1f} {pct:6.1f}%  {label} (stg={stg} th={th} bM={bM} bN={bN})")
    print()
