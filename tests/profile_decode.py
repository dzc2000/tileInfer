"""
Decode profiling: 用 torch.profiler 精确定位 decode 各算子耗时。

用法 (2×A100):
    torchrun --nproc_per_node=2 tests/profile_decode.py \
        --model /mnt/models/Qwen3.6-27B --tp-size 2

    # 导出 Chrome trace
    torchrun --nproc_per_node=2 tests/profile_decode.py \
        --model /mnt/models/Qwen3.6-27B --tp-size 2 --trace trace.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault(
    "TILELANG_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".tilelang_cache"),
)

import torch
from torch.profiler import profile, ProfilerActivity


def is_rank0():
    from engine.parallel import get_tp_rank
    return get_tp_rank() == 0


def main():
    parser = argparse.ArgumentParser(description="Decode Profiling")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128, help="Decode tokens to profile")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA Graph")
    parser.add_argument("--top-k", type=int, default=30, help="Top-K operators to show")
    parser.add_argument("--trace", type=str, default=None, help="Save Chrome trace JSON")
    args = parser.parse_args()

    from engine.llm import LLM
    from engine.sampling_params import SamplingParams

    if is_rank0():
        print(f"\n{'='*80}")
        print(f"  Decode Profiling | TP={args.tp_size} | "
              f"{'Eager' if args.enforce_eager else 'CUDA Graph'}")
        print(f"{'='*80}")

    llm = LLM(
        model=args.model,
        tp_size=args.tp_size,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
        max_model_len=4096,
    )

    prompt = "Explain quantum computing in simple terms."
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Warmup
    if is_rank0():
        print(f"\n  Warmup...")
    llm.generate([prompt], sp)

    # Profile
    if is_rank0():
        print(f"  Profiling {args.max_tokens} decode tokens...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        outputs = llm.generate([prompt], sp)

    if is_rank0():
        prompt_len = len(outputs[0]["prompt_token_ids"])
        completion_tokens = len(outputs[0]["token_ids"])

        print(f"\n  Prompt: {prompt_len} tokens")
        print(f"  Generated: {completion_tokens} tokens")

        # Per-step estimate
        total_cuda = sum(e.cuda_time_total for e in prof.key_averages()) / 1000
        per_step_ms = total_cuda / max(completion_tokens, 1)
        print(f"  Total CUDA time: {total_cuda:.1f} ms")
        print(f"  Est. per-step: {per_step_ms:.2f} ms/token")
        print(f"  Est. throughput: {1000/per_step_ms:.1f} tok/s" if per_step_ms > 0 else "")

        # Category breakdown
        print(f"\n{'='*80}")
        print(f"  Operator Category Breakdown")
        print(f"{'='*80}")

        events = prof.key_averages()

        categories = {
            "GEMM/Linear": ["gemm", "linear", "mm", "addmm", "matmul"],
            "Elementwise": ["elementwise", "mul", "add", "div", "sub", "sigmoid", "softmax", "silu", "gelu"],
            "Reduction": ["reduce", "sum", "max", "argmax", "norm", "rms"],
            "Attention": ["attention", "flash", "gqa", "sdpa", "varlen", "paged"],
            "DeltaNet": ["deltanet", "recurrent", "delta", "conv1d", "flashqla"],
            "RoPE": ["rope", "rotary"],
            "Communication": ["all_reduce", "all_gather", "broadcast", "nccl", "wait"],
            "Memory": ["copy", "view", "reshape", "slice", "cat", "contiguous", "clone", "to"],
            "Embedding": ["embedding", "embed", "gather"],
            "Other": [],
        }

        cat_times = {cat: 0.0 for cat in categories}
        cat_counts = {cat: 0 for cat in categories}

        for e in events:
            if e.cuda_time_total == 0:
                continue
            name_lower = e.key.lower()
            matched = False
            for cat, keywords in categories.items():
                if any(kw in name_lower for kw in keywords):
                    cat_times[cat] += e.cuda_time_total / 1000
                    cat_counts[cat] += e.count
                    matched = True
                    break
            if not matched:
                cat_times["Other"] += e.cuda_time_total / 1000
                cat_counts["Other"] += e.count

        total = sum(cat_times.values())
        print(f"  {'Category':<18s} | {'CUDA(ms)':>10s} | {'%':>6s} | {'Calls':>8s}")
        print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}")

        for cat in sorted(cat_times, key=lambda c: cat_times[c], reverse=True):
            t = cat_times[cat]
            if t > 0:
                pct = t / total * 100 if total > 0 else 0
                print(f"  {cat:<18s} | {t:>10.3f} | {pct:>5.1f}% | {cat_counts[cat]:>8d}")

        print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}")
        print(f"  {'TOTAL':<18s} | {total:>10.3f} | {'100%':>6s} |")

        # Top operators
        print(f"\n{'='*80}")
        print(f"  Top {args.top_k} Operators by CUDA Time")
        print(f"{'='*80}")

        cuda_events = sorted(
            [e for e in events if e.cuda_time_total > 0],
            key=lambda e: e.cuda_time_total,
            reverse=True,
        )[:args.top_k]

        print(f"  {'Name':<55s} | {'CUDA(ms)':>10s} | {'Calls':>6s} | {'Avg(us)':>10s}")
        print(f"  {'-'*55}-+-{'-'*10}-+-{'-'*6}-+-{'-'*10}")

        for e in cuda_events:
            name = e.key[:55]
            cuda_ms = e.cuda_time_total / 1000
            calls = e.count
            avg_us = e.cuda_time_total / e.count if e.count > 0 else 0
            print(f"  {name:<55s} | {cuda_ms:>10.3f} | {calls:>6d} | {avg_us:>10.1f}")

        # Chrome trace
        if args.trace:
            prof.export_chrome_trace(args.trace)
            print(f"\n  Chrome trace saved to {args.trace}")
            print(f"  Open at chrome://tracing or perfetto.dev")

        print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
