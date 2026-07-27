"""
Decode profiling 脚本: 用 torch.profiler 精确定位 39ms/token 的耗时分布。

用法 (2×A100):
    torchrun --nproc_per_node=2 tests/profile_decode.py \
        --model /mnt/models/Qwen3.6-27B --tp-size 2

输出:
  - 按 operator 分类的耗时统计
  - Top-20 最耗时的 kernel
  - DeltaNet vs Attention vs MLP 时间分布
  - 通信 (all_reduce) 开销
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
from torch.profiler import profile, ProfilerActivity, record_function


def is_rank0():
    from engine.parallel import get_tp_rank
    return get_tp_rank() == 0


def profile_decode(model_runner, seqs, num_steps=10):
    """Profile decode steps, breakdown by operator category."""
    from engine.context import set_context, reset_context

    # Prepare decode context
    input_ids, positions, ctx = model_runner.prepare_decode(seqs)

    # Warmup (trigger JIT + CUDA Graph if enabled)
    set_context(ctx)
    try:
        for _ in range(3):
            model_runner.run(seqs, is_prefill=False)
    finally:
        reset_context()

    # Profile
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for step in range(num_steps):
            with record_function(f"decode_step_{step}"):
                model_runner.run(seqs, is_prefill=False)

    return prof


def print_top_operators(prof, top_k=30):
    """Print top operators by CUDA time."""
    print(f"\n{'='*80}")
    print(f"  Top {top_k} Operators by CUDA Time")
    print(f"{'='*80}")

    events = prof.key_averages()
    cuda_events = sorted(
        [e for e in events if e.cuda_time_total > 0],
        key=lambda e: e.cuda_time_total,
        reverse=True,
    )[:top_k]

    print(f"  {'Name':<55s} | {'CUDA(ms)':>10s} | {'Calls':>6s} | {'Avg(us)':>10s}")
    print(f"  {'-'*55}-+-{'-'*10}-+-{'-'*6}-+-{'-'*10}")

    for e in cuda_events:
        name = e.key[:55]
        cuda_ms = e.cuda_time_total / 1000
        calls = e.count
        avg_us = e.cuda_time_total / e.count if e.count > 0 else 0
        print(f"  {name:<55s} | {cuda_ms:>10.3f} | {calls:>6d} | {avg_us:>10.1f}")


def print_category_breakdown(prof):
    """Breakdown by operator category."""
    print(f"\n{'='*80}")
    print(f"  Operator Category Breakdown")
    print(f"{'='*80}")

    events = prof.key_averages()

    categories = {
        "GEMM/Linear": ["gemm", "linear", "mm", "addmm"],
        "Elementwise": ["elementwise", "mul", "add", "div", "sub", "sigmoid", "softmax"],
        "Reduction": ["reduce", "sum", "max", "argmax", "norm"],
        "Attention": ["attention", "flash", "gqa", "sdpa", "varlen"],
        "DeltaNet": ["deltanet", "recurrent", "delta", "conv1d"],
        "RMSNorm": ["rms_norm", "rmsnorm", "layernorm"],
        "RoPE": ["rope", "rotary"],
        "Communication": ["all_reduce", "all_gather", "broadcast", "nccl", "wait"],
        "Memory": ["copy", "view", "reshape", "slice", "cat", "contiguous"],
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


def print_step_breakdown(prof, num_steps):
    """Per-step timing."""
    print(f"\n{'='*80}")
    print(f"  Per-Step Timing ({num_steps} decode steps)")
    print(f"{'='*80}")

    events = prof.key_averages()
    step_times = []

    for e in events:
        if e.key.startswith("decode_step_"):
            step_times.append(e.cuda_time_total / 1000)

    if step_times:
        avg = sum(step_times) / len(step_times)
        print(f"  Steps: {len(step_times)}")
        print(f"  Total CUDA: {sum(step_times):.2f} ms")
        print(f"  Avg/step:   {avg:.2f} ms")
        print(f"  Min/step:   {min(step_times):.2f} ms")
        print(f"  Max/step:   {max(step_times):.2f} ms")
        print(f"  Est. tok/s: {1000/avg:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Decode Profiling")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10, help="Profiled decode steps")
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

    # Prefill a prompt to set up KV cache + DeltaNet state
    from engine.sequence import Sequence
    from engine.context import set_context, reset_context

    prompt = "Explain quantum computing in simple terms."
    token_ids = llm.tokenizer.encode(prompt, add_special_tokens=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.num_steps + 5)
    seq = Sequence(token_ids, sp)

    # Manually run prefill
    for s in [seq]:
        if s.seq_id not in llm.model_runner.seq_to_slot:
            llm.model_runner.allocate_deltanet_slot(s.seq_id)
    llm.scheduler.add(seq)

    # Run prefill
    seqs_batch, is_prefill = llm.scheduler.schedule()
    llm.model_runner.run(seqs_batch, is_prefill=True)
    llm.scheduler.postprocess(seqs_batch, [-1], is_prefill=True)

    # Now seq is in decode phase
    decode_seqs = [seq]

    if is_rank0():
        print(f"\n  Prompt: {len(token_ids)} tokens")
        print(f"  Profiling {args.num_steps} decode steps...")

    # Profile decode
    prof = profile_decode(llm.model_runner, decode_seqs, args.num_steps)

    if is_rank0():
        print_step_breakdown(prof, args.num_steps)
        print_category_breakdown(prof)
        print_top_operators(prof, args.top_k)

        # Chrome trace
        if args.trace:
            prof.export_chrome_trace(args.trace)
            print(f"\nChrome trace saved to {args.trace}")

        print(f"\n{'='*80}")
        print("  Done. Use --trace trace.json to export Chrome trace for visualization.")
        print("  Load in chrome://tracing or perfetto.dev")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
