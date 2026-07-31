"""
Multi-GPU Tensor Parallelism Benchmark.

Measures TP scaling efficiency across 1/2/4 GPUs.
Compares tileInfer TP vs HuggingFace device_map="auto".

Metrics:
  - Decode throughput (tok/s) per GPU count
  - TP scaling efficiency: (TP=N throughput) / (TP=1 throughput * N)
  - Peak VRAM per rank (GB)
  - Communication overhead ratio

Usage:
    # Single GPU baseline
    python tests/bench_multi_gpu.py --model /path/to/Qwen3.6-27B --tp-size 1

    # 2-GPU TP
    torchrun --nproc_per_node=2 tests/bench_multi_gpu.py --model /path/to/Qwen3.6-27B --tp-size 2

    # 4-GPU TP
    torchrun --nproc_per_node=4 tests/bench_multi_gpu.py --model /path/to/Qwen3.6-27B --tp-size 4

    # Full scaling sweep (run each separately, then aggregate)
    python tests/bench_multi_gpu.py --model /path/to/Qwen3.6-27B --tp-size 1 --output results_tp1.json
    torchrun --nproc_per_node=2 tests/bench_multi_gpu.py --model /path/to/Qwen3.6-27B --tp-size 2 --output results_tp2.json
    torchrun --nproc_per_node=4 tests/bench_multi_gpu.py --model /path/to/Qwen3.6-27B --tp-size 4 --output results_tp4.json
    python tests/bench_multi_gpu.py --aggregate results_tp1.json results_tp2.json results_tp4.json
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault(
    "TILELANG_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".tilelang_cache"),
)

import torch


def bench_tileinfer_tp(model_path: str, tp_size: int, prompt: str,
                       max_new_tokens: int, num_runs: int, warmup_runs: int) -> dict:
    """Benchmark tileInfer with given TP size."""
    from engine.parallel import get_tp_rank, init_distributed, barrier
    from engine.llm import LLM
    from engine.sampling_params import SamplingParams

    rank = get_tp_rank() if tp_size > 1 else 0

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  tileInfer TP={tp_size} (Rank {rank})")
        print(f"{'='*60}")

    llm = LLM(model=model_path, tp_size=tp_size)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    # Warmup (includes CUDA Graph capture on first run)
    for _ in range(warmup_runs + 1):
        _ = llm.generate([prompt], params)

    # Timed runs
    prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
    all_timings = []
    for run in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Measure prefill separately (max_tokens=1)
        p_start = torch.cuda.Event(enable_timing=True)
        p_end = torch.cuda.Event(enable_timing=True)
        p_start.record()
        _ = llm.generate([prompt], prefill_params)
        p_end.record()
        torch.cuda.synchronize()
        prefill_ms = p_start.elapsed_time(p_end)

        # Measure full generate (prefill + decode)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        outputs = llm.generate([prompt], params)

        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        completion_tokens = len(outputs[0]["token_ids"])
        prompt_len = len(outputs[0]["prompt_token_ids"])

        # decode_ms = total - prefill; decode_tokens excludes the first token
        # already produced during the prefill-only run
        decode_ms = max(total_ms - prefill_ms, 0.0)
        decode_tokens = max(completion_tokens - 1, 1)
        decode_tok_s = decode_tokens / decode_ms * 1000 if decode_ms > 0 else 0
        mean_itl_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        timing = {
            "prefill_ms": round(prefill_ms, 2),
            "total_ms": round(total_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "decode_tokens": decode_tokens,
            "decode_tok_s": round(decode_tok_s, 1),
            "mean_itl_ms": round(mean_itl_ms, 2),
            "peak_vram_gb": round(peak_vram, 2),
        }
        all_timings.append(timing)

        if rank == 0:
            print(f"  run {run+1}: {decode_tok_s:.1f} tok/s, ITL={mean_itl_ms:.2f}ms, "
                  f"VRAM={peak_vram:.1f}GB")

    barrier()

    # Aggregate (only rank 0 saves)
    tok_s = [t["decode_tok_s"] for t in all_timings]
    itl = [t["mean_itl_ms"] for t in all_timings]
    vram = [t["peak_vram_gb"] for t in all_timings]

    result = {
        "method": f"tileInfer_TP{tp_size}",
        "tp_size": tp_size,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "num_runs": num_runs,
        "decode_tok_s": {
            "mean": round(sum(tok_s) / len(tok_s), 1),
            "median": round(sorted(tok_s)[len(tok_s) // 2], 1),
            "min": round(min(tok_s), 1),
            "max": round(max(tok_s), 1),
        },
        "mean_itl_ms": {
            "mean": round(sum(itl) / len(itl), 2),
        },
        "peak_vram_gb": {
            "mean": round(sum(vram) / len(vram), 2),
        },
    }

    return result


def bench_hf_device_map_auto(model_path: str, prompt: str,
                             max_new_tokens: int, num_runs: int,
                             warmup_runs: int) -> dict:
    """Benchmark HuggingFace with device_map='auto' (pipeline parallelism baseline)."""
    print(f"\n{'='*60}")
    print(f"  HuggingFace device_map='auto' (Pipeline Parallel)")
    print(f"{'='*60}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.requires_grad_(False)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    prompt_len = input_ids.shape[1]

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

    # Timed runs
    all_timings = []
    for run in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        with torch.no_grad():
            outputs = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True,
            )

        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        decode_tokens = max_new_tokens
        decode_ms = total_ms * 0.9  # rough estimate
        decode_tok_s = decode_tokens / decode_ms * 1000 if decode_ms > 0 else 0
        mean_itl_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        timing = {
            "total_ms": round(total_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "decode_tokens": decode_tokens,
            "decode_tok_s": round(decode_tok_s, 1),
            "mean_itl_ms": round(mean_itl_ms, 2),
            "peak_vram_gb": round(peak_vram, 2),
        }
        all_timings.append(timing)
        print(f"  run {run+1}: {decode_tok_s:.1f} tok/s, ITL={mean_itl_ms:.2f}ms")

    del model
    torch.cuda.empty_cache()

    tok_s = [t["decode_tok_s"] for t in all_timings]
    itl = [t["mean_itl_ms"] for t in all_timings]
    vram = [t["peak_vram_gb"] for t in all_timings]

    return {
        "method": "HF_device_map_auto",
        "tp_size": torch.cuda.device_count(),
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "num_runs": num_runs,
        "decode_tok_s": {
            "mean": round(sum(tok_s) / len(tok_s), 1),
            "median": round(sorted(tok_s)[len(tok_s) // 2], 1),
        },
        "mean_itl_ms": {
            "mean": round(sum(itl) / len(itl), 2),
        },
        "peak_vram_gb": {
            "mean": round(sum(vram) / len(vram), 2),
        },
    }


def aggregate_results(result_files: list[str]):
    """Aggregate results from multiple TP runs and compute scaling efficiency."""
    print(f"\n{'='*60}")
    print("  TP SCALING EFFICIENCY")
    print(f"{'='*60}")

    all_results = []
    for f in result_files:
        with open(f) as fp:
            all_results.append(json.load(fp))

    # Sort by tp_size
    all_results.sort(key=lambda x: x.get("tp_size", 1))

    baseline_tok_s = None
    print(f"\n  {'TP Size':>8s} | {'tok/s':>10s} | {'ITL (ms)':>10s} | {'VRAM (GB)':>10s} | {'Efficiency':>10s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for r in all_results:
        tp = r.get("tp_size", 1)
        tok_s = r["decode_tok_s"]["mean"]
        itl = r["mean_itl_ms"]["mean"]
        vram = r["peak_vram_gb"]["mean"]

        if tp == 1:
            baseline_tok_s = tok_s
            efficiency = 1.0
        elif baseline_tok_s and baseline_tok_s > 0:
            # Ideal: N * baseline. Actual: tok_s. Efficiency = actual / ideal
            efficiency = tok_s / (baseline_tok_s * tp)
        else:
            efficiency = 0.0

        print(f"  {tp:>8d} | {tok_s:>10.1f} | {itl:>10.2f} | {vram:>10.1f} | {efficiency:>9.1%}")

    # Save aggregated
    output = {"scaling_results": all_results}
    out_file = "bench_multi_gpu_scaling.json"
    with open(out_file, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nScaling results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU TP Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.6-27B")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--aggregate", nargs="+", help="Aggregate multiple result files")
    args = parser.parse_args()

    # Aggregate mode
    if args.aggregate:
        aggregate_results(args.aggregate)
        return

    output_file = args.output or f"bench_tp{args.tp_size}_results.json"

    results = {
        "model": args.model,
        "tp_size": args.tp_size,
        "benchmarks": [],
    }

    # tileInfer TP benchmark
    ti_result = bench_tileinfer_tp(
        args.model, args.tp_size, args.prompt,
        args.max_tokens, args.num_runs, args.warmup_runs,
    )
    results["benchmarks"].append(ti_result)

    # HF baseline (only for TP=1 or device_map=auto)
    if not args.skip_hf and args.tp_size == 1:
        hf_result = bench_hf_device_map_auto(
            args.model, args.prompt,
            args.max_tokens, args.num_runs, args.warmup_runs,
        )
        results["benchmarks"].append(hf_result)

    # Save (only rank 0 for TP>1)
    from engine.parallel import get_tp_rank
    if args.tp_size == 1 or get_tp_rank() == 0:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
