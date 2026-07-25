"""
Run all benchmarks and generate a summary report.

Usage:
    # Single GPU benchmarks only
    python tests/run_all_benchmarks.py --model /path/to/Qwen3.6-27B

    # Skip HF baseline (faster)
    python tests/run_all_benchmarks.py --model /path/to/Qwen3.6-27B --skip-hf

    # Quick mode (fewer runs)
    python tests/run_all_benchmarks.py --model /path/to/Qwen3.6-27B --quick

    # With multi-GPU TP benchmarks
    python tests/run_all_benchmarks.py --model /path/to/Qwen3.6-27B --tp-sizes 1,2,4
"""
import argparse
import json
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_command(cmd: str, description: str) -> float:
    """Run a benchmark command and return elapsed time."""
    print(f"\n{'#'*60}")
    print(f"# {description}")
    print(f"# Command: {cmd}")
    print(f"{'#'*60}")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"  WARNING: {description} failed with return code {result.returncode}")
    else:
        print(f"  {description} completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run All Benchmarks")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--skip-hf", action="store_true", help="Skip HF baselines")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer runs)")
    parser.add_argument("--tp-sizes", type=str, default="1",
                        help="Comma-separated TP sizes, e.g. '1,2,4'")
    parser.add_argument("--output-dir", type=str, default="bench_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    num_runs = 3 if args.quick else 10
    warmup = 1 if args.quick else 3
    max_tokens = 64 if args.quick else 128
    skip_hf = "--skip-hf" if args.skip_hf else ""
    tp_sizes = [int(x) for x in args.tp_sizes.split(",")]

    total_start = time.time()
    results_summary = []

    # ============================================================
    # 1. Single-GPU Benchmark
    # ============================================================
    cmd = (f"python tests/bench_single_gpu.py "
           f"--model {args.model} "
           f"--max-tokens {max_tokens} "
           f"--num-runs {num_runs} "
           f"--warmup-runs {warmup} "
           f"{skip_hf} "
           f"--output {args.output_dir}/bench_single_gpu.json")
    run_command(cmd, "Single-GPU Benchmark (tileInfer vs HF)")
    results_summary.append("bench_single_gpu.json")

    # ============================================================
    # 2. Prefill Throughput Benchmark
    # ============================================================
    cmd = (f"python tests/bench_prefill.py "
           f"--model {args.model} "
           f"--num-runs {num_runs} "
           f"{skip_hf} "
           f"--output {args.output_dir}/bench_prefill.json")
    run_command(cmd, "Prefill Throughput Benchmark")
    results_summary.append("bench_prefill.json")

    # ============================================================
    # 3. Batched Decode Benchmark
    # ============================================================
    cmd = (f"python tests/bench_batched_decode.py "
           f"--model {args.model} "
           f"--max-tokens {max_tokens} "
           f"--num-runs {num_runs} "
           f"{skip_hf} "
           f"--output {args.output_dir}/bench_batched.json")
    run_command(cmd, "Batched Decode Benchmark")
    results_summary.append("bench_batched.json")

    # ============================================================
    # 4. Multi-GPU TP Benchmark (for each TP size)
    # ============================================================
    tp_result_files = []
    for tp in tp_sizes:
        if tp == 1:
            cmd_prefix = "python"
        else:
            cmd_prefix = f"torchrun --nproc_per_node={tp}"

        output_file = f"{args.output_dir}/bench_tp{tp}.json"
        cmd = (f"{cmd_prefix} tests/bench_multi_gpu.py "
               f"--model {args.model} "
               f"--tp-size {tp} "
               f"--max-tokens {max_tokens} "
               f"--num-runs {num_runs} "
               f"--warmup-runs {warmup} "
               f"{skip_hf} "
               f"--output {output_file}")
        run_command(cmd, f"Multi-GPU TP={tp} Benchmark")
        tp_result_files.append(output_file)

    # Aggregate TP scaling if multiple sizes
    if len(tp_result_files) > 1:
        agg_cmd = (f"python tests/bench_multi_gpu.py "
                   f"--aggregate {' '.join(tp_result_files)}")
        run_command(agg_cmd, "TP Scaling Aggregation")

    # ============================================================
    # Final Summary
    # ============================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL BENCHMARKS COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Results directory: {args.output_dir}/")
    print(f"{'='*60}")

    # Print summary from each result file
    for result_file in results_summary:
        path = os.path.join(args.output_dir, result_file)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"\n  --- {result_file} ---")
            if "benchmarks" in data:
                for bm in data["benchmarks"]:
                    method = bm.get("method", "unknown")
                    if "decode_tok_s" in bm:
                        tok_s = bm["decode_tok_s"]
                        if isinstance(tok_s, dict):
                            print(f"    {method}: {tok_s.get('mean', 0):.1f} tok/s (mean)")
                        else:
                            print(f"    {method}: {tok_s} tok/s")
                    elif "throughput_tok_s" in bm:
                        print(f"    {method} (seq={bm.get('seq_len', '?')}): "
                              f"{bm['throughput_tok_s']:.1f} tok/s")

    print(f"\nFull results saved in {args.output_dir}/")


if __name__ == "__main__":
    main()
