"""
Continuous Batching Decode Throughput Benchmark.

Measures aggregate decode throughput with varying batch sizes.
Compares tileInfer (continuous batching + paged KV cache) vs
HuggingFace (static batched decode).

Batch sizes: 1, 4, 8, 16, 32

Metrics:
  - Aggregate throughput (tok/s across all sequences)
  - Per-sequence throughput (tok/s per user)
  - Mean ITL (ms)

Usage:
    python tests/bench_batched_decode.py --model /path/to/Qwen3.6-27B
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer


BATCH_SIZES = [1, 4, 8, 16, 32]
PROMPT = "Write a short essay about artificial intelligence."


def bench_hf_batched(model_path: str, tokenizer, batch_sizes: list,
                     max_new_tokens: int, num_runs: int) -> list:
    """Benchmark HuggingFace batched decode (lockstep, same prompt for all)."""
    print(f"\n{'='*60}")
    print("  HuggingFace Batched Decode (lockstep)")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    input_ids_single = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda")

    results = []
    for bsz in batch_sizes:
        # Replicate prompt for batch
        input_ids = input_ids_single.expand(bsz, -1)

        # Warmup
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        # Timed runs
        latencies = []
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            with torch.no_grad():
                _ = model.generate(
                    input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                    use_cache=True,
                )

            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

        mean_ms = sum(latencies) / len(latencies)
        total_tokens = bsz * max_new_tokens
        agg_tok_s = total_tokens / mean_ms * 1000
        per_user_tok_s = max_new_tokens / mean_ms * 1000
        mean_itl_ms = mean_ms / max_new_tokens

        result = {
            "method": "HF_batched",
            "batch_size": bsz,
            "total_ms": round(mean_ms, 2),
            "aggregate_tok_s": round(agg_tok_s, 1),
            "per_user_tok_s": round(per_user_tok_s, 1),
            "mean_itl_ms": round(mean_itl_ms, 2),
        }
        results.append(result)
        print(f"  bsz={bsz:3d}: agg={agg_tok_s:8.1f} tok/s, "
              f"per_user={per_user_tok_s:8.1f} tok/s, ITL={mean_itl_ms:.2f}ms")

    del model
    torch.cuda.empty_cache()
    return results


def bench_tileinfer_batched(model_path: str, tokenizer, batch_sizes: list,
                            max_new_tokens: int, num_runs: int) -> list:
    """Benchmark tileInfer continuous batching decode."""
    print(f"\n{'='*60}")
    print("  tileInfer Continuous Batching Decode")
    print(f"{'='*60}")

    from engine.llm import LLM
    from engine.sampling_params import SamplingParams

    llm = LLM(model=model_path)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    results = []
    for bsz in batch_sizes:
        prompts = [PROMPT] * bsz

        # Warmup
        _ = llm.generate(prompts, params)

        # Timed runs
        latencies = []
        total_tokens_list = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            outputs = llm.generate(prompts, params)

            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

            # Count actual tokens (may vary due to EOS)
            total_tokens = sum(len(o["token_ids"]) for o in outputs)
            total_tokens_list.append(total_tokens)

        mean_ms = sum(latencies) / len(latencies)
        mean_total_tokens = sum(total_tokens_list) / len(total_tokens_list)
        agg_tok_s = mean_total_tokens / mean_ms * 1000
        per_user_tok_s = agg_tok_s / bsz
        mean_itl_ms = mean_ms / (mean_total_tokens / bsz)

        result = {
            "method": "tileInfer_batched",
            "batch_size": bsz,
            "total_ms": round(mean_ms, 2),
            "aggregate_tok_s": round(agg_tok_s, 1),
            "per_user_tok_s": round(per_user_tok_s, 1),
            "mean_itl_ms": round(mean_itl_ms, 2),
        }
        results.append(result)
        print(f"  bsz={bsz:3d}: agg={agg_tok_s:8.1f} tok/s, "
              f"per_user={per_user_tok_s:8.1f} tok/s, ITL={mean_itl_ms:.2f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Batched Decode Throughput Benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--output", type=str, default="bench_batched_results.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = {
        "model": args.model,
        "max_new_tokens": args.max_tokens,
        "num_runs": args.num_runs,
        "benchmarks": [],
    }

    # HF baseline
    if not args.skip_hf:
        hf_results = bench_hf_batched(
            args.model, tokenizer, BATCH_SIZES, args.max_tokens, args.num_runs,
        )
        results["benchmarks"].extend(hf_results)

    # tileInfer
    ti_results = bench_tileinfer_batched(
        args.model, tokenizer, BATCH_SIZES, args.max_tokens, args.num_runs,
    )
    results["benchmarks"].extend(ti_results)

    # Summary table
    print(f"\n{'='*60}")
    print("  BATCHED DECODE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':20s} | {'BSZ':>4s} | {'Agg tok/s':>10s} | {'Per-user':>10s} | {'ITL':>8s}")
    print(f"  {'-'*20}-+-{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for bm in results["benchmarks"]:
        print(f"  {bm['method']:20s} | {bm['batch_size']:>4d} | {bm['aggregate_tok_s']:>10.1f} | "
              f"{bm['per_user_tok_s']:>10.1f} | {bm['mean_itl_ms']:>6.2f}ms")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
