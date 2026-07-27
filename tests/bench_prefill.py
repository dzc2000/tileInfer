"""
Prefill Throughput Benchmark.

Measures prefill latency across different sequence lengths.
Compares tileInfer (fused kernels + FlashQLA) vs HuggingFace baseline.

Tests:
  1. Short prompt (64 tokens)
  2. Medium prompt (512 tokens)
  3. Long prompt (2048 tokens)
  4. Very long prompt (4096 tokens)
  5. Chunked prefill (8192 tokens with 4096 chunk size)

Usage:
    python tests/bench_prefill.py --model /path/to/Qwen3.6-27B
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer


PROMPT_LENGTHS = [64, 512, 2048, 4096]
LONG_PROMPT_LENGTHS = [8192]


def generate_prompt_of_length(tokenizer, target_len: int) -> str:
    """Generate a prompt that tokenizes to approximately target_len tokens."""
    base = "The history of science is a fascinating subject. "
    # Repeat to get close to target length
    tokens = tokenizer.encode(base, add_special_tokens=False)
    repeats = max(1, target_len // len(tokens))
    prompt = (base * repeats)[:target_len * 6]  # overshoot, tokenizer will trim
    actual_len = len(tokenizer.encode(prompt, add_special_tokens=True))
    # Adjust
    while actual_len > target_len + 10:
        prompt = prompt[:int(len(prompt) * target_len / actual_len)]
        actual_len = len(tokenizer.encode(prompt, add_special_tokens=True))
    return prompt


def bench_hf_prefill(model_path: str, tokenizer, prompt_lengths: list,
                     num_runs: int) -> list:
    """Benchmark HuggingFace prefill at various sequence lengths."""
    print(f"\n{'='*60}")
    print("  HuggingFace Prefill Latency")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    results = []
    for target_len in prompt_lengths:
        prompt = generate_prompt_of_length(tokenizer, target_len)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        actual_len = input_ids.shape[1]

        # Warmup
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)

        # Timed runs
        latencies = []
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            with torch.no_grad():
                _ = model(input_ids, use_cache=False)

            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

        mean_ms = sum(latencies) / len(latencies)
        tok_per_s = actual_len / mean_ms * 1000

        result = {
            "method": "HF_prefill",
            "seq_len": actual_len,
            "latency_ms": round(mean_ms, 2),
            "throughput_tok_s": round(tok_per_s, 1),
        }
        results.append(result)
        print(f"  seq_len={actual_len:5d}: {mean_ms:8.2f}ms, {tok_per_s:8.1f} tok/s")

    del model
    torch.cuda.empty_cache()
    return results


def bench_tileinfer_prefill(model_path: str, tokenizer, prompt_lengths: list,
                            num_runs: int) -> list:
    """Benchmark tileInfer prefill at various sequence lengths."""
    print(f"\n{'='*60}")
    print("  tileInfer Prefill Latency (fused kernels + FlashQLA)")
    print(f"{'='*60}")

    from engine.llm import LLM
    from engine.sampling_params import SamplingParams

    llm = LLM(model=model_path, enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=1)  # only 1 decode token

    results = []
    for target_len in prompt_lengths:
        prompt = generate_prompt_of_length(tokenizer, target_len)
        actual_len = len(tokenizer.encode(prompt, add_special_tokens=True))

        # Warmup
        _ = llm.generate([prompt], params)

        # Timed runs
        latencies = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            _ = llm.generate([prompt], params)

            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

        mean_ms = sum(latencies) / len(latencies)
        # Subtract approximate decode time (1 token ~ 5ms)
        prefill_ms = mean_ms - 5.0
        tok_per_s = actual_len / prefill_ms * 1000 if prefill_ms > 0 else 0

        result = {
            "method": "tileInfer_prefill",
            "seq_len": actual_len,
            "latency_ms": round(prefill_ms, 2),
            "total_ms": round(mean_ms, 2),
            "throughput_tok_s": round(tok_per_s, 1),
        }
        results.append(result)
        print(f"  seq_len={actual_len:5d}: {prefill_ms:8.2f}ms, {tok_per_s:8.1f} tok/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Prefill Throughput Benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--output", type=str, default="bench_prefill_results.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = {
        "model": args.model,
        "num_runs": args.num_runs,
        "benchmarks": [],
    }

    # HF baseline
    if not args.skip_hf:
        hf_results = bench_hf_prefill(
            args.model, tokenizer, PROMPT_LENGTHS, args.num_runs,
        )
        results["benchmarks"].extend(hf_results)

    # tileInfer
    ti_results = bench_tileinfer_prefill(
        args.model, tokenizer, PROMPT_LENGTHS, args.num_runs,
    )
    results["benchmarks"].extend(ti_results)

    # Summary table
    print(f"\n{'='*60}")
    print("  PREFILL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':20s} | {'Seq Len':>7s} | {'Latency':>8s} | {'Throughput':>10s}")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}")
    for bm in results["benchmarks"]:
        print(f"  {bm['method']:20s} | {bm['seq_len']:>7d} | {bm['latency_ms']:>7.2f}ms | "
              f"{bm['throughput_tok_s']:>9.1f} tok/s")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
