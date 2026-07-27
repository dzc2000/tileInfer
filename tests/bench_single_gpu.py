"""
Single-GPU Benchmark: tileInfer vs HuggingFace Transformers (native Qwen3.6).

Compares:
  1. HuggingFace Transformers eager decode (baseline)
  2. tileInfer engine (fused kernels + paged KV cache)

Metrics:
  - Prefill latency (ms)
  - Decode throughput (tok/s)
  - Mean ITL (inter-token latency, ms)
  - Peak VRAM (GB)

Usage:
    python tests/bench_single_gpu.py --model /path/to/Qwen3.6-27B
    python tests/bench_single_gpu.py --model /path/to/Qwen3.6-27B --max-tokens 128 --num-runs 10
"""
import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# HuggingFace Baseline
# ============================================================
def bench_hf_eager(model_path: str, prompt: str, max_new_tokens: int,
                   num_runs: int, warmup_runs: int) -> dict:
    """Benchmark HuggingFace Transformers eager decode."""
    print("\n" + "=" * 60)
    print("  HuggingFace Transformers (eager) - Baseline")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    prompt_len = input_ids.shape[1]

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()

    # Timed runs
    all_timings = []
    for run in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Prefill
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        prefill_start.record()

        with torch.no_grad():
            outputs = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True,
            )
        prefill_end.record()
        torch.cuda.synchronize()

        total_ms = prefill_start.elapsed_time(prefill_end)
        # Approximate: HF generate does prefill + decode together
        # We estimate prefill as ~prompt_len * hidden / bandwidth
        # and decode as (total - prefill_estimate)
        # More accurate: use HF's built-in timing if available
        decode_tokens = max_new_tokens
        # Rough split: prefill is fast, decode dominates
        prefill_ms = prompt_len * 0.05  # rough estimate
        decode_ms = total_ms - prefill_ms
        decode_tok_s = decode_tokens / decode_ms * 1000 if decode_ms > 0 else 0
        mean_itl_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        timing = {
            "prefill_ms": round(prefill_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "decode_tokens": decode_tokens,
            "decode_tok_s": round(decode_tok_s, 1),
            "mean_itl_ms": round(mean_itl_ms, 2),
            "total_ms": round(total_ms, 2),
            "peak_vram_gb": round(peak_vram, 2),
        }
        all_timings.append(timing)
        print(f"  run {run+1}: {decode_tok_s:.1f} tok/s, ITL={mean_itl_ms:.2f}ms, "
              f"VRAM={peak_vram:.1f}GB")

    # Free model
    del model
    torch.cuda.empty_cache()

    return _aggregate_results("HF_eager", all_timings, prompt_len, max_new_tokens)


def bench_hf_manual(model_path: str, prompt: str, max_new_tokens: int,
                    num_runs: int, warmup_runs: int) -> dict:
    """Benchmark HuggingFace with manual decode loop (accurate timing)."""
    print("\n" + "=" * 60)
    print("  HuggingFace Transformers (manual decode) - Baseline")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    prompt_len = input_ids.shape[1]
    device = "cuda"

    # Warmup
    for _ in range(warmup_runs):
        _run_hf_manual(model, input_ids, max_new_tokens, device)
    torch.cuda.synchronize()

    # Timed runs
    all_timings = []
    for run in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        timing = _run_hf_manual(model, input_ids, max_new_tokens, device)
        timing["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
        all_timings.append(timing)
        print(f"  run {run+1}: {timing['decode_tok_s']:.1f} tok/s, "
              f"ITL={timing['mean_itl_ms']:.2f}ms, VRAM={timing['peak_vram_gb']:.1f}GB")

    del model
    torch.cuda.empty_cache()

    return _aggregate_results("HF_manual", all_timings, prompt_len, max_new_tokens)


def _run_hf_manual(model, input_ids, max_new_tokens, device):
    """Single run of HF manual decode with CUDA event timing."""
    batch_size, prompt_len = input_ids.shape
    eos_token_id = 151645

    # Prefill
    cache_position = torch.arange(prompt_len, device=device)
    position_ids = cache_position.unsqueeze(0)

    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_start.record()

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        for layer in model.model.layers:
            hidden = layer(
                hidden, cache_position=cache_position,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=True,
            )
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)

    next_token = logits[:, -1:, :].argmax(dim=-1)
    prefill_end.record()

    generated_ids = [next_token]
    cur_pos = prompt_len

    # Decode
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)
    decode_start.record()

    decode_tokens = 0
    for step in range(max_new_tokens - 1):
        cache_position = torch.tensor([cur_pos], device=device)
        position_ids = cache_position.unsqueeze(0)

        with torch.no_grad():
            hidden = model.model.embed_tokens(next_token)
            position_embeddings = model.model.rotary_emb(hidden, position_ids)
            for layer in model.model.layers:
                hidden = layer(
                    hidden, cache_position=cache_position,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=True,
                )
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)

        next_token = logits[:, -1:, :].argmax(dim=-1)
        generated_ids.append(next_token)
        cur_pos += 1
        decode_tokens += 1

        if decode_tokens % 8 == 0:
            recent = torch.cat(generated_ids[-8:], dim=-1)
            if (recent == eos_token_id).any().item():
                break

    decode_end.record()
    torch.cuda.synchronize()

    prefill_ms = prefill_start.elapsed_time(prefill_end)
    decode_ms = decode_start.elapsed_time(decode_end)
    decode_tok_s = decode_tokens / decode_ms * 1000 if decode_ms > 0 else 0
    mean_itl_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0

    return {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tokens": decode_tokens,
        "decode_tok_s": round(decode_tok_s, 1),
        "mean_itl_ms": round(mean_itl_ms, 2),
    }


# ============================================================
# tileInfer Engine
# ============================================================
def bench_tileinfer(model_path: str, prompt: str, max_new_tokens: int,
                    num_runs: int, warmup_runs: int) -> dict:
    """Benchmark tileInfer engine with paged KV cache."""
    print("\n" + "=" * 60)
    print("  tileInfer Engine (fused kernels + paged KV cache)")
    print("=" * 60)

    from engine.llm import LLM
    from engine.sampling_params import SamplingParams

    llm = LLM(model=model_path)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    # Warmup (includes CUDA Graph capture on first run)
    for _ in range(warmup_runs + 1):
        _ = llm.generate([prompt], params)

    # Timed runs
    all_timings = []
    for run in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        outputs = llm.generate([prompt], params)

        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        completion_tokens = len(outputs[0]["token_ids"])
        prompt_len = len(outputs[0]["prompt_token_ids"])

        # Estimate prefill vs decode split
        # Prefill is typically very fast with fused kernels
        prefill_ms = prompt_len * 0.02  # rough estimate
        decode_ms = total_ms - prefill_ms
        decode_tok_s = completion_tokens / decode_ms * 1000 if decode_ms > 0 else 0
        mean_itl_ms = decode_ms / completion_tokens if completion_tokens > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        timing = {
            "prefill_ms": round(prefill_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "decode_tokens": completion_tokens,
            "decode_tok_s": round(decode_tok_s, 1),
            "mean_itl_ms": round(mean_itl_ms, 2),
            "total_ms": round(total_ms, 2),
            "peak_vram_gb": round(peak_vram, 2),
        }
        all_timings.append(timing)
        print(f"  run {run+1}: {decode_tok_s:.1f} tok/s, ITL={mean_itl_ms:.2f}ms, "
              f"VRAM={peak_vram:.1f}GB")

    return _aggregate_results("tileInfer", all_timings, prompt_len, max_new_tokens)


# ============================================================
# Aggregation
# ============================================================
def _aggregate_results(method: str, all_timings: list, prompt_len: int,
                       max_new_tokens: int) -> dict:
    """Compute mean/median/p95/p99 statistics."""
    tok_s = [t["decode_tok_s"] for t in all_timings]
    itl = [t["mean_itl_ms"] for t in all_timings]
    prefill = [t["prefill_ms"] for t in all_timings]
    vram = [t.get("peak_vram_gb", 0) for t in all_timings]

    tok_s_sorted = sorted(tok_s)
    itl_sorted = sorted(itl)

    return {
        "method": method,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "num_runs": len(all_timings),
        "decode_tok_s": {
            "mean": round(sum(tok_s) / len(tok_s), 1),
            "median": round(tok_s_sorted[len(tok_s_sorted) // 2], 1),
            "p95": round(tok_s_sorted[int(len(tok_s_sorted) * 0.95)], 1),
            "min": round(min(tok_s), 1),
            "max": round(max(tok_s), 1),
        },
        "mean_itl_ms": {
            "mean": round(sum(itl) / len(itl), 2),
            "median": round(itl_sorted[len(itl_sorted) // 2], 2),
            "p95": round(itl_sorted[int(len(itl_sorted) * 0.95)], 2),
        },
        "prefill_ms": {
            "mean": round(sum(prefill) / len(prefill), 2),
        },
        "peak_vram_gb": {
            "mean": round(sum(vram) / len(vram), 2) if any(vram) else 0,
        },
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Single-GPU Benchmark: tileInfer vs HF")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--skip-hf", action="store_true", help="Skip HF baseline (saves time)")
    parser.add_argument("--output", type=str, default="bench_single_gpu_results.json")
    args = parser.parse_args()

    results = {
        "model": args.model,
        "prompt": args.prompt[:80] + "..." if len(args.prompt) > 80 else args.prompt,
        "max_new_tokens": args.max_tokens,
        "num_runs": args.num_runs,
        "benchmarks": [],
    }

    # HF manual decode baseline (most accurate)
    if not args.skip_hf:
        hf_result = bench_hf_manual(
            args.model, args.prompt, args.max_tokens,
            args.num_runs, args.warmup_runs,
        )
        results["benchmarks"].append(hf_result)

    # tileInfer engine
    ti_result = bench_tileinfer(
        args.model, args.prompt, args.max_tokens,
        args.num_runs, args.warmup_runs,
    )
    results["benchmarks"].append(ti_result)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for bm in results["benchmarks"]:
        method = bm["method"]
        tok_s = bm["decode_tok_s"]["mean"]
        itl = bm["mean_itl_ms"]["mean"]
        vram = bm["peak_vram_gb"]["mean"]
        print(f"  {method:20s}: {tok_s:8.1f} tok/s | ITL={itl:6.2f}ms | VRAM={vram:5.1f}GB")

    # Speedup
    if len(results["benchmarks"]) == 2:
        hf_tok = results["benchmarks"][0]["decode_tok_s"]["mean"]
        ti_tok = results["benchmarks"][1]["decode_tok_s"]["mean"]
        speedup = ti_tok / hf_tok if hf_tok > 0 else 0
        print(f"\n  Speedup: {speedup:.2f}x")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
