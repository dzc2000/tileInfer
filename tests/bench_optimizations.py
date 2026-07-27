"""
优化效果基准测试脚本 (2×A100 40GB)。

测量维度：
  1. Prefill 吞吐 (tok/s)
  2. Decode 吞吐 (tok/s) — CUDA Graph vs Eager
  3. Fused lm_head+argmax vs 原始 lm_head+argmax 逐项对比
  4. TP 通信开销占比

用法 (2×A100):
    # CUDA Graph 路径 (默认)
    torchrun --nproc_per_node=2 tests/bench_optimizations.py \
        --model /path/to/Qwen3.6-27B --tp-size 2

    # Eager 模式 (触发 fused lm_head 非CUDA Graph路径)
    torchrun --nproc_per_node=2 tests/bench_optimizations.py \
        --model /path/to/Qwen3.6-27B --tp-size 2 --enforce-eager

    # 单卡基线
    python tests/bench_optimizations.py --model /path/to/Qwen3.6-27B --tp-size 1
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault(
    "TILELANG_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".tilelang_cache"),
)

import torch
from engine.parallel import get_tp_rank, get_tp_world_size, is_tp_active, barrier


def is_rank0():
    return get_tp_rank() == 0


def bench_prefill(llm, prompt, params, num_runs, warmup_runs):
    """测量 prefill 吞吐 (仅单序列)。"""
    from engine.sampling_params import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    # Warmup
    for _ in range(warmup_runs):
        llm.generate([prompt], sp)

    # 测量: 用 max_tokens=1 让 prefill 占主导
    timings = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        llm.generate([prompt], sp)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        timings.append(ms)

    prompt_len = len(llm.tokenizer.encode(prompt, add_special_tokens=True))
    prefill_ms = sum(timings) / len(timings)
    prefill_tok_s = prompt_len / (prefill_ms / 1000)

    return {
        "prompt_len": prompt_len,
        "prefill_ms": round(prefill_ms, 2),
        "prefill_tok_s": round(prefill_tok_s, 1),
        "runs": timings,
    }


def bench_decode(llm, prompt, max_tokens, num_runs, warmup_runs):
    """测量 decode 吞吐 (固定 prefill 后只测 decode)。"""
    from engine.sampling_params import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # Warmup
    for _ in range(warmup_runs):
        llm.generate([prompt], sp)

    timings = []
    itl_list = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = llm.generate([prompt], sp)
        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        completion_tokens = len(outputs[0]["token_ids"])
        prompt_len = len(outputs[0]["prompt_token_ids"])

        # 粗估 prefill 开销 (prefill 吞吐约 2000-4000 tok/s)
        prefill_est_ms = prompt_len / 3000 * 1000
        decode_ms = max(total_ms - prefill_est_ms, 1)
        decode_tok_s = completion_tokens / (decode_ms / 1000)
        mean_itl = decode_ms / completion_tokens

        timings.append(decode_tok_s)
        itl_list.append(mean_itl)

    return {
        "prompt_len": prompt_len,
        "decode_tokens": max_tokens,
        "decode_tok_s_mean": round(sum(timings) / len(timings), 1),
        "decode_tok_s_median": round(sorted(timings)[len(timings) // 2], 1),
        "decode_tok_s_min": round(min(timings), 1),
        "decode_tok_s_max": round(max(timings), 1),
        "mean_itl_ms": round(sum(itl_list) / len(itl_list), 2),
    }


def bench_fused_lm_head(model_runner, num_runs=50):
    """直接对比 fused lm_head+argmax vs 原始 lm_head+argmax。"""
    from kernels.lm_head_topk import fused_lm_head_argmax, fused_lm_head_argmax_with_max
    from engine.parallel import tp_distributed_argmax, tp_distributed_argmax_fused

    hidden_size = model_runner.hidden_size
    weight = model_runner.model.lm_head.weight
    vocab_shard = weight.shape[0]
    dtype = weight.dtype
    device = weight.device

    results = {}

    # 测试不同 batch size
    for bsz in [1, 4, 8, 16, 32, 64]:
        hidden = torch.randn(bsz, hidden_size, dtype=dtype, device=device)

        # Warmup
        for _ in range(5):
            _ = fused_lm_head_argmax(hidden, weight)
        torch.cuda.synchronize()

        # --- 原始路径: lm_head -> argmax ---
        ms_orig = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            logits = torch.nn.functional.linear(hidden, weight)
            if is_tp_active():
                # 模拟 all_gather
                import torch.distributed as dist
                from engine.parallel import get_tp_group
                gathered = [torch.empty_like(logits) for _ in range(get_tp_world_size())]
                dist.all_gather(gathered, logits, group=get_tp_group())
                logits = torch.cat(gathered, dim=-1)
            tokens_orig = logits.argmax(dim=-1)
            end.record()
            torch.cuda.synchronize()
            ms_orig.append(start.elapsed_time(end))

        # --- Fused 路径 ---
        ms_fused = []
        for _ in range(5):
            _ = fused_lm_head_argmax(hidden, weight)
        torch.cuda.synchronize()

        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if is_tp_active():
                local_idx, local_max = fused_lm_head_argmax_with_max(hidden, weight)
                tokens_fused = tp_distributed_argmax_fused(local_idx, local_max, vocab_shard)
            else:
                tokens_fused = fused_lm_head_argmax(hidden, weight)
            end.record()
            torch.cuda.synchronize()
            ms_fused.append(start.elapsed_time(end))

        results[f"bsz_{bsz}"] = {
            "orig_ms": round(sum(ms_orig) / len(ms_orig), 3),
            "fused_ms": round(sum(ms_fused) / len(ms_fused), 3),
            "speedup": round(
                (sum(ms_orig) / len(ms_orig)) / (sum(ms_fused) / len(ms_fused)), 2
            ),
        }

    return results


def bench_vram():
    """测量峰值显存。"""
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    reserved = torch.cuda.max_memory_reserved() / (1024**3)
    return {
        "peak_allocated_gb": round(peak, 2),
        "peak_reserved_gb": round(reserved, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="优化效果基准测试")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--tp-size", type=int, default=2, help="TP size")
    parser.add_argument("--max-tokens", type=int, default=256, help="Decode token 数")
    parser.add_argument("--num-runs", type=int, default=10, help="每项测试运行次数")
    parser.add_argument("--warmup-runs", type=int, default=3, help="预热次数")
    parser.add_argument("--enforce-eager", action="store_true", help="禁用 CUDA Graph")
    parser.add_argument("--skip-lm-head-bench", action="store_true", help="跳过 fused lm_head 微基准")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 文件")
    args = parser.parse_args()

    from engine.llm import LLM
    from engine.sampling_params import SamplingParams

    prompt_short = "Explain quantum computing in simple terms."
    prompt_long = (
        "Write a detailed essay about the history of artificial intelligence, "
        "covering its origins in the 1950s, the AI winters, the deep learning "
        "revolution, and future prospects. Include key figures, milestones, "
        "and technological breakthroughs that shaped the field. "
    ) * 4  # 约 800 tokens

    # 初始化引擎
    if is_rank0():
        print(f"\n{'='*70}")
        print(f"  优化基准测试 | TP={args.tp_size} | "
              f"{'Eager' if args.enforce_eager else 'CUDA Graph'}")
        print(f"{'='*70}")

    llm = LLM(
        model=args.model,
        tp_size=args.tp_size,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
        max_model_len=4096,
    )

    results = {
        "config": {
            "model": args.model,
            "tp_size": args.tp_size,
            "enforce_eager": args.enforce_eager,
            "max_tokens": args.max_tokens,
            "num_runs": args.num_runs,
        },
    }

    # 1. Prefill 吞吐
    if is_rank0():
        print(f"\n--- Prefill 吞吐 (短 prompt) ---")
    r = bench_prefill(llm, prompt_short, SamplingParams(max_tokens=1),
                      args.num_runs, args.warmup_runs)
    if is_rank0():
        print(f"  Prompt len: {r['prompt_len']} tokens")
        print(f"  Prefill: {r['prefill_ms']:.2f} ms ({r['prefill_tok_s']:.0f} tok/s)")
    results["prefill_short"] = r

    if is_rank0():
        print(f"\n--- Prefill 吞吐 (长 prompt ~800 tokens) ---")
    r = bench_prefill(llm, prompt_long, SamplingParams(max_tokens=1),
                      args.num_runs, args.warmup_runs)
    if is_rank0():
        print(f"  Prompt len: {r['prompt_len']} tokens")
        print(f"  Prefill: {r['prefill_ms']:.2f} ms ({r['prefill_tok_s']:.0f} tok/s)")
    results["prefill_long"] = r

    # 2. Decode 吞吐
    if is_rank0():
        print(f"\n--- Decode 吞吐 ({args.max_tokens} tokens, 短 prompt) ---")
    r = bench_decode(llm, prompt_short, args.max_tokens, args.num_runs, args.warmup_runs)
    if is_rank0():
        print(f"  Decode: {r['decode_tok_s_mean']:.1f} tok/s (median: {r['decode_tok_s_median']:.1f})")
        print(f"  ITL: {r['mean_itl_ms']:.2f} ms/token")
    results["decode_short"] = r

    if is_rank0():
        print(f"\n--- Decode 吞吐 ({args.max_tokens} tokens, 长 prompt) ---")
    r = bench_decode(llm, prompt_long, args.max_tokens, args.num_runs, args.warmup_runs)
    if is_rank0():
        print(f"  Decode: {r['decode_tok_s_mean']:.1f} tok/s (median: {r['decode_tok_s_median']:.1f})")
        print(f"  ITL: {r['mean_itl_ms']:.2f} ms/token")
    results["decode_long"] = r

    # 3. Fused lm_head 微基准
    if not args.skip_lm_head_bench:
        if is_rank0():
            print(f"\n--- Fused LM_Head+Argmax vs 原始 LM_Head+Argmax ---")
            print(f"  {'Batch':>6s} | {'原始(ms)':>10s} | {'Fused(ms)':>10s} | {'加速比':>8s}")
            print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

        r = bench_fused_lm_head(llm.model_runner, num_runs=50)
        for key, val in r.items():
            bsz = key.split("_")[1]
            if is_rank0():
                print(f"  {bsz:>6s} | {val['orig_ms']:>10.3f} | {val['fused_ms']:>10.3f} | "
                      f"{val['speedup']:>7.2f}x")
        results["fused_lm_head"] = r

    # 4. 显存
    vram = bench_vram()
    if is_rank0():
        print(f"\n--- 显存 (Rank {get_tp_rank()}) ---")
        print(f"  Peak allocated: {vram['peak_allocated_gb']:.2f} GB")
        print(f"  Peak reserved:  {vram['peak_reserved_gb']:.2f} GB")
    results["vram"] = vram

    barrier()

    # 保存结果
    if is_rank0() or not is_tp_active():
        output_file = args.output or (
            f"bench_opt_tp{args.tp_size}_"
            f"{'eager' if args.enforce_eager else 'graph'}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 {output_file}")

    # 总结
    if is_rank0():
        print(f"\n{'='*70}")
        print(f"  总结 (TP={args.tp_size}, {'Eager' if args.enforce_eager else 'CUDA Graph'})")
        print(f"{'='*70}")
        print(f"  Prefill 吞吐:  {results['prefill_short']['prefill_tok_s']:.0f} tok/s (短) / "
              f"{results['prefill_long']['prefill_tok_s']:.0f} tok/s (长)")
        print(f"  Decode 吞吐:   {results['decode_short']['decode_tok_s_mean']:.1f} tok/s (短) / "
              f"{results['decode_long']['decode_tok_s_mean']:.1f} tok/s (长)")
        if not args.skip_lm_head_bench:
            bsz1 = results["fused_lm_head"]["bsz_1"]
            bsz64 = results["fused_lm_head"]["bsz_64"]
            print(f"  Fused lm_head:  {bsz1['speedup']:.2f}x (bsz=1) / "
                  f"{bsz64['speedup']:.2f}x (bsz=64)")
        print(f"  峰值显存:     {vram['peak_allocated_gb']:.2f} GB")
        print()


if __name__ == "__main__":
    main()
