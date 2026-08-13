"""
Qwen3.6-27B Inference Engine - Entry Point.

Single GPU:
    python run_engine.py --model /path/to/Qwen3.6-27B --prompt "Hello, world!"

Multi-GPU (TP=2):
    torchrun --nproc_per_node=2 run_engine.py --model /path/to/Qwen3.6-27B --tp-size 2

Multi-GPU (TP=4):
    torchrun --nproc_per_node=4 run_engine.py --model /path/to/Qwen3.6-27B --tp-size 4
"""
import os
os.environ.setdefault(
    "TILELANG_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tilelang_cache"),
)

import argparse
import time

from engine.llm import LLM
from engine.sampling_params import SamplingParams
from engine.parallel import get_tp_rank


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-27B Inference Engine")
    parser.add_argument("--model", type=str, required=True, help="Path to Qwen3.6-27B model")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallelism size (number of GPUs)")
    parser.add_argument("--num-scheduler-steps", type=int, default=1,
                        help="Multi-step scheduling: decode steps per scheduler invocation (1 = disabled)")
    parser.add_argument("--no-prefix-caching", action="store_true",
                        help="Disable prefix caching")
    parser.add_argument("--guided-json", action="store_true",
                        help="Constrain output to valid JSON")
    parser.add_argument("--guided-regex", type=str, default=None,
                        help="Constrain output to match a regular expression")
    parser.add_argument("--guided-choice", type=str, default=None,
                        help="Comma-separated choices to constrain output to one of")
    args = parser.parse_args()

    # Initialize engine
    print(f"Initializing engine with model: {args.model}, tp_size: {args.tp_size}")
    llm = LLM(
        model=args.model,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        tp_size=args.tp_size,
        num_scheduler_steps=args.num_scheduler_steps,
        enable_prefix_caching=not args.no_prefix_caching,
    )

    # Set sampling params
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        guided_json=True if args.guided_json else None,
        guided_regex=args.guided_regex,
        guided_choice=(args.guided_choice.split(",")
                       if args.guided_choice else []),
    )

    # Generate (only rank 0 prints results)
    if get_tp_rank() == 0:
        print(f"\nPrompt: {args.prompt}\n")
    start = time.time()
    outputs = llm.generate([args.prompt], params)
    elapsed = time.time() - start

    if get_tp_rank() == 0:
        for output in outputs:
            print(f"Generated: {output['text']}")
            print(f"  Prompt tokens: {len(output['prompt_token_ids'])}, "
                  f"Completion tokens: {len(output['token_ids'])}")

        print(f"\nTotal time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
