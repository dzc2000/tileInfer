"""Qwen3.6 Inference Engine - Top-level API.

Supports single-GPU and multi-GPU (Tensor Parallelism) inference.

Usage (single GPU):
    from engine.llm import LLM, SamplingParams
    llm = LLM("/path/to/Qwen3.6-27B")
    outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=256))

Usage (multi-GPU, TP=2):
    python -m torch.distributed.launch --nproc_per_node=2 run_engine.py \\
        --model /path/to/Qwen3.6-27B --tp-size 2
"""
from engine.llm_engine import LLMEngine
from engine.sampling_params import SamplingParams


class LLM(LLMEngine):
    """High-level API for Qwen3.6-27B inference engine.

    For multi-GPU inference, launch with torch.distributed:
        torchrun --nproc_per_node=2 run_engine.py --model ... --tp-size 2
    """
    pass
