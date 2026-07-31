"""
Qwen3.6-27B Inference Engine Configuration.

Adapted from nano-vllm for the hybrid DeltaNet + GQA architecture.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Config:
    model: str
    # Scheduling
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 64
    max_model_len: int = 32768
    # Memory
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1  # auto-computed
    swap_space_bytes: int = 0  # CPU offload for SWAP preemption (0 = disabled)
    # Model
    enforce_eager: bool = False
    # CUDA Graph: max decode batch size captured (powers-of-two buckets up to this)
    cuda_graph_max_batch_size: int = 256
    hf_config: object = None
    eos: list = None  # populated from hf_config in __post_init__
    # Chunked prefill
    max_prefill_chunk_tokens: int = 4096
    # Tensor parallelism
    tp_size: int = 1
    # NCCL
    nccl_timeout: float = 600.0  # seconds

    def __post_init__(self):
        from transformers import AutoConfig
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        # Qwen3.6 is multimodal (Qwen3_5Config); text model config is nested in text_config.
        # Unwrap it so downstream code can access num_hidden_layers, hidden_size, etc. directly.
        if hasattr(self.hf_config, 'text_config'):
            self.hf_config = self.hf_config.text_config
        max_pos = getattr(self.hf_config, 'max_position_embeddings', 32768)
        self.max_model_len = min(self.max_model_len, max_pos)
        # Support multiple EOS tokens (Qwen3.6 has <|im_end|>, <|endoftext|>, etc.)
        eos_id = getattr(self.hf_config, 'eos_token_id', 151645)
        if isinstance(eos_id, list):
            self.eos = eos_id
        else:
            self.eos = [eos_id]
        self._validate_tp_size()

    def _validate_tp_size(self):
        """Ensure tp_size evenly divides every sharded dimension.

        A tp_size that does not divide a head/intermediate count would produce a
        zero-sized shard (e.g. num_key_value_heads=4 with tp_size=8 -> 0 KV heads
        per rank) and fail silently or crash deep inside the kernels. Fail fast
        with a clear message instead.
        """
        if self.tp_size <= 1:
            return
        sharded = {
            "num_attention_heads": getattr(self.hf_config, 'num_attention_heads', 24),
            "num_key_value_heads": getattr(self.hf_config, 'num_key_value_heads', 4),
            "deltanet_num_k_heads": getattr(self.hf_config, 'linear_num_key_heads', 16),
            "deltanet_num_v_heads": getattr(self.hf_config, 'linear_num_value_heads', 48),
            "intermediate_size": getattr(self.hf_config, 'intermediate_size', 17408),
        }
        for name, value in sharded.items():
            if value % self.tp_size != 0:
                raise ValueError(
                    f"tp_size={self.tp_size} does not evenly divide {name}={value}. "
                    f"Choose a tp_size that divides all of: {sharded}."
                )
