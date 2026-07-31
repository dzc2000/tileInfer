"""Sampling parameters for generation requests."""
from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    ignore_eos: bool = False
    repetition_penalty: float = 1.0  # 1.0 = no penalty
    stop: list = field(default_factory=list)  # stop strings (generation halts when any is matched)
    n: int = 1  # number of completions to generate per prompt

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
