# tileInfer

A high-performance inference engine for **Qwen3.6-27B**, built on custom TileLang kernels with INT8 weight-only quantization. Achieves **36.6 tok/s** decode throughput on 2× A100 40GB (TP=2).

## Features

| Feature | Description |
|---------|-------------|
| **INT8 Weight-Only Quantization** | Per-channel W8A16 quantization, 2× weight-memory reduction. Decode uses register-level dequantize GEMV; prefill dequantizes to BF16 on the fly for GEMM |
| **BitBLAS-style GEMV Kernel** | Register-level dequantize GEMV with `T.vectorized` 128-bit loads and warp shuffle reduction |
| **Fused TileLang Kernels** | Custom kernels for RMSNorm, RoPE, SiLU, GQA attention, DeltaNet recurrent states |
| **Mixed Batching** | Decode + prefill sequences coexist in one step (serial prefill, parallel decode) |
| **Paged KV Cache** | Block-based KV cache management with RECOMPUTE preemption |
| **Continuous Batching** | Dynamic batching with iterative scheduler (no recursion) |
| **Chunked Prefill** | Split long prefill sequences into chunks to avoid memory spikes |
| **Prefix Caching** | Reuse KV-cache blocks + DeltaNet recurrent state across requests sharing a token prefix |
| **Constrained Decoding** | `logit_bias`, allow/deny token lists, guided choice / JSON / regex |
| **Multi-Step Scheduling** | Run several decode steps per scheduler invocation to amortize Python overhead |
| **Tensor Parallelism** | Multi-GPU inference via column/row parallel weight sharding and All-Reduce |
| **CUDA Graph** | Decode step capture for minimal kernel launch overhead |
| **Streaming API** | `generate_stream()` for token-by-token output |
| **OpenAI-Compatible Server** | `/v1/chat/completions` with SSE streaming support |
| **Distributed Sampling** | Top-k All-Gather instead of full-vocab communication (99% reduction) |
| **Kernel Disk Cache** | TileLang kernels compiled once and cached to `.tilelang_cache/` |

## Performance

| Configuration | BF16 | INT8 W8A16 |
|---------------|------|-----------|
| 2× A100 40GB, TP=2 | 25.5 tok/s | **36.6 tok/s (+43%)** |
| VRAM per rank | 26.3 GB | 14.9 GB |
| ITL (inter-token latency) | 39.2 ms | 27.3 ms |

## Architecture

Qwen3.6-27B uses a hybrid architecture:
- **48 DeltaNet layers** (linear attention with recurrent state)
- **16 full attention layers** (GQA with KV cache)
- **24 Q heads / 4 KV heads** (group ratio = 6)

```
Input
  |
  v
[Embed + RoPE]
  |
  v
[Layer 0-47: DeltaNet]  --> chunk_gated_delta_rule (prefill)
  |                          FlashQLA (sm90+) or TileLang kernel (sm80)
  |                          fused causal_conv1d + recurrent (decode)
  v
[Layer 48-63: GQA Attention] --> TileLang flashattn varlen (prefill)
  |                               TileLang GQA decode with paged KV cache
  v
[Norm + LM Head]
  |
  v
Output
```

## Installation

```bash
git clone git@github.com:<user>/tileInfer.git
cd tileInfer

pip install -r requirements.txt

# (Optional) FlashQLA for DeltaNet prefill on Hopper GPUs (sm90+)
git clone https://github.com/QwenLM/FlashQLA.git
cd FlashQLA && pip install -v . && cd ..

pip install -e .
```

> **Note:** TileLang kernels are compiled on first run and cached to `.tilelang_cache/`.
> Subsequent runs load from cache, reducing startup time from ~90s to ~10s.

## Quick Start

### CLI

```bash
# Single GPU
python run_engine.py --model /path/to/Qwen3.6-27B --prompt "Hello" --max-tokens 256

# Multi-GPU (TP=2)
torchrun --nproc_per_node=2 run_engine.py --model /path/to/Qwen3.6-27B --tp-size 2 --prompt "Hello"
```

### Python API

```python
from engine.llm import LLM
from engine.sampling_params import SamplingParams

llm = LLM(model="/path/to/Qwen3.6-27B", tp_size=2)
params = SamplingParams(temperature=0.6, max_tokens=256)

# Batch generation
outputs = llm.generate(["Hello", "Tell me a joke"], params)
for o in outputs:
    print(o["text"])

# Streaming
for idx, text, done in llm.generate_stream(["Hello"], params):
    print(text, end="", flush=True)
```

### OpenAI-Compatible API Server

```bash
torchrun --nproc_per_node=2 api_server.py --model /path/to/Qwen3.6-27B --tp-size 2 --port 8000
```

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'

# Streaming (SSE)
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hi"}],"stream":true}'

# List models
curl http://localhost:8000/v1/models
```

## Project Structure

```
tileInfer/
├── engine/                 # Inference engine core
│   ├── llm_engine.py       # Main orchestrator (scheduler + model runner)
│   ├── llm.py              # High-level API (generate / generate_stream)
│   ├── model_runner.py     # Model loading, KV cache, CUDA Graph, INT8 quantization
│   ├── scheduler.py        # Mixed batching scheduler (iterative state machine)
│   ├── block_manager.py    # Paged KV cache block manager with SWAP support
│   ├── parallel.py         # Tensor parallelism (NCCL All-Reduce, weight sharding)
│   ├── config.py           # Engine configuration
│   ├── sampling_params.py  # Generation parameters (temperature, top_k, stop, rep_penalty)
│   ├── sequence.py         # Sequence state management
│   └── context.py          # Execution context for layer forward
├── model/
│   └── qwen_36.py          # Qwen3.6 model (ModelState, fused forward, TP, config-driven)
├── kernels/                # TileLang custom CUDA kernels
│   ├── splitk_gemv.py      # INT8/BF16 GEMV with online dequantization
│   ├── rms_norm.py         # Fused RMSNorm
│   ├── qknorm_rope.py      # Fused QK-norm + RoPE
│   ├── act_mul.py          # Fused SiLU + elementwise multiply
│   ├── residual_rmsnorm.py # Fused residual + RMSNorm
│   ├── gqa_varlen.py       # GQA flash attention (variable length, prefill)
│   ├── gqa_decode_paged.py # GQA decode with paged KV cache
│   ├── kvcache_store.py    # KV cache store kernel
│   ├── causal_conv1d.py    # Causal 1D convolution for DeltaNet
│   ├── deltanet_fused.py   # Fused DeltaNet post-projection + recurrent (decode)
│   ├── gated_delta_rule_prefill.py  # Chunked gated delta rule (sm80 prefill)
│   ├── lm_head_topk.py     # Fused LM head + top-k sampling
│   └── varlen_utils.py     # Variable-length sequence utilities
├── api_server.py           # OpenAI-compatible HTTP server (SSE streaming)
├── tests/                  # Benchmarks & correctness tests
│   ├── bench_multi_gpu.py      # Multi-GPU TP scaling benchmark
│   ├── bench_single_gpu.py     # Single-GPU throughput benchmark
│   ├── bench_prefill.py        # Prefill latency across sequence lengths
│   ├── bench_batched_decode.py # Continuous batching throughput
│   ├── profile_decode.py       # CUDA profiling for decode path
│   ├── test_correctness.py     # Output correctness verification
│   ├── test_features.py        # Feature tests (stop, rep_penalty, multi-prompt)
│   └── test_oom.py             # OOM recovery / preemption test
├── run_engine.py           # CLI entry point
└── setup.py                # Package setup
```

## Optimizations

### Quantization
1. **INT8 Weight-Only (W8A16)** — Per-channel quantization, 2× weight-memory reduction. Decode (M=1) runs the BitBLAS-style dequantize GEMV kernel; prefill (M>1) falls back to BF16 GEMM with on-the-fly dequantization
2. **BitBLAS-style GEMV** — Pure register (`alloc_local`) design, no shared memory sync, warp shuffle reduction, `T.vectorized(8)` for 128-bit loads

### Kernel Fusion
3. **Fused QK-norm + RoPE** — ~18 PyTorch ops → 2 CUDA launches per attention layer
4. **Fused SiLU Gate** — 4 ops → 1 kernel per layer
5. **Fused Residual + RMSNorm** — Eliminates 64 separate add kernels per token
6. **Fused DeltaNet Post-projection** — 17 ops → 1 kernel per layer (decode)
7. **Fused LM Head + Top-k** — Single kernel for logits computation and sampling

### Scheduling
8. **Mixed Batching** — Decode + prefill in same step, iterative scheduler (no recursion)
9. **Serial Prefill** — One prefill per step for DeltaNet state compatibility
10. **RECOMPUTE Preemption** — Under memory pressure, release a sequence's blocks and re-prefill it later (no CPU swap space involved)
11. **Distributed Top-k Sampling** — Only gather top-k logits across TP ranks (vs full vocab)

### Infrastructure
12. **CUDA Graph** — Decode capture with static buffers, O(log n) bucket selection
13. **Weight Deduplication** — Views into concatenated weights, saving ~30GB VRAM
14. **Config-driven Architecture** — All model constants read from HuggingFace config
15. **ModelState Encapsulation** — No global mutable state, supports unpatch/repatch

## Configuration

```python
from engine.llm import LLM

llm = LLM(
    model="/path/to/Qwen3.6-27B",
    tp_size=2,                  # Tensor parallelism size
    max_num_seqs=64,            # Max concurrent sequences
    max_model_len=32768,        # Max sequence length
    gpu_memory_utilization=0.9, # GPU memory fraction for KV cache
    enforce_eager=False,        # Set True to disable CUDA graph (debugging)
    enable_prefix_caching=True, # Reuse KV/DeltaNet state across shared prefixes
    num_scheduler_steps=1,      # Decode steps per scheduler invocation (multi-step)
)
```

## Prefix Caching / Constrained Decoding / Multi-Step Scheduling

```python
from engine.llm import LLM
from engine.sampling_params import SamplingParams

llm = LLM(model="/path/to/Qwen3.6-27B", enable_prefix_caching=True,
          num_scheduler_steps=8)

# Prefix caching: requests sharing a token prefix (e.g. a system prompt) reuse
# cached KV blocks and DeltaNet recurrent state instead of re-prefilling.
for prompt in ["<system>...</system> question A",
               "<system>...</system> question B"]:
    print(llm.generate([prompt], SamplingParams(max_tokens=64))[0]["text"])

# Constrained decoding — guided JSON (valid JSON output only).
json_out = llm.generate(
    ["Return a JSON object"],
    SamplingParams(max_tokens=128, guided_json=True),
)[0]["text"]

# Guided choice.
choice_out = llm.generate(
    ["Classify the sentiment"],
    SamplingParams(max_tokens=8, guided_choice=["positive", "negative"]),
)[0]["text"]

# Guided regex (supported subset: literals, classes, quantifiers, alternation).
regex_out = llm.generate(
    ["Write a phone number"],
    SamplingParams(max_tokens=16, guided_regex=r"\d{3}-\d{4}"),
)[0]["text"]

# logit_bias / allowed / denied token ids.
bias_out = llm.generate(
    ["Hello"],
    SamplingParams(max_tokens=64, logit_bias={1234: 5.0},
                   bad_token_ids=[151645]),
)[0]["text"]
```

The OpenAI-compatible server exposes the same knobs via request fields
(`guided_json`, `guided_regex`, `guided_choice`, `logit_bias`,
`allowed_token_ids`, `bad_token_ids`) and maps
`response_format: {"type": "json_object"}` to JSON mode automatically.

## Tensor Parallelism

| TP Size | Q heads/rank | KV heads/rank | K heads/rank | V heads/rank | Weight/rank |
|---------|-------------|---------------|--------------|--------------|-------------|
| 1 | 24 | 4 | 16 | 48 | ~54 GB |
| 2 | 12 | 2 | 8 | 24 | ~27 GB |
| 4 | 6 | 1 | 4 | 12 | ~13.5 GB |

## Benchmarks

```bash
# Multi-GPU decode throughput
torchrun --nproc_per_node=2 tests/bench_multi_gpu.py \
    --model /path/to/Qwen3.6-27B --tp-size 2 --max-tokens 256 --num-runs 10

# CUDA profiling
torchrun --nproc_per_node=2 tests/profile_decode.py \
    --model /path/to/Qwen3.6-27B --tp-size 2 --max-tokens 64

# Correctness tests
torchrun --nproc_per_node=2 tests/test_correctness.py
torchrun --nproc_per_node=2 tests/test_features.py
torchrun --nproc_per_node=2 tests/test_oom.py
```

## Requirements

- Python 3.10+
- CUDA 11.8+ or 12.1+
- PyTorch 2.0+ (with CUDA)
- transformers >= 4.40.0
- tilelang >= 0.1.0
- FlashQLA (optional, sm90+ only; sm80 uses built-in TileLang kernel)

### Hardware

| GPU | TP Size | VRAM/rank (INT8) | Decode Speed |
|-----|---------|-------------------|--------------|
| A100 40GB × 2 | 2 | ~15 GB | 36.6 tok/s |
| A100 80GB × 1 | 1 | ~27 GB | ~18 tok/s |
| H100 80GB × 2 | 2 | ~15 GB | ~55 tok/s (est.) |

## License

MIT
