# tileInfer

A high-performance inference engine for **Qwen3.6-27B**, built on custom TileLang kernels and vLLM-style serving features. Supports both Hopper (sm90+) and Ampere (sm80) GPUs.

## Features

| Feature | Description |
|---------|-------------|
| **Fused TileLang Kernels** | Custom CUDA kernels for RMSNorm, RoPE, SiLU, GQA attention, DeltaNet recurrent states, and more |
| **DeltaNet Prefill (dual backend)** | FlashQLA `chunk_gated_delta_rule` on sm90+; TileLang chunked gated delta rule kernel on sm80 (A100) |
| **Paged KV Cache** | Block-based KV cache management, inspired by vLLM |
| **Continuous Batching** | Dynamic batching of requests with different sequence lengths |
| **Chunked Prefill** | Split long prefill sequences into chunks to avoid memory spikes |
| **Tensor Parallelism** | Multi-GPU inference via column/row parallel weight sharding and All-Reduce |
| **CUDA Graph** | Decode step capture for minimal kernel launch overhead |
| **Kernel Disk Cache** | TileLang kernels are compiled once and cached to `.tilelang_cache/` for fast subsequent startups |

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
# Clone the repository
git clone <repo-url>
cd tileInfer

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Install FlashQLA for DeltaNet prefill on Hopper GPUs (sm90+)
# On Ampere (A100, sm80), the built-in TileLang kernel is used automatically.
git clone https://github.com/QwenLM/FlashQLA.git
cd FlashQLA
pip install -v .
cd ..

# Install tileInfer
pip install -e .
```

> **Note:** TileLang kernels are compiled on first run and cached to `.tilelang_cache/`.
> Subsequent runs load from cache, reducing startup time from ~90s to ~10s.

## Quick Start

### Single GPU

```bash
python run_engine.py \
    --model /path/to/Qwen3.6-27B \
    --prompt "Explain quantum computing in simple terms." \
    --max-tokens 256
```

### Multi-GPU (Tensor Parallelism)

```bash
# 2 GPUs
torchrun --nproc_per_node=2 run_engine.py \
    --model /path/to/Qwen3.6-27B \
    --tp-size 2 \
    --prompt "Hello, world!"

# 4 GPUs
torchrun --nproc_per_node=4 run_engine.py \
    --model /path/to/Qwen3.6-27B \
    --tp-size 4 \
    --prompt "Hello, world!"
```

### Python API

```python
from engine.llm import LLM
from engine.sampling_params import SamplingParams

# Single GPU
llm = LLM(model="/path/to/Qwen3.6-27B")

# Multi-GPU
llm = LLM(model="/path/to/Qwen3.6-27B", tp_size=2)

params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, world!"], params)
print(outputs[0]["text"])
```

## Project Structure

```
tileInfer/
├── engine/                 # Inference engine core
│   ├── llm_engine.py       # Main orchestrator (scheduler + model runner)
│   ├── llm.py              # High-level API
│   ├── model_runner.py     # Model loading, KV cache allocation
│   ├── scheduler.py        # Continuous batching scheduler
│   ├── block_manager.py    # Paged KV cache block manager
│   ├── parallel.py         # Tensor parallelism (NCCL All-Reduce, weight sharding)
│   ├── config.py           # Engine configuration
│   ├── sampling_params.py  # Generation parameters
│   ├── sequence.py         # Sequence state management
│   └── context.py          # Execution context for layer forward
├── model/
│   ├── qwen_36.py          # Qwen3.6 model patcher (fused forward, TP support)
│   └── generate.py         # Manual decode loop with CUDA graph
├── kernels/                # TileLang custom CUDA kernels
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
│   ├── splitk_gemv.py      # Split-K GEMV for linear layers
│   ├── lm_head_topk.py     # Fused LM head + top-k sampling
│   └── varlen_utils.py     # Variable-length sequence utilities
├── tests/                  # Benchmarks
│   ├── bench_single_gpu.py     # Single-GPU throughput vs HF baseline
│   ├── bench_multi_gpu.py      # Multi-GPU TP scaling efficiency
│   ├── bench_prefill.py        # Prefill latency across sequence lengths
│   ├── bench_batched_decode.py # Continuous batching throughput
│   └── run_all_benchmarks.py   # Run all benchmarks and aggregate
├── run_engine.py           # CLI entry point
└── setup.py                # Package setup
```

## Benchmarks

Run all benchmarks with a single command:

```bash
python tests/run_all_benchmarks.py --model /path/to/Qwen3.6-27B
```

Or run individual benchmarks:

```bash
# Single-GPU decode throughput (requires model fits in one GPU)
python tests/bench_single_gpu.py --model /path/to/Qwen3.6-27B

# Multi-GPU TP scaling (use torchrun for TP > 1)
torchrun --nproc_per_node=2 tests/bench_multi_gpu.py --model /path/to/Qwen3.6-27B --tp-size 2
torchrun --nproc_per_node=4 tests/bench_multi_gpu.py --model /path/to/Qwen3.6-27B --tp-size 4

# Prefill throughput across sequence lengths
torchrun --nproc_per_node=2 tests/bench_prefill.py --model /path/to/Qwen3.6-27B

# Continuous batching throughput
torchrun --nproc_per_node=2 tests/bench_batched_decode.py --model /path/to/Qwen3.6-27B
```

## Optimizations

1. **Weight Deduplication** — Replace original params with views into concatenated weights, saving ~30GB VRAM
2. **Weight Pre-concatenation** — 4->1 GEMV for DeltaNet, 2->1 for MLP, 3->1 for attention
3. **Fused QK-norm + RoPE** — ~18 PyTorch ops -> 2 CUDA launches per attention layer
4. **Fused SiLU Gate** — 4 ops -> 1 kernel per attention layer
5. **Fused Residual + RMSNorm** — Eliminates 64 separate add kernels per token
6. **Fused DeltaNet Post-projection** — 17 ops -> 1 kernel per layer (decode)
7. **Chunked Gated Delta Rule** — TileLang kernel for DeltaNet prefill on sm80 (A100), FlashQLA on sm90+
8. **TileLang GQA Flash Attention** — Variable-length attention for full attention layers
9. **Tensor Parallelism** — Column/row parallel weight sharding for multi-GPU scaling

## Configuration

```python
from engine.llm import LLM

llm = LLM(
    model="/path/to/Qwen3.6-27B",
    max_num_seqs=512,          # Max concurrent sequences
    max_model_len=4096,        # Max sequence length
    gpu_memory_utilization=0.9, # GPU memory fraction for KV cache
    enforce_eager=True,         # Disable CUDA graph (for debugging)
    tp_size=1,                  # Tensor parallelism size
)
```

## Tensor Parallelism

| TP Size | Q heads/rank | KV heads/rank | K heads/rank | V heads/rank | Intermediate/rank | Weight/rank |
|---------|-------------|---------------|--------------|--------------|-------------------|-------------|
| 1 | 24 | 4 | 16 | 48 | 17408 | ~54GB |
| 2 | 12 | 2 | 8 | 24 | 8704 | ~27GB |
| 4 | 6 | 1 | 4 | 12 | 4352 | ~13.5GB |
| 8 | 3 | 1 | 2 | 6 | 2176 | ~6.8GB |

## Requirements

See [requirements.txt](requirements.txt) for Python dependencies.

- Python 3.10+
- CUDA 11.8+ or 12.1+
- PyTorch 2.0+ (with CUDA)
- transformers >= 4.40.0
- tilelang >= 0.1.0
- FlashQLA (optional, sm90+ only; sm80 uses built-in TileLang kernel)

### Hardware

| GPU | TP Size | VRAM/rank | Notes |
|-----|---------|-----------|-------|
| A100 40GB | 2 | ~27GB | Uses TileLang DeltaNet prefill kernel |
| A100 80GB | 1 | ~54GB | Uses TileLang DeltaNet prefill kernel |
| H100 80GB | 1 | ~54GB | Uses FlashQLA (if installed) |
| H100 80GB | 2 | ~27GB | Uses FlashQLA (if installed) |

## License

MIT
