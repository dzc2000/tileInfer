"""
Qwen3.6-27B Dense Model Patcher

Optimizations:
  1. Weight deduplication: replace original params with views into concatenated weights
     Saves ~30GB VRAM (81GB -> ~54GB), reduces HBM controller pressure
  2. Weight pre-concatenation: 4->1 GEMV for DeltaNet proj, 2->1 for MLP, 3->1 for attention
  3. Fused QK-norm + partial RoPE: replaces ~18 PyTorch ops with 2 CUDA launches
     per attention layer. Saves ~192 kernel executions over 16 layers.
  4. Fused sigmoid gate multiply: replaces 4 PyTorch ops with 1 CUDA kernel
     per attention layer. Saves 48 kernel executions.
  5. Fused residual+RMSNorm: eliminates 64 separate add kernels per token
  6. Fused DeltaNet post-projection + recurrent: 17 ops -> 1 CUDA kernel per layer
  7. Pre-computed constants: -exp(A_log), dt_bias.float() computed once at patch time
  8. FlashQLA for DeltaNet prefill (chunk_gated_delta_rule)
  9. TileLang GQA varlen flash attention for full attention prefill
  10. Tensor Parallelism support for multi-GPU inference

Monkey-patches layer.forward() for all 64 layers of Qwen3.6-27B:
  - 48 DeltaNet layers (Qwen3_6GatedDeltaNet) -> fused_deltanet_forward()
  - 16 full attention layers (Qwen3_6Attention) -> fused_attention_forward()

Qwen3.6-27B architecture (same as Qwen3.5):

  DeltaNet layer (linear_attn: Qwen3_6GatedDeltaNet):
    in_proj_qkv: [10240, 5120] -- fused Q(2048)+K(2048)+V(6144)
    in_proj_a:   [48, 5120]    -- alpha (num_v_heads=48)
    in_proj_b:   [48, 5120]    -- beta (num_v_heads=48)
    in_proj_z:   [6144, 5120]  -- gate z (value_dim=6144)
    A_log:       [48]          -- log-space decay
    dt_bias:     [48]          -- delta-time bias
    conv1d:      Conv1d(10240, 1, 4, groups=10240) -- depthwise
    norm:        RMSNormGated([128]) -- per head_v_dim
    out_proj:    [5120, 6144]  -- output

  Attention layer (self_attn: Qwen3_6Attention):
    q_proj:      [12288, 5120] -- 48*256 (24 heads * 256 * 2 for gating)
    k_proj:      [1024, 5120]  -- 4 KV heads * 256
    v_proj:      [1024, 5120]  -- 4 KV heads * 256
    o_proj:      [5120, 6144]  -- hidden_size x (24 * 256)
    q_norm:      [256]         -- per head dim
    k_norm:      [256]         -- per head dim

  Both layer types share:
    mlp.gate_proj: [17408, 5120]
    mlp.up_proj:   [17408, 5120]
    mlp.down_proj: [5120, 17408]
    input_layernorm: [5120]
    post_attention_layernorm: [5120]

Architecture constants and the DeltaNet layer pattern are read from the HF
config in patch_model(); the hardcoded values below are only fallback defaults
for Qwen3.6-27B used when the config omits a field.
"""
import gc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang.language as T

from kernels.rms_norm import rmsnorm
from kernels.residual_rmsnorm import fused_residual_rmsnorm
from kernels.deltanet_fused import fused_postproj_recurrent
from kernels.causal_conv1d import causal_conv1d_update_fn as causal_conv1d_update, causal_conv1d_prefill
from kernels.act_mul import fused_silu_mul, fused_sigmoid_mul
from kernels.qknorm_rope import fused_qknorm_rope
from kernels.splitk_gemv import bf16_linear_forward
from kernels.gqa_varlen import flashattn as tilelang_flashattn
from kernels.gqa_decode_paged import gqa_decode_paged_fn
from kernels.kvcache_store import store_kvcache
try:
    from flash_qla import chunk_gated_delta_rule
except (ImportError, ValueError):
    from kernels.gated_delta_rule_prefill import chunk_gated_delta_rule_tilelang as chunk_gated_delta_rule
from engine.parallel import (
    get_tp_rank, get_tp_world_size, is_tp_active,
    tp_all_reduce, tp_all_reduce_async, shard_weight_col, shard_weight_row, shard_bias_col,
)


# Default fallback DeltaNet pattern: [DeltaNet, DeltaNet, DeltaNet, Attention] x 16.
# Used only when the HF config does not expose ``layer_types``.
_DEFAULT_DELTANET_PATTERN = [True, True, True, False] * 16  # True = DeltaNet


class ModelState:
    """Encapsulates runtime state previously held in module-level globals.

    Holds the paged KV cache and the paged DeltaNet state pools.
    ``patch_model`` accepts an optional instance; the legacy
    ``set_paged_kv_cache`` / ``set_deltanet_pools`` helpers mutate a shared
    singleton so existing callers (model_runner.py, llm_engine.py) work
    unchanged.
    """
    __slots__ = (
        'paged_kv_cache',            # [2, num_attn_layers, num_blocks, block_size, num_kv_heads, head_dim]
        'deltanet_recurrent_pool',   # [num_dn_layers, max_slots, num_v_heads, Dk, Dv]
        'deltanet_conv_pool',        # [num_dn_layers, max_slots, conv_dim, kernel_size]
        'seq_to_slot',               # dict: seq_id -> slot
    )

    def __init__(self):
        self.paged_kv_cache = None
        self.deltanet_recurrent_pool = None
        self.deltanet_conv_pool = None
        self.seq_to_slot = None


# Shared singleton updated by the backward-compatible setters below. fused_forward
# closures capture this instance (or an explicit one passed to patch_model).
_default_state = ModelState()


def set_paged_kv_cache(kv_cache):
    """Set the paged KV cache reference (updates the default ModelState singleton)."""
    _default_state.paged_kv_cache = kv_cache


def set_deltanet_pools(recurrent_pool, conv_pool, seq_to_slot):
    """Set the paged DeltaNet state pools (updates the default ModelState singleton)."""
    _default_state.deltanet_recurrent_pool = recurrent_pool
    _default_state.deltanet_conv_pool = conv_pool
    _default_state.seq_to_slot = seq_to_slot


class _DeltaNetCacheView:
    """View into paged DeltaNet state for a specific batch."""
    __slots__ = ('conv_state', 'recurrent_state', 'dn_idx', 'slots')

    def __init__(self, conv_state, recurrent_state, dn_idx=None, slots=None):
        self.conv_state = conv_state
        self.recurrent_state = recurrent_state
        self.dn_idx = dn_idx
        self.slots = slots


def _get_deltanet_cache(state: ModelState, dn_idx: int, bsz: int,
                        tp_conv_dim: int, tp_num_v_heads: int,
                        head_k_dim: int, head_v_dim: int) -> _DeltaNetCacheView:
    """Get DeltaNet cache for this layer from the paged pool in ``state``.

    ``dn_idx`` is precomputed at patch time (O(1) lookup) rather than rescanned
    on every forward.
    """
    from engine.context import get_context
    ctx = get_context()
    if hasattr(ctx, 'deltanet_slots') and ctx.deltanet_slots is not None:
        slots = ctx.deltanet_slots  # [bsz] tensor of slot indices
        # Always use fancy indexing (tensor index) for CUDA Graph compatibility.
        # .item() would freeze the slot value during graph capture, causing all
        # replays to write to the same slot regardless of the actual sequence.
        # Fancy indexing returns a COPY; writeback is required after modification.
        conv_state = state.deltanet_conv_pool[dn_idx, slots]        # [bsz, conv_dim, kernel_size]
        recurrent_state = state.deltanet_recurrent_pool[dn_idx, slots]  # [bsz, num_v_heads, Dk, Dv]
    else:
        slots = None
        conv_state = torch.zeros(bsz, tp_conv_dim, 4, device="cuda", dtype=torch.bfloat16)
        recurrent_state = torch.zeros(bsz, tp_num_v_heads, head_k_dim, head_v_dim,
                                       device="cuda", dtype=torch.bfloat16)
    return _DeltaNetCacheView(conv_state, recurrent_state, dn_idx, slots)


def _writeback_deltanet_state(state: ModelState, dn_idx: int, slots, final_state):
    """Write back DeltaNet recurrent state to paged pool after prefill."""
    if state.deltanet_recurrent_pool is not None and slots is not None:
        state.deltanet_recurrent_pool[dn_idx, slots] = final_state.to(
            state.deltanet_recurrent_pool.dtype)


def quantize_weight_int8(w: torch.Tensor, align: int = 64):
    N, K = w.shape
    w_f32 = w.float()
    amax = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    scale = amax / 127.0
    w_int8 = (w_f32 / scale).round().clamp(-128, 127).to(torch.int8)
    scale = scale.squeeze(1).contiguous()
    N_padded = ((N + align - 1) // align) * align
    if N_padded > N:
        pad_n = N_padded - N
        w_int8 = torch.cat([w_int8, torch.zeros(pad_n, K, dtype=torch.int8, device=w.device)], dim=0)
        scale = torch.cat([scale, torch.ones(pad_n, dtype=torch.float32, device=w.device)], dim=0).contiguous()
    return w_int8, scale


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _compute_deltanet_pattern(config) -> list:
    """Derive the per-layer DeltaNet pattern from the HF config.

    Reads ``config.layer_types`` (values: 'linear_attention' / 'full_attention')
    when available; otherwise falls back to [DeltaNet, DeltaNet, DeltaNet, Attention] x 16.
    """
    layer_types = getattr(config, 'layer_types', None)
    if layer_types is not None:
        return [t == 'linear_attention' for t in layer_types]
    return list(_DEFAULT_DELTANET_PATTERN)


def is_deltanet_layer(layer_idx: int, pattern: Optional[list] = None) -> bool:
    """Return True if ``layer_idx`` is a DeltaNet layer.

    ``pattern`` defaults to the module fallback pattern for backward
    compatibility; ``patch_model`` passes the config-derived pattern explicitly.
    """
    if pattern is None:
        pattern = _DEFAULT_DELTANET_PATTERN
    return pattern[layer_idx]


def load_model(
    model_id: str = "Qwen/Qwen3.6-27B",
    cache_dir: str = "/cache/models",
    device: str = "cuda",
    hf_token: Optional[str] = None,
) -> nn.Module:
    """Load Qwen3.6-27B in BF16."""
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_id} in BF16 on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    vram_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"Model loaded: {vram_gb:.1f}GB VRAM")

    return model


def patch_model(model: nn.Module, device: str = "cuda",
                state: Optional[ModelState] = None) -> nn.Module:
    """Patch all layers with fused CUDA kernels.

    Args:
        model: The HuggingFace model (may be on CPU)
        device: Target device for sharded weights (e.g. "cuda:0")
        state: Optional ``ModelState`` holding the paged KV cache and DeltaNet
            pools. When ``None``, the shared default singleton (updated by
            ``set_paged_kv_cache`` / ``set_deltanet_pools``) is used so that
            fused_forward closures observe live cache references.
    """
    if state is None:
        state = _default_state

    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config

    # ===== Config-driven architecture constants (no longer hardcoded) =====
    hidden_size = config.hidden_size
    eps = getattr(config, 'rms_norm_eps', 1e-6)
    intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers

    # DeltaNet dimensions (fallbacks match Qwen3.6-27B)
    num_k_heads = getattr(config, 'linear_num_key_heads', 16)
    num_v_heads = getattr(config, 'linear_num_value_heads', 48)
    head_k_dim = getattr(config, 'linear_key_head_dim', 128)
    head_v_dim = getattr(config, 'linear_value_head_dim', 128)

    # Full attention dimensions (fallbacks match Qwen3.6-27B)
    num_q_heads = getattr(config, 'num_attention_heads', 24)
    num_kv_heads = getattr(config, 'num_key_value_heads', 4)
    head_dim = getattr(config, 'head_dim', 256)

    # DeltaNet pattern from config (fallback to hardcoded).
    deltanet_pattern = _compute_deltanet_pattern(config)

    # ===== Precompute layer-index lookup tables (O(1) at forward time) =====
    # Replaces the previous per-forward O(n) scans in _get_attention_layer_idx
    # and _get_deltanet_cache. Each layer captures its own index as a closure
    # variable, so the tables themselves are only needed here.
    layer_idx_to_attn_idx: dict = {}
    layer_idx_to_dn_idx: dict = {}
    attn_idx = 0
    dn_idx = 0
    for layer_idx in range(num_layers):
        if deltanet_pattern[layer_idx]:
            layer_idx_to_dn_idx[layer_idx] = dn_idx
            dn_idx += 1
        else:
            layer_idx_to_attn_idx[layer_idx] = attn_idx
            attn_idx += 1

    print(f"Patching {num_layers} layers (weight dedup + fused QKnorm+RoPE + sigmoid gate)...")

    for layer_idx in range(num_layers):
        if layer_idx % 8 == 0:
            print(f"  Patching layer {layer_idx}/{num_layers}...")
        layer = model.model.layers[layer_idx]

        next_norm_w = None
        if layer_idx < num_layers - 1:
            next_norm_w = model.model.layers[layer_idx + 1].input_layernorm.weight

        if is_deltanet_layer(layer_idx, deltanet_pattern):
            fwd = _create_fused_deltanet_forward(
                layer, layer_idx,
                hidden_size=hidden_size,
                eps=eps,
                intermediate_size=intermediate_size,
                next_layer_norm_w=next_norm_w,
                device=device,
                state=state,
                dn_idx=layer_idx_to_dn_idx[layer_idx],
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
            )
        else:
            fwd = _create_fused_attention_forward(
                layer, layer_idx,
                hidden_size=hidden_size,
                eps=eps,
                intermediate_size=intermediate_size,
                next_layer_norm_w=next_norm_w,
                device=device,
                state=state,
                attn_idx=layer_idx_to_attn_idx[layer_idx],
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )

        # Mark as patched and install the fused forward. The _fused marker is
        # used by verify_patch() / unpatch_model() instead of introspecting
        # closures.
        layer._fused = True
        layer.forward = fwd

    _patch_final_norm(model, eps)

    gc.collect()
    torch.cuda.empty_cache()

    vram_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"Patching complete. Paged KV cache mode, VRAM: {vram_gb:.1f}GB")
    return model


def unpatch_model(model: nn.Module) -> nn.Module:
    """Restore original ``forward`` methods on all patched layers.

    Removes the instance ``forward`` attribute set by ``patch_model`` so that
    attribute lookup falls back to the class-defined method, and clears the
    ``_fused`` marker. Safe to call on an already-unpatched model.
    """
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    num_layers = config.num_hidden_layers

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        if getattr(layer, '_fused', False):
            try:
                del layer.forward
            except AttributeError:
                pass
            layer._fused = False

    final_norm = getattr(model.model, 'norm', None)
    if final_norm is not None and getattr(final_norm, '_fused', False):
        try:
            del final_norm.forward
        except AttributeError:
            pass
        final_norm._fused = False

    return model


def _create_fused_deltanet_forward(
    layer: nn.Module,
    layer_idx: int,
    hidden_size: int,
    eps: float,
    intermediate_size: int,
    next_layer_norm_w=None,
    device: str = "cuda",
    state: Optional[ModelState] = None,
    dn_idx: int = 0,
    num_k_heads: int = 16,
    num_v_heads: int = 48,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
):
    """Create fused forward for a Qwen3_6GatedDeltaNet layer.

    Optimizations:
    - Pre-concatenated projection weights: 4->1 GEMV for DeltaNet proj
    - Pre-concatenated MLP weights: 2->1 GEMV for MLP
    - Fused residual+RMSNorm between attention and MLP
    - Pre-computed -exp(A) and dt_bias.float()
    - Tensor Parallelism: column-parallel for projections, row-parallel for out/down
    """
    if state is None:
        state = _default_state

    # Norm weights (replicated, not sharded)
    input_ln_w = layer.input_layernorm.weight
    post_ln_w = layer.post_attention_layernorm.weight

    attn = layer.linear_attn  # Qwen3_6GatedDeltaNet

    # ===== TP context =====
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()

    # TP-adjusted dimensions (all derived from config-driven constants)
    tp_num_k_heads = num_k_heads // tp_size
    tp_num_v_heads = num_v_heads // tp_size
    tp_key_dim = tp_num_k_heads * head_k_dim
    tp_value_dim = tp_num_v_heads * head_v_dim
    tp_conv_dim = tp_key_dim * 2 + tp_value_dim  # Q+K+V channels
    tp_intermediate_size = intermediate_size // tp_size

    # Full (unsharded) key dim, used for slicing Q/K/V segments of the
    # pre-concatenated projection / conv1d weights.
    key_dim = num_k_heads * head_k_dim

    inv_sqrt_dk = head_k_dim ** -0.5  # 1/sqrt(head_k_dim)

    # ===== PRE-CONCATENATE DELTANET PROJECTION WEIGHTS + DEDUP + SHARD =====
    # Full: w_qkv[10240,5120] + w_a[48,5120] + w_b[48,5120] + w_z[6144,5120] = [16480,5120]
    # Column-parallel: shard each sub-weight along dim=0 (output dim)
    w_qkv_full = attn.in_proj_qkv.weight    # [10240, 5120]
    w_a_full = attn.in_proj_a.weight         # [48, 5120]
    w_b_full = attn.in_proj_b.weight         # [48, 5120]
    w_z_full = attn.in_proj_z.weight         # [6144, 5120]

    # Shard QKV weight: must split Q/K/V segments separately to preserve layout
    # w_qkv_full [10240, 5120] = [Q(2048) | K(2048) | V(6144)]
    q_part = w_qkv_full[:key_dim]                        # [2048, 5120]
    k_part = w_qkv_full[key_dim:key_dim * 2]             # [2048, 5120]
    v_part = w_qkv_full[key_dim * 2:]                    # [6144, 5120]
    w_qkv_shard = torch.cat([
        shard_weight_col(q_part, tp_rank, tp_size),
        shard_weight_col(k_part, tp_rank, tp_size),
        shard_weight_col(v_part, tp_rank, tp_size),
    ], dim=0).contiguous()                               # [10240/tp, 5120]
    w_a_shard = shard_weight_col(w_a_full, tp_rank, tp_size)      # [48/tp, 5120]
    w_b_shard = shard_weight_col(w_b_full, tp_rank, tp_size)      # [48/tp, 5120]
    w_z_shard = shard_weight_col(w_z_full, tp_rank, tp_size)      # [6144/tp, 5120]

    w_delta_proj = torch.cat([w_qkv_shard, w_a_shard, w_b_shard, w_z_shard], dim=0).contiguous()

    del w_qkv_full, w_a_full, w_b_full, w_z_full, w_qkv_shard, w_a_shard, w_b_shard, w_z_shard

    _GEMV_BLOCK_M = 64
    if w_delta_proj.shape[0] % _GEMV_BLOCK_M != 0:
        pad_rows = _GEMV_BLOCK_M - (w_delta_proj.shape[0] % _GEMV_BLOCK_M)
        w_delta_proj = torch.cat([
            w_delta_proj,
            torch.zeros(pad_rows, w_delta_proj.shape[1], dtype=w_delta_proj.dtype),
        ], dim=0).contiguous()

    attn.in_proj_qkv = None
    attn.in_proj_a = None
    attn.in_proj_b = None
    attn.in_proj_z = None

    # TP-adjusted split indices
    delta_split_qkv = tp_conv_dim
    delta_split_a = delta_split_qkv + tp_num_v_heads
    delta_split_b = delta_split_a + tp_num_v_heads
    delta_proj_dim = delta_split_b + tp_value_dim

    # ===== PRE-COMPUTE CONSTANTS (sharded) =====
    neg_A_exp = shard_bias_col(-attn.A_log.float().exp(), tp_rank, tp_size)  # [48/tp] fp32
    dt_bias_f = shard_bias_col(attn.dt_bias.float(), tp_rank, tp_size)       # [48/tp] fp32
    # bf16 copies for decode kernel (fused_postproj_recurrent requires bf16)
    neg_A_exp_bf16 = neg_A_exp.to(torch.bfloat16)
    dt_bias_f_bf16 = dt_bias_f.to(torch.bfloat16)

    # Conv1d weights (sharded along conv_dim = channel dim)
    # Must split Q/K/V segments separately to preserve layout (same as in_proj_qkv)
    conv1d = attn.conv1d
    conv_w_full = conv1d.weight.squeeze(1)  # [10240, 4] = [Q(2048)|K(2048)|V(6144), 4]
    conv_q = conv_w_full[:key_dim]
    conv_k = conv_w_full[key_dim:key_dim * 2]
    conv_v = conv_w_full[key_dim * 2:]
    conv_w = torch.cat([
        shard_weight_col(conv_q, tp_rank, tp_size),
        shard_weight_col(conv_k, tp_rank, tp_size),
        shard_weight_col(conv_v, tp_rank, tp_size),
    ], dim=0).contiguous()  # [10240/tp, 4]
    conv_b_full = getattr(conv1d, 'bias', None)
    conv_b = shard_bias_col(conv_b_full, tp_rank, tp_size) if conv_b_full is not None else None

    # Output norm (replicated per v_head, not sharded)
    norm_w = attn.norm.weight  # [128] (head_v_dim, same across TP ranks)

    # Output projection (row-parallel: split input dim = value_dim)
    w_out_full = attn.out_proj.weight  # [5120, 6144]
    w_out = shard_weight_row(w_out_full, tp_rank, tp_size)  # [5120, 6144/tp]

    # ===== PRE-CONCATENATE MLP WEIGHTS + DEDUP + SHARD =====
    mlp = layer.mlp
    # gate_proj + up_proj: column-parallel (split intermediate dim)
    w_gate_full = mlp.gate_proj.weight   # [17408, 5120]
    w_up_full = mlp.up_proj.weight       # [17408, 5120]
    w_gate_shard = shard_weight_col(w_gate_full, tp_rank, tp_size)
    w_up_shard = shard_weight_col(w_up_full, tp_rank, tp_size)
    w_mlp_gate_up = torch.cat([w_gate_shard, w_up_shard], dim=0).contiguous()

    # down_proj: row-parallel (split intermediate dim)
    w_mlp_down = shard_weight_row(mlp.down_proj.weight, tp_rank, tp_size)  # [5120, 17408/tp]

    del w_gate_full, w_up_full, w_gate_shard, w_up_shard
    mlp.gate_proj = None
    mlp.up_proj = None
    mlp.down_proj = None

    input_ln_w = input_ln_w.to(device)
    post_ln_w = post_ln_w.to(device)
    w_delta_proj = w_delta_proj.to(device)
    neg_A_exp = neg_A_exp.to(device)
    dt_bias_f = dt_bias_f.to(device)
    neg_A_exp_bf16 = neg_A_exp_bf16.to(device)
    dt_bias_f_bf16 = dt_bias_f_bf16.to(device)
    conv_w = conv_w.to(device)
    if conv_b is not None:
        conv_b = conv_b.to(device)
    norm_w = norm_w.to(device)
    w_out = w_out.to(device)
    w_mlp_gate_up = w_mlp_gate_up.to(device)
    w_mlp_down = w_mlp_down.to(device)
    if next_layer_norm_w is not None:
        next_layer_norm_w = next_layer_norm_w.to(device)

    w_delta_proj_q, w_delta_proj_s = quantize_weight_int8(w_delta_proj)
    del w_delta_proj
    w_out_q, w_out_s = quantize_weight_int8(w_out)
    del w_out
    w_mlp_gate_up_q, w_mlp_gate_up_s = quantize_weight_int8(w_mlp_gate_up)
    del w_mlp_gate_up
    w_mlp_down_q, w_mlp_down_s = quantize_weight_int8(w_mlp_down)
    del w_mlp_down

    def fused_forward(
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        x_normed_input=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        from engine.context import get_context as _get_ctx
        _ctx = _get_ctx()
        _is_prefill = _ctx is not None and _ctx.is_prefill

        if q_len > 1 or _is_prefill:
            # ====== PREFILL PATH - FlashQLA + tilelang kernels ======
            residual = hidden_states
            x_normed = rmsnorm(hidden_states, input_ln_w, eps, add_one_to_weight=True)
            x_2d = x_normed.reshape(-1, hidden_size)

            # Single GEMV for all DeltaNet projections (column-parallel, no comm)
            proj_all = bf16_linear_forward(x_2d, w_delta_proj_q, w_delta_proj_s)  # [B*T, delta_proj_dim]
            proj_all = proj_all.reshape(bsz, q_len, -1)
            qkv = proj_all[:, :, :delta_split_qkv]                # [B, T, tp_conv_dim]
            alpha = proj_all[:, :, delta_split_qkv:delta_split_a]  # [B, T, tp_num_v_heads]
            beta_raw = proj_all[:, :, delta_split_a:delta_split_b]  # [B, T, tp_num_v_heads]
            z = proj_all[:, :, delta_split_b:delta_proj_dim].contiguous()  # [B, T, tp_value_dim]

            # Causal conv1d for prefill (parallel, single kernel launch)
            dn_cache = _get_deltanet_cache(state, dn_idx, bsz,
                                           tp_conv_dim, tp_num_v_heads,
                                           head_k_dim, head_v_dim)
            K_conv = conv_w.shape[1]
            # Prepend conv_state for boundary continuity in chunked prefill.
            # For the first chunk conv_state is zero, giving the same result as
            # the kernel's internal zero-padding.
            prefix = dn_cache.conv_state[:, :, 1:K_conv].transpose(1, 2)  # [B, K-1, conv_dim]
            qkv_extended = torch.cat([prefix, qkv], dim=1)
            qkv_conv_extended = causal_conv1d_prefill(
                qkv_extended, conv_w, conv_b, apply_silu=True,
            )
            qkv_conv = qkv_conv_extended[:, K_conv - 1:, :]  # [B, T, tp_conv_dim]

            # Write conv_state for subsequent decode steps.
            # The decode kernel (causal_conv1d_update) stores RAW input values
            # in the state buffer — it shifts them, inserts the new raw token,
            # computes the weighted sum, and THEN applies SiLU.  Therefore we
            # must save the raw projection output (qkv), NOT the post-SiLU
            # conv output (qkv_conv).
            if dn_cache.slots is not None:
                K = conv_w.shape[1]  # kernel size (typically 4)
                num_to_save = min(K - 1, q_len)
                if num_to_save > 0:
                    last_tokens = qkv[:, -num_to_save:, :].transpose(1, 2)  # [B, conv_dim, num_to_save]
                    dn_cache.conv_state[:, :, K - num_to_save:K] = last_tokens
                    state.deltanet_conv_pool[dn_cache.dn_idx, dn_cache.slots] = dn_cache.conv_state

            # Extract Q, K, V from conv output (TP-adjusted dimensions)
            q_conv = qkv_conv[:, :, :tp_key_dim].reshape(bsz, q_len, tp_num_k_heads, head_k_dim)
            k_conv = qkv_conv[:, :, tp_key_dim:tp_key_dim*2].reshape(bsz, q_len, tp_num_k_heads, head_k_dim)
            v_conv = qkv_conv[:, :, tp_key_dim*2:].reshape(bsz, q_len, tp_num_v_heads, head_v_dim)

            # L2 normalize Q and K along head_dim (matches decode kernel's
            # fused_postproj_recurrent which normalizes internally).
            # Scale=inv_sqrt_dk is applied to the output, equivalent to scaling Q here.
            # Reuse the fp32 copies as scratch: in-place div_ avoids allocating
            # an extra division-result tensor per tensor.
            q_conv_f = q_conv.float()
            q_conv_f.div_(q_conv_f.norm(dim=-1, keepdim=True).clamp_(min=1e-6))
            q_conv = q_conv_f.to(q_conv.dtype)
            k_conv_f = k_conv.float()
            k_conv_f.div_(k_conv_f.norm(dim=-1, keepdim=True).clamp_(min=1e-6))
            k_conv = k_conv_f.to(k_conv.dtype)

            # Compute gate (g) and beta for FlashQLA (TP-adjusted).
            # alpha_f is a fresh fp32 copy not needed beyond this block, so we
            # reuse its storage for the softplus input and then for g, shaving
            # two intermediate allocations.
            alpha_f = alpha.float()
            neg_a_exp_expanded = neg_A_exp.unsqueeze(0).unsqueeze(0)  # [1, 1, tp_num_v_heads]
            dt_bias_expanded = dt_bias_f.unsqueeze(0).unsqueeze(0)    # [1, 1, tp_num_v_heads]
            sp = alpha_f
            sp.add_(dt_bias_expanded)                                # sp = alpha + dt_bias
            # softplus(sp) with overflow guard. log1p(exp(x)) == log(1 + exp(x)).
            log_term = torch.log1p(torch.exp(sp))
            torch.where(sp > 20.0, sp, log_term, out=sp)
            g = torch.mul(sp, neg_a_exp_expanded, out=sp)            # [B, T, tp_num_v_heads]
            beta = torch.sigmoid(beta_raw.float())                   # [B, T, tp_num_v_heads]

            # FlashQLA chunk_gated_delta_rule
            initial_state = None
            if dn_cache.slots is not None:
                initial_state = dn_cache.recurrent_state

            o_flash, final_state = chunk_gated_delta_rule(
                q=q_conv,           # [B, T, tp_num_k_heads, head_k_dim]
                k=k_conv,           # [B, T, tp_num_k_heads, head_k_dim]
                v=v_conv,           # [B, T, tp_num_v_heads, head_v_dim]
                g=g,                # [B, T, tp_num_v_heads]
                beta=beta,          # [B, T, tp_num_v_heads]
                scale=inv_sqrt_dk,
                initial_state=initial_state,
                output_final_state=True,
            )

            # Update recurrent state
            _writeback_deltanet_state(state, dn_idx, dn_cache.slots, final_state)

            # RMSNorm + gating (TP-adjusted)
            z_3d = z.reshape(bsz, q_len, tp_num_v_heads, head_v_dim)
            o_normed = rmsnorm(o_flash.reshape(bsz * q_len * tp_num_v_heads, head_v_dim),
                               norm_w, eps, add_one_to_weight=False)
            o_normed = o_normed.reshape(bsz, q_len, tp_num_v_heads, head_v_dim)
            # Gate = o_normed * SiLU(z) = o_normed * z * sigmoid(z).
            # fused_silu_mul(z, o_normed) computes z * sigmoid(z) * o_normed in a
            # single kernel (previously sigmoid_mul + an extra elementwise mul),
            # matching the decode path which fuses this inside the recurrent kernel.
            attn_output = fused_silu_mul(z_3d, o_normed)
            attn_output = attn_output.reshape(bsz, q_len, tp_value_dim)

            # Output projection (row-parallel -> All-Reduce)
            o_2d = attn_output.reshape(-1, tp_value_dim)
            attn_output = bf16_linear_forward(o_2d, w_out_q, w_out_s).reshape(bsz, q_len, hidden_size)
            ar_work = None
            if is_tp_active():
                ar_work = tp_all_reduce_async(attn_output)

            # Fused residual + RMSNorm
            if ar_work is not None:
                ar_work.wait()
            hidden_states, x_normed = fused_residual_rmsnorm(
                residual, attn_output, post_ln_w, eps, add_one_to_weight=True,
            )

            # MLP block (column-parallel gate_up, row-parallel down -> All-Reduce)
            x_2d = x_normed.reshape(-1, hidden_size)
            gate_up = bf16_linear_forward(x_2d, w_mlp_gate_up_q, w_mlp_gate_up_s)
            gate_out = gate_up[:, :tp_intermediate_size]
            up_out = gate_up[:, tp_intermediate_size:]
            mlp_mid = fused_silu_mul(gate_out, up_out)
            down_out = bf16_linear_forward(
                mlp_mid.reshape(-1, tp_intermediate_size), w_mlp_down_q, w_mlp_down_s,
            ).reshape(bsz, q_len, hidden_size)
            ar_work2 = None
            if is_tp_active():
                ar_work2 = tp_all_reduce_async(down_out)

            if ar_work2 is not None:
                ar_work2.wait()
            hidden_states = hidden_states + down_out
            return hidden_states

        # ============================================================
        # DECODE PATH (q_len == 1)  - fused kernels
        # ============================================================
        residual = hidden_states

        # Use pre-computed norm from previous layer if available
        if x_normed_input is not None:
            x_normed = x_normed_input
        else:
            x_normed = rmsnorm(hidden_states, input_ln_w, eps, add_one_to_weight=True)
        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for all DeltaNet projections (column-parallel) =====
        proj_all = bf16_linear_forward(x_2d, w_delta_proj_q, w_delta_proj_s)  # [M, delta_proj_dim_padded]
        if bsz == 1:
            _p = proj_all.view(-1)
            qkv = _p[:delta_split_qkv].view(1, -1)
            alpha = _p[delta_split_qkv:delta_split_a].view(1, -1)
            beta_raw = _p[delta_split_a:delta_split_b].view(1, -1)
            z = _p[delta_split_b:delta_proj_dim].view(1, -1)
        else:
            qkv = proj_all[:, :delta_split_qkv].contiguous()
            alpha = proj_all[:, delta_split_qkv:delta_split_a].contiguous()
            beta_raw = proj_all[:, delta_split_a:delta_split_b].contiguous()
            z = proj_all[:, delta_split_b:delta_proj_dim].contiguous()

        dn_cache = _get_deltanet_cache(state, dn_idx, bsz,
                                       tp_conv_dim, tp_num_v_heads,
                                       head_k_dim, head_v_dim)

        # Fused causal conv1d update (TP-adjusted conv_dim)
        qkv_flat = qkv.reshape(bsz, tp_conv_dim)
        qkv_conv = causal_conv1d_update(
            qkv_flat, dn_cache.conv_state, conv_w, conv_b, apply_silu=True,
        )

        # ===== FUSED POST-PROJ + RECURRENT (TP-adjusted) =====
        z_3d = z.reshape(bsz, tp_num_v_heads, head_v_dim)
        attn_output, dn_cache.recurrent_state = fused_postproj_recurrent(
            qkv_conv, alpha, beta_raw, neg_A_exp_bf16, dt_bias_f_bf16,
            dn_cache.recurrent_state,
            num_k_heads=tp_num_k_heads, num_v_heads=tp_num_v_heads, head_dim=head_k_dim,
            norm_weight=norm_w, z=z_3d, norm_eps=eps,
        )
        attn_output = attn_output.reshape(bsz, 1, tp_value_dim)

        # Output projection (row-parallel -> All-Reduce)
        o_2d = attn_output.reshape(-1, tp_value_dim)
        attn_output = bf16_linear_forward(o_2d, w_out_q, w_out_s).reshape(bsz, q_len, hidden_size)
        ar_work = None
        if is_tp_active():
            ar_work = tp_all_reduce_async(attn_output)

        # Overlap state write-backs with NCCL communication
        if dn_cache.slots is not None:
            state.deltanet_conv_pool[dn_cache.dn_idx, dn_cache.slots] = dn_cache.conv_state
            state.deltanet_recurrent_pool[dn_cache.dn_idx, dn_cache.slots] = dn_cache.recurrent_state

        # ===== FUSED RESIDUAL + RMSNORM =====
        if ar_work is not None:
            ar_work.wait()
        hidden_states, x_normed = fused_residual_rmsnorm(
            residual, attn_output, post_ln_w, eps, add_one_to_weight=True,
        )

        # ============================================================
        # MLP block
        # ============================================================
        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for gate+up projections (column-parallel) =====
        gate_up = bf16_linear_forward(x_2d, w_mlp_gate_up_q, w_mlp_gate_up_s)
        if bsz == 1:
            _gu = gate_up.view(-1)
            gate_out = _gu[:tp_intermediate_size].view(1, -1)
            up_out = _gu[tp_intermediate_size:tp_intermediate_size * 2].view(1, -1)
        else:
            gate_out = gate_up[:, :tp_intermediate_size].contiguous()
            up_out = gate_up[:, tp_intermediate_size:].contiguous()
        mlp_mid = fused_silu_mul(gate_out, up_out)

        # down_proj (row-parallel -> All-Reduce)
        down_out = bf16_linear_forward(
            mlp_mid.reshape(-1, tp_intermediate_size), w_mlp_down_q, w_mlp_down_s,
        ).reshape(bsz, q_len, hidden_size)
        ar_work2 = None
        if is_tp_active():
            ar_work2 = tp_all_reduce_async(down_out)

        # Inter-layer chaining: fuse residual-add + next layer's input norm
        if ar_work2 is not None:
            ar_work2.wait()
        if next_layer_norm_w is not None and q_len == 1:
            hidden_states, x_normed_next = fused_residual_rmsnorm(
                hidden_states, down_out, next_layer_norm_w, eps, add_one_to_weight=True,
            )
            return hidden_states, x_normed_next
        else:
            hidden_states = hidden_states + down_out
            return hidden_states

    return fused_forward


def _create_fused_attention_forward(
    layer: nn.Module,
    layer_idx: int,
    hidden_size: int,
    eps: float,
    intermediate_size: int,
    next_layer_norm_w=None,
    device: str = "cuda",
    state: Optional[ModelState] = None,
    attn_idx: int = 0,
    num_q_heads: int = 24,
    num_kv_heads: int = 4,
    head_dim: int = 256,
):
    """Create fused forward for a Qwen3_6Attention layer.

    Optimizations:
    - Weight dedup: original params replaced with views into concatenated
    - Fused QK-norm + partial RoPE: 2 CUDA launches instead of ~18 PyTorch ops
    - Fused sigmoid gate multiply: 1 CUDA kernel instead of 4 PyTorch ops
    - Pre-concatenated QKV weights: 3 GEMVs -> 1
    - Pre-concatenated MLP weights: 2 GEMVs -> 1
    - Fused residual+RMSNorm between attention and MLP
    - Tensor Parallelism: column-parallel for QKV/gate_up, row-parallel for O/down
    """
    if state is None:
        state = _default_state

    input_ln_w = layer.input_layernorm.weight
    post_ln_w = layer.post_attention_layernorm.weight
    self_attn = layer.self_attn
    mlp = layer.mlp

    # ===== TP context =====
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()

    # TP-adjusted dimensions (all derived from config-driven constants)
    tp_num_q_heads = num_q_heads // tp_size
    tp_num_kv_heads = num_kv_heads // tp_size
    tp_q_out_dim = tp_num_q_heads * head_dim
    tp_kv_out_dim = tp_num_kv_heads * head_dim
    tp_q_proj_dim = tp_num_q_heads * head_dim * 2  # Q + gate
    tp_intermediate_size = intermediate_size // tp_size

    # ===== PRE-CONCATENATE ATTENTION QKV WEIGHTS + DEDUP + SHARD =====
    # Column-parallel: shard each sub-weight along dim=0 (output dim)
    w_q_full = self_attn.q_proj.weight   # [12288, 5120]
    w_k_full = self_attn.k_proj.weight   # [1024, 5120]
    w_v_full = self_attn.v_proj.weight   # [1024, 5120]

    w_q_shard = shard_weight_col(w_q_full, tp_rank, tp_size)  # [12288/tp, 5120]
    w_k_shard = shard_weight_col(w_k_full, tp_rank, tp_size)  # [1024/tp, 5120]
    w_v_shard = shard_weight_col(w_v_full, tp_rank, tp_size)  # [1024/tp, 5120]

    w_attn_qkv = torch.cat([w_q_shard, w_k_shard, w_v_shard], dim=0).contiguous()

    del w_q_full, w_k_full, w_v_full, w_q_shard, w_k_shard, w_v_shard
    self_attn.q_proj = None
    self_attn.k_proj = None
    self_attn.v_proj = None

    # TP-adjusted split indices
    attn_split_q = tp_q_proj_dim
    attn_split_k = attn_split_q + tp_kv_out_dim
    attn_qkv_dim = attn_split_k + tp_kv_out_dim

    # O projection: row-parallel (split input = q_proj_dim / 2 = q_out_dim)
    w_o = shard_weight_row(self_attn.o_proj.weight, tp_rank, tp_size)  # [5120, 6144/tp]
    del self_attn.o_proj.weight
    self_attn.o_proj = None

    # Pre-extract norm weights for fused QKnorm+RoPE (replicated, not sharded)
    q_norm_w = self_attn.q_norm.weight if hasattr(self_attn, 'q_norm') and self_attn.q_norm is not None else None
    k_norm_w = self_attn.k_norm.weight if hasattr(self_attn, 'k_norm') and self_attn.k_norm is not None else None

    # ===== PRE-CONCATENATE MLP WEIGHTS + DEDUP + SHARD =====
    w_gate_full = mlp.gate_proj.weight   # [17408, 5120]
    w_up_full = mlp.up_proj.weight       # [17408, 5120]
    w_gate_shard = shard_weight_col(w_gate_full, tp_rank, tp_size)
    w_up_shard = shard_weight_col(w_up_full, tp_rank, tp_size)
    w_mlp_gate_up = torch.cat([w_gate_shard, w_up_shard], dim=0).contiguous()

    # down_proj: row-parallel (split intermediate dim)
    w_mlp_down = shard_weight_row(mlp.down_proj.weight, tp_rank, tp_size)  # [5120, 17408/tp]

    del w_gate_full, w_up_full, w_gate_shard, w_up_shard
    mlp.gate_proj = None
    mlp.up_proj = None
    mlp.down_proj = None

    input_ln_w = input_ln_w.to(device)
    post_ln_w = post_ln_w.to(device)
    w_attn_qkv = w_attn_qkv.to(device)
    w_o = w_o.to(device)
    if q_norm_w is not None:
        q_norm_w = q_norm_w.to(device)
    if k_norm_w is not None:
        k_norm_w = k_norm_w.to(device)
    w_mlp_gate_up = w_mlp_gate_up.to(device)
    w_mlp_down = w_mlp_down.to(device)
    if next_layer_norm_w is not None:
        next_layer_norm_w = next_layer_norm_w.to(device)

    w_attn_qkv_q, w_attn_qkv_s = quantize_weight_int8(w_attn_qkv)
    del w_attn_qkv
    w_o_q, w_o_s = quantize_weight_int8(w_o)
    del w_o
    w_mlp_gate_up_q, w_mlp_gate_up_s = quantize_weight_int8(w_mlp_gate_up)
    del w_mlp_gate_up
    w_mlp_down_q, w_mlp_down_s = quantize_weight_int8(w_mlp_down)
    del w_mlp_down

    def fused_forward(
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        x_normed_input=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # ============================================================
        # Attention block
        # ============================================================
        residual = hidden_states

        # Use pre-computed norm from previous layer if available
        if x_normed_input is not None and q_len == 1:
            x_normed = x_normed_input
        else:
            x_normed = rmsnorm(hidden_states, input_ln_w, eps, add_one_to_weight=True)
        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for Q+K+V (column-parallel, no comm) =====
        qkv_all = bf16_linear_forward(x_2d, w_attn_qkv_q, w_attn_qkv_s)  # [M, attn_qkv_dim]
        q_full = qkv_all[:, :attn_split_q]           # [M, tp_q_proj_dim]
        k = qkv_all[:, attn_split_q:attn_split_k]    # [M, tp_kv_out_dim]
        v = qkv_all[:, attn_split_k:]                 # [M, tp_kv_out_dim]

        # q_proj output is tp_q_proj_dim = tp_num_q_heads * head_dim * 2 (interleaved Q + gate)
        q_full_heads = q_full.reshape(bsz, q_len, tp_num_q_heads, head_dim * 2)
        q, q_gate = q_full_heads.chunk(2, dim=-1)
        q_gate = q_gate.reshape(bsz, q_len, -1)

        # Reshape Q/K/V to [B, num_heads, S, head_dim] (TP-adjusted)
        q = q.transpose(1, 2).contiguous()
        k = k.reshape(bsz, q_len, tp_num_kv_heads, head_dim).transpose(1, 2).contiguous()
        v = v.reshape(bsz, q_len, tp_num_kv_heads, head_dim).transpose(1, 2).contiguous()

        # RoPE position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
        elif hasattr(self_attn, 'rotary_emb'):
            cos, sin = self_attn.rotary_emb(v, position_ids)
        else:
            cos, sin = None, None

        if q_len == 1 and bsz == 1 and cos is not None and q_norm_w is not None:
            # ===== FUSED QK-NORM + PARTIAL ROPE (single-sequence decode only) =====
            cos_flat = cos.reshape(-1)[:head_dim] if cos.numel() > head_dim else cos.reshape(-1)
            sin_flat = sin.reshape(-1)[:head_dim] if sin.numel() > head_dim else sin.reshape(-1)
            q = fused_qknorm_rope(q, q_norm_w, cos_flat, sin_flat, eps)
            k = fused_qknorm_rope(k, k_norm_w, cos_flat, sin_flat, eps)
        else:
            # Prefill or batched decode: use separate norm + RoPE ops
            if q_norm_w is not None:
                q = rmsnorm(q, q_norm_w, eps, add_one_to_weight=True)
            if k_norm_w is not None:
                k = rmsnorm(k, k_norm_w, eps, add_one_to_weight=True)
            if cos is not None:
                rope_dim = cos.shape[-1]
                cos_r = cos.unsqueeze(1)
                sin_r = sin.unsqueeze(1)
                if rope_dim < head_dim:
                    q_rope, q_pass = q[..., :rope_dim], q[..., rope_dim:]
                    k_rope, k_pass = k[..., :rope_dim], k[..., rope_dim:]
                    q_rope = (q_rope * cos_r) + (_rotate_half(q_rope) * sin_r)
                    k_rope = (k_rope * cos_r) + (_rotate_half(k_rope) * sin_r)
                    q = torch.cat([q_rope, q_pass], dim=-1)
                    k = torch.cat([k_rope, k_pass], dim=-1)
                else:
                    q = (q * cos_r) + (_rotate_half(q) * sin_r)
                    k = (k * cos_r) + (_rotate_half(k) * sin_r)

        # Paged KV cache: store K/V
        from engine.context import get_context
        ctx = get_context()

        # Store K/V into paged cache (TP-adjusted: tp_num_kv_heads)
        if q_len == 1:
            k_store = k.squeeze(2)
            v_store = v.squeeze(2)
        else:
            k_store = k.transpose(1, 2).reshape(-1, tp_num_kv_heads, head_dim).contiguous()
            v_store = v.transpose(1, 2).reshape(-1, tp_num_kv_heads, head_dim).contiguous()
        # attn_idx is precomputed at patch time (O(1) closure lookup).
        k_cache = state.paged_kv_cache[0, attn_idx].reshape(-1, tp_num_kv_heads * head_dim)
        v_cache = state.paged_kv_cache[1, attn_idx].reshape(-1, tp_num_kv_heads * head_dim)
        store_kvcache(k_store, v_store, k_cache, v_cache, ctx.slot_mapping)

        if q_len > 1 or (ctx is not None and ctx.is_prefill):
            # ===== PREFILL: TileLang GQA varlen flash attention ======
            groups = tp_num_q_heads // tp_num_kv_heads  # same ratio = 6
            q_unpad = q.transpose(1, 2).reshape(bsz * q_len, tp_num_q_heads, head_dim).contiguous()

            # For chunked prefill continuation (num_cached_tokens > 0), the
            # attention must see the full KV context (cached prefix + current
            # chunk). Gather K/V from the paged cache which already contains
            # both the prefix (from prior chunks) and the current chunk
            # (stored above via store_kvcache).
            full_kv_len = ctx.max_seqlen_k if ctx is not None else q_len
            if full_kv_len > q_len and ctx is not None and ctx.block_tables is not None:
                k_cache_paged = state.paged_kv_cache[0, attn_idx]
                v_cache_paged = state.paged_kv_cache[1, attn_idx]
                page_size = state.paged_kv_cache.shape[3]
                positions = torch.arange(full_kv_len, device=q.device)
                block_indices = positions // page_size
                offsets_in_block = positions % page_size
                physical_blocks = ctx.block_tables[0, block_indices]
                k_unpad = k_cache_paged[physical_blocks, offsets_in_block].reshape(
                    full_kv_len, tp_num_kv_heads, head_dim).contiguous()
                v_unpad = v_cache_paged[physical_blocks, offsets_in_block].reshape(
                    full_kv_len, tp_num_kv_heads, head_dim).contiguous()
                cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
                cu_seqlens_k = torch.tensor([0, full_kv_len], dtype=torch.int32, device=q.device)
                kernel = tilelang_flashattn(
                    1, groups, q_len, full_kv_len,
                    tp_num_q_heads, head_dim, is_causal=True,
                    block_M=128, block_N=64, num_stages=2, threads=128,
                    dtype=T.bfloat16 if q.dtype == torch.bfloat16 else T.float16,
                )
                attn_out_unpad = kernel(q_unpad, k_unpad, v_unpad,
                                        cu_seqlens_q, cu_seqlens_k, q_len)
            else:
                # First chunk (no cached prefix): K/V from current computation
                k_unpad = k.transpose(1, 2).reshape(bsz * q_len, tp_num_kv_heads, head_dim).contiguous()
                v_unpad = v.transpose(1, 2).reshape(bsz * q_len, tp_num_kv_heads, head_dim).contiguous()
                cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len,
                                           dtype=torch.int32, device=q.device)
                kernel = tilelang_flashattn(
                    bsz, groups, bsz * q_len, bsz * q_len,
                    tp_num_q_heads, head_dim, is_causal=True,
                    block_M=128, block_N=64, num_stages=2, threads=128,
                    dtype=T.bfloat16 if q.dtype == torch.bfloat16 else T.float16,
                )
                attn_out_unpad = kernel(q_unpad, k_unpad, v_unpad, cu_seqlens, cu_seqlens, q_len)

            attn_output = attn_out_unpad.reshape(bsz, q_len, tp_num_q_heads, head_dim)
            attn_output = attn_output.reshape(bsz, q_len, tp_q_out_dim)
        else:
            # ===== DECODE: TileLang GQA paged attention ======
            k_cache_paged = state.paged_kv_cache[0, attn_idx]
            v_cache_paged = state.paged_kv_cache[1, attn_idx]
            q_decode = q.squeeze(2).contiguous()
            o = gqa_decode_paged_fn(
                q_decode, k_cache_paged, v_cache_paged,
                block_table=ctx.block_tables,
                seqlen_kv=ctx.context_lens,
                page_size=state.paged_kv_cache.shape[3],
            )
            attn_output = o.reshape(bsz, q_len, tp_q_out_dim)

        # ===== FUSED SIGMOID GATE MULTIPLY =====
        attn_output = fused_sigmoid_mul(attn_output, q_gate)

        # Output projection (row-parallel -> All-Reduce)
        o_2d = attn_output.reshape(-1, tp_q_out_dim)
        attn_output = bf16_linear_forward(o_2d, w_o_q, w_o_s).reshape(bsz, q_len, hidden_size)
        ar_work = None
        if is_tp_active():
            ar_work = tp_all_reduce_async(attn_output)

        # ===== FUSED RESIDUAL + RMSNORM =====
        if ar_work is not None:
            ar_work.wait()
        hidden_states, x_normed = fused_residual_rmsnorm(
            residual, attn_output, post_ln_w, eps, add_one_to_weight=True,
        )

        # ============================================================
        # MLP block
        # ============================================================
        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for gate+up projections (column-parallel) =====
        gate_up = bf16_linear_forward(x_2d, w_mlp_gate_up_q, w_mlp_gate_up_s)
        if gate_up.shape[0] == 1:
            _gu = gate_up.view(-1)
            gate_out = _gu[:tp_intermediate_size].view(1, -1)
            up_out = _gu[tp_intermediate_size:tp_intermediate_size * 2].view(1, -1)
        else:
            gate_out = gate_up[:, :tp_intermediate_size].contiguous()
            up_out = gate_up[:, tp_intermediate_size:].contiguous()
        mlp_mid = fused_silu_mul(gate_out, up_out)

        # down_proj (row-parallel -> All-Reduce)
        down_out = bf16_linear_forward(
            mlp_mid.reshape(-1, tp_intermediate_size), w_mlp_down_q, w_mlp_down_s,
        ).reshape(bsz, q_len, hidden_size)
        ar_work2 = None
        if is_tp_active():
            ar_work2 = tp_all_reduce_async(down_out)

        # Inter-layer chaining: fuse residual-add + next layer's input norm
        if ar_work2 is not None:
            ar_work2.wait()
        if next_layer_norm_w is not None and q_len == 1:
            hidden_states, x_normed_next = fused_residual_rmsnorm(
                hidden_states, down_out, next_layer_norm_w, eps, add_one_to_weight=True,
            )
            return hidden_states, x_normed_next
        else:
            hidden_states = hidden_states + down_out
            return hidden_states

    return fused_forward


def _patch_final_norm(model: nn.Module, eps: float):
    """Patch final RMSNorm to use CUDA kernel."""
    final_norm = model.model.norm
    final_norm_w = final_norm.weight

    def fused_norm_forward(hidden_states):
        return rmsnorm(hidden_states, final_norm_w, eps, add_one_to_weight=True)

    final_norm._fused = True
    final_norm.forward = fused_norm_forward


def verify_patch(model: nn.Module) -> bool:
    """Verify that all layers have been patched.

    Uses the ``_fused`` marker attribute set by ``patch_model`` rather than
    introspecting closure cells, which is both faster and more reliable.
    """
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    num_layers = config.num_hidden_layers

    patched = 0
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        if getattr(layer, '_fused', False):
            patched += 1

    print(f"Patch verification: {patched}/{num_layers} layers patched")
    return patched == num_layers
