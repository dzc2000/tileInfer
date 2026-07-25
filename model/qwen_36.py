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
    tp_all_reduce, shard_weight_col, shard_weight_row, shard_bias_col,
)


# Layer pattern: [DeltaNet, DeltaNet, DeltaNet, Attention] x 16
DELTANET_PATTERN = [True, True, True, False] * 16  # True = DeltaNet

# DeltaNet dimension constants (from model config)
NUM_K_HEADS = 16
HEAD_K_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM  # 2048
NUM_V_HEADS = 48
HEAD_V_DIM = 128
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM  # 6144
CONV_DIM = KEY_DIM + KEY_DIM + VALUE_DIM  # 10240 (Q+K+V channels for conv1d)

# Full attention constants
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
Q_OUT_DIM = NUM_Q_HEADS * HEAD_DIM  # 6144
KV_OUT_DIM = NUM_KV_HEADS * HEAD_DIM  # 1024
Q_PROJ_DIM = 12288  # 2 * Q_OUT_DIM (includes gate for output gating)

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408

# Pre-computed constants
INV_SQRT_DK = HEAD_K_DIM ** -0.5  # 1/sqrt(128)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Global paged KV cache reference (set by ModelRunner)
_paged_kv_cache = None  # [2, num_attention_layers, num_blocks, block_size, num_kv_heads, head_dim]

# Layer index mapping
def _get_attention_layer_idx(layer_idx: int) -> int:
    """Map global layer index to attention layer index (0-based)."""
    count = 0
    for i in range(layer_idx):
        if not DELTANET_PATTERN[i]:
            count += 1
    return count

def set_paged_kv_cache(kv_cache):
    """Set the global paged KV cache reference."""
    global _paged_kv_cache
    _paged_kv_cache = kv_cache

# Global paged DeltaNet state pools (set by ModelRunner)
_deltanet_recurrent_pool = None  # [num_dn_layers, max_slots, num_v_heads, Dk, Dv]
_deltanet_conv_pool = None       # [num_dn_layers, max_slots, conv_dim, kernel_size]
_seq_to_slot = None              # dict: seq_id -> slot

def set_deltanet_pools(recurrent_pool, conv_pool, seq_to_slot):
    """Set the global paged DeltaNet state pools."""
    global _deltanet_recurrent_pool, _deltanet_conv_pool, _seq_to_slot
    _deltanet_recurrent_pool = recurrent_pool
    _deltanet_conv_pool = conv_pool
    _seq_to_slot = seq_to_slot

class _DeltaNetCacheView:
    """View into paged DeltaNet state for a specific batch."""
    __slots__ = ('conv_state', 'recurrent_state', 'dn_idx', 'slots')
    def __init__(self, conv_state, recurrent_state, dn_idx=None, slots=None):
        self.conv_state = conv_state
        self.recurrent_state = recurrent_state
        self.dn_idx = dn_idx
        self.slots = slots

def _get_deltanet_cache(layer_idx, bsz):
    """Get DeltaNet cache for this layer from paged pool."""
    dn_idx = 0
    for i in range(layer_idx):
        if DELTANET_PATTERN[i]:
            dn_idx += 1

    from engine.context import get_context
    ctx = get_context()
    if hasattr(ctx, 'deltanet_slots') and ctx.deltanet_slots is not None:
        slots = ctx.deltanet_slots  # [bsz] tensor of slot indices
        # Always use fancy indexing (tensor index) for CUDA Graph compatibility.
        # .item() would freeze the slot value during graph capture, causing all
        # replays to write to the same slot regardless of the actual sequence.
        # Fancy indexing returns a COPY; writeback is required after modification.
        conv_state = _deltanet_conv_pool[dn_idx, slots]  # [bsz, conv_dim, kernel_size]
        recurrent_state = _deltanet_recurrent_pool[dn_idx, slots]  # [bsz, num_v_heads, Dk, Dv]
    else:
        slots = None
        tp_size = get_tp_world_size()
        tp_conv_dim = CONV_DIM // tp_size
        tp_num_v_heads = NUM_V_HEADS // tp_size
        conv_state = torch.zeros(bsz, tp_conv_dim, 4, device="cuda", dtype=torch.bfloat16)
        recurrent_state = torch.zeros(bsz, tp_num_v_heads, HEAD_K_DIM, HEAD_V_DIM,
                                       device="cuda", dtype=torch.bfloat16)
    return _DeltaNetCacheView(conv_state, recurrent_state, dn_idx, slots)

def _writeback_deltanet_state(dn_idx, slots, final_state, dn_cache):
    """Write back DeltaNet recurrent state to paged pool after prefill."""
    if _deltanet_recurrent_pool is not None and slots is not None:
        _deltanet_recurrent_pool[dn_idx, slots] = final_state.to(_deltanet_recurrent_pool.dtype)

# Split indices for pre-concatenated DeltaNet projection weights
# w_delta_proj = cat(w_qkv[10240], w_a[48], w_b[48], w_z[6144]) = [16480, 5120]
DELTA_SPLIT_QKV = CONV_DIM              # 10240
DELTA_SPLIT_A = DELTA_SPLIT_QKV + NUM_V_HEADS  # 10288
DELTA_SPLIT_B = DELTA_SPLIT_A + NUM_V_HEADS    # 10336
DELTA_PROJ_DIM = DELTA_SPLIT_B + VALUE_DIM     # 16480

# Split indices for pre-concatenated attention QKV weights
# w_attn_qkv = cat(q_proj[12288], k_proj[1024], v_proj[1024]) = [14336, 5120]
ATTN_SPLIT_Q = Q_PROJ_DIM               # 12288
ATTN_SPLIT_K = ATTN_SPLIT_Q + KV_OUT_DIM  # 13312
ATTN_QKV_DIM = ATTN_SPLIT_K + KV_OUT_DIM  # 14336


def is_deltanet_layer(layer_idx: int) -> bool:
    return DELTANET_PATTERN[layer_idx]


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


def patch_model(model: nn.Module, device: str = "cuda") -> nn.Module:
    """Patch all 64 layers with fused CUDA kernels.

    Args:
        model: The HuggingFace model (may be on CPU)
        device: Target device for sharded weights (e.g. "cuda:0")
    """
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    hidden_size = config.hidden_size
    eps = getattr(config, 'rms_norm_eps', 1e-6)
    intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers

    print(f"Patching {num_layers} layers (weight dedup + fused QKnorm+RoPE + sigmoid gate)...")

    for layer_idx in range(num_layers):
        if layer_idx % 8 == 0:
            print(f"  Patching layer {layer_idx}/{num_layers}...")
        layer = model.model.layers[layer_idx]

        next_norm_w = None
        if layer_idx < num_layers - 1:
            next_norm_w = model.model.layers[layer_idx + 1].input_layernorm.weight

        if is_deltanet_layer(layer_idx):
            fwd = _create_fused_deltanet_forward(
                layer, layer_idx,
                hidden_size=hidden_size,
                eps=eps,
                intermediate_size=intermediate_size,
                next_layer_norm_w=next_norm_w,
                device=device,
            )
        else:
            fwd = _create_fused_attention_forward(
                layer, layer_idx,
                hidden_size=hidden_size,
                eps=eps,
                intermediate_size=intermediate_size,
                next_layer_norm_w=next_norm_w,
                device=device,
            )

        layer.forward = fwd

    _patch_final_norm(model, eps)

    gc.collect()
    torch.cuda.empty_cache()

    vram_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"Patching complete. Paged KV cache mode, VRAM: {vram_gb:.1f}GB")
    return model


def _create_fused_deltanet_forward(
    layer: nn.Module,
    layer_idx: int,
    hidden_size: int,
    eps: float,
    intermediate_size: int,
    next_layer_norm_w=None,
    device: str = "cuda",
):
    """Create fused forward for a Qwen3_6GatedDeltaNet layer.

    Optimizations:
    - Pre-concatenated projection weights: 4->1 GEMV for DeltaNet proj
    - Pre-concatenated MLP weights: 2->1 GEMV for MLP
    - Fused residual+RMSNorm between attention and MLP
    - Pre-computed -exp(A) and dt_bias.float()
    - Tensor Parallelism: column-parallel for projections, row-parallel for out/down
    """
    # Norm weights (replicated, not sharded)
    input_ln_w = layer.input_layernorm.weight
    post_ln_w = layer.post_attention_layernorm.weight

    attn = layer.linear_attn  # Qwen3_6GatedDeltaNet

    # ===== TP context =====
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()

    # TP-adjusted dimensions
    tp_num_k_heads = NUM_K_HEADS // tp_size
    tp_num_v_heads = NUM_V_HEADS // tp_size
    tp_key_dim = tp_num_k_heads * HEAD_K_DIM
    tp_value_dim = tp_num_v_heads * HEAD_V_DIM
    tp_conv_dim = tp_key_dim * 2 + tp_value_dim  # Q+K+V channels
    tp_intermediate_size = INTERMEDIATE_SIZE // tp_size

    # ===== PRE-CONCATENATE DELTANET PROJECTION WEIGHTS + DEDUP + SHARD =====
    # Full: w_qkv[10240,5120] + w_a[48,5120] + w_b[48,5120] + w_z[6144,5120] = [16480,5120]
    # Column-parallel: shard each sub-weight along dim=0 (output dim)
    w_qkv_full = attn.in_proj_qkv.weight    # [10240, 5120]
    w_a_full = attn.in_proj_a.weight         # [48, 5120]
    w_b_full = attn.in_proj_b.weight         # [48, 5120]
    w_z_full = attn.in_proj_z.weight         # [6144, 5120]

    # Shard QKV weight: must split Q/K/V segments separately to preserve layout
    # w_qkv_full [10240, 5120] = [Q(2048) | K(2048) | V(6144)]
    q_part = w_qkv_full[:KEY_DIM]                        # [2048, 5120]
    k_part = w_qkv_full[KEY_DIM:KEY_DIM * 2]             # [2048, 5120]
    v_part = w_qkv_full[KEY_DIM * 2:]                    # [6144, 5120]
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
    conv_q = conv_w_full[:KEY_DIM]
    conv_k = conv_w_full[KEY_DIM:KEY_DIM * 2]
    conv_v = conv_w_full[KEY_DIM * 2:]
    conv_w = torch.cat([
        shard_weight_col(conv_q, tp_rank, tp_size),
        shard_weight_col(conv_k, tp_rank, tp_size),
        shard_weight_col(conv_v, tp_rank, tp_size),
    ], dim=0).contiguous()  # [10240/tp, 4]
    conv_b_full = getattr(conv1d, 'bias', None)
    conv_b = shard_bias_col(conv_b_full, tp_rank, tp_size) if conv_b_full is not None else None

    # Output norm (replicated per v_head, not sharded)
    norm_w = attn.norm.weight  # [128] (head_v_dim, same across TP ranks)

    # Output projection (row-parallel: split input dim = VALUE_DIM)
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

        if q_len > 1:
            # ====== PREFILL PATH - FlashQLA + tilelang kernels ======
            residual = hidden_states
            x_normed = rmsnorm(hidden_states, input_ln_w, eps, add_one_to_weight=True)
            x_2d = x_normed.reshape(-1, hidden_size)

            # Single GEMV for all DeltaNet projections (column-parallel, no comm)
            proj_all = bf16_linear_forward(x_2d, w_delta_proj)  # [B*T, delta_proj_dim]
            proj_all = proj_all.reshape(bsz, q_len, -1)
            qkv = proj_all[:, :, :delta_split_qkv]                # [B, T, tp_conv_dim]
            alpha = proj_all[:, :, delta_split_qkv:delta_split_a]  # [B, T, tp_num_v_heads]
            beta_raw = proj_all[:, :, delta_split_a:delta_split_b]  # [B, T, tp_num_v_heads]
            z = proj_all[:, :, delta_split_b:].contiguous()         # [B, T, tp_value_dim]

            # Causal conv1d for prefill (parallel, single kernel launch)
            dn_cache = _get_deltanet_cache(layer_idx, bsz)
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
                    _deltanet_conv_pool[dn_cache.dn_idx, dn_cache.slots] = dn_cache.conv_state

            # Extract Q, K, V from conv output (TP-adjusted dimensions)
            q_conv = qkv_conv[:, :, :tp_key_dim].reshape(bsz, q_len, tp_num_k_heads, HEAD_K_DIM)
            k_conv = qkv_conv[:, :, tp_key_dim:tp_key_dim*2].reshape(bsz, q_len, tp_num_k_heads, HEAD_K_DIM)
            v_conv = qkv_conv[:, :, tp_key_dim*2:].reshape(bsz, q_len, tp_num_v_heads, HEAD_V_DIM)

            # L2 normalize Q and K along head_dim (matches decode kernel's
            # fused_postproj_recurrent which normalizes internally).
            # Scale=INV_SQRT_DK is applied to the output, equivalent to scaling Q here.
            q_conv_f = q_conv.float()
            k_conv_f = k_conv.float()
            q_conv = (q_conv_f / q_conv_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)).to(q_conv.dtype)
            k_conv = (k_conv_f / k_conv_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)).to(k_conv.dtype)

            # Compute gate (g) and beta for FlashQLA (TP-adjusted)
            alpha_f = alpha.float()
            neg_a_exp_expanded = neg_A_exp.unsqueeze(0).unsqueeze(0)  # [1, 1, tp_num_v_heads]
            dt_bias_expanded = dt_bias_f.unsqueeze(0).unsqueeze(0)    # [1, 1, tp_num_v_heads]
            sp_input = alpha_f + dt_bias_expanded
            sp = torch.where(sp_input > 20.0, sp_input, torch.log(1.0 + torch.exp(sp_input)))
            g = neg_a_exp_expanded * sp  # [B, T, tp_num_v_heads]
            beta = torch.sigmoid(beta_raw.float())  # [B, T, tp_num_v_heads]

            # FlashQLA chunk_gated_delta_rule
            initial_state = None
            if dn_cache.slots is not None:
                initial_state = dn_cache.recurrent_state

            o_flash, final_state = chunk_gated_delta_rule(
                q=q_conv,           # [B, T, tp_num_k_heads, HEAD_K_DIM]
                k=k_conv,           # [B, T, tp_num_k_heads, HEAD_K_DIM]
                v=v_conv,           # [B, T, tp_num_v_heads, HEAD_V_DIM]
                g=g,                # [B, T, tp_num_v_heads]
                beta=beta,          # [B, T, tp_num_v_heads]
                scale=INV_SQRT_DK,
                initial_state=initial_state,
                output_final_state=True,
            )

            # Update recurrent state
            _writeback_deltanet_state(dn_cache.dn_idx, dn_cache.slots, final_state, dn_cache)

            # RMSNorm + gating (TP-adjusted)
            z_3d = z.reshape(bsz, q_len, tp_num_v_heads, HEAD_V_DIM)
            o_normed = rmsnorm(o_flash.reshape(bsz * q_len * tp_num_v_heads, HEAD_V_DIM),
                               norm_w, eps, add_one_to_weight=False)
            o_normed = o_normed.reshape(bsz, q_len, tp_num_v_heads, HEAD_V_DIM)
            # Gate = o_normed * SiLU(z) = o_normed * z * sigmoid(z).
            # fused_silu_mul(z, o_normed) computes z * sigmoid(z) * o_normed in a
            # single kernel (previously sigmoid_mul + an extra elementwise mul),
            # matching the decode path which fuses this inside the recurrent kernel.
            attn_output = fused_silu_mul(z_3d, o_normed)
            attn_output = attn_output.reshape(bsz, q_len, tp_value_dim)

            # Output projection (row-parallel -> All-Reduce)
            o_2d = attn_output.reshape(-1, tp_value_dim)
            attn_output = bf16_linear_forward(o_2d, w_out).reshape(bsz, q_len, hidden_size)
            if is_tp_active():
                tp_all_reduce(attn_output)

            # Fused residual + RMSNorm
            hidden_states, x_normed = fused_residual_rmsnorm(
                residual, attn_output, post_ln_w, eps, add_one_to_weight=True,
            )

            # MLP block (column-parallel gate_up, row-parallel down -> All-Reduce)
            x_2d = x_normed.reshape(-1, hidden_size)
            gate_up = bf16_linear_forward(x_2d, w_mlp_gate_up)
            gate_out = gate_up[:, :tp_intermediate_size]
            up_out = gate_up[:, tp_intermediate_size:]
            mlp_mid = fused_silu_mul(gate_out, up_out)
            down_out = bf16_linear_forward(
                mlp_mid.reshape(-1, tp_intermediate_size), w_mlp_down,
            ).reshape(bsz, q_len, hidden_size)
            if is_tp_active():
                tp_all_reduce(down_out)

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
        proj_all = bf16_linear_forward(x_2d, w_delta_proj)  # [M, delta_proj_dim]
        qkv = proj_all[:, :delta_split_qkv].contiguous()                # [M, tp_conv_dim]
        alpha = proj_all[:, delta_split_qkv:delta_split_a].contiguous()  # [M, tp_num_v_heads]
        beta_raw = proj_all[:, delta_split_a:delta_split_b].contiguous()  # [M, tp_num_v_heads]
        z = proj_all[:, delta_split_b:].contiguous()         # [M, tp_value_dim]

        dn_cache = _get_deltanet_cache(layer_idx, bsz)

        # Fused causal conv1d update (TP-adjusted conv_dim)
        qkv_flat = qkv.reshape(bsz, tp_conv_dim)
        qkv_conv = causal_conv1d_update(
            qkv_flat, dn_cache.conv_state, conv_w, conv_b, apply_silu=True,
        )

        # Write back conv state to paged pool (fancy indexing creates a copy)
        if dn_cache.slots is not None:
            _deltanet_conv_pool[dn_cache.dn_idx, dn_cache.slots] = dn_cache.conv_state

        # ===== FUSED POST-PROJ + RECURRENT (TP-adjusted) =====
        z_3d = z.reshape(bsz, tp_num_v_heads, HEAD_V_DIM)
        attn_output, dn_cache.recurrent_state = fused_postproj_recurrent(
            qkv_conv, alpha, beta_raw, neg_A_exp_bf16, dt_bias_f_bf16,
            dn_cache.recurrent_state,
            num_k_heads=tp_num_k_heads, num_v_heads=tp_num_v_heads, head_dim=HEAD_K_DIM,
            norm_weight=norm_w, z=z_3d, norm_eps=eps,
        )
        attn_output = attn_output.reshape(bsz, 1, tp_value_dim)

        # Write back recurrent state to paged pool
        if dn_cache.slots is not None:
            _deltanet_recurrent_pool[dn_cache.dn_idx, dn_cache.slots] = dn_cache.recurrent_state

        # Output projection (row-parallel -> All-Reduce)
        o_2d = attn_output.reshape(-1, tp_value_dim)
        attn_output = bf16_linear_forward(o_2d, w_out).reshape(bsz, q_len, hidden_size)
        if is_tp_active():
            tp_all_reduce(attn_output)

        # ===== FUSED RESIDUAL + RMSNORM =====
        hidden_states, x_normed = fused_residual_rmsnorm(
            residual, attn_output, post_ln_w, eps, add_one_to_weight=True,
        )

        # ============================================================
        # MLP block
        # ============================================================
        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for gate+up projections (column-parallel) =====
        gate_up = bf16_linear_forward(x_2d, w_mlp_gate_up)
        gate_out = gate_up[:, :tp_intermediate_size]
        up_out = gate_up[:, tp_intermediate_size:]

        mlp_mid = fused_silu_mul(gate_out, up_out)

        # down_proj (row-parallel -> All-Reduce)
        down_out = bf16_linear_forward(
            mlp_mid.reshape(-1, tp_intermediate_size), w_mlp_down,
        ).reshape(bsz, q_len, hidden_size)
        if is_tp_active():
            tp_all_reduce(down_out)

        # Inter-layer chaining: fuse residual-add + next layer's input norm
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
    input_ln_w = layer.input_layernorm.weight
    post_ln_w = layer.post_attention_layernorm.weight
    self_attn = layer.self_attn
    mlp = layer.mlp

    # ===== TP context =====
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()

    # TP-adjusted dimensions
    tp_num_q_heads = NUM_Q_HEADS // tp_size
    tp_num_kv_heads = NUM_KV_HEADS // tp_size
    tp_q_out_dim = tp_num_q_heads * HEAD_DIM
    tp_kv_out_dim = tp_num_kv_heads * HEAD_DIM
    tp_q_proj_dim = tp_num_q_heads * HEAD_DIM * 2  # Q + gate
    tp_intermediate_size = INTERMEDIATE_SIZE // tp_size

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

    # O projection: row-parallel (split input = Q_OUT_DIM)
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
        qkv_all = bf16_linear_forward(x_2d, w_attn_qkv)  # [M, attn_qkv_dim]
        q_full = qkv_all[:, :attn_split_q]           # [M, tp_q_proj_dim]
        k = qkv_all[:, attn_split_q:attn_split_k]    # [M, tp_kv_out_dim]
        v = qkv_all[:, attn_split_k:]                 # [M, tp_kv_out_dim]

        # q_proj output is tp_q_proj_dim = tp_num_q_heads * HEAD_DIM * 2 (interleaved Q + gate)
        q_full_heads = q_full.reshape(bsz, q_len, tp_num_q_heads, HEAD_DIM * 2)
        q, q_gate = q_full_heads.chunk(2, dim=-1)
        q_gate = q_gate.reshape(bsz, q_len, -1)

        # Reshape Q/K/V to [B, num_heads, S, head_dim] (TP-adjusted)
        q = q.transpose(1, 2).contiguous()
        k = k.reshape(bsz, q_len, tp_num_kv_heads, HEAD_DIM).transpose(1, 2).contiguous()
        v = v.reshape(bsz, q_len, tp_num_kv_heads, HEAD_DIM).transpose(1, 2).contiguous()

        # RoPE position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
        elif hasattr(self_attn, 'rotary_emb'):
            cos, sin = self_attn.rotary_emb(v, position_ids)
        else:
            cos, sin = None, None

        if q_len == 1 and bsz == 1 and cos is not None and q_norm_w is not None:
            # ===== FUSED QK-NORM + PARTIAL ROPE (single-sequence decode only) =====
            cos_flat = cos.reshape(-1)[:HEAD_DIM] if cos.numel() > HEAD_DIM else cos.reshape(-1)
            sin_flat = sin.reshape(-1)[:HEAD_DIM] if sin.numel() > HEAD_DIM else sin.reshape(-1)
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
                if rope_dim < HEAD_DIM:
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
        k_store = k.transpose(1, 2).reshape(-1, tp_num_kv_heads, HEAD_DIM).contiguous()
        v_store = v.transpose(1, 2).reshape(-1, tp_num_kv_heads, HEAD_DIM).contiguous()
        attn_layer_idx = _get_attention_layer_idx(layer_idx)
        k_cache = _paged_kv_cache[0, attn_layer_idx].reshape(-1, tp_num_kv_heads * HEAD_DIM)
        v_cache = _paged_kv_cache[1, attn_layer_idx].reshape(-1, tp_num_kv_heads * HEAD_DIM)
        store_kvcache(k_store, v_store, k_cache, v_cache, ctx.slot_mapping)

        if q_len > 1:
            # ===== PREFILL: TileLang GQA varlen flash attention ======
            groups = tp_num_q_heads // tp_num_kv_heads  # same ratio = 6
            q_unpad = q.transpose(1, 2).reshape(bsz * q_len, tp_num_q_heads, HEAD_DIM).contiguous()

            # For chunked prefill continuation (num_cached_tokens > 0), the
            # attention must see the full KV context (cached prefix + current
            # chunk). Gather K/V from the paged cache which already contains
            # both the prefix (from prior chunks) and the current chunk
            # (stored above via store_kvcache).
            full_kv_len = ctx.max_seqlen_k if ctx is not None else q_len
            if full_kv_len > q_len and ctx is not None and ctx.block_tables is not None:
                k_cache_paged = _paged_kv_cache[0, attn_layer_idx]
                v_cache_paged = _paged_kv_cache[1, attn_layer_idx]
                page_size = _paged_kv_cache.shape[3]
                positions = torch.arange(full_kv_len, device=q.device)
                block_indices = positions // page_size
                offsets_in_block = positions % page_size
                physical_blocks = ctx.block_tables[0, block_indices]
                k_unpad = k_cache_paged[physical_blocks, offsets_in_block].reshape(
                    full_kv_len, tp_num_kv_heads, HEAD_DIM).contiguous()
                v_unpad = v_cache_paged[physical_blocks, offsets_in_block].reshape(
                    full_kv_len, tp_num_kv_heads, HEAD_DIM).contiguous()
                cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
                cu_seqlens_k = torch.tensor([0, full_kv_len], dtype=torch.int32, device=q.device)
                kernel = tilelang_flashattn(
                    1, groups, q_len, full_kv_len,
                    tp_num_q_heads, HEAD_DIM, is_causal=True,
                    block_M=128, block_N=64, num_stages=2, threads=128,
                    dtype=T.bfloat16 if q.dtype == torch.bfloat16 else T.float16,
                )
                attn_out_unpad = kernel(q_unpad, k_unpad, v_unpad,
                                        cu_seqlens_q, cu_seqlens_k, q_len)
            else:
                # First chunk (no cached prefix): K/V from current computation
                k_unpad = k.transpose(1, 2).reshape(bsz * q_len, tp_num_kv_heads, HEAD_DIM).contiguous()
                v_unpad = v.transpose(1, 2).reshape(bsz * q_len, tp_num_kv_heads, HEAD_DIM).contiguous()
                cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len,
                                           dtype=torch.int32, device=q.device)
                kernel = tilelang_flashattn(
                    bsz, groups, bsz * q_len, bsz * q_len,
                    tp_num_q_heads, HEAD_DIM, is_causal=True,
                    block_M=128, block_N=64, num_stages=2, threads=128,
                    dtype=T.bfloat16 if q.dtype == torch.bfloat16 else T.float16,
                )
                attn_out_unpad = kernel(q_unpad, k_unpad, v_unpad, cu_seqlens, cu_seqlens, q_len)

            attn_output = attn_out_unpad.reshape(bsz, q_len, tp_num_q_heads, HEAD_DIM)
            attn_output = attn_output.reshape(bsz, q_len, tp_q_out_dim)
        else:
            # ===== DECODE: TileLang GQA paged attention ======
            k_cache_paged = _paged_kv_cache[0, attn_layer_idx]
            v_cache_paged = _paged_kv_cache[1, attn_layer_idx]
            q_decode = q.squeeze(2).contiguous()
            o = gqa_decode_paged_fn(
                q_decode, k_cache_paged, v_cache_paged,
                block_table=ctx.block_tables,
                seqlen_kv=ctx.context_lens,
                page_size=_paged_kv_cache.shape[3],
            )
            attn_output = o.reshape(bsz, q_len, tp_q_out_dim)

        # ===== FUSED SIGMOID GATE MULTIPLY =====
        attn_output = fused_sigmoid_mul(attn_output, q_gate)

        # Output projection (row-parallel -> All-Reduce)
        o_2d = attn_output.reshape(-1, tp_q_out_dim)
        attn_output = bf16_linear_forward(o_2d, w_o).reshape(bsz, q_len, hidden_size)
        if is_tp_active():
            tp_all_reduce(attn_output)

        # ===== FUSED RESIDUAL + RMSNORM =====
        hidden_states, x_normed = fused_residual_rmsnorm(
            residual, attn_output, post_ln_w, eps, add_one_to_weight=True,
        )

        # ============================================================
        # MLP block
        # ============================================================
        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for gate+up (column-parallel) =====
        gate_up = bf16_linear_forward(x_2d, w_mlp_gate_up)
        gate_out = gate_up[:, :tp_intermediate_size]
        up_out = gate_up[:, tp_intermediate_size:]

        mlp_mid = fused_silu_mul(gate_out, up_out)

        # down_proj (row-parallel -> All-Reduce)
        down_out = bf16_linear_forward(
            mlp_mid.reshape(-1, tp_intermediate_size), w_mlp_down,
        ).reshape(bsz, q_len, hidden_size)
        if is_tp_active():
            tp_all_reduce(down_out)

        # Inter-layer chaining: fuse residual-add + next layer's input norm
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

    final_norm.forward = fused_norm_forward


def verify_patch(model: nn.Module) -> bool:
    """Verify that all layers have been patched."""
    num_layers = model.config.num_hidden_layers
    patched = 0
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        fwd = layer.forward
        if callable(fwd) and hasattr(fwd, '__closure__') and fwd.__closure__ is not None:
            patched += 1

    print(f"Patch verification: {patched}/{num_layers} layers patched")
    return patched == num_layers