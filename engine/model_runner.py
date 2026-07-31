"""
ModelRunner: executes model forward pass with paged KV cache.

Handles:
- Preparing input tensors from scheduled sequences
- Paged KV cache allocation and storage
- Prefill with FlashQLA (DeltaNet) + flash_attn_varlen_func (Attention)
- Decode with flash_attn_with_kvcache (Attention)
- DeltaNet recurrent state management (paged)
- Sampling
- Tensor Parallelism support
"""
import bisect
from collections import deque

import torch
from engine.config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.context import AttentionContext, set_context, reset_context
from engine.parallel import (get_tp_rank, get_tp_world_size, is_tp_active,
                             tp_broadcast, tp_distributed_argmax, tp_distributed_argmax_fused)
from kernels.lm_head_topk import fused_lm_head_argmax, fused_lm_head_argmax_with_max


class CUDAGraphRunner:
    """Holds static buffers and a captured CUDA graph for one decode batch size."""
    __slots__ = (
        'batch_size', 'graph', 'ctx',
        'static_input_ids', 'static_positions', 'static_slot_mapping',
        'static_block_tables', 'static_context_lens', 'static_deltanet_slots',
        'static_logits',
    )

    def __init__(self):
        self.batch_size = 0
        self.graph = None
        self.ctx = None
        self.static_input_ids = None
        self.static_positions = None
        self.static_slot_mapping = None
        self.static_block_tables = None
        self.static_context_lens = None
        self.static_deltanet_slots = None
        self.static_logits = None


class ModelRunner:
    def __init__(self, config: Config):
        self.config = config
        self.hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        Sequence.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager

        # Model architecture constants
        self.num_layers = self.hf_config.num_hidden_layers
        self.hidden_size = self.hf_config.hidden_size
        # dtype may be a string ("bfloat16") or a torch.dtype object
        _dt = getattr(self.hf_config, 'torch_dtype', None) or getattr(self.hf_config, 'dtype', None)
        if isinstance(_dt, str):
            self.dtype = getattr(torch, _dt, torch.bfloat16)
        elif isinstance(_dt, torch.dtype):
            self.dtype = _dt
        else:
            self.dtype = torch.bfloat16

        # TP context
        self.tp_size = config.tp_size
        self.tp_rank = get_tp_rank()

        # Attention layer config (full model dimensions)
        self.num_q_heads = self.hf_config.num_attention_heads
        self.num_kv_heads = self.hf_config.num_key_value_heads
        self.head_dim = getattr(self.hf_config, 'head_dim',
                                self.hidden_size // self.num_q_heads)

        # DeltaNet layer config (full model dimensions, from config)
        self.num_k_heads = getattr(self.hf_config, 'linear_num_key_heads', 16)
        self.num_v_heads = getattr(self.hf_config, 'linear_num_value_heads', 48)
        self.head_k_dim = getattr(self.hf_config, 'linear_key_head_dim', 128)
        self.head_v_dim = getattr(self.hf_config, 'linear_value_head_dim', 128)

        # TP-adjusted dimensions (per rank)
        self.tp_num_q_heads = self.num_q_heads // self.tp_size
        self.tp_num_kv_heads = self.num_kv_heads // self.tp_size
        self.tp_num_k_heads = self.num_k_heads // self.tp_size
        self.tp_num_v_heads = self.num_v_heads // self.tp_size
        self.tp_conv_dim = self.tp_num_k_heads * self.head_k_dim * 2 + self.tp_num_v_heads * self.head_v_dim
        self.tp_intermediate_size = self.hf_config.intermediate_size // self.tp_size

        # Layer pattern: read from config's layer_types (linear_attention / full_attention)
        layer_types = getattr(self.hf_config, 'layer_types', None)
        if layer_types is not None:
            self.deltanet_pattern = [t == 'linear_attention' for t in layer_types]
        else:
            # Fallback: [DeltaNet, DeltaNet, DeltaNet, Attention] x 16
            self.deltanet_pattern = [True, True, True, False] * 16
        self.num_deltanet_layers = sum(self.deltanet_pattern)
        self.num_attention_layers = self.num_layers - self.num_deltanet_layers

        # KV cache and state pools (allocated later)
        from collections import deque
        self.kv_cache = None
        self.deltanet_recurrent_pool = None
        self.deltanet_conv_pool = None
        self.free_deltanet_slots: deque[int] = deque()
        self.seq_to_slot: dict[int, int] = {}

        self.model = None

        # CUDA Graph state (populated by capture_decode_graphs)
        self.cuda_graphs: dict[int, CUDAGraphRunner] = {}
        self.graph_pool = None
        self.padding_deltanet_slot = -1
        self.max_blocks_per_seq = (config.max_model_len // self.block_size) + 2
        self._sorted_buckets: list[int] = []

        # Pinned-memory tensors for decode (allocated in capture_decode_graphs)
        self._pinned_input_ids = None
        self._pinned_positions = None
        self._pinned_slot_mapping = None
        self._pinned_context_lens = None
        self._pinned_deltanet_slots = None
        self._pinned_block_tables = None

        # Cache for chunked prefill gather indices: (seq_id, start) -> (block_table_copy, slot_mapping_tensor)
        self._prefill_gather_cache: dict[tuple[int, int], tuple[list[int], torch.Tensor]] = {}

    def load_model(self):
        """Load Qwen3.6-27B model and apply tilelang kernels.

        Loading strategy for TP on memory-constrained GPUs (e.g. 2x A100 40GB):
        1. Load full model on CPU (low_cpu_mem_usage avoids peak 2x memory)
        2. patch_model() shards weights for TP and moves shards to target GPU
        3. Move remaining model parts (embed, norm, lm_head) to GPU
        4. Free CPU memory
        """
        import gc
        from transformers import AutoModelForCausalLM
        from model.qwen_36 import patch_model

        device = f"cuda:{self.tp_rank}" if self.tp_size > 1 else "cuda"
        print(f"[Rank {self.tp_rank}] Loading {self.config.model} in BF16 (CPU -> shard -> {device})...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.requires_grad_(False)

        self._normalize_model_structure()

        patch_model(self.model, device=device)

        self.model.model.embed_tokens.to(device)
        self.model.model.norm.to(device)
        self.model.model.rotary_emb.to(device)
        if hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
            if self.tp_size > 1:
                from engine.parallel import shard_weight_col
                lm_w = self.model.lm_head.weight.data
                self.model.lm_head.weight.data = shard_weight_col(
                    lm_w, self.tp_rank, self.tp_size).contiguous().to(device)
            else:
                self.model.lm_head.to(device)

        for layer in self.model.model.layers:
            for param in layer.parameters():
                param.data = torch.empty(0)
            for buf in layer.buffers():
                buf.data = torch.empty(0)

        gc.collect()
        torch.cuda.empty_cache()

        vram_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"[Rank {self.tp_rank}] Model loaded: {vram_gb:.1f}GB VRAM")

    def _normalize_model_structure(self):
        """Ensure model has the expected structure: model.model.layers, model.lm_head.

        Handles three cases:
        1. ForCausalLM wrapper: model.model.layers exists -> no-op
        2. Multimodal wrapper: model.model.model.layers exists -> unwrap one level
        3. Base model only (no .model attr): wrap it
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
            inner = self.model.model
            if hasattr(inner, 'model') and hasattr(inner.model, 'layers'):
                lm_head = getattr(self.model, 'lm_head', None)
                self.model.model = inner.model
                if lm_head is None and hasattr(inner, 'lm_head'):
                    self.model.lm_head = inner.lm_head
                return

        base = self.model
        class _Wrapper:
            pass
        wrapper = _Wrapper()
        wrapper.model = base
        wrapper.lm_head = getattr(base, 'lm_head', None)
        wrapper.config = base.config
        self.model = wrapper

    def allocate_kv_cache(self):
        """Allocate paged KV cache and DeltaNet state pools (TP-aware)."""
        config = self.config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.max_memory_allocated()
        current = torch.cuda.memory_allocated()

        # Attention KV cache: per block per attn layer
        # [2, num_attn_layers, num_blocks, block_size, tp_num_kv_heads, head_dim]
        block_bytes_per_attn_layer = (
            2 * self.block_size * self.tp_num_kv_heads * self.head_dim * 2  # 2=K+V, 2=bf16 bytes
        )
        total_attn_block_bytes = block_bytes_per_attn_layer * self.num_attention_layers

        # DeltaNet state pools (TP-adjusted)
        # +1 extra slot reserved for CUDA Graph padding (padding_deltanet_slot)
        max_slots = config.max_num_seqs + 1
        tp_conv_dim = self.tp_conv_dim
        # Read conv1d kernel size from hf_config (Qwen3.6 exposes it as
        # conv_kernel_size). Fallback to 4 if not present.
        kernel_size = getattr(self.hf_config, 'conv_kernel_size', None)
        if kernel_size is None:
            kernel_size = getattr(self.hf_config, 'ssm_conv_kernel', None)
        if kernel_size is None:
            kernel_size = 4

        deltanet_state_bytes = (
            max_slots * self.tp_num_v_heads * self.head_k_dim * self.head_v_dim * 2
            * self.num_deltanet_layers
        )
        deltanet_conv_bytes = (
            max_slots * tp_conv_dim * kernel_size * 2 * self.num_deltanet_layers
        )

        # Compute available memory for KV cache blocks
        available_bytes = int(total * config.gpu_memory_utilization) - used
        available_bytes -= deltanet_state_bytes + deltanet_conv_bytes

        num_blocks = max(1, available_bytes // total_attn_block_bytes)
        config.num_kvcache_blocks = num_blocks

        # Allocate paged KV cache (TP-adjusted: tp_num_kv_heads per rank)
        self.kv_cache = torch.zeros(
            2, self.num_attention_layers, num_blocks, self.block_size,
            self.tp_num_kv_heads, self.head_dim,
            device="cuda", dtype=torch.bfloat16,
        )

        # Allocate DeltaNet recurrent state pool (TP-adjusted: tp_num_v_heads)
        self.deltanet_recurrent_pool = torch.zeros(
            self.num_deltanet_layers, max_slots, self.tp_num_v_heads,
            self.head_k_dim, self.head_v_dim,
            device="cuda", dtype=torch.bfloat16,
        )

        # Allocate DeltaNet conv state pool (TP-adjusted: tp_conv_dim)
        self.deltanet_conv_pool = torch.zeros(
            self.num_deltanet_layers, max_slots, tp_conv_dim, kernel_size,
            device="cuda", dtype=torch.bfloat16,
        )

        # Free slot tracking
        self.free_deltanet_slots = deque(range(max_slots))
        self.seq_to_slot = {}

        # Clamp max_blocks_per_seq to the actual number of blocks available.
        # __init__ sets it to (max_model_len // block_size) + 2 for correctness,
        # but when GPU memory is scarce (few blocks), the static block_table
        # must not exceed the actual cache size.
        self.max_blocks_per_seq = min(self.max_blocks_per_seq, num_blocks)

        # Register with model module
        from model.qwen_36 import set_paged_kv_cache, set_deltanet_pools
        set_paged_kv_cache(self.kv_cache)
        set_deltanet_pools(self.deltanet_recurrent_pool, self.deltanet_conv_pool, self.seq_to_slot)

        cache_gb = (self.kv_cache.numel() * 2 + self.deltanet_recurrent_pool.numel() * 2
                    + self.deltanet_conv_pool.numel() * 2) / (1024**3)
        print(f"[Rank {self.tp_rank}] KV cache: {num_blocks} blocks x {self.block_size} tokens/block, "
              f"{self.num_attention_layers} attention layers, {self.tp_num_kv_heads} KV heads/rank")
        print(f"[Rank {self.tp_rank}] DeltaNet: {max_slots} state slots, {self.num_deltanet_layers} layers, "
              f"{self.tp_num_v_heads} v_heads/rank")
        print(f"[Rank {self.tp_rank}] Cache memory: {cache_gb:.1f}GB")

    def allocate_deltanet_slot(self, seq_id: int) -> int:
        """Allocate a DeltaNet state slot for a sequence."""
        if seq_id in self.seq_to_slot:
            return self.seq_to_slot[seq_id]
        assert self.free_deltanet_slots, "No free DeltaNet state slots"
        slot = self.free_deltanet_slots.popleft()
        self.seq_to_slot[seq_id] = slot
        return slot

    def free_deltanet_slot(self, seq_id: int):
        """Free a DeltaNet state slot and invalidate prefill gather cache for this seq."""
        slot = self.seq_to_slot.pop(seq_id, None)
        if slot is not None:
            self.deltanet_recurrent_pool[:, slot].zero_()
            self.deltanet_conv_pool[:, slot].zero_()
            self.free_deltanet_slots.append(slot)
        # Invalidate any cached prefill gather indices (block_table may change after preemption).
        keys_to_remove = [k for k in self._prefill_gather_cache if k[0] == seq_id]
        for k in keys_to_remove:
            del self._prefill_gather_cache[k]

    @torch.inference_mode()
    def _decode_forward_logits(self, input_ids: torch.Tensor, positions: torch.Tensor,
                               ctx: AttentionContext) -> torch.Tensor:
        """Run the decode forward pass up to logits (no sampling).

        This is the unit captured by CUDA Graph: embed -> layers -> norm -> lm_head.
        All KV-cache / DeltaNet metadata is read from ``ctx`` (which, during capture,
        references the static buffers so the recorded kernels point at stable memory).

        Returns LOCAL logits shard [bsz, 1, vocab/tp] without all-gather.
        The caller is responsible for distributed reduction (argmax or gather).
        """
        set_context(ctx)
        try:
            hidden = self.model.model.embed_tokens(input_ids)
            hidden = hidden.unsqueeze(1)                 # [bsz, 1, hidden]
            position_ids = positions.unsqueeze(1)        # [bsz, 1]
            position_embeddings = self.model.model.rotary_emb(hidden, position_ids)

            x_normed_next = None
            for layer in self.model.model.layers:
                result = layer(
                    hidden,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    x_normed_input=x_normed_next,
                )
                if isinstance(result, tuple) and len(result) == 2:
                    hidden, x_normed_next = result
                else:
                    hidden = result
                    x_normed_next = None

            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)          # [bsz, 1, vocab/tp]
        finally:
            reset_context()
        return logits

    def _all_gather_logits(self, logits_shard: torch.Tensor) -> torch.Tensor:
        """All-Gather sharded logits across TP ranks to reconstruct full vocab logits."""
        import torch.distributed as dist
        from engine.parallel import get_tp_group
        tp_size = self.tp_size
        if not hasattr(self, '_gather_buffers') or self._gather_buffers[0].shape != logits_shard.shape:
            self._gather_buffers = [torch.empty_like(logits_shard) for _ in range(tp_size)]
        dist.all_gather(self._gather_buffers, logits_shard, group=get_tp_group())
        return torch.cat(self._gather_buffers, dim=-1)

    def _graph_buckets(self) -> list[int]:
        """Powers-of-two batch sizes up to the configured capture limit."""
        max_b = min(self.config.max_num_seqs, self.config.cuda_graph_max_batch_size)
        buckets = []
        b = 1
        while b <= max_b:
            buckets.append(b)
            b *= 2
        return buckets

    def capture_decode_graphs(self):
        """Capture CUDA graphs for decode at power-of-two batch sizes.

        Must be called after ``allocate_kv_cache`` (needs the real KV cache and
        DeltaNet pools registered with the model module). A dedicated padding
        DeltaNet slot is reserved so padded rows never corrupt real sequence state.
        """
        if self.enforce_eager:
            return

        buckets = self._graph_buckets()
        if not buckets:
            return

        # Reserve the last slot for CUDA Graph padding (deterministic across ranks).
        self.padding_deltanet_slot = self.config.max_num_seqs
        try:
            self.free_deltanet_slots.remove(self.padding_deltanet_slot)
        except ValueError:
            pass

        max_blocks = self.max_blocks_per_seq
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.cuda_graphs = {}

        for bsz in buckets:
            runner = CUDAGraphRunner()
            runner.batch_size = bsz
            runner.static_input_ids = torch.zeros(bsz, dtype=torch.int64, device="cuda")
            runner.static_positions = torch.zeros(bsz, dtype=torch.int64, device="cuda")
            # Padded rows skip KV writes (-1) and use the dummy DeltaNet slot.
            runner.static_slot_mapping = torch.full((bsz,), -1, dtype=torch.int32, device="cuda")
            runner.static_block_tables = torch.zeros(bsz, max_blocks, dtype=torch.int32, device="cuda")
            runner.static_context_lens = torch.ones(bsz, dtype=torch.int32, device="cuda")
            runner.static_deltanet_slots = torch.full(
                (bsz,), self.padding_deltanet_slot, dtype=torch.int32, device="cuda")

            ctx = AttentionContext()
            ctx.slot_mapping = runner.static_slot_mapping
            ctx.block_tables = runner.static_block_tables
            ctx.context_lens = runner.static_context_lens
            ctx.deltanet_slots = runner.static_deltanet_slots
            ctx.max_seqlen_q = 1
            ctx.max_seqlen_k = 1
            ctx.is_prefill = False
            runner.ctx = ctx

            # Warmup on a side stream to trigger tilelang JIT for this shape.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._decode_forward_logits(
                        runner.static_input_ids, runner.static_positions, ctx)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.graph_pool):
                runner.static_logits = self._decode_forward_logits(
                    runner.static_input_ids, runner.static_positions, ctx)
            runner.graph = g
            self.cuda_graphs[bsz] = runner

        # Pre-sort bucket sizes for O(log n) bisect_left lookup in _select_bucket.
        self._sorted_buckets = sorted(self.cuda_graphs)

        # Pre-allocate CPU pinned-memory tensors reused every decode step to
        # avoid per-step Python list + torch.tensor allocations in
        # _prepare_decode_into_graph. Sized to the largest captured bucket.
        max_bsz = buckets[-1]
        self._pinned_input_ids = torch.empty(max_bsz, dtype=torch.int64, pin_memory=True)
        self._pinned_positions = torch.empty(max_bsz, dtype=torch.int64, pin_memory=True)
        self._pinned_slot_mapping = torch.empty(max_bsz, dtype=torch.int32, pin_memory=True)
        self._pinned_context_lens = torch.empty(max_bsz, dtype=torch.int32, pin_memory=True)
        self._pinned_deltanet_slots = torch.empty(max_bsz, dtype=torch.int32, pin_memory=True)
        self._pinned_block_tables = torch.empty(
            max_bsz, max_blocks, dtype=torch.int32, pin_memory=True)

        print(f"[Rank {self.tp_rank}] Captured CUDA graphs for decode batch sizes: {buckets}")

    def _select_bucket(self, n: int) -> int | None:
        """Smallest captured batch size >= n, or None if n exceeds all buckets."""
        buckets = self._sorted_buckets
        idx = bisect.bisect_left(buckets, n)
        if idx < len(buckets):
            return buckets[idx]
        return None

    def _run_decode_graph(self, input_ids: torch.Tensor, positions: torch.Tensor,
                          ctx: AttentionContext, n: int) -> torch.Tensor:
        """Copy real decode inputs into static buffers, replay the graph, return logits[:n]."""
        bsz = self._select_bucket(n)
        runner = self.cuda_graphs[bsz]

        runner.static_input_ids[:n].copy_(input_ids)
        runner.static_positions[:n].copy_(positions)
        runner.static_slot_mapping[:n].copy_(ctx.slot_mapping)
        width = ctx.block_tables.shape[1]
        runner.static_block_tables[:n, :width].copy_(ctx.block_tables)
        runner.static_context_lens[:n].copy_(ctx.context_lens)
        runner.static_deltanet_slots[:n].copy_(ctx.deltanet_slots)

        # Reset padded rows so they cannot corrupt real state.
        if n < bsz:
            runner.static_slot_mapping[n:].fill_(-1)
            runner.static_deltanet_slots[n:].fill_(self.padding_deltanet_slot)
            runner.static_context_lens[n:].fill_(1)
            runner.static_block_tables[n:].fill_(-1)

        # Clear residual columns beyond current width to prevent stale block ids
        # from a previous (wider) batch from being read by the paged attention kernel.
        max_blocks = runner.static_block_tables.shape[1]
        if width < max_blocks:
            runner.static_block_tables[:n, width:].fill_(-1)

        runner.graph.replay()
        return runner.static_logits[:n]

    def _prepare_decode_into_graph(self, seqs: list[Sequence], runner: CUDAGraphRunner, n: int):
        """Prepare decode inputs directly into CUDA graph static buffers.

        Avoids creating intermediate GPU tensors and the subsequent D2D copy.
        Reuses pre-allocated pinned-memory CPU tensors for async H2D transfer.
        """
        bsz = runner.batch_size
        block_size = self.block_size

        # Reuse pre-allocated pinned tensors (no per-step allocation).
        pids = self._pinned_input_ids
        ppos = self._pinned_positions
        pslots = self._pinned_slot_mapping
        pctx = self._pinned_context_lens
        pdn = self._pinned_deltanet_slots
        pbt = self._pinned_block_tables

        for i, seq in enumerate(seqs):
            pids[i] = seq.last_token
            pos = seq.num_tokens - 1
            ppos[i] = pos
            pctx[i] = seq.num_tokens
            block_idx = pos // block_size
            block_offset = pos % block_size
            physical_block = seq.block_table[block_idx]
            pslots[i] = physical_block * block_size + block_offset
            pdn[i] = self.seq_to_slot[seq.seq_id]
            bt_len = len(seq.block_table)
            pbt[i, :bt_len] = torch.tensor(seq.block_table, dtype=torch.int32)
            pbt[i, bt_len:].fill_(-1)

        # Fill padding rows in pinned block_tables (only rows that will be copied).
        if n < bsz:
            pbt[n:bsz].fill_(-1)

        runner.static_input_ids[:n].copy_(pids[:n], non_blocking=True)
        runner.static_positions[:n].copy_(ppos[:n], non_blocking=True)
        runner.static_slot_mapping[:n].copy_(pslots[:n], non_blocking=True)
        runner.static_context_lens[:n].copy_(pctx[:n], non_blocking=True)
        runner.static_deltanet_slots[:n].copy_(pdn[:n], non_blocking=True)
        runner.static_block_tables.copy_(pbt[:bsz], non_blocking=True)

        if n < bsz:
            runner.static_slot_mapping[n:].fill_(-1)
            runner.static_deltanet_slots[n:].fill_(self.padding_deltanet_slot)
            runner.static_context_lens[n:].fill_(1)
            runner.static_block_tables[n:].fill_(-1)

    def _build_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """Build padded block_tables tensor for paged attention."""
        max_blocks = max(len(seq.block_table) for seq in seqs)
        tables = [
            seq.block_table + [-1] * (max_blocks - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(tables, dtype=torch.int32, device="cuda")

    def _build_slot_mapping_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """Build slot_mapping for prefill: maps each token position to its physical slot in KV cache.

        Caches per-(seq_id, start) so that re-prefill of the same chunk (e.g. after
        preemption with the same block_table) reuses the precomputed tensor.
        """
        slot_mapping = []
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            cache_key = (seq.seq_id, start)
            cached = self._prefill_gather_cache.get(cache_key)
            if cached is not None and cached[0] == seq.block_table:
                # Block table unchanged — reuse cached gather indices.
                slot_mapping.append(cached[1])
                continue

            # Rebuild gather indices for this chunk.
            chunk_slots = []
            for pos in range(start, end):
                block_idx = pos // self.block_size
                block_offset = pos % self.block_size
                if block_idx < len(seq.block_table):
                    physical_block = seq.block_table[block_idx]
                    chunk_slots.append(physical_block * self.block_size + block_offset)
                else:
                    chunk_slots.append(-1)  # shouldn't happen
            chunk_tensor = torch.tensor(chunk_slots, dtype=torch.int32, device="cuda")
            self._prefill_gather_cache[cache_key] = (list(seq.block_table), chunk_tensor)
            slot_mapping.append(chunk_tensor)

        return torch.cat(slot_mapping) if len(slot_mapping) > 1 else slot_mapping[0]

    def _build_slot_mapping_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """Build slot_mapping for decode: one slot per sequence (the new token)."""
        slot_mapping = []
        for seq in seqs:
            pos = seq.num_tokens - 1  # position of the new token
            block_idx = pos // self.block_size
            block_offset = pos % self.block_size
            physical_block = seq.block_table[block_idx]
            slot_mapping.append(physical_block * self.block_size + block_offset)
        return torch.tensor(slot_mapping, dtype=torch.int32, device="cuda")

    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor, AttentionContext]:
        """Prepare input tensors for a prefill step.

        Builds varlen-style inputs by concatenating all sequences' tokens.
        Currently processes single sequence at a time for DeltaNet state
        simplicity. Multi-sequence prefill support requires changes to
        _get_deltanet_cache in qwen_36.py to handle multi-batch state.
        """
        if len(seqs) > 1:
            raise ValueError(
                "Multi-sequence prefill not yet supported - "
                "requires multi-batch DeltaNet state management."
            )
        if not seqs:
            raise ValueError("No sequences to prefill")

        seq = seqs[0]
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q

        # Token IDs for this chunk
        input_ids = torch.tensor(seq.token_ids[start:end], dtype=torch.int64, device="cuda")
        positions = torch.tensor(list(range(start, end)), dtype=torch.int64, device="cuda")

        # cu_seqlens for varlen attention
        cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.tensor([0, end], dtype=torch.int32, device="cuda")

        slot_mapping = self._build_slot_mapping_prefill(seqs)
        block_tables = self._build_block_tables(seqs)

        # DeltaNet slots
        deltanet_slots = torch.tensor(
            [self.seq_to_slot[seq.seq_id] for seq in seqs],
            dtype=torch.int32, device="cuda",
        )

        ctx = AttentionContext()
        ctx.slot_mapping = slot_mapping
        ctx.block_tables = block_tables
        ctx.context_lens = None
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = seqlen_q
        ctx.max_seqlen_k = end
        ctx.deltanet_slots = deltanet_slots
        ctx.is_prefill = True

        return input_ids, positions, ctx

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor, AttentionContext]:
        """Prepare input tensors for a decode step."""
        input_ids = []
        positions = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(seq.num_tokens - 1)
            context_lens.append(seq.num_tokens)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
        positions = torch.tensor(positions, dtype=torch.int64, device="cuda")
        context_lens = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
        slot_mapping = self._build_slot_mapping_decode(seqs)
        block_tables = self._build_block_tables(seqs)

        # DeltaNet slots
        deltanet_slots = torch.tensor(
            [self.seq_to_slot[seq.seq_id] for seq in seqs],
            dtype=torch.int32, device="cuda",
        )

        ctx = AttentionContext()
        ctx.slot_mapping = slot_mapping
        ctx.block_tables = block_tables
        ctx.context_lens = context_lens
        ctx.cu_seqlens_q = None
        ctx.cu_seqlens_k = None
        ctx.max_seqlen_q = 1
        ctx.max_seqlen_k = context_lens.max().item()
        ctx.deltanet_slots = deltanet_slots
        ctx.is_prefill = False

        return input_ids, positions, ctx

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Execute one step: prepare inputs, run model, sample tokens.

        Wraps _run_impl with CUDA OOM recovery: on OutOfMemoryError, clears the
        CUDA cache, preempts the last-added sequence to reduce batch size, and
        retries. After max_retries failed attempts, raises a clear error.
        """
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                return self._run_impl(seqs, is_prefill)
            except torch.cuda.OutOfMemoryError:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"CUDA OOM after {max_retries} retries with "
                        f"batch_size={len(seqs)}, is_prefill={is_prefill}. "
                        f"Try reducing max_num_seqs, max_model_len, or "
                        f"max_prefill_chunk_tokens."
                    ) from None
                # Clear fragmented cache and preempt the last sequence.
                torch.cuda.empty_cache()
                if len(seqs) > 1:
                    preempted = seqs.pop()
                    self.free_deltanet_slot(preempted.seq_id)
                # Retry with reduced batch (or same batch after cache clear).

    def _run_impl(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Core execution logic for one step (without OOM handling)."""
        # Allocate DeltaNet slots for new sequences
        for seq in seqs:
            if seq.seq_id not in self.seq_to_slot:
                self.allocate_deltanet_slot(seq.seq_id)

        # Zero DeltaNet state when re-prefilling from scratch (e.g. after preemption).
        # Preemption releases KV blocks and resets num_cached_tokens to 0, but the
        # DeltaNet slot retains stale recurrent state. Without this, FlashQLA would
        # start from the old state instead of zero, producing incorrect outputs.
        if is_prefill:
            for seq in seqs:
                if seq.num_cached_tokens == 0 and seq.seq_id in self.seq_to_slot:
                    slot = self.seq_to_slot[seq.seq_id]
                    self.deltanet_recurrent_pool[:, slot].zero_()
                    self.deltanet_conv_pool[:, slot].zero_()

        # CUDA Graph fast path: prepare inputs directly into static buffers,
        # avoiding intermediate tensor allocations and H2D transfers.
        if not is_prefill and not self.enforce_eager and self.cuda_graphs:
            n = len(seqs)
            bsz = self._select_bucket(n)
            if bsz is not None:
                runner = self.cuda_graphs[bsz]
                self._prepare_decode_into_graph(seqs, runner, n)
                # Restore the decode context that was active during graph capture.
                # The prefill step may have left a stale context with context_lens=None;
                # without this, gqa_decode_paged_fn gets seqlen_kv=None.
                set_context(runner.ctx)
                runner.graph.replay()
                logits = runner.static_logits[:n]
                token_ids = self._sample_decode(logits, seqs)
                all_greedy = all(seq.temperature < 1e-6 for seq in seqs)
                if is_tp_active() and not all_greedy:
                    token_ids = self._sync_tokens(token_ids)
                return token_ids

        # Prepare inputs
        if is_prefill:
            input_ids, positions, ctx = self.prepare_prefill(seqs)
        else:
            input_ids, positions, ctx = self.prepare_decode(seqs)

        # Set context for model forward
        set_context(ctx)

        try:
            # Forward pass
            hidden = self.model.model.embed_tokens(input_ids)

            # Reshape hidden for batch processing
            if is_prefill:
                # Prefill: single sequence, [seqlen, hidden] -> [1, seqlen, hidden] (bsz=1)
                hidden = hidden.unsqueeze(0)
                position_ids = positions.unsqueeze(0)  # [1, seqlen]
            else:
                # Decode: multiple sequences, [bsz, hidden] -> [bsz, 1, hidden]
                hidden = hidden.unsqueeze(1)
                position_ids = positions.unsqueeze(1)  # [bsz, 1]

            # Get position embeddings from rotary_emb
            position_embeddings = self.model.model.rotary_emb(hidden, position_ids)

            # Forward through layers.
            # During decode (q_len == 1) each layer returns (hidden, x_normed_next),
            # where x_normed_next is the next layer's input RMSNorm already fused into
            # the current layer's residual-add. Threading it through avoids recomputing
            # one RMSNorm per layer (the same optimization model/generate.py uses).
            # During prefill (q_len > 1) layers return a single tensor, so x_normed
            # stays None and every layer computes its own norm.
            x_normed_next = None
            for layer in self.model.model.layers:
                result = layer(
                    hidden,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    x_normed_input=x_normed_next,
                )
                if isinstance(result, tuple) and len(result) == 2:
                    hidden, x_normed_next = result
                else:
                    hidden = result
                    x_normed_next = None

            # Final norm + LM head (only needed if we need to sample a token)
            # For chunked prefill that's not the final chunk, skip sampling
            need_sample = True
            if is_prefill:
                seq = seqs[0]
                remaining_after = seq.num_tokens - seq.num_cached_tokens - seq.num_scheduled_tokens
                if remaining_after > 0:
                    # Chunked prefill: not the final chunk, no need to sample
                    need_sample = False

            if need_sample:
                hidden = self.model.model.norm(hidden)

                if is_prefill:
                    token_ids = self._sample_prefill_fused(hidden, seqs)
                else:
                    all_greedy = all(seq.temperature < 1e-6 for seq in seqs)
                    has_rep_penalty = any(seq.repetition_penalty > 1.0 for seq in seqs)
                    # Fused greedy argmax can't apply repetition penalty; fall back
                    # to materialized logits when penalty is active.
                    if all_greedy and not has_rep_penalty:
                        token_ids = self._decode_greedy_fused(hidden, seqs)
                    else:
                        logits = self.model.lm_head(hidden)
                        token_ids = self._sample_decode(logits, seqs)
            else:
                token_ids = [-1] * len(seqs)
        finally:
            reset_context()

        if is_tp_active():
            all_greedy = all(seq.temperature < 1e-6 for seq in seqs)
            if not all_greedy:
                token_ids = self._sync_tokens(token_ids)

        return token_ids

    def _sync_tokens(self, token_ids: list[int]) -> list[int]:
        """Broadcast sampled token ids from rank 0 so all TP ranks stay in lockstep."""
        tensor = torch.tensor(token_ids, dtype=torch.int64, device="cuda")
        tp_broadcast(tensor)
        return tensor.tolist()

    def _sample(self, logits: torch.Tensor, seqs: list[Sequence],
                is_prefill: bool, ctx: AttentionContext) -> list[int]:
        """Sample tokens from logits."""
        if is_prefill:
            last_logits = logits[0, -1, :]
            seq = seqs[0]
            if seq.temperature < 1e-6:
                return [last_logits.argmax().item()]
            scaled = last_logits / seq.temperature
            probs = torch.softmax(scaled, dim=-1)
            probs = self._apply_top_k_top_p(probs, seq.top_k, seq.top_p)
            return [torch.multinomial(probs, 1).item()]

        if logits.dim() == 3:
            last_logits = logits.squeeze(1)
        else:
            last_logits = logits

        token_ids = []
        for i, seq in enumerate(seqs):
            logits_i = last_logits[i]

            if seq.temperature < 1e-6:
                token_ids.append(logits_i.argmax().item())
            else:
                scaled = logits_i / seq.temperature
                probs = torch.softmax(scaled, dim=-1)
                probs = self._apply_top_k_top_p(probs, seq.top_k, seq.top_p)
                token_ids.append(torch.multinomial(probs, 1).item())

        return token_ids

    def _apply_repetition_penalty(self, logits: torch.Tensor, seqs: list[Sequence]):
        """Apply repetition penalty to logits in-place before sampling.

        For tokens that appeared in the sequence's context:
            logit = logit / repetition_penalty   if logit > 0
            logit = logit * repetition_penalty   if logit <= 0

        Works on both full-vocab logits and sharded logits (TP). When sharded,
        each rank only penalizes tokens within its vocab shard.
        """
        if not any(seq.repetition_penalty > 1.0 for seq in seqs):
            return

        is_2d = logits.dim() == 2
        vocab_shard = logits.shape[-1]
        vocab_offset = self.tp_rank * vocab_shard if self.tp_size > 1 else 0

        for i, seq in enumerate(seqs):
            if seq.repetition_penalty <= 1.0:
                continue
            logits_i = logits[i] if is_2d else logits
            appeared = set(seq.token_ids)
            local_indices = [
                t - vocab_offset for t in appeared
                if 0 <= t - vocab_offset < vocab_shard
            ]
            if not local_indices:
                continue
            idx_tensor = torch.tensor(local_indices, dtype=torch.int64, device=logits_i.device)
            gathered = logits_i[idx_tensor]
            penalized = torch.where(
                gathered > 0,
                gathered / seq.repetition_penalty,
                gathered * seq.repetition_penalty,
            )
            logits_i[idx_tensor] = penalized

    def _sample_decode(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        """Optimized decode sampling: uses distributed argmax for greedy, avoids .item() syncs."""
        if logits.dim() == 3:
            logits = logits.squeeze(1)

        # Apply repetition penalty before any sampling/argmax.
        self._apply_repetition_penalty(logits, seqs)

        all_greedy = all(seq.temperature < 1e-6 for seq in seqs)

        if all_greedy:
            if self.tp_size > 1:
                token_tensor = tp_distributed_argmax(logits)
            else:
                token_tensor = logits.argmax(dim=-1)
            return token_tensor.tolist()

        # Non-greedy: use distributed top-k to avoid all-gathering full logits.
        if self.tp_size > 1:
            return self._distributed_sample_topk(logits, seqs)

        token_ids = []
        for i, seq in enumerate(seqs):
            logits_i = logits[i]
            if seq.temperature < 1e-6:
                token_ids.append(logits_i.argmax().item())
            else:
                scaled = logits_i / seq.temperature
                probs = torch.softmax(scaled, dim=-1)
                probs = self._apply_top_k_top_p(probs, seq.top_k, seq.top_p)
                token_ids.append(torch.multinomial(probs, 1).item())
        return token_ids

    def _distributed_sample_topk(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        """Distributed non-greedy sampling via top-k to reduce communication.

        Instead of All-Gathering full logits (vocab_size * bf16 per rank), each
        rank computes local top-k logits + indices, All-Gathers only the small
        (k * tp_size) candidate set, finds the global top-k, then samples.

        Communication drops from O(vocab_size) to O(k * tp_size).
        """
        import torch.distributed as dist
        from engine.parallel import get_tp_group

        bsz = logits.shape[0]
        vocab_shard = logits.shape[1]

        # Choose k large enough for all sequences' top_k / top_p.
        k = max((seq.top_k for seq in seqs if seq.top_k > 0), default=20)
        if any(seq.top_p < 1.0 and seq.top_k <= 0 for seq in seqs):
            k = max(k, 50)
        k = min(k, vocab_shard)

        # Local top-k.
        local_vals, local_idx = logits.topk(k, dim=-1)
        # Convert local indices to global token IDs.
        vocab_offset = self.tp_rank * vocab_shard
        local_idx = local_idx + vocab_offset

        # All-Gather top-k (vals, idx) — tiny compared to full logits.
        all_vals = [torch.empty_like(local_vals) for _ in range(self.tp_size)]
        all_idx = [torch.empty_like(local_idx) for _ in range(self.tp_size)]
        dist.all_gather(all_vals, local_vals, group=get_tp_group())
        dist.all_gather(all_idx, local_idx, group=get_tp_group())

        gathered_vals = torch.cat(all_vals, dim=-1)   # [bsz, k * tp]
        gathered_idx = torch.cat(all_idx, dim=-1)     # [bsz, k * tp]

        # Global top-k from the gathered candidate set.
        global_vals, topk_pos = gathered_vals.topk(k, dim=-1)
        global_idx = gathered_idx.gather(1, topk_pos)

        token_ids = []
        for i, seq in enumerate(seqs):
            vals_i = global_vals[i]
            idx_i = global_idx[i]

            if seq.temperature < 1e-6:
                token_ids.append(idx_i[vals_i.argmax()].item())
                continue

            scaled = vals_i / seq.temperature
            probs = torch.softmax(scaled, dim=-1)

            # Further truncate to seq.top_k if specified and smaller than k.
            if seq.top_k > 0 and seq.top_k < probs.numel():
                tk_vals, tk_pos = torch.topk(probs, seq.top_k)
                probs = torch.zeros_like(probs)
                probs[tk_pos] = tk_vals

            # Apply top-p on the candidate set.
            if seq.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                mask = cumsum - sorted_probs > seq.top_p
                sorted_probs[mask] = 0
                sorted_probs /= sorted_probs.sum()
                probs = torch.zeros_like(probs)
                probs[sorted_indices] = sorted_probs

            sampled = torch.multinomial(probs, 1)
            token_ids.append(idx_i[sampled].item())

        return token_ids

    def _decode_greedy_fused(self, hidden: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        """Fused lm_head + argmax for greedy decode (non-CUDA Graph path).

        Avoids materializing the [bsz, vocab/tp] logits tensor entirely.
        For TP > 1, uses fused_lm_head_argmax_with_max + distributed reduction.
        """
        hidden_2d = hidden.squeeze(1)
        weight = self.model.lm_head.weight

        if self.tp_size > 1:
            local_idx, local_max = fused_lm_head_argmax_with_max(hidden_2d, weight)
            vocab_shard = weight.shape[0]
            token_tensor = tp_distributed_argmax_fused(local_idx, local_max, vocab_shard)
        else:
            token_tensor = fused_lm_head_argmax(hidden_2d, weight)

        return token_tensor.tolist()

    def _sample_prefill_fused(self, hidden: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        """Fused lm_head + argmax for prefill (non-CUDA Graph path).

        For greedy without repetition penalty: uses 1D fused kernel on the last
        token's hidden state.
        For sampling or when repetition penalty is active: falls back to full
        logits materialization.
        """
        seq = seqs[0]
        last_hidden = hidden[0, -1, :]
        has_rep_penalty = seq.repetition_penalty > 1.0

        # Fast path: greedy without repetition penalty can use fused argmax.
        if seq.temperature < 1e-6 and not has_rep_penalty:
            if self.tp_size > 1:
                local_idx, local_max = fused_lm_head_argmax_with_max(
                    last_hidden.unsqueeze(0), self.model.lm_head.weight)
                vocab_shard = self.model.lm_head.weight.shape[0]
                token_tensor = tp_distributed_argmax_fused(local_idx, local_max, vocab_shard)
                return [token_tensor[0].item()]
            else:
                token_tensor = fused_lm_head_argmax(last_hidden, self.model.lm_head.weight)
                return [token_tensor[0].item()]

        # Materialize logits (needed for sampling or repetition penalty).
        logits = self.model.lm_head(hidden)
        if self.tp_size > 1:
            logits = self._all_gather_logits(logits)
        last_logits = logits[0, -1, :]

        # Apply repetition penalty on the last token's logits.
        if has_rep_penalty:
            self._apply_repetition_penalty(last_logits.unsqueeze(0), [seq])

        if seq.temperature < 1e-6:
            return [last_logits.argmax().item()]

        scaled = last_logits / seq.temperature
        probs = torch.softmax(scaled, dim=-1)
        probs = self._apply_top_k_top_p(probs, seq.top_k, seq.top_p)
        return [torch.multinomial(probs, 1).item()]

    @staticmethod
    def _apply_top_k_top_p(probs: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """Apply top-k truncation first, then top-p (nucleus) truncation."""
        if top_k > 0:
            k = min(top_k, probs.numel())
            topk_vals, topk_indices = torch.topk(probs, k)
            probs = torch.zeros_like(probs)
            probs.scatter_(0, topk_indices, topk_vals)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs /= sorted_probs.sum()
            probs = torch.zeros_like(probs)
            probs.scatter_(0, sorted_indices, sorted_probs)

        return probs
