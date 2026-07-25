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
from collections import deque

import torch
from engine.config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.context import AttentionContext, set_context, reset_context
from engine.parallel import get_tp_rank, get_tp_world_size, is_tp_active, tp_broadcast


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
        max_slots = config.max_num_seqs
        tp_conv_dim = self.tp_conv_dim
        # NOTE: kernel_size is hardcoded to 4 (DeltaNet conv1d kernel size).
        # conv1d weights are only available after model patching, so we cannot
        # read it from the weight tensor here. Update if the model uses a
        # different conv kernel size.
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

        # Update max_blocks_per_seq to match the kernel's compile-time num_pages.
        # gqa_decode_paged compiles block_table with shape [batch, num_blocks],
        # so CUDA Graph static buffers must use the same width.
        self.max_blocks_per_seq = num_blocks

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
        """Free a DeltaNet state slot."""
        slot = self.seq_to_slot.pop(seq_id, None)
        if slot is not None:
            self.deltanet_recurrent_pool[:, slot].zero_()
            self.deltanet_conv_pool[:, slot].zero_()
            self.free_deltanet_slots.append(slot)

    @torch.inference_mode()
    def _decode_forward_logits(self, input_ids: torch.Tensor, positions: torch.Tensor,
                               ctx: AttentionContext) -> torch.Tensor:
        """Run the decode forward pass up to logits (no sampling).

        This is the unit captured by CUDA Graph: embed -> layers -> norm -> lm_head.
        All KV-cache / DeltaNet metadata is read from ``ctx`` (which, during capture,
        references the static buffers so the recorded kernels point at stable memory).
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
            if self.tp_size > 1:
                logits = self._all_gather_logits(logits)
        finally:
            reset_context()
        return logits

    def _all_gather_logits(self, logits_shard: torch.Tensor) -> torch.Tensor:
        """All-Gather sharded logits across TP ranks to reconstruct full vocab logits."""
        import torch.distributed as dist
        from engine.parallel import get_tp_group
        tp_size = self.tp_size
        gather_list = [torch.empty_like(logits_shard) for _ in range(tp_size)]
        dist.all_gather(gather_list, logits_shard, group=get_tp_group())
        return torch.cat(gather_list, dim=-1)

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

        # Reserve a dummy DeltaNet slot for padded rows (deterministic across ranks).
        self.padding_deltanet_slot = self.config.max_num_seqs - 1
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

        print(f"[Rank {self.tp_rank}] Captured CUDA graphs for decode batch sizes: {buckets}")

    def _select_bucket(self, n: int) -> int | None:
        """Smallest captured batch size >= n, or None if n exceeds all buckets."""
        for b in sorted(self.cuda_graphs):
            if b >= n:
                return b
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

    def _build_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """Build padded block_tables tensor for paged attention."""
        max_blocks = max(len(seq.block_table) for seq in seqs)
        tables = [
            seq.block_table + [-1] * (max_blocks - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(tables, dtype=torch.int32, device="cuda")

    def _build_slot_mapping_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """Build slot_mapping for prefill: maps each token position to its physical slot in KV cache."""
        slot_mapping = []
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            for pos in range(start, end):
                block_idx = pos // self.block_size
                block_offset = pos % self.block_size
                if block_idx < len(seq.block_table):
                    physical_block = seq.block_table[block_idx]
                    slot_mapping.append(physical_block * self.block_size + block_offset)
                else:
                    slot_mapping.append(-1)  # shouldn't happen
        return torch.tensor(slot_mapping, dtype=torch.int32, device="cuda")

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
        """Execute one step: prepare inputs, run model, sample tokens."""
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

        # Prepare inputs
        if is_prefill:
            input_ids, positions, ctx = self.prepare_prefill(seqs)
        else:
            input_ids, positions, ctx = self.prepare_decode(seqs)

        # CUDA Graph fast path for decode: replay a captured graph keyed by the
        # padded batch size, then sample eagerly (sampling has host-side branches
        # and RNG, so it stays outside the graph).
        if not is_prefill and not self.enforce_eager and self.cuda_graphs:
            n = len(seqs)
            if self._select_bucket(n) is not None:
                logits = self._run_decode_graph(input_ids, positions, ctx, n)
                token_ids = self._sample(logits, seqs, is_prefill=False, ctx=ctx)
                if is_tp_active():
                    token_ids = self._sync_tokens(token_ids)
                return token_ids

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
                
                logits = self.model.lm_head(hidden)
                if self.tp_size > 1:
                    logits = self._all_gather_logits(logits)

                if is_prefill or any(seq.temperature >= 1e-6 or seq.top_p < 1.0 or seq.top_k > 0 for seq in seqs):
                    token_ids = self._sample(logits, seqs, is_prefill, ctx)
                else:
                    if logits.dim() == 3:
                        logits = logits.squeeze(1)
                    token_ids = logits.argmax(dim=-1).tolist()
            else:
                token_ids = [-1] * len(seqs)  # placeholder, won't be used
        finally:
            reset_context()

        # Under tensor parallelism every rank must agree on the sampled tokens.
        # Sampling draws from per-rank RNG state, so without synchronization the
        # ranks would diverge (different tokens -> different block tables /
        # sequence lengths -> desynchronized collectives). Broadcast rank 0's
        # tokens to all ranks. (Greedy/placeholder tokens are already identical,
        # so this is a no-op for correctness in those cases.)
        if is_tp_active():
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
