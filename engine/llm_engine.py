"""
LLMEngine: main orchestrator for Qwen3.6-27B inference.

Coordinates Scheduler, BlockManager, and ModelRunner to process
batches of requests with continuous batching, chunked prefill,
and mixed batching. Supports Tensor Parallelism for multi-GPU inference.

Usage:
    from engine.llm_engine import LLMEngine
    from engine.sampling_params import SamplingParams

    engine = LLMEngine("/path/to/Qwen3.6-27B")
    params = SamplingParams(temperature=0.6, max_tokens=256)
    outputs = engine.generate(["Hello, world!"], params)

Streaming:
    for token in engine.generate_stream(["Hello"], params):
        print(token, end="", flush=True)
"""
import torch
from transformers import AutoTokenizer
from engine.config import Config
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner
from engine.sequence import Sequence, SequenceStatus
from engine.sampling_params import SamplingParams
from engine.parallel import init_distributed, get_tp_rank, barrier


class LLMEngine:
    def __init__(self, model: str, **kwargs):
        self.config = Config(model=model, **kwargs)

        # Initialize TP (must be done before model loading)
        init_distributed(self.config.tp_size, timeout=self.config.nccl_timeout)

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # Initialize model runner
        self.model_runner = ModelRunner(self.config)
        self.model_runner.load_model()

        # Warmup to measure peak memory
        self._warmup()

        # Allocate KV cache after warmup
        self.model_runner.allocate_kv_cache()

        # Synchronize all ranks after cache allocation
        barrier()

        # Capture CUDA graphs for decode (skipped when enforce_eager=True)
        self.model_runner.capture_decode_graphs()
        barrier()

        # Initialize scheduler with tokenizer (for stop string checks)
        self.scheduler = Scheduler(self.config, tokenizer=self.tokenizer)
        self.scheduler.on_preempt = lambda seq: self.model_runner.free_deltanet_slot(seq.seq_id)
        # Prefix-cache callbacks (DeltaNet state snapshots live in ModelRunner).
        self.scheduler.on_prefix_hit = (
            lambda seq, matched, n: self.model_runner.prefix_state_exists(matched))
        self.scheduler.on_prefix_save = self.model_runner.save_prefix_state
        self.scheduler.on_block_freed = self.model_runner.evict_prefix_state

        # Track all sequences for result collection and cleanup
        self._all_seqs: dict[int, Sequence] = {}

    def _warmup(self):
        """Warmup model: trigger tilelang JIT compilation and measure peak memory."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        from engine.context import AttentionContext, set_context, reset_context
        from model.qwen_36 import set_paged_kv_cache, set_deltanet_pools

        # Allocate minimal temporary pools for warmup
        tp_num_kv_heads = self.model_runner.tp_num_kv_heads
        tp_num_v_heads = self.model_runner.tp_num_v_heads
        tp_conv_dim = self.model_runner.tp_conv_dim

        temp_kv_cache = torch.zeros(
            2, self.model_runner.num_attention_layers,
            1, self.model_runner.block_size,  # 1 block, block_size from config
            tp_num_kv_heads, self.model_runner.head_dim,
            device="cuda", dtype=torch.bfloat16,
        )
        temp_recurrent = torch.zeros(
            self.model_runner.num_deltanet_layers, 1,
            tp_num_v_heads, 128, 128,
            device="cuda", dtype=torch.bfloat16,
        )
        temp_conv = torch.zeros(
            self.model_runner.num_deltanet_layers, 1,
            tp_conv_dim, 4,
            device="cuda", dtype=torch.bfloat16,
        )
        temp_slots = {0: 0}

        set_paged_kv_cache(temp_kv_cache)
        set_deltanet_pools(temp_recurrent, temp_conv, temp_slots)

        # Minimal context for warmup
        ctx = AttentionContext()
        ctx.slot_mapping = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        ctx.block_tables = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        ctx.context_lens = torch.tensor([2], dtype=torch.int32, device="cuda")
        ctx.cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        ctx.cu_seqlens_k = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
        ctx.max_seqlen_q = 2
        ctx.max_seqlen_k = 2
        ctx.deltanet_slots = torch.tensor([0], dtype=torch.int32, device="cuda")
        ctx.is_prefill = True
        set_context(ctx)

        # Run a minimal prefill (2 tokens) through the full model
        # This triggers JIT compilation for all fused kernels
        with torch.no_grad():
            dummy_ids = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
            pos_ids = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
            hidden = self.model_runner.model.model.embed_tokens(dummy_ids)
            pos_emb = self.model_runner.model.model.rotary_emb(hidden, pos_ids)
            for layer in self.model_runner.model.model.layers:
                hidden = layer(
                    hidden,
                    position_ids=pos_ids,
                    position_embeddings=pos_emb,
                    use_cache=False,
                )
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
            hidden = self.model_runner.model.model.norm(hidden)

        reset_context()
        torch.cuda.empty_cache()

    def add_request(self, prompt: str, sampling_params: SamplingParams | None = None):
        """Add a single request to the engine."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        # Validate prompt length
        max_len = self.config.max_model_len
        if len(token_ids) > max_len:
            raise ValueError(
                f"Prompt has {len(token_ids)} tokens, exceeds max_model_len={max_len}. "
                f"Truncate the prompt or increase max_model_len."
            )
        if len(token_ids) == 0:
            raise ValueError("Prompt produced no tokens after tokenization")

        seq = Sequence(token_ids, sampling_params)
        # Build the constrained-decoding constraint (if any) from sampling params.
        from engine.constraints import make_constraint
        seq.constraint = make_constraint(sampling_params, self.tokenizer)
        self.scheduler.add(seq)
        self._all_seqs[seq.seq_id] = seq
        return seq

    def step(self) -> list[Sequence]:
        """Run one scheduling step. Returns list of sequences that finished this step."""
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return []
        self.model_runner.oom_preempted.clear()

        # Multi-step scheduling: run several decode steps on the same batch to
        # amortize per-step Python/scheduling overhead. Only applies to pure
        # decode steps (no prefill sequence mixed in).
        if (not is_prefill and self.config.num_scheduler_steps > 1
                and all(not s.is_prefill for s in seqs)):
            return self._multi_step_decode(seqs)

        token_ids = self.model_runner.run(seqs, is_prefill)

        self._handle_preempted()

        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        return self._finalize_finished(seqs)

    def _handle_preempted(self):
        """Re-queue sequences dropped due to OOM during execution."""
        for preempted_seq in self.model_runner.oom_preempted:
            preempted_seq.status = SequenceStatus.WAITING
            preempted_seq.is_prefill = True
            preempted_seq.num_cached_tokens = 0
            preempted_seq.prefix_restore_blocks = None
            self.scheduler.deallocate(preempted_seq)
            self.scheduler.waiting.appendleft(preempted_seq)
            if preempted_seq.seq_id in self.scheduler.running:
                del self.scheduler.running[preempted_seq.seq_id]
        self.model_runner.oom_preempted.clear()

    def _finalize_finished(self, seqs: list[Sequence]) -> list[Sequence]:
        """Free DeltaNet slots and tracking state for finished sequences."""
        finished = [seq for seq in seqs if seq.is_finished]
        for seq in finished:
            self.model_runner.free_deltanet_slot(seq.seq_id)
            self._all_seqs.pop(seq.seq_id, None)
        return finished

    def _multi_step_decode(self, seqs: list[Sequence]) -> list[Sequence]:
        """Run ``num_scheduler_steps`` decode steps on one scheduled batch."""
        batch: list[Sequence] = list(seqs)
        finished: list[Sequence] = []

        for _ in range(self.config.num_scheduler_steps):
            if not batch:
                break
            token_ids = self.model_runner.run(batch, False)
            self._handle_preempted()
            self.scheduler.postprocess(batch, token_ids, False)
            finished.extend(self._finalize_finished(
                [s for s in batch if s.is_finished]))
            batch = [s for s in batch if not s.is_finished]
            if batch:
                self.scheduler.advance_decode(batch)

        return finished

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[dict]:
        """Generate completions for a list of prompts.

        Returns list of dicts with 'text', 'prompt', 'token_ids', 'prompt_token_ids' keys.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        seqs = []
        for prompt in prompts:
            seq = self.add_request(prompt, sampling_params)
            seqs.append(seq)

        try:
            while not self.scheduler.is_finished():
                self.step()
        finally:
            self._cleanup_seqs(seqs)

        outputs = []
        for seq in seqs:
            completion_ids = seq.completion_token_ids
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            outputs.append({
                "text": text,
                "prompt": self.tokenizer.decode(seq.prompt_token_ids, skip_special_tokens=True),
                "token_ids": completion_ids,
                "prompt_token_ids": seq.prompt_token_ids,
            })

        return outputs

    def generate_stream(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | None = None,
    ):
        """Generate completions with streaming output.

        Yields (seq_index, token_text, is_finished) for each token generated.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        seqs = []
        for prompt in prompts:
            seq = self.add_request(prompt, sampling_params)
            seqs.append(seq)

        # Track which tokens have been yielded per sequence
        yielded_count = [0] * len(seqs)

        try:
            while not self.scheduler.is_finished():
                finished_seqs = self.step()

                # Yield newly generated tokens
                for i, seq in enumerate(seqs):
                    if seq.is_finished:
                        continue
                    new_count = len(seq.completion_token_ids)
                    while yielded_count[i] < new_count:
                        token_id = seq.completion_token_ids[yielded_count[i]]
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        yielded_count[i] += 1
                        yield (i, token_text, False)

                # Yield final token for finished sequences
                for seq in finished_seqs:
                    i = seqs.index(seq)
                    new_count = len(seq.completion_token_ids)
                    while yielded_count[i] < new_count:
                        token_id = seq.completion_token_ids[yielded_count[i]]
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        yielded_count[i] += 1
                        yield (i, token_text, True)
        finally:
            self._cleanup_seqs(seqs)

    def generate_token_ids(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[int]]:
        """Generate token IDs for prompts. Returns list of completion token ID lists."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        seqs = []
        for prompt in prompts:
            seq = self.add_request(prompt, sampling_params)
            seqs.append(seq)

        try:
            while not self.scheduler.is_finished():
                self.step()
        finally:
            self._cleanup_seqs(seqs)

        return [seq.completion_token_ids for seq in seqs]

    def _cleanup_seqs(self, seqs: list[Sequence]):
        """Clean up all sequences: free DeltaNet slots and remove from tracking."""
        for seq in seqs:
            self.model_runner.free_deltanet_slot(seq.seq_id)
            self._all_seqs.pop(seq.seq_id, None)
