"""
Scheduler with Continuous Batching, Chunked Prefill, Mixed Batching,
Prefix Caching, and Multi-Step Scheduling.

Decides which sequences to run each step:
- Mixed batching: decode + prefill sequences can coexist in one step
- Chunked prefill: long prompts are split into chunks
- RECOMPUTE preemption: release blocks and re-prefill under memory pressure
- Prefix caching: reuse KV-cache blocks (and DeltaNet recurrent state) when
  a request shares a token prefix with an already-prefilled sequence
- Multi-step scheduling: run several decode steps per scheduler invocation
"""
from collections import deque, OrderedDict
from engine.config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager
from engine.prefix_cache import PrefixCache
from engine.constraints import advance_constraint, constraint_complete


class Scheduler:
    def __init__(self, config: Config, tokenizer=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_prefill_chunk_tokens = config.max_prefill_chunk_tokens
        self.eos: list = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: OrderedDict[int, Sequence] = OrderedDict()
        self.on_preempt = None
        self.tokenizer = tokenizer
        # Limit concurrent preemptions to avoid thundering herd
        self.max_preempt_count = max(1, config.max_num_seqs // 4)

        # --- Prefix caching ---
        self.enable_prefix_caching = config.enable_prefix_caching
        self.prefix_cache = PrefixCache(self.block_size)
        # Callbacks wired by LLMEngine:
        #   on_prefix_hit(seq, block_chain, num_tokens) -> bool (restore state)
        #   on_prefix_save(seq, block_chain)
        #   on_block_freed(block_id)
        self.on_prefix_hit = None
        self.on_prefix_save = None
        self.on_block_freed = None

        # Forward eviction notifications from the block manager.
        self.block_manager.on_block_freed = self._on_block_freed

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _on_block_freed(self, block_id: int):
        self.prefix_cache.evict_block(block_id)
        if self.on_block_freed is not None:
            self.on_block_freed(block_id)

    def deallocate(self, seq: Sequence):
        """Release a sequence's blocks and evict stale prefix-cache entries."""
        self.block_manager.deallocate(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Schedule sequences for the next step.

        Returns (scheduled_seqs, is_prefill).
        Supports mixed batching: decode sequences are scheduled first,
        then remaining token budget is used for prefill sequences.
        When only prefill sequences are scheduled, is_prefill=True.
        When any decode sequences are present, is_prefill=False.
        """
        # Iterative scheduling: try decode first, then fill with prefill.
        # No recursion — use a simple loop with explicit phase tracking.
        decode_seqs: list[Sequence] = []
        prefill_seqs: list[Sequence] = []
        num_batched_tokens = 0

        # === Phase 1: Schedule decode sequences (continuous batching) ===
        preempted = 0
        decode_list = list(self.running.values())
        for seq in decode_list:
            if len(decode_seqs) + len(prefill_seqs) >= self.max_num_seqs:
                break
            if num_batched_tokens >= self.max_num_batched_tokens:
                break

            # Ensure we can append a new token's KV
            while not self.block_manager.can_append(seq):
                if preempted >= self.max_preempt_count:
                    break
                freed = self._try_preempt(seq)
                if not freed:
                    break
                preempted += 1
                if not self.block_manager.can_append(seq):
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                del self.running[seq.seq_id]
                decode_seqs.append(seq)
                num_batched_tokens += 1
                continue
            # If we get here, we couldn't schedule this seq
            break

        # === Phase 2: Schedule prefill sequences with remaining budget ===
        # DeltaNet recurrent state requires sequential per-sequence processing,
        # so we prefill one sequence at a time (vLLM-style serial prefill).
        remaining_budget = self.max_num_batched_tokens - num_batched_tokens
        if self.waiting and remaining_budget > 0 and len(prefill_seqs) == 0:
            if len(decode_seqs) + len(prefill_seqs) < self.max_num_seqs:
                seq = self.waiting[0]

                if not seq.block_table:
                    # Fresh sequence: attempt prefix-cache reuse, else allocate.
                    if not self._allocate_blocks_for_new_seq(seq):
                        # Not enough memory this step — defer, don't schedule.
                        remaining_budget = 0

                if seq.block_table:
                    # num_tokens remaining to prefill (after any cached prefix)
                    num_tokens = seq.num_tokens - seq.num_cached_tokens

                    if num_tokens > 0 and remaining_budget > 0:
                        # Chunked prefill: limit chunk size
                        chunk_limit = min(remaining_budget,
                                          self.max_prefill_chunk_tokens)
                        seq.num_scheduled_tokens = min(num_tokens, chunk_limit)
                        remaining_budget -= seq.num_scheduled_tokens
                        num_batched_tokens += seq.num_scheduled_tokens

                        # If fully prefilled, move to running
                        if (seq.num_cached_tokens + seq.num_scheduled_tokens
                                >= seq.num_tokens):
                            seq.status = SequenceStatus.RUNNING
                            self.waiting.popleft()
                            self.running[seq.seq_id] = seq

                        prefill_seqs.append(seq)

        scheduled = prefill_seqs + decode_seqs

        # Restore decode seqs to running dict (they were popped for scheduling)
        for seq in decode_seqs:
            self.running[seq.seq_id] = seq

        if not scheduled:
            return [], False

        # is_prefill = True only if ALL sequences are prefill (no decode)
        is_prefill = len(decode_seqs) == 0
        return scheduled, is_prefill

    def _allocate_blocks_for_new_seq(self, seq: Sequence) -> bool:
        """Allocate blocks for a brand-new sequence, reusing a cached prefix.

        Returns True if block allocation succeeded, False if the sequence must
        be deferred to a later step (insufficient free blocks).
        """
        # 1) Try to reuse a cached prefix.
        if self.enable_prefix_caching:
            matched, n = self.prefix_cache.find_longest_prefix(seq.token_ids)
            if matched:
                needed = seq.num_blocks - len(matched)
                if self.block_manager.num_free_blocks >= needed:
                    # Restore the DeltaNet recurrent state for the matched
                    # prefix. If no snapshot is available we must recompute.
                    if self.on_prefix_hit is None or \
                            self.on_prefix_hit(seq, matched, n):
                        self.block_manager.share_blocks(seq, matched)
                        seq.num_cached_tokens = n
                        seq.prefix_restore_blocks = tuple(matched)
                        self.block_manager.allocate_extra(seq)
                        return True
                    # Fall through: restore failed -> full recompute.

        # 2) Full allocation from scratch.
        if self.block_manager.can_allocate(seq):
            self.block_manager.allocate(seq)
            return True

        return False

    def advance_decode(self, seqs: list[Sequence]):
        """Prepare a decode batch for the next multi-step decode sub-step.

        Allocates a KV slot for each sequence's next token (may append a new
        block) and marks the sub-step as decode. Called between consecutive
        decode sub-steps of multi-step scheduling.
        """
        for seq in seqs:
            if not self.block_manager.can_append(seq):
                # Rare: out of blocks mid multi-step. Preempt if possible,
                # otherwise the caller will drop the sequence.
                if not self._try_preempt(seq):
                    continue
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)

    def _try_preempt(self, seq: Sequence) -> bool:
        """Try to free memory by preempting a running sequence (RECOMPUTE).

        Returns True if memory was freed, False otherwise.
        """
        if not self.running:
            self._preempt_seq(seq)
            return True

        preempt_id, preempt_seq = next(reversed(self.running.items()))
        del self.running[preempt_id]
        self._preempt_seq(preempt_seq)
        return True

    def _preempt_seq(self, seq: Sequence):
        """Preempt a sequence: release all blocks, reset to waiting for recompute."""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_cached_tokens = 0
        seq.prefix_restore_blocks = None
        self.block_manager.deallocate(seq)
        if self.on_preempt is not None:
            self.on_preempt(seq)
        self.waiting.appendleft(seq)

    def preempt(self, seq: Sequence):
        """Public preempt interface."""
        self._preempt_seq(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """Update sequences after a step: append tokens, check EOS/stop strings."""
        for i, seq in enumerate(seqs):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # Save prefix-cache state at block boundaries reached during prefill.
            if (self.enable_prefix_caching and seq.is_prefill
                    and seq.num_cached_tokens > 0
                    and seq.num_cached_tokens % self.block_size == 0):
                num_blocks = seq.num_cached_tokens // self.block_size
                self.prefix_cache.add(seq.token_ids, seq.block_table)
                if self.on_prefix_save is not None:
                    self.on_prefix_save(seq, tuple(seq.block_table[:num_blocks]))

            # If still in prefill (chunked), don't append decode token yet
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            # Append the sampled token
            if i < len(token_ids):
                token_id = token_ids[i]
            else:
                continue

            # Skip placeholder tokens from non-final prefill chunks
            if token_id == -1:
                continue

            seq.append_token(token_id)
            advance_constraint(seq, token_id)

            # Check EOS (supports multiple EOS tokens)
            finished = False
            if not seq.ignore_eos and token_id in self.eos:
                finished = True
            if seq.num_completion_tokens >= seq.max_tokens:
                finished = True

            # Guided generation reached a terminal grammar state.
            if not finished and constraint_complete(seq):
                finished = True

            # Check stop strings
            if not finished and seq.stop and self.tokenizer is not None:
                text = self.tokenizer.decode(seq.completion_token_ids,
                                             skip_special_tokens=True)
                for stop_str in seq.stop:
                    if stop_str in text:
                        finished = True
                        break

            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq.seq_id in self.running:
                    del self.running[seq.seq_id]
