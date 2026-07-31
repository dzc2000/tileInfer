"""
Scheduler with Continuous Batching, Chunked Prefill, and Mixed Batching.

Decides which sequences to run each step:
- Mixed batching: decode + prefill sequences can coexist in one step
- Chunked prefill: long prompts are split into chunks
- RECOMPUTE preemption: release blocks and re-prefill under memory pressure
"""
from collections import deque, OrderedDict
from engine.config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config, tokenizer=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_prefill_chunk_tokens = config.max_prefill_chunk_tokens
        self.eos: list = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size,
            swap_space_bytes=config.swap_space_bytes,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: OrderedDict[int, Sequence] = OrderedDict()
        self.on_preempt = None
        self.tokenizer = tokenizer
        # Limit concurrent preemptions to avoid thundering herd
        self.max_preempt_count = max(1, config.max_num_seqs // 4)

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

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
        # === Phase 2: Schedule ONE prefill sequence per step ===
        # DeltaNet recurrent state requires sequential per-sequence processing,
        # so we prefill one sequence at a time (vLLM-style serial prefill).
        remaining_budget = self.max_num_batched_tokens - num_batched_tokens
        if self.waiting and remaining_budget > 0 and len(prefill_seqs) == 0:
            if len(decode_seqs) + len(prefill_seqs) < self.max_num_seqs:
                seq = self.waiting[0]

                # First time: allocate blocks
                if not seq.block_table:
                    if self.block_manager.can_allocate(seq):
                        self.block_manager.allocate(seq)
                        num_tokens = seq.num_tokens
                    else:
                        num_tokens = 0
                else:
                    # Already partially prefilled (chunked prefill continuation)
                    num_tokens = seq.num_tokens - seq.num_cached_tokens

                if num_tokens > 0:
                    # Chunked prefill: limit chunk size
                    chunk_limit = min(remaining_budget, self.max_prefill_chunk_tokens)
                    seq.num_scheduled_tokens = min(num_tokens, chunk_limit)
                    remaining_budget -= seq.num_scheduled_tokens
                    num_batched_tokens += seq.num_scheduled_tokens

                    # If fully prefilled, move to running
                    if seq.num_cached_tokens + seq.num_scheduled_tokens >= seq.num_tokens:
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

            # Check EOS (supports multiple EOS tokens)
            finished = False
            if not seq.ignore_eos and token_id in self.eos:
                finished = True
            if seq.num_completion_tokens >= seq.max_tokens:
                finished = True

            # Check stop strings
            if not finished and seq.stop and self.tokenizer is not None:
                text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True)
                for stop_str in seq.stop:
                    if stop_str in text:
                        finished = True
                        break

            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq.seq_id in self.running:
                    del self.running[seq.seq_id]
