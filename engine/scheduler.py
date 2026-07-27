"""
Scheduler with Continuous Batching and Chunked Prefill.

Decides which sequences to run each step:
- Prefill: schedule waiting sequences (chunked if too long)
- Decode: schedule running sequences (continuous batching)
- Preemption: evict sequences when memory is insufficient
"""
from collections import deque
from engine.config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_prefill_chunk_tokens = config.max_prefill_chunk_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        from collections import OrderedDict
        self.waiting: deque[Sequence] = deque()
        self.running: OrderedDict[int, Sequence] = OrderedDict()  # seq_id -> Sequence for O(1) removal
        self._schedule_depth = 0
        self.on_preempt = None

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Schedule sequences for the next step.

        Returns (scheduled_seqs, is_prefill).
        Prefill and decode are never mixed in the same step.
        """
        # Recursion guard: prevent infinite retry loop (max 1 retry)
        self._schedule_depth += 1
        if self._schedule_depth > 2:
            raise RuntimeError(
                "Schedule retry limit exceeded. Sequences may be too large "
                "for available KV cache blocks."
            )

        scheduled_seqs = []
        num_batched_tokens = 0

        # === PREFILL PATH ===
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # First time: allocate blocks
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break  # not enough memory
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # Already partially prefilled (chunked prefill continuation)
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Chunked prefill: limit chunk size for the first sequence
            chunk_limit = min(remaining, self.max_prefill_chunk_tokens)
            if num_tokens > chunk_limit and scheduled_seqs:
                break  # only allow chunked prefill for the first seq in batch

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, chunk_limit)
            num_batched_tokens += seq.num_scheduled_tokens

            # If fully prefilled, move to running
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running[seq.seq_id] = seq

            scheduled_seqs.append(seq)

            # The model runner currently supports a single sequence per prefill
            # step (DeltaNet recurrent state is tracked per-sequence and the
            # prefill attention path is single-batch). Schedule one prefill at a
            # time; batching multiple prefill sequences would crash the runner.
            break

        if scheduled_seqs:
            self._schedule_depth = 0
            return scheduled_seqs, True

        # === DECODE PATH (Continuous Batching) ===
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq_id, seq = next(iter(self.running.items()))
            del self.running[seq_id]
            # Check if we can append a new token's KV
            while not self.block_manager.can_append(seq):
                if self.running:
                    preempt_id, preempt_seq = next(reversed(self.running.items()))
                    del self.running[preempt_id]
                    self.preempt(preempt_seq)
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # If no decode sequences could be scheduled (all were preempted),
        # try to schedule prefill from waiting queue instead.
        if not scheduled_seqs:
            return self.schedule()  # retry as prefill

        for seq in reversed(scheduled_seqs):
            self.running[seq.seq_id] = seq
        self._schedule_depth = 0
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """Evict a sequence: release blocks and put back in waiting queue."""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        if self.on_preempt is not None:
            self.on_preempt(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """Update sequences after a step: append tokens, check EOS."""
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

            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq.seq_id in self.running:
                    del self.running[seq.seq_id]
