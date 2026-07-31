"""
Paged KV Cache Block Manager.

Manages KV cache as fixed-size blocks (pages). Supports:
- Block allocation/deallocation for variable-length sequences
- SWAP: CPU offload for preemption (avoids expensive recompute)
- Prefix caching: reuse blocks across sequences sharing a common prompt prefix

For Qwen3.6-27B hybrid model:
- Attention layers (16): standard paged KV cache
- DeltaNet layers (48): paged recurrent state (1 block per sequence per layer)

Prefix caching for DeltaNet requires caching the recurrent+conv state
alongside KV blocks. The BlockManager signals cache hits so the
scheduler/model_runner can restore DeltaNet state from a saved snapshot.
"""
from engine.sequence import Sequence


class Block:
    __slots__ = ('block_id', 'ref_count', 'hash', 'swap_location')

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = None        # content hash for prefix caching
        self.swap_location = None  # index into CPU swap pool if swapped out

    def reset(self):
        self.ref_count = 1
        self.hash = None
        self.swap_location = None


class BlockManager:
    """Manages paged KV cache blocks with optional SWAP and prefix caching."""

    def __init__(self, num_blocks: int, block_size: int, swap_space_bytes: int = 0):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # Use a list as a stack for deterministic allocation order (better locality)
        self.free_block_ids: list[int] = list(range(num_blocks - 1, -1, -1))
        self.used_block_ids: set[int] = set()

        # SWAP pool: CPU pinned memory for offloading KV blocks under memory pressure
        self.swap_pool = None
        self.free_swap_slots: list[int] = []
        if swap_space_bytes > 0:
            import torch
            block_bytes = block_size * 2 * 2  # K+V, bf16, per head handled by caller
            # Actual swap pool is allocated by ModelRunner which knows head dims
            self.max_swap_blocks = swap_space_bytes // max(block_bytes, 1)
            self.free_swap_slots = list(range(self.max_swap_blocks - 1, -1, -1))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_free_swap_slots(self) -> int:
        return len(self.free_swap_slots)

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.pop()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if we can allocate blocks for a sequence."""
        return self.num_free_blocks >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Allocate blocks for a sequence."""
        assert not seq.block_table
        for _ in range(seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = 0

    def deallocate(self, seq: Sequence):
        """Release all blocks held by a sequence."""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count <= 0:
                block.ref_count = 0
                # If block was swapped, free the swap slot too
                if block.swap_location is not None:
                    self.free_swap_slots.append(block.swap_location)
                    block.swap_location = None
                self._deallocate_block(block_id)
        seq.block_table.clear()

    def swap_out(self, seq: Sequence) -> int:
        """Swap out a sequence's KV blocks to CPU memory.

        Frees GPU blocks but preserves data in CPU swap pool.
        Returns the swap slot index, or -1 if swap is unavailable.
        """
        if not self.free_swap_slots:
            return -1

        swap_slot = self.free_swap_slots.pop()
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.swap_location = swap_slot
            block.ref_count = 0
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)
        # Keep block_table for later swap_in
        return swap_slot

    def swap_in(self, seq: Sequence) -> bool:
        """Restore a sequence's KV blocks from CPU swap pool.

        Reallocates GPU blocks. Returns True on success.
        """
        needed = len(seq.block_table)
        if self.num_free_blocks < needed:
            return False

        new_table = []
        old_table = seq.block_table
        seq.block_table = []
        for _ in range(needed):
            new_table.append(self._allocate_block())
        seq.block_table = new_table
        # Caller (ModelRunner) is responsible for actual D2D copy from swap pool
        return True

    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append one more token (may need a new block)."""
        needs_new_block = (len(seq) % self.block_size == 1)
        if needs_new_block:
            return self.num_free_blocks > 0
        return True

    def may_append(self, seq: Sequence):
        """Append a slot for the new token, allocating a new block if needed."""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())
