"""
Paged KV Cache Block Manager.

Manages KV cache as fixed-size blocks (pages). Supports:
- Block allocation/deallocation for variable-length sequences
- RECOMPUTE preemption: release blocks and re-prefill from scratch

For Qwen3.6-27B hybrid model:
- Attention layers (16): standard paged KV cache
- DeltaNet layers (48): paged recurrent state (1 block per sequence per layer)
"""
from engine.sequence import Sequence


class Block:
    __slots__ = ('block_id', 'ref_count')

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0

    def reset(self):
        self.ref_count = 1


class BlockManager:
    """Manages paged KV cache blocks."""

    def __init__(self, num_blocks: int, block_size: int, swap_space_bytes: int = 0):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: list[int] = list(range(num_blocks - 1, -1, -1))
        self.used_block_ids: set[int] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

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
                self._deallocate_block(block_id)
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append one more token (may need a new block)."""
        needs_new_block = (len(seq) % self.block_size == 0)
        if needs_new_block:
            return self.num_free_blocks > 0
        return True

    def may_append(self, seq: Sequence):
        """Append a slot for the new token, allocating a new block if needed."""
        if len(seq) % self.block_size == 0:
            seq.block_table.append(self._allocate_block())
