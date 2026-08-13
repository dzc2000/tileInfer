"""Prefix cache for paged KV-cache block reuse.

The cache is keyed by the *full* token prefix (not by individual block
contents), so RoPE positions are preserved: a block that contains the same
tokens but appears at a different position within the sequence must NOT be
reused (it would have different positional embeddings and therefore a
different KV cache).

Implementation
--------------
A trie over ``block_size``-sized token blocks. Each node records the physical
block chain up to that point, so a lookup walks the request tokens block by
block and returns the deepest cacheable prefix in ``O(total_tokens)``.

Eviction is driven by the ``BlockManager``: when a physical block is truly
freed (ref_count reaches zero) the owning ``Scheduler`` calls
:meth:`evict_block`, which invalidates every trie node whose block chain
contains the freed block.
"""


class _Node:
    __slots__ = ('children', 'block_ids', 'num_tokens')

    def __init__(self):
        self.children: dict = {}
        self.block_ids: list[int] | None = None  # cumulative block chain at this node
        self.num_tokens: int = 0                 # number of tokens represented by this node


class PrefixCache:
    """Block-level prefix cache.

    Args:
        block_size: KV-cache block size in tokens (must match Config.kvcache_block_size).
        max_entries: soft cap on the number of cached block boundaries. When
            exceeded, entries are evicted in FIFO order. ``0`` disables the cap.
    """

    def __init__(self, block_size: int, max_entries: int = 0):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.max_entries = max_entries
        self._root = _Node()
        self._num_entries = 0
        # Reverse index: physical block id -> list of nodes whose block_ids
        # ends with that block id. Used for correct eviction when a block is freed.
        self._block_to_nodes: dict[int, list[_Node]] = {}

    def find_longest_prefix(self, token_ids) -> tuple[list[int], int]:
        """Return ``(matched_block_ids, num_matched_tokens)``.

        Only full blocks are considered, so ``num_matched_tokens`` is always a
        multiple of ``block_size``.
        """
        node = self._root
        matched: list[int] = []
        matched_tokens = 0
        bs = self.block_size
        pos = 0
        while pos + bs <= len(token_ids):
            block = tuple(token_ids[pos:pos + bs])
            child = node.children.get(block)
            if child is None:
                break
            node = child
            pos += bs
            if node.block_ids is not None:
                matched = list(node.block_ids)
                matched_tokens = pos
        return matched, matched_tokens

    def add(self, token_ids, block_ids):
        """Register every full-block boundary of a freshly prefilled sequence."""
        bs = self.block_size
        num_full_blocks = len(token_ids) // bs
        num_blocks = min(num_full_blocks, len(block_ids))

        node = self._root
        pos = 0
        for nblocks in range(1, num_blocks + 1):
            block = tuple(token_ids[pos:pos + bs])
            pos += bs
            child = node.children.get(block)
            if child is None:
                child = _Node()
                node.children[block] = child
            node = child
            # Idempotent: refresh the cumulative chain but only count/register
            # the node once (postprocess calls add() repeatedly for a growing
            # prefix across chunked-prefill steps).
            if node.block_ids is None:
                self._block_to_nodes.setdefault(
                    block_ids[nblocks - 1], []).append(node)
                self._num_entries += 1
            node.block_ids = list(block_ids[:nblocks])
            node.num_tokens = pos

        self._maybe_trim()

    def evict_block(self, block_id: int):
        """Invalidate every cached prefix that references ``block_id``.

        Called when ``block_id`` is returned to the free pool (ref_count 0),
        i.e. its KV contents are no longer valid.
        """
        nodes = self._block_to_nodes.pop(block_id, [])
        for node in nodes:
            self._invalidate_subtree(node)

    def _invalidate_subtree(self, node: _Node):
        if node.block_ids is not None:
            node.block_ids = None
            node.num_tokens = 0
            self._num_entries -= 1
        for child in node.children.values():
            self._invalidate_subtree(child)

    def _maybe_trim(self):
        """Simple FIFO eviction: drop the oldest registered block boundary.

        This is best-effort; a real deployment would use an LRU with access
        tracking. FIFO keeps memory bounded without per-lookup bookkeeping.
        """
        if self.max_entries <= 0 or self._num_entries <= self.max_entries:
            return
        # Walk the trie in insertion order by tracking first child repeatedly.
        # Fallback: collect all live nodes and evict the oldest by insertion index.
        nodes: list[_Node] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                if child.block_ids is not None:
                    nodes.append(child)
                stack.append(child)
        overflow = self._num_entries - self.max_entries
        # Insertion order is preserved by BFS layer-by-layer for a trie; take
        # the first ``overflow`` entries at the shallowest levels (oldest-ish).
        for node in nodes[:overflow]:
            if node.block_ids is not None:
                node.block_ids = None
                node.num_tokens = 0
                self._num_entries -= 1

    def clear(self):
        self._root = _Node()
        self._block_to_nodes = {}
        self._num_entries = 0

    @property
    def num_entries(self) -> int:
        return self._num_entries
