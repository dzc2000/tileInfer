"""AttentionContext: thread-local context for passing paged KV cache metadata to model forward."""
import threading

_context = threading.local()


class AttentionContext:
    """Holds paged KV cache metadata for the current forward pass.

    Set by ModelRunner before calling model.forward(), read by
    attention layers to access slot_mapping, block_tables, etc.
    """
    __slots__ = (
        'slot_mapping',       # [total_tokens] flat slot indices for kvcache_store
        'block_tables',       # [bsz, max_num_blocks_per_seq] block table for flash_attn
        'context_lens',       # [bsz] sequence lengths for decode flash_attn_with_kvcache
        'cu_seqlens_q',       # [bsz+1] cumulative sequence lengths for prefill
        'cu_seqlens_k',       # [bsz+1] cumulative sequence lengths for prefill
        'max_seqlen_q',       # int max query sequence length
        'max_seqlen_k',       # int max key sequence length
        'deltanet_slots',     # [bsz] slot indices for DeltaNet state pool
        'is_prefill',         # bool: whether this is a prefill step
    )

    def __init__(self):
        self.slot_mapping = None
        self.block_tables = None
        self.context_lens = None
        self.cu_seqlens_q = None
        self.cu_seqlens_k = None
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0
        self.deltanet_slots = None
        self.is_prefill = False


def get_context() -> AttentionContext | None:
    """Get the current attention context."""
    return getattr(_context, 'attn_ctx', None)


def set_context(ctx: AttentionContext | None):
    """Set the current attention context."""
    _context.attn_ctx = ctx


def reset_context():
    """Reset the attention context after forward pass."""
    _context.attn_ctx = None
