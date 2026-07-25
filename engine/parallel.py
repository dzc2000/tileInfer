"""Tensor Parallelism utilities for multi-GPU inference.

Provides:
- Process group initialization (NCCL backend)
- TP rank / world_size accessors
- All-Reduce communication primitive
- Weight sharding helpers (column-parallel, row-parallel)
"""
import torch
import torch.distributed as dist

# Module-level TP state
_tp_rank: int = 0
_tp_world_size: int = 1
_tp_group = None
_initialized: bool = False


def init_distributed(tp_size: int = 1, backend: str = "nccl"):
    """Initialize tensor parallelism.

    Must be called before model loading when tp_size > 1.
    Sets CUDA device to the local rank.
    """
    global _tp_rank, _tp_world_size, _tp_group, _initialized

    if tp_size <= 1:
        _initialized = True
        return

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    _tp_world_size = dist.get_world_size()
    _tp_rank = dist.get_rank()

    if _tp_world_size != tp_size:
        raise ValueError(
            f"tp_size={tp_size} but world_size={_tp_world_size}. "
            f"Launch with exactly {tp_size} processes (e.g. torchrun --nproc_per_node={tp_size})."
        )

    _tp_group = dist.group.WORLD
    _initialized = True

    # Bind each process to its GPU
    torch.cuda.set_device(_tp_rank)
    print(f"[TP] Rank {_tp_rank}/{_tp_world_size} initialized on cuda:{_tp_rank}")


def is_initialized() -> bool:
    return _initialized


def get_tp_rank() -> int:
    return _tp_rank


def get_tp_world_size() -> int:
    return _tp_world_size


def get_tp_group():
    return _tp_group


def is_tp_active() -> bool:
    return _tp_world_size > 1


def tp_all_reduce(tensor: torch.Tensor):
    """In-place All-Reduce (sum) across TP group."""
    if _tp_world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_tp_group)


def tp_broadcast(tensor: torch.Tensor, src: int = 0):
    """In-place broadcast a tensor from src rank to all TP ranks."""
    if _tp_world_size > 1:
        dist.broadcast(tensor, src=src, group=_tp_group)


def shard_weight_col(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Shard weight along dim=0 (column-parallel: split output dim).

    weight: [out_dim, in_dim] -> [out_dim // world_size, in_dim]
    """
    if world_size <= 1:
        return weight
    shard_size = weight.shape[0] // world_size
    return weight[rank * shard_size : (rank + 1) * shard_size, :].contiguous()


def shard_weight_row(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Shard weight along dim=1 (row-parallel: split input dim).

    weight: [out_dim, in_dim] -> [out_dim, in_dim // world_size]
    """
    if world_size <= 1:
        return weight
    shard_size = weight.shape[1] // world_size
    return weight[:, rank * shard_size : (rank + 1) * shard_size].contiguous()


def shard_bias_col(bias: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Shard bias for column-parallel (split output dim)."""
    if world_size <= 1:
        return bias
    shard_size = bias.shape[0] // world_size
    return bias[rank * shard_size : (rank + 1) * shard_size].contiguous()


def barrier():
    """Synchronize all TP ranks."""
    if _tp_world_size > 1:
        dist.barrier(group=_tp_group)
