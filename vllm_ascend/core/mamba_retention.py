# SPDX-License-Identifier: Apache-2.0
"""Sparse, block-aligned recurrent-state checkpoints for the 0.22.1 engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


@dataclass(frozen=True)
class MambaRetentionConfig:
    # None preserves existing caching; 0 retains only known reuse boundaries.
    # Positive intervals are measured in tokens, not blocks.
    interval: int | None = None
    diagnostics: bool = False
    prefer_reuse_boundaries: bool = False

    def __post_init__(self):
        if self.interval is not None and (type(self.interval) is not int or self.interval < 0):
            raise ValueError("mamba_cache_retention.interval must be null or a nonnegative integer")
        if type(self.diagnostics) is not bool:
            raise ValueError("mamba_cache_retention.diagnostics must be a boolean")
        if type(self.prefer_reuse_boundaries) is not bool:
            raise ValueError("mamba_cache_retention.prefer_reuse_boundaries must be a boolean")
        if self.prefer_reuse_boundaries and self.interval is None:
            raise ValueError("prefer_reuse_boundaries requires an explicit retention interval")


def insert_free_block_before(queue: FreeKVCacheBlockQueue, block: KVCacheBlock, before: KVCacheBlock) -> None:
    """Insert a detached, unreferenced block without changing its cache hash."""
    assert block.ref_cnt == 0 and not block.is_null
    previous = before.prev_free_block
    assert previous is not None
    previous.next_free_block = block
    block.prev_free_block = previous
    block.next_free_block = before
    before.prev_free_block = block
    queue.num_free_blocks += 1


@dataclass
class PeriodicCheckpointIndex:
    """Per-pool probation, bounded by physical blocks rather than requests.

    Only optional periodic checkpoints enter probation. Prompt/decode/junction
    states and checkpoints actually reused by a request use ordinary LRU.
    Hash checks prevent stale metadata from classifying a recycled block.
    """

    hashes: dict[int, bytes] = field(default_factory=dict)
    available: dict[int, KVCacheBlock] = field(default_factory=dict)

    def discard(self, block: KVCacheBlock) -> None:
        self.hashes.pop(block.block_id, None)
        self.available.pop(block.block_id, None)

    def remember(self, block: KVCacheBlock) -> None:
        self.discard(block)
        assert block.block_hash is not None
        self.hashes[block.block_id] = block.block_hash

    def contains(self, block: KVCacheBlock) -> bool:
        return block.block_hash is not None and self.hashes.get(block.block_id) == block.block_hash

    def release(self, block: KVCacheBlock) -> None:
        if self.contains(block):
            assert block.ref_cnt == 0
            self.available[block.block_id] = block

    def take(self) -> KVCacheBlock | None:
        while self.available:
            block = self.available.pop(next(iter(self.available)))
            if block.ref_cnt == 0 and self.contains(block):
                return block
        return None

    def clear(self) -> None:
        self.hashes.clear()
        self.available.clear()


def reuse_boundaries(prompt_tokens: int, block_size: int, shared_boundary: int = 0) -> tuple[int, ...]:
    """Keep both identical-prompt replay and append-only continuation boundaries.

    Replay recomputes the last token for logits, hence prompt_tokens - 1.
    At exact block boundaries this differs from the continuation checkpoint.
    """
    return tuple(
        sorted({max(0, n) // block_size * block_size for n in (prompt_tokens - 1, prompt_tokens, shared_boundary)})
    )


def checkpoint_mask(
    start: int, end: int, block_size: int, interval: int, boundaries: tuple[int, ...], prompt_tokens: int
) -> list[bool]:
    return [
        (i + 1) * block_size in boundaries
        or (i + 1) * block_size > prompt_tokens
        or (interval > 0 and (i + 1) * block_size % interval == 0)
        for i in range(start, end)
    ]


def cap_at_checkpoint(computed: int, scheduled: int, interval: int, boundaries: tuple[int, ...]) -> int:
    """Produce a real final state at each retained boundary, never relabel one."""
    candidates = [boundary for boundary in boundaries if computed < boundary < computed + scheduled]
    if interval > 0:
        next_interval = (computed // interval + 1) * interval
        if next_interval < computed + scheduled:
            candidates.append(next_interval)
    return min(candidates) - computed if candidates else scheduled


@dataclass
class MambaRetentionState:
    config: MambaRetentionConfig
    candidate_tokens: int = 0
    lookups: int = 0
    full_attention_hit_tokens: int = 0
    hybrid_hit_tokens: int = 0
    checkpoint_gap_tokens: int = 0
    retained_blocks: int = 0
    skipped_blocks: int = 0
    recycled_uncached_blocks: int = 0
    preferred_periodic_reclaims: int = 0
    promoted_checkpoints: int = 0
    checkpoints: PeriodicCheckpointIndex = field(default_factory=PeriodicCheckpointIndex)

    def snapshot(self) -> dict:
        # Never recursively serialize block objects (the free list is cyclic).
        result = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("config", "checkpoints")}
        result["config"] = asdict(self.config)
        result["periodic_blocks"] = len(self.checkpoints.hashes)
        result["free_periodic_blocks"] = len(self.checkpoints.available)
        return result
