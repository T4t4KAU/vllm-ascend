# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission and reusable plans for mixed ForkAttention decode batches."""

from dataclasses import dataclass, replace

import numpy as np

from vllm_ascend.attention.fork_plan import MAX_DESCRIPTOR_VALUE, validate_seq_lengths

KERNEL_PAGE_SIZE = 128
MAX_BATCH = 8
MAX_SPLITS = 16
LONG_PREFIX_TOKENS = 65536
SHORT_PREFIX_SPLITS = 4


@dataclass(frozen=True)
class ForkAttentionConfig:
    enabled: bool = False
    min_shared_tokens: int = 32768
    diagnostics: bool = False

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.diagnostics) is not bool:
            raise ValueError("fork_attention enabled and diagnostics must be booleans")
        if type(self.min_shared_tokens) is not int or self.min_shared_tokens < KERNEL_PAGE_SIZE:
            raise ValueError("fork_attention min_shared_tokens must be an integer >= 128")
        if self.min_shared_tokens % KERNEL_PAGE_SIZE:
            raise ValueError("fork_attention min_shared_tokens must be page aligned (128 tokens)")


@dataclass(frozen=True)
class ForkBatchPlan:
    batch_size: int
    block_rows: tuple[tuple[int, ...], ...]
    query_ends: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    query_indices: tuple[int, ...]
    merge_indices: tuple[tuple[int, ...], ...]
    prefix_tokens: tuple[int, ...]
    cohorts: tuple[tuple[int, ...], ...]

    def with_lengths(self, lengths):
        # Every query's private tail is one of the final batch_size segments.
        tails = tuple(int(n) - p for n, p in zip(lengths, self.prefix_tokens))
        return replace(self, kv_lengths=self.kv_lengths[: -self.batch_size] + tails)


class ForkBatchPlanner:
    """One last-plan cache per attention metadata builder, never per layer.

    Revalidate the active physical page table on every step. Request IDs and
    sequence lengths alone cannot detect page reassignment or batch reordering.
    Reuse descriptors and query maps while pages and page counts are unchanged.
    """

    def __init__(self, config: ForkAttentionConfig):
        self.config = config
        self._rows = None
        self._counts = None
        self._plan = None
        self.rebuilds = 0
        self.reuses = 0

    def build(self, rows: np.ndarray, lengths) -> ForkBatchPlan | None:
        lengths = validate_seq_lengths(lengths)
        batch = len(lengths)
        if not 2 <= batch <= MAX_BATCH or np.count_nonzero(lengths > self.config.min_shared_tokens) < 2:
            return None
        if rows.ndim != 2 or not np.issubdtype(rows.dtype, np.integer) or rows.shape[0] != batch:
            raise ValueError("expected an integer physical page table for the actual requests")
        counts = (lengths + KERNEL_PAGE_SIZE - 1) // KERNEL_PAGE_SIZE
        width = int(counts.max())
        if np.any(lengths <= 0) or width > rows.shape[1]:
            raise ValueError("invalid sequence length or insufficient page capacity")
        live = rows[:, :width]
        if self._rows is not None and np.array_equal(counts, self._counts) and np.array_equal(live, self._rows):
            self.reuses += 1
            return self._plan.with_lengths(lengths) if self._plan is not None else None
        valid = np.arange(width)[None, :] < counts[:, None]
        if np.any(((live < 0) | (live > MAX_DESCRIPTOR_VALUE)) & valid):
            raise ValueError("active physical page IDs must be nonnegative int32 integers")
        self.rebuilds += 1
        self._rows, self._counts = live.copy(), counts.copy()
        self._plan = self._build(live, lengths, counts)
        return self._plan

    def _build(self, rows, lengths, counts):
        minimum = self.config.min_shared_tokens // KERNEL_PAGE_SIZE
        groups = {}
        for index, length in enumerate(lengths):
            if length > self.config.min_shared_tokens:
                groups.setdefault(rows[index, :minimum].tobytes(), []).append(index)
        cohorts = [group for group in groups.values() if len(group) >= 2]
        if not cohorts:
            return None
        batch = len(lengths)
        segments, kv_lengths, query_ends, query_indices = [], [], [], []
        merge_indices = [[] for _ in lengths]
        prefix_tokens = [0] * batch
        for cohort in cohorts:
            limit = int(min((lengths[q] - 1) // KERNEL_PAGE_SIZE for q in cohort))
            shared = np.all(rows[cohort, :limit] == rows[cohort[0], :limit], axis=0)
            different = np.flatnonzero(~shared)
            pages = int(different[0]) if len(different) else limit
            prefix = pages * KERNEL_PAGE_SIZE
            splits = min(pages, MAX_SPLITS if prefix >= LONG_PREFIX_TOKENS else SHORT_PREFIX_SPLITS)
            for part in range(splits):
                start, end = part * pages // splits, (part + 1) * pages // splits
                segments.append(tuple(rows[cohort[0], start:end].tolist()))
                kv_lengths.append((end - start) * KERNEL_PAGE_SIZE)
                for q in cohort:
                    merge_indices[q].append(len(query_indices))
                    query_indices.append(q)
                    prefix_tokens[q] = prefix
                query_ends.append(len(query_indices))
        # Unrelated queries remain independent full-length tail sequences.
        for q in range(batch):
            start = prefix_tokens[q] // KERNEL_PAGE_SIZE
            segments.append(tuple(rows[q, start : counts[q]].tolist()))
            kv_lengths.append(int(lengths[q]) - prefix_tokens[q])
            merge_indices[q].append(len(query_indices))
            query_indices.append(q)
            query_ends.append(len(query_indices))
        return ForkBatchPlan(
            batch,
            tuple(segments),
            tuple(query_ends),
            tuple(kv_lengths),
            tuple(query_indices),
            tuple(map(tuple, merge_indices)),
            tuple(prefix_tokens),
            tuple(map(tuple, cohorts)),
        )
