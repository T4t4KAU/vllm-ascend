# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU planning for one cohort of single-token ForkAttention queries."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from numbers import Integral

import numpy as np

MAX_PREFIX_SPLITS = 32
MAX_DESCRIPTOR_VALUE = (1 << 31) - 1


def validate_seq_lengths(lengths):
    """Validate CPU lengths before arithmetic or a cached-plan lookup."""
    values = np.asarray(lengths)
    if values.ndim != 1 or (values.size and not np.issubdtype(values.dtype, np.integer)):
        raise ValueError("sequence lengths must be one-dimensional integers")
    if np.any(values <= 0) or np.any(values > MAX_DESCRIPTOR_VALUE):
        raise ValueError("sequence lengths must be positive int32 values")
    # Widen before rounding up page counts (including int32/uint32 inputs).
    return values.astype(np.int64, copy=False)


def _page_tuple(row):
    # Convert NumPy scalars in C instead of boxing each page in a Python loop.
    return tuple(row.tolist()) if isinstance(row, np.ndarray) else tuple(row)


@dataclass(frozen=True)
class ForkAttentionPlan:
    """A shared, page-aligned prefix followed by a private tail per query.

    Page identities, not session IDs or token-count hints, establish sharing.
    The current decode token always belongs to the tail. All segments therefore
    have a nonempty key range, including at an exact page boundary.
    """

    batch_size: int
    block_size: int
    prefix_tokens: int
    prefix_splits: int
    block_rows: tuple[tuple[int, ...], ...]
    query_ends: tuple[int, ...]
    kv_lengths: tuple[int, ...]

    def __post_init__(self):
        # This public value object is also accepted by graph-update APIs.
        # Validate here so construction/replace cannot bypass the planner and
        # launch a kernel with negative pages or incomplete partial outputs.
        dimensions = (self.batch_size, self.block_size, self.prefix_tokens, self.prefix_splits)
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in dimensions):
            raise ValueError("plan dimensions must be positive integers")
        if self.batch_size < 2 or not 1 <= self.prefix_splits <= MAX_PREFIX_SPLITS:
            raise ValueError("invalid cohort or prefix split count")
        if self.prefix_tokens % self.block_size:
            raise ValueError("prefix must contain complete pages")
        if not all(type(x) is tuple for x in (self.block_rows, self.query_ends, self.kv_lengths)):
            raise ValueError("plan descriptors must be immutable tuples")
        if any(isinstance(n, bool) or not isinstance(n, Integral) for n in self.query_ends):
            raise ValueError("query segment ends must be integers")
        batch, splits = self.batch_size, self.prefix_splits
        shared_ends = tuple((i + 1) * batch for i in range(splits)) + tuple(
            splits * batch + i + 1 for i in range(batch)
        )
        separate_ends = tuple(range(1, (splits + 1) * batch + 1))
        if self.query_ends not in (shared_ends, separate_ends):
            raise ValueError("query segments must cover each prefix part and private tail exactly once")
        if len(self.block_rows) != len(self.query_ends) or len(self.kv_lengths) != len(self.query_ends):
            raise ValueError("segment descriptor counts differ")
        lengths = validate_seq_lengths(self.kv_lengths)
        for row, length in zip(self.block_rows, lengths):
            if type(row) is not tuple or len(row) != (length + self.block_size - 1) // self.block_size:
                raise ValueError("each segment requires an immutable row with exactly its live pages")
            if any(
                isinstance(p, bool) or not isinstance(p, Integral) or not 0 <= p <= MAX_DESCRIPTOR_VALUE for p in row
            ):
                raise ValueError("physical page IDs must be nonnegative int32 integers")
        prefix_lengths = lengths[:-batch]
        if np.any(prefix_lengths % self.block_size):
            raise ValueError("prefix segments must contain complete pages")
        if self.query_ends == separate_ends:
            prefix_lengths = prefix_lengths.reshape(splits, batch)
        if np.any(prefix_lengths.sum(axis=0) != self.prefix_tokens):
            raise ValueError("prefix segment lengths do not match prefix_tokens")

    @property
    def num_parts(self) -> int:
        return self.prefix_splits + 1


def build_fork_plan(
    block_rows: Sequence[Sequence[int]],
    seq_lengths: Sequence[int],
    block_size: int,
    *,
    prefix_splits: int = 8,
    min_shared_tokens: int = 1024,
    share_prefix_queries: bool = True,
) -> ForkAttentionPlan | None:
    """Plan an explicitly selected cohort; unrelated cohorts return ``None``.

    Inputs must be CPU metadata. This function does not read device tensors.
    The caller must rebuild after page reassignment or sequence-length changes.
    Split count is an experimental tuning parameter, not an admission policy.
    Set share_prefix_queries=False only for the split-without-sharing control:
    it repeats the prefix page descriptors and executes each query separately.
    """
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in (block_size, min_shared_tokens, prefix_splits)):
        raise ValueError("block_size, min_shared_tokens and prefix_splits must be integers")
    if block_size <= 0 or min_shared_tokens < 0:
        raise ValueError("block_size must be positive and min_shared_tokens nonnegative")
    if not 1 <= prefix_splits <= MAX_PREFIX_SPLITS:
        raise ValueError(f"prefix_splits must be in [1, {MAX_PREFIX_SPLITS}]")
    if type(share_prefix_queries) is not bool:
        raise ValueError("share_prefix_queries must be a boolean")
    seq_lengths = validate_seq_lengths(seq_lengths).tolist()
    batch = len(seq_lengths)
    if len(block_rows) != batch:
        raise ValueError("block-table and sequence-length row counts differ")
    array_input = isinstance(block_rows, np.ndarray)
    if array_input and (block_rows.ndim != 2 or not np.issubdtype(block_rows.dtype, np.integer)):
        raise ValueError("physical page IDs must form a two-dimensional integer table")
    for row, length in zip(block_rows, seq_lengths):
        if length <= 0 or len(row) < ceil(length / block_size):
            raise ValueError("each sequence requires a positive length and enough pages")
        if not array_input:
            active = np.asarray(row[: ceil(length / block_size)])
            if active.ndim != 1 or not np.issubdtype(active.dtype, np.integer):
                raise ValueError("physical page IDs must be one-dimensional integers")
            if np.any(active < 0):
                raise ValueError("active physical page IDs must be nonnegative")
    if array_input and batch:
        counts = (np.asarray(seq_lengths) + block_size - 1) // block_size
        width = int(counts.max())
        valid = np.arange(width)[None, :] < counts[:, None]
        if np.any((block_rows[:, :width] < 0) & valid):
            raise ValueError("active physical page IDs must be nonnegative")
    if batch < 2:
        return None
    limit = min((length - 1) // block_size for length in seq_lengths)
    # The runner owns a rectangular NumPy block table. Compare pages in native
    # loops rather than paying Python dispatch once per token block per request.
    if array_input:
        shared = np.all(block_rows[:, :limit] == block_rows[:1, :limit], axis=0)
    else:
        first = np.asarray(block_rows[0][:limit])
        shared = np.ones(limit, dtype=np.bool_)
        for row in block_rows[1:]:
            shared &= np.asarray(row[:limit]) == first
    different = np.flatnonzero(~shared)
    common = int(different[0]) if len(different) else limit
    if common == 0 or common * block_size < min_shared_tokens:
        return None

    splits = min(prefix_splits, common)
    rows, lengths, query_ends = [], [], []
    for index in range(splits):
        start, end = index * common // splits, (index + 1) * common // splits
        for query in range(1 if share_prefix_queries else batch):
            rows.append(_page_tuple(block_rows[0][start:end]))
            lengths.append((end - start) * block_size)
            query_ends.append((index + 1) * batch if share_prefix_queries else index * batch + query + 1)
    for index, (row, length) in enumerate(zip(block_rows, seq_lengths)):
        rows.append(_page_tuple(row[common : ceil(length / block_size)]))
        lengths.append(length - common * block_size)
        query_ends.append(splits * batch + index + 1)
    return ForkAttentionPlan(
        batch, block_size, common * block_size, splits, tuple(rows), tuple(query_ends), tuple(lengths)
    )
