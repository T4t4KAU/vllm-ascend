# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm_ascend.attention.fork_batch import ForkAttentionConfig, ForkBatchPlanner


def planner():
    return ForkBatchPlanner(ForkAttentionConfig(enabled=True, min_shared_tokens=128))


def check_coverage(plan, rows, lengths):
    for q, length in enumerate(lengths):
        pages, tokens = [], 0
        start = 0
        for segment, end in enumerate(plan.query_ends):
            if q in plan.query_indices[start:end]:
                pages.extend(plan.block_rows[segment])
                tokens += plan.kv_lengths[segment]
            start = end
        assert pages == rows[q, : (length + 127) // 128].tolist()
        assert tokens == length
        assert all(plan.query_indices[index] == q for index in plan.merge_indices[q])


def test_multiple_noncontiguous_cohorts_and_independent_request():
    rows = np.array([[2, 3, 8], [4, 5, 9], [2, 3, 10], [4, 6, 11], [7, 12, 13]])
    lengths = [300, 310, 320, 330, 340]
    plan = planner().build(rows, lengths)
    assert plan.cohorts == ((0, 2), (1, 3))
    assert plan.prefix_tokens == (256, 128, 256, 128, 0)
    check_coverage(plan, rows, lengths)


def test_growing_lengths_reuse_descriptors_but_not_old_tail_lengths():
    builder = planner()
    rows = np.array([[2, 3, 8], [2, 3, 9]])
    first = builder.build(rows, [300, 310])
    second = builder.build(rows, [301, 311])
    assert second.block_rows is first.block_rows
    assert second.query_indices is first.query_indices
    assert second.kv_lengths[-2:] == (45, 55)
    assert builder.reuses == 1
    check_coverage(second, rows, [301, 311])


def test_reassignment_reorder_rollback_and_boundary_invalidate_topology():
    builder = planner()
    rows = np.array([[2, 3, 8], [2, 3, 9], [7, 4, 10]])
    for lengths in ([255, 250, 280], [256, 256, 280], [257, 257, 281], [200, 200, 200]):
        plan = builder.build(rows, lengths)
        check_coverage(plan, rows, lengths)
    reordered = rows[[2, 0, 1]]
    plan = builder.build(reordered, [300] * 3)
    assert plan.cohorts == ((1, 2),)
    reordered[2, 0] = 12
    assert builder.build(reordered, [300] * 3) is None
    assert builder.build(reordered, [301] * 3) is None


def test_inactive_padding_is_ignored_and_live_negative_page_is_rejected():
    rows = np.array([[2, 8, -1], [2, 9, -1]])
    assert planner().build(rows, [200, 201])
    with pytest.raises(ValueError, match="nonnegative"):
        planner().build(rows, [300, 301])


@pytest.mark.parametrize("length", [1, 127, 128])
def test_short_prefix_falls_back(length):
    assert planner().build(np.array([[2], [2]]), [length, length]) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"diagnostics": "false"},
        {"min_shared_tokens": 0},
        {"min_shared_tokens": 129},
        {"min_shared_tokens": 128.0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ForkAttentionConfig(**kwargs)


@pytest.mark.parametrize("lengths", [[129.5, 130.5], [True, True], [[129], [130]], [129, 1 << 31]])
def test_invalid_lengths_cannot_bypass_validation_on_cache_hit(lengths):
    builder = planner()
    rows = np.array([[1, 2], [1, 3]], dtype=np.int32)
    original = builder.build(rows, [129, 130])
    with pytest.raises(ValueError, match="sequence lengths"):
        builder.build(rows, lengths)
    assert builder.reuses == 0
    assert builder.build(rows, [129, 130]) == original


def test_wide_page_ids_are_rejected_before_int32_device_conversion():
    rows = np.array([[1 << 31, 2], [1 << 31, 3]], dtype=np.int64)
    with pytest.raises(ValueError, match="int32"):
        planner().build(rows, [129, 130])


def test_int32_page_count_rounding_cannot_overflow():
    rows = np.array([[1, 2], [1, 3]], dtype=np.int32)
    lengths = np.full(2, np.iinfo(np.int32).max, dtype=np.int32)
    with pytest.raises(ValueError, match="insufficient page capacity"):
        planner().build(rows, lengths)
