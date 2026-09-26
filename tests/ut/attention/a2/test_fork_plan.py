# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import numpy as np
import pytest

from vllm_ascend.attention.fork_plan import build_fork_plan


def test_shared_segments_and_private_tails_cover_original_sequences():
    rows = [[7, 3, 11, 2, 9], [7, 3, 11, 4, 8], [7, 3, 11, 6, 5]]
    lengths = [513, 520, 600]
    plan = build_fork_plan(rows, lengths, 128, prefix_splits=2, min_shared_tokens=128)
    assert plan is not None
    assert plan.prefix_tokens == 384
    assert plan.kv_lengths == (128, 256, 129, 136, 216)
    assert plan.query_ends == (3, 6, 7, 8, 9)
    prefix_pages = sum((list(row) for row in plan.block_rows[: plan.prefix_splits]), [])
    for query, row in enumerate(rows):
        assert prefix_pages + list(plan.block_rows[plan.prefix_splits + query]) == row
        assert (
            sum(plan.kv_lengths[: plan.prefix_splits]) + plan.kv_lengths[plan.prefix_splits + query] == lengths[query]
        )


def test_equal_lengths_do_not_imply_shared_pages():
    assert build_fork_plan([[2, 3], [4, 5]], [129, 129], 128, min_shared_tokens=128) is None


def test_shared_suffix_is_not_a_shared_prefix():
    assert build_fork_plan([[2, 3, 8], [4, 3, 8]], [300, 300], 128, min_shared_tokens=128) is None


@pytest.mark.parametrize("length, prefix", [(127, None), (128, None), (129, 128), (256, 128), (257, 256)])
def test_current_token_remains_in_nonempty_tail(length, prefix):
    plan = build_fork_plan([[2, 3, 4], [2, 3, 5]], [length, length], 128, min_shared_tokens=0)
    assert (plan.prefix_tokens if plan else None) == prefix
    if plan:
        assert all(length > 0 for length in plan.kv_lengths)


def test_minimum_applies_to_complete_shared_pages():
    assert build_fork_plan([[1, 2], [1, 3]], [200, 200], 128, min_shared_tokens=129) is None


def test_reordered_pages_and_sequence_rollback_require_new_plan():
    rows = [[7, 3, 11, 2], [7, 3, 11, 4]]
    original = build_fork_plan(rows, [400, 400], 128, min_shared_tokens=128)
    rows[1][1] = 6
    changed = build_fork_plan(rows, [400, 400], 128, min_shared_tokens=128)
    rollback = build_fork_plan(rows, [128, 400], 128, min_shared_tokens=128)
    assert original.prefix_tokens == 384
    assert changed.prefix_tokens == 128
    assert rollback is None


@pytest.mark.parametrize(
    "rows,lengths,block,splits",
    [
        ([[1]], [0], 128, 1),
        ([[1]], [129], 128, 1),
        ([[-1]], [1], 128, 1),
        ([[1]], [1, 1], 128, 1),
        ([[1]], [1], 0, 1),
        ([[1]], [1], 128, 33),
    ],
)
def test_invalid_metadata_rejected(rows, lengths, block, splits):
    with pytest.raises(ValueError):
        build_fork_plan(rows, lengths, block, prefix_splits=splits)


def test_padding_pages_are_not_considered_live():
    plan = build_fork_plan([[2, 3, -1], [2, 4, -1]], [129, 129], 128, min_shared_tokens=128)
    assert plan is not None
    assert plan.block_rows == ((2,), (3,), (4,))


def test_empty_and_single_query_are_ineligible():
    assert build_fork_plan([], [], 128) is None
    assert build_fork_plan([[2]], [1], 128) is None


def test_unshared_control_repeats_descriptors_without_changing_query_order():
    rows = [[7, 3, 11, 2], [7, 3, 11, 4]]
    plan = build_fork_plan(rows, [400, 400], 128, prefix_splits=2, min_shared_tokens=128, share_prefix_queries=False)
    assert plan.block_rows == ((7,), (7,), (3, 11), (3, 11), (2,), (4,))
    assert plan.query_ends == (1, 2, 3, 4, 5, 6)
    assert plan.kv_lengths == (128, 128, 256, 256, 16, 16)
    assert plan.num_parts == 3


def test_numpy_page_table_matches_sequence_input():
    rows = [[7, 3, 11, 2], [7, 3, 11, 4]]
    args = dict(seq_lengths=[400, 400], block_size=128, prefix_splits=2, min_shared_tokens=128)
    assert build_fork_plan(rows, **args) == build_fork_plan(np.asarray(rows, dtype=np.int32), **args)


def test_fractional_physical_page_ids_are_rejected():
    with pytest.raises(ValueError, match="integers"):
        build_fork_plan([[0.5, 1], [0.5, 2]], [129, 129], 128)


def test_numpy_validation_masks_padding_but_rejects_live_negative_pages():
    rows = np.array([[2, 3, -1], [2, 4, -1]], dtype=np.int32)
    assert build_fork_plan(rows, [129, 129], 128, min_shared_tokens=128) is not None
    with pytest.raises(ValueError, match="nonnegative"):
        build_fork_plan(rows, [300, 129], 128, min_shared_tokens=128)
    with pytest.raises(ValueError, match="nonnegative"):
        build_fork_plan(np.array([[-1]], dtype=np.int32), [1], 128)


@pytest.mark.parametrize("lengths", [[129.5, 130.5], [True, True], [[129], [130]], [129, 1 << 31]])
def test_invalid_lengths_fail_before_page_arithmetic(lengths):
    with pytest.raises(ValueError, match="sequence lengths"):
        build_fork_plan([[1, 2], [1, 3]], lengths, 128, min_shared_tokens=128)


@pytest.mark.parametrize(
    "changes",
    [
        {"block_rows": ((-1,), (2,), (3,))},
        {"block_rows": ((1 << 31,), (2,), (3,))},
        {"block_rows": ((1.5,), (2,), (3,))},
        {"block_rows": ((1,), (), (3,))},
        {"block_rows": ([1], (2,), (3,))},
        {"query_ends": (2, 2, 4)},
        {"query_ends": (2.0, 3.0, 4.0)},
        {"kv_lengths": (128, 0, 2)},
        {"kv_lengths": (128, 1.5, 2)},
        {"kv_lengths": (128, 1)},
        {"prefix_tokens": 256},
        {"prefix_splits": 0},
    ],
)
def test_plan_construction_cannot_bypass_descriptor_validation(changes):
    plan = build_fork_plan([[1, 2], [1, 3]], [129, 130], 128, min_shared_tokens=128)
    with pytest.raises(ValueError):
        replace(plan, **changes)
