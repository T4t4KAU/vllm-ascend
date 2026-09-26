# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.fork_plan import build_fork_plan
from vllm_ascend.ops.fork_attention import ForkAttention, ForkAttentionGraph
from vllm_ascend.ops.triton.fork_attention import merge_fork_partials


def make_case(batch, dim, block, dtype):
    torch.manual_seed(71)
    heads, kv_heads, prefix_pages, tail_pages = 16, 4, 3, 2
    pages = prefix_pages + batch * tail_pages
    permutation = torch.randperm(pages).tolist()
    rows = [
        permutation[:prefix_pages] + permutation[prefix_pages + i * tail_pages : prefix_pages + (i + 1) * tail_pages]
        for i in range(batch)
    ]
    lengths = [prefix_pages * block + 1 + (i * (block + 17)) % (2 * block) for i in range(batch)]
    key = torch.randn(pages, block, kv_heads, dim, dtype=dtype)
    value = torch.randn_like(key)
    query = torch.randn(batch, heads, dim, dtype=dtype)
    return query, key, value, rows, lengths


def dense_reference(query, key, value, rows, lengths):
    outputs = []
    ratio = query.shape[1] // key.shape[2]
    for i, (row, length) in enumerate(zip(rows, lengths)):
        k = key[row].flatten(0, 1)[:length].float().repeat_interleave(ratio, dim=1).transpose(0, 1)
        v = value[row].flatten(0, 1)[:length].float().repeat_interleave(ratio, dim=1).transpose(0, 1)
        logits = torch.einsum("hd,hsd->hs", query[i].float(), k) / math.sqrt(query.shape[-1])
        outputs.append(torch.einsum("hs,hsd->hd", logits.softmax(-1), v))
    return torch.stack(outputs)


@pytest.mark.parametrize("batch,dim,block", [(2, 64, 128), (3, 128, 128), (4, 256, 128), (8, 256, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("splits", [1, 3])
def test_fork_matches_dense_with_fragmented_pages_and_unequal_tails(batch, dim, block, dtype, splits):
    q, k, v, rows, lengths = make_case(batch, dim, block, dtype)
    reference = dense_reference(q, k, v, rows, lengths)
    plan = build_fork_plan(rows, lengths, block, prefix_splits=splits, min_shared_tokens=128)
    qn, kn, vn = (t.npu() for t in (q, k, v))
    op = ForkAttention(qn, kn, vn, plan)
    result = op(qn).cpu().float()
    torch.testing.assert_close(result, reference, atol=2e-3 if dtype == torch.bfloat16 else 3e-4, rtol=2e-2)


def test_graph_replay_consumes_new_queries_and_kv_values():
    q, k, v, rows, lengths = make_case(4, 256, 128, torch.bfloat16)
    plan = build_fork_plan(rows, lengths, 128, prefix_splits=3, min_shared_tokens=128)
    qn, kn, vn = (t.npu() for t in (q, k, v))
    op = ForkAttention(qn, kn, vn, plan)
    for _ in range(3):
        op(qn)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = op(qn)
    for seed in (79, 83, 89):
        torch.manual_seed(seed)
        q.normal_()
        v.normal_()
        qn.copy_(q)
        vn.copy_(v)
        graph.replay()
        expected = dense_reference(q, k, v, rows, lengths)
        torch.testing.assert_close(output.cpu().float(), expected, atol=2e-3, rtol=2e-2)


def test_private_tail_cannot_affect_sibling_output():
    q, k, v, rows, lengths = make_case(4, 256, 128, torch.bfloat16)
    plan = build_fork_plan(rows, lengths, 128, prefix_splits=2, min_shared_tokens=128)
    qn, kn, vn = (t.npu() for t in (q, k, v))
    op = ForkAttention(qn, kn, vn, plan)
    before = op(qn).clone()
    vn[rows[0][3]] = 100
    after = op(qn).clone()
    torch.testing.assert_close(before[1:], after[1:], atol=0, rtol=0)
    assert (before[0] - after[0]).abs().max().item() > 0.01


def test_page_capacity_and_query_stride_rejected():
    q, k, v, rows, lengths = make_case(2, 64, 128, torch.bfloat16)
    plan = build_fork_plan(rows, lengths, 128, min_shared_tokens=128)
    qn, kn, vn = (t.npu() for t in (q, k, v))
    with pytest.raises(ValueError, match="page ID"):
        ForkAttention(qn, kn[:1], vn[:1], plan)
    with pytest.raises(ValueError, match="query heads"):
        ForkAttention(qn[:, :0], kn, vn, plan)
    op = ForkAttention(qn, kn, vn, plan)
    with pytest.raises(ValueError, match="contiguous"):
        op(torch.empty(2, 16, 128, dtype=qn.dtype, device=qn.device)[..., ::2])


def test_merge_33_parts_with_large_lse_is_finite_and_accurate():
    torch.manual_seed(97)
    parts, rows, dim = 33, 32, 256
    values = torch.randn(parts, rows, dim, dtype=torch.bfloat16)
    lse = torch.randn(parts, rows) * 10 + 1000
    expected = (values.float() * lse.softmax(0).unsqueeze(-1)).sum(0)
    values_npu, lse_npu = values.npu(), lse.npu()
    result = torch.empty((rows, dim), dtype=torch.bfloat16, device="npu")
    merge_fork_partials[(rows,)](values_npu, lse_npu, result, rows, dim, parts, 64)
    torch.testing.assert_close(result.cpu().float(), expected, atol=2e-2, rtol=1e-2)


def test_unsplit_hybrid_manager_pages_are_rejected():
    q, k, v, rows, lengths = make_case(4, 256, 1024, torch.bfloat16)
    plan = build_fork_plan(rows, lengths, 1024)
    with pytest.raises(ValueError, match="128-token kernel pages"):
        ForkAttention(q.npu(), k.npu(), v.npu(), plan)


def test_graph_task_updates_lengths_pages_and_rollback_without_recapture():
    q, k, v, rows, lengths = make_case(4, 256, 128, torch.bfloat16)
    plan = build_fork_plan(rows, lengths, 128, prefix_splits=2, min_shared_tokens=128)
    qn, kn, vn = (t.npu() for t in (q, k, v))
    op = ForkAttention(qn, kn, vn, plan, max_pages_per_segment=8)
    graph = ForkAttentionGraph(op, qn)
    for step, length in enumerate([401, 511, 512, 513, 257, 530]):
        changed_rows = [row[:] for row in rows]
        if step % 2:
            changed_rows[0][3:], changed_rows[1][3:] = changed_rows[1][3:], changed_rows[0][3:]
        changed_lengths = [length] * 4
        updated = build_fork_plan(changed_rows, changed_lengths, 128, prefix_splits=2, min_shared_tokens=128)
        q.normal_()
        qn.copy_(q)
        result = graph.replay(updated)
        expected = dense_reference(q, k, v, changed_rows, changed_lengths)
        torch.testing.assert_close(result.cpu().float(), expected, atol=2e-3, rtol=2e-2)
    incompatible = build_fork_plan(rows, lengths, 128, prefix_splits=3, min_shared_tokens=128)
    with pytest.raises(ValueError, match="different graph"):
        graph.replay(incompatible)
    # A rejected update does not poison the existing graph or metadata.
    torch.testing.assert_close(graph.replay().cpu().float(), expected, atol=2e-3, rtol=2e-2)


def test_current_device_is_checked_before_kernel_launch_or_graph_update():
    q, k, v, rows, lengths = make_case(2, 64, 128, torch.bfloat16)
    plan = build_fork_plan(rows, lengths, 128, min_shared_tokens=128)
    qn, kn, vn = (t.npu() for t in (q, k, v))
    op = ForkAttention(qn, kn, vn, plan)
    graph = ForkAttentionGraph(op, qn)
    with patch.object(torch.npu, "current_device", return_value=qn.device.index + 1):
        for call in (
            lambda: ForkAttention(qn, kn, vn, plan),
            lambda: op(qn),
            lambda: ForkAttentionGraph(op, qn),
            lambda: graph.replay(plan),
        ):
            with pytest.raises(ValueError, match="current NPU device"):
                call()
    with pytest.raises(ValueError, match="ForkAttentionPlan"):
        graph.replay(object())
    torch.testing.assert_close(
        graph.replay().cpu().float(), dense_reference(q, k, v, rows, lengths), atol=2e-3, rtol=2e-2
    )
