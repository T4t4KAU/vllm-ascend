# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu
from vllm.triton_utils import triton

from vllm_ascend.attention.fork_batch import ForkAttentionConfig, ForkBatchPlanner
from vllm_ascend.ops.fork_decode import MERGE_HEAD_TILE, ForkCapturedAttention, ForkDecodeBuffers
from vllm_ascend.ops.triton.fork_attention import merge_fork_batch


def dense_reference(query, key, value, rows, lengths):
    outputs = []
    ratio = query.shape[1] // key.shape[2]
    for q, row, length in zip(query, rows, lengths):
        k = key[row].flatten(0, 1)[:length].float().repeat_interleave(ratio, dim=1).transpose(0, 1)
        v = value[row].flatten(0, 1)[:length].float().repeat_interleave(ratio, dim=1).transpose(0, 1)
        logits = torch.einsum("hd,hsd->hs", q.float(), k) * query.shape[-1] ** -0.5
        outputs.append(torch.einsum("hs,hsd->hd", logits.softmax(-1), v))
    return torch.stack(outputs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [7, 16])
def test_grouped_merge_replay_with_head_masks_and_sparse_partials(dtype, heads):
    """Merge real partials; padded slots and native fallback must not leak NaNs."""
    generator = torch.Generator().manual_seed(223)
    batch, capacity, parts, dim = 8, 136, 17, 256
    values = torch.randn(capacity, heads, dim, generator=generator).to(dtype)
    lse = torch.randn(capacity, heads, generator=generator) * 10 + 1000
    values[-1] = float("nan")
    lse[-1] = float("nan")
    mapping = torch.full((batch, parts), -1, dtype=torch.int32)
    for query, size in enumerate((1, 5, 17, 3)):
        slots = torch.randperm(parts, generator=generator)[:size]
        mapping[query, slots] = torch.randperm(capacity - 1, generator=generator)[:size].int()
    vn, ln, mn = values.npu(), lse.npu(), mapping.npu()
    count = torch.tensor([int((mapping >= 0).sum())], dtype=torch.int32, device="npu")
    output = torch.empty(batch, heads, dim, dtype=dtype, device="npu")

    def merge():
        merge_fork_batch[(batch, triton.cdiv(heads, MERGE_HEAD_TILE))](
            vn, ln, output, mn, count, heads, dim, parts, triton.next_power_of_2(parts), MERGE_HEAD_TILE
        )

    for _ in range(3):
        merge()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        merge()

    for step in range(2):
        if step:
            values[:-1].neg_()
            lse[:-1].sub_(2000)
            mapping = mapping.flip(0).contiguous()
            vn.copy_(values)
            ln.copy_(lse)
            mn.copy_(mapping)
        expected = torch.zeros(batch, heads, dim, dtype=torch.float64)
        for query in range(batch):
            indices = mapping[query][mapping[query] >= 0].long()
            if indices.numel():
                weights = lse[indices].double().softmax(0)
                expected[query] = (values[indices].double() * weights[:, :, None]).sum(0)
        count.fill_(int((mapping >= 0).sum()))
        graph.replay()
        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(output.cpu(), expected.to(dtype), atol=tolerance, rtol=tolerance)

        # Count changes on the same graph: skip stores, then resume next step.
        sentinel = torch.randn(batch, heads, dim, generator=generator).to(dtype)
        sentinel[0, 0, 0] = float("nan")
        output.copy_(sentinel)
        count.zero_()
        graph.replay()
        torch.testing.assert_close(output.cpu(), sentinel, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_captured_native_fork_switch_with_shared_buffers_and_padded_queries(dtype):
    torch.manual_seed(201)
    device, batch, heads, kv_heads, dim = "npu", 4, 16, 4, 256
    buffers = ForkDecodeBuffers(device, dtype, heads, kv_heads, dim, 1024)
    planner = ForkBatchPlanner(ForkAttentionConfig(enabled=True, min_shared_tokens=128))
    q = torch.randn(batch, heads, dim, dtype=dtype)
    qn = q.npu()
    caches = [torch.randn(2, 32, 128, kv_heads, dim, dtype=dtype) for _ in range(2)]
    caches_npu = [cache.npu() for cache in caches]
    outputs = [torch.empty_like(qn) for _ in caches]
    lses = [torch.empty(1, dtype=dtype, device=device) for _ in caches]
    rows = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.int32)
    table = torch.tensor(rows, device=device)
    lengths = [300] * batch
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device=device), 1)
    kwargs = dict(
        block_table=table,
        block_size=128,
        actual_seq_lengths=list(range(1, batch + 1)),
        actual_seq_lengths_kv=lengths,
        atten_mask=mask,
        input_layout="TND",
        num_heads=heads,
        num_key_value_heads=kv_heads,
        scale=dim**-0.5,
        sparse_mode=3,
    )
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        qn,
        caches_npu[0][0].flatten(2),
        caches_npu[0][1].flatten(2),
        **kwargs,
    )
    records = [
        ForkCapturedAttention(buffers, qn, kv[0].flatten(2), kv[1].flatten(2), out, kwargs, lse, workspace)
        for kv, out, lse in zip(caches_npu, outputs, lses)
    ]
    for record in records:
        record.attention()
    torch.npu.synchronize()
    graph, stream = torch.npu.NPUGraph(), torch.npu.current_stream()
    events, handles = [], []
    with torch.npu.graph(graph):
        capture_stream = torch.npu.current_stream()
        for record in records:
            event = torch.npu.ExternalEvent()
            event.wait(capture_stream)
            event.reset(capture_stream)
            events.append(event)
            handles.append(record.capture(capture_stream))
    update_stream = torch.npu.Stream()
    # Two disjoint groups, one group plus an unrelated query, native fallback,
    # full sharing, rollback and a changed page mapping on the same graph.
    cases = [
        ([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 9, 10], [5, 6, 11, 12]], [300, 301, 302, 303]),
        ([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 9, 10]], [310, 311, 312]),
        (rows.tolist(), [300] * 4),
        ([[1, 2, 3, i] for i in (4, 8, 12, 16)], [400, 401, 402, 403]),
        ([[1, 2, 3, i] for i in (4, 8, 12, 16)], [401, 402, 403, 404]),
        ([[1, 2, 3, i] for i in (16, 8, 12, 4)], [256, 257, 258, 259]),
        (rows[:1].tolist(), [200]),
    ]
    for case_rows, case_lengths in cases:
        live = len(case_lengths)
        q.normal_()
        qn.copy_(q)
        current = np.asarray(case_rows, dtype=np.int32)
        host_table = np.zeros_like(rows)
        host_table[:live] = current
        table.copy_(torch.from_numpy(host_table))
        metadata = SimpleNamespace(
            fork_plan=planner.build(current, case_lengths),
            block_tables=table,
            actual_seq_lengths_q=list(range(1, batch + 1)),
            seq_lens_list=case_lengths + [0] * (batch - live),
        )
        # Match vLLM: previous replay complete, then replay is enqueued BEFORE
        # graph-task updates, and external events release each attention layer.
        stream.synchronize()
        graph.replay()
        with torch.npu.stream(update_stream):
            for record, event, handle in zip(records, events, handles):
                record.update(metadata, update_stream, handle, event)
        for cache, output in zip(caches, outputs):
            expected = dense_reference(q[:live], cache[0], cache[1], current.tolist(), case_lengths)
            torch.testing.assert_close(
                output[:live].cpu().float(), expected, atol=2e-3 if dtype == torch.bfloat16 else 3e-4, rtol=2e-2
            )
        assert records[0].buffers is records[1].buffers


@pytest.mark.parametrize("batch,prefix", [(3, 32768), (8, 65536)])
def test_serving_pool_long_prefix_matches_native_fia(batch, prefix):
    torch.manual_seed(211)
    heads, kv_heads, dim, block = 16, 4, 256, 128
    shared = prefix // block
    pages = shared + batch * 2
    permutation = torch.randperm(pages).tolist()
    rows = np.array(
        [permutation[:shared] + permutation[shared + 2 * q : shared + 2 * q + 2] for q in range(batch)], dtype=np.int32
    )
    lengths = [prefix + 129 + q for q in range(batch)]
    q = torch.randn(batch, heads, dim, device="npu", dtype=torch.bfloat16)
    k = torch.randn(pages, block, kv_heads * dim, device="npu", dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.tensor(rows, device="npu")
    mask = torch.triu(torch.ones((2048, 2048), device="npu", dtype=torch.bool), 1)
    kwargs = dict(
        block_table=table,
        block_size=block,
        actual_seq_lengths=list(range(1, batch + 1)),
        actual_seq_lengths_kv=lengths,
        num_heads=heads,
        num_key_value_heads=kv_heads,
        scale=dim**-0.5,
        atten_mask=mask,
        sparse_mode=3,
        input_layout="TND",
    )
    reference, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, **kwargs)
    planner = ForkBatchPlanner(ForkAttentionConfig(enabled=True))
    plan = planner.build(rows, lengths)
    assert plan is not None
    buffers = ForkDecodeBuffers(q.device, q.dtype, heads, kv_heads, dim, prefix + 256)
    output = torch.empty_like(q)
    buffers.run(q, k, v, output, SimpleNamespace(fork_plan=plan), dim**-0.5)
    torch.testing.assert_close(output.cpu().float(), reference.cpu().float(), atol=2e-3, rtol=2e-2)

    # Exercise serving's update-after-replay ordering at the same long prefixes,
    # including a native fallback between two shared executions on one graph.
    buffers.prepare(SimpleNamespace(fork_plan=None))
    lse = torch.empty(1, dtype=q.dtype, device=q.device)
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(q, k, v, **kwargs)
    record = ForkCapturedAttention(buffers, q, k, v, output, kwargs, lse, workspace)
    record.attention()
    torch.npu.synchronize()
    graph, stream = torch.npu.NPUGraph(), torch.npu.current_stream()
    event = torch.npu.ExternalEvent()
    with torch.npu.graph(graph):
        capture_stream = torch.npu.current_stream()
        event.wait(capture_stream)
        event.reset(capture_stream)
        handle = record.capture(capture_stream)
    update_stream = torch.npu.Stream()
    for offset, use_fork in ((0, True), (1, False), (-1, True)):
        q.normal_()
        current_lengths = [length + offset for length in lengths]
        # Mutate a private page and its mapping so stale descriptors/KV are visible.
        rows[0, -2:] = rows[0, -2:][::-1].copy()
        table.copy_(torch.from_numpy(rows))
        v[int(rows[0, -1])].normal_()
        native_kwargs = dict(kwargs, actual_seq_lengths_kv=current_lengths)
        expected, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, **native_kwargs)
        metadata = SimpleNamespace(
            fork_plan=planner.build(rows, current_lengths) if use_fork else None,
            block_tables=table,
            actual_seq_lengths_q=list(range(1, batch + 1)),
            seq_lens_list=current_lengths,
        )
        if use_fork:
            assert metadata.fork_plan is not None
        stream.synchronize()
        graph.replay()
        with torch.npu.stream(update_stream):
            record.update(metadata, update_stream, handle, event)
        torch.testing.assert_close(output.cpu().float(), expected.cpu().float(), atol=2e-3, rtol=2e-2)
