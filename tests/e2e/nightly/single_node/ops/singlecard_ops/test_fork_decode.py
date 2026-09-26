# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.attention.fork_batch import ForkAttentionConfig, ForkBatchPlanner
from vllm_ascend.ops.fork_decode import ForkCapturedAttention, ForkDecodeBuffers


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
    reference, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
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
    planner = ForkBatchPlanner(ForkAttentionConfig(enabled=True))
    plan = planner.build(rows, lengths)
    assert plan is not None
    buffers = ForkDecodeBuffers(q.device, q.dtype, heads, kv_heads, dim, prefix + 256)
    output = torch.empty_like(q)
    buffers.run(q, k, v, output, SimpleNamespace(fork_plan=plan), dim**-0.5)
    torch.testing.assert_close(output.cpu().float(), reference.cpu().float(), atol=2e-3, rtol=2e-2)
