# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental NPU ForkAttention for an already validated decode cohort.

Uses one paged FIA invocation for shared-prefix splits and private tails. Each
shared split sees all cohort queries as query rows of one sequence. No KV data
is gathered or copied. Serving uses the multi-cohort buffers in fork_decode.py.
Use ForkAttentionGraph for standalone length/page updates between replays;
a plain NPUGraph capture keeps its original plan.
"""

import torch
import torch_npu
from vllm.triton_utils import triton

from vllm_ascend.attention.fork_plan import ForkAttentionPlan
from vllm_ascend.ops.triton.fork_attention import merge_fork_partials, pack_fork_queries

QUERY_COPY_TILE = 1024
SUPPORTED_HEAD_SIZES = (64, 128, 256)
KERNEL_BLOCK_SIZE = 128
MAX_COHORT_QUERIES = 8


class ForkAttention:
    """Preallocated execution workspace for a fixed, nonempty cohort plan.

    Prepare outside graph capture. A plain graph capture fixes pages and lengths.
    ForkAttentionGraph explicitly updates that metadata; cohort and buffer
    capacities still stay fixed. Instances are not reentrant.
    """

    def __init__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: ForkAttentionPlan,
        *,
        max_pages_per_segment: int | None = None,
    ):
        self._validate(query, key, value, plan)
        self.plan = plan
        self.query_shape = query.shape
        self.dtype, self.device = query.dtype, query.device
        batch, heads, dim = query.shape
        self.output = torch.empty_like(query)
        self.packed_query = torch.empty((plan.num_parts * batch, heads, dim), dtype=query.dtype, device=query.device)
        self.partial_output = torch.empty_like(self.packed_query)
        self.partial_lse = torch.empty((plan.num_parts * batch, heads, 1), dtype=torch.float32, device=query.device)
        width = max(map(len, plan.block_rows))
        if max_pages_per_segment is not None:
            if max_pages_per_segment < width:
                raise ValueError("segment capacity is smaller than the initial plan")
            width = max_pages_per_segment
        rows = [list(row) + [0] * (width - len(row)) for row in plan.block_rows]
        self.block_table = torch.tensor(rows, dtype=torch.int32, device=query.device)
        self.key, self.value = key.flatten(2), value.flatten(2)
        self.cache_pages = key.shape[0]
        self.fia_kwargs = dict(
            block_table=self.block_table,
            actual_seq_lengths=list(plan.query_ends),
            actual_seq_lengths_kv=list(plan.kv_lengths),
            num_heads=heads,
            num_key_value_heads=key.shape[2],
            input_layout="TND",
            block_size=plan.block_size,
            scale=dim**-0.5,
            sparse_mode=0,
            softmax_lse_flag=True,
        )
        self.workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            self.packed_query, self.key, self.value, **self.fia_kwargs
        )

    @staticmethod
    def _validate(query, key, value, plan):
        if not isinstance(plan, ForkAttentionPlan):
            raise ValueError("a validated, eligible ForkAttentionPlan is required")
        if query.ndim != 3 or key.ndim != 4 or key.shape != value.shape:
            raise ValueError("expected Q [batch, heads, dim] and matching K/V [pages, block, heads, dim]")
        if query.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("ForkAttention requires BF16 or FP16")
        if query.device.type != "npu" or any(t.device != query.device or t.dtype != query.dtype for t in (key, value)):
            raise ValueError("Q/K/V must have the same NPU device and dtype")
        if torch.npu.current_device() != query.device.index:
            raise ValueError("the current NPU device must match the query device")
        if not all(t.is_contiguous() for t in (query, key, value)):
            raise ValueError("this prototype requires contiguous Q/K/V")
        if query.shape[0] != plan.batch_size or key.shape[1] != plan.block_size:
            raise ValueError("query batch or cache block size does not match plan")
        if plan.block_size != KERNEL_BLOCK_SIZE:
            raise ValueError("use 128-token kernel pages after splitting hybrid manager blocks")
        if not 2 <= plan.batch_size <= MAX_COHORT_QUERIES:
            raise ValueError(f"this prototype requires 2 to {MAX_COHORT_QUERIES} queries")
        if query.shape[-1] not in SUPPORTED_HEAD_SIZES or query.shape[-1] != key.shape[-1]:
            raise ValueError("unsupported or inconsistent head dimension")
        if key.shape[2] <= 0 or query.shape[1] <= 0 or query.shape[1] % key.shape[2]:
            raise ValueError("query heads must be a multiple of KV heads")
        if max(max(row) for row in plan.block_rows) >= key.shape[0]:
            raise ValueError("physical page ID exceeds cache capacity")

    def __call__(self, query: torch.Tensor) -> torch.Tensor:
        if query.shape != self.query_shape or query.dtype != self.dtype or query.device != self.device:
            raise ValueError("query shape, dtype and device must match the prepared workspace")
        if not query.is_contiguous():
            raise ValueError("query must be contiguous")
        if torch.npu.current_device() != self.device.index:
            raise ValueError("the current NPU device must match the prepared workspace")
        self._pack(query)
        self._attention()
        self._merge()
        return self.output

    def _pack(self, query):
        batch, heads, dim = query.shape
        pack_fork_queries[(self.packed_query.shape[0], triton.cdiv(heads * dim, QUERY_COPY_TILE))](
            query, self.packed_query, batch, heads * dim, QUERY_COPY_TILE
        )

    def _attention(self):
        torch_npu.npu_fused_infer_attention_score.out(
            self.packed_query,
            self.key,
            self.value,
            **self.fia_kwargs,
            workspace=self.workspace,
            out=[self.partial_output, self.partial_lse],
        )

    def _merge(self):
        batch, heads, dim = self.query_shape
        merge_fork_partials[(batch * heads,)](
            self.partial_output,
            self.partial_lse,
            self.output,
            batch * heads,
            dim,
            self.plan.num_parts,
            triton.next_power_of_2(self.plan.num_parts),
        )


class ForkAttentionGraph:
    """Replay one cohort capacity with updated FIA tasks and fixed buffers.

    This uses the same graph-task update mechanism as AscendAttention. It does
    not implement serving admission or switch between Fork/native model graphs.
    The caller owns query/cache writes and supplies kernel-page CPU metadata.
    Calls must use the stream that created this object; concurrent calls and
    concurrent mutation of its query/cache buffers are not supported.
    """

    def __init__(self, operator: ForkAttention, query: torch.Tensor):
        self.operator = operator
        self.query = query
        for _ in range(3):
            operator(query)
        torch.npu.synchronize()
        self.stream = torch.npu.current_stream()
        self.update_stream = torch.npu.Stream(device=query.device)
        self.event = torch.npu.ExternalEvent()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            operator._pack(query)
            stream = torch.npu.current_stream()
            self.event.wait(stream)
            self.event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            operator._attention()
            self.handle = torch.npu.graph_task_group_end(stream)
            operator._merge()

    def _prepare_update(self, plan):
        op = self.operator
        original = op.plan
        if (plan.batch_size, plan.block_size, plan.prefix_splits, plan.query_ends) != (
            original.batch_size,
            original.block_size,
            original.prefix_splits,
            original.query_ends,
        ):
            raise ValueError("plan requires a different graph cohort or split capacity")
        if len(plan.block_rows) != op.block_table.shape[0] or max(map(len, plan.block_rows)) > op.block_table.shape[1]:
            raise ValueError("plan exceeds captured block-table capacity")
        if max(max(row) for row in plan.block_rows) >= op.cache_pages:
            raise ValueError("physical page ID exceeds cache capacity")
        width = op.block_table.shape[1]
        rows = [list(row) + [0] * (width - len(row)) for row in plan.block_rows]
        return torch.tensor(rows, dtype=torch.int32, device="cpu", pin_memory=True)

    def replay(self, plan: ForkAttentionPlan | None = None) -> torch.Tensor:
        if plan is not None and not isinstance(plan, ForkAttentionPlan):
            raise ValueError("a validated, eligible ForkAttentionPlan is required")
        if torch.npu.current_device() != self.operator.device.index:
            raise ValueError("the current NPU device must match the prepared workspace")
        if torch.npu.current_stream() != self.stream:
            raise ValueError("graph replay must use its owning stream")
        op = self.operator
        changed = plan is not None and plan != op.plan
        host_table = self._prepare_update(plan) if changed else None
        # Complete the preceding replay before mutating its table or task.
        self.update_stream.wait_stream(self.stream)
        with torch.npu.stream(self.update_stream):
            if changed:
                op.block_table.copy_(host_table, non_blocking=True)
                op.fia_kwargs["actual_seq_lengths_kv"] = list(plan.kv_lengths)
                torch.npu.graph_task_update_begin(self.update_stream, self.handle)
                op._attention()
                torch.npu.graph_task_update_end(self.update_stream)
                op.plan = plan
            self.event.record(self.update_stream)
        self.graph.replay()
        return op.output
