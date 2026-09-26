# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ForkAttention buffers shared by the layers and graphs of one KV group."""

import torch
import torch_npu
from vllm.logger import logger
from vllm.triton_utils import triton

from vllm_ascend.attention.fork_batch import KERNEL_PAGE_SIZE, MAX_BATCH, MAX_SPLITS
from vllm_ascend.ops.triton.fork_attention import merge_fork_batch, pack_fork_batch
from vllm_ascend.utils import weak_ref_tensors

QUERY_COPY_TILE = 1024


class ForkDecodeBuffers:
    """Allocate before capture; engine execution serializes use of this pool.

    Only small descriptors cross H2D when topology changes. Ordinary token
    growth changes FIA's CPU length arguments, reusing all descriptor buffers.
    One instance belongs to a metadata builder, shared across its layers and
    all graph batch sizes. It must not be used by concurrent model executions.
    """

    def __init__(self, device, dtype, heads, kv_heads, dim, max_model_len, diagnostics=False):
        self.heads, self.kv_heads, self.dim = heads, kv_heads, dim
        self.parts = MAX_SPLITS + 1
        capacity = MAX_BATCH * self.parts
        pages = (max_model_len + KERNEL_PAGE_SIZE - 1) // KERNEL_PAGE_SIZE
        self.packed = torch.empty((capacity, heads, dim), dtype=dtype, device=device)
        self.partial = torch.empty_like(self.packed)
        self.lse = torch.empty((capacity, heads, 1), dtype=torch.float32, device=device)
        self.table = torch.zeros((capacity, pages), dtype=torch.int32, device=device)
        self.query_map = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.merge_map = torch.full((MAX_BATCH, self.parts), -1, dtype=torch.int32, device=device)
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        # The workspace API returns a conservative maximum. Reserve the full
        # supported descriptor capacity once, outside the capture memory pool.
        dummy_kv = torch.empty((1, KERNEL_PAGE_SIZE, kv_heads * dim), dtype=dtype, device=device)
        self.workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            self.packed,
            dummy_kv,
            dummy_kv,
            block_table=self.table,
            block_size=KERNEL_PAGE_SIZE,
            actual_seq_lengths=list(range(1, capacity + 1)),
            actual_seq_lengths_kv=[max_model_len] * capacity,
            num_heads=heads,
            num_key_value_heads=kv_heads,
            input_layout="TND",
            scale=dim**-0.5,
            sparse_mode=0,
            softmax_lse_flag=True,
        )
        self._metadata = None
        self._rows = None
        self._mapping = None
        self._count = 0
        self.diagnostics = diagnostics
        self._executed_metadata = None
        self.batches = self.selected = self.graph_batches = 0
        # Compile the two no-op variants before model graph capture begins.
        self.pack(self.packed[:MAX_BATCH])
        self.merge(self.partial[:MAX_BATCH])

    @staticmethod
    def _copy(destination, values):
        host = torch.tensor(values, dtype=torch.int32, device="cpu", pin_memory=True)
        destination.copy_(host, non_blocking=True)

    def prepare(self, metadata):
        if metadata is self._metadata:
            return
        plan = metadata.fork_plan
        count = len(plan.query_indices) if plan else 0
        if plan is not None:
            if plan.block_rows is not self._rows:
                rows = [list(row) + [0] * (self.table.shape[1] - len(row)) for row in plan.block_rows]
                self._copy(self.table[: len(rows)], rows)
                self._rows = plan.block_rows
            mapping = (plan.query_indices, plan.merge_indices)
            if mapping != self._mapping:
                self._copy(self.query_map[:count], plan.query_indices)
                merge = [list(row) + [-1] * (self.parts - len(row)) for row in plan.merge_indices]
                merge += [[-1] * self.parts for _ in range(MAX_BATCH - plan.batch_size)]
                self._copy(self.merge_map, merge)
                self._mapping = mapping
        if count != self._count:
            self._copy(self.count, [count])
            self._count = count
        self._metadata = metadata

    def pack(self, query):
        pack_fork_batch[(self.packed.shape[0], triton.cdiv(self.heads * self.dim, QUERY_COPY_TILE))](
            query,
            self.packed,
            self.query_map,
            self.count,
            self.heads * self.dim,
            QUERY_COPY_TILE,
        )

    def attention(self, key, value, plan, scale):
        count = len(plan.query_indices)
        torch_npu.npu_fused_infer_attention_score.out(
            self.packed[:count],
            key,
            value,
            block_table=self.table[: len(plan.block_rows)],
            block_size=KERNEL_PAGE_SIZE,
            actual_seq_lengths=list(plan.query_ends),
            actual_seq_lengths_kv=list(plan.kv_lengths),
            num_heads=self.heads,
            num_key_value_heads=self.kv_heads,
            input_layout="TND",
            scale=scale,
            sparse_mode=0,
            softmax_lse_flag=True,
            workspace=self.workspace,
            out=[self.partial[:count], self.lse[:count]],
        )

    def merge(self, output):
        merge_fork_batch[(output.shape[0], self.heads)](
            self.partial,
            self.lse,
            output,
            self.merge_map,
            self.count,
            self.heads,
            self.dim,
            self.parts,
            triton.next_power_of_2(self.parts),
        )

    def run(self, query, key, value, output, metadata, scale):
        self.record_execution(metadata, graph=False)
        self.prepare(metadata)
        self.pack(query)
        self.attention(key, value, metadata.fork_plan, scale)
        self.merge(output)
        return output

    def record_execution(self, metadata, *, graph):
        if not self.diagnostics or metadata is self._executed_metadata:
            return
        self._executed_metadata = metadata
        self.batches += 1
        selected = metadata.fork_plan is not None
        self.selected += int(selected)
        self.graph_batches += int(graph)
        if self.batches % 256 == 0 or selected and self.selected == 1:
            logger.info(
                "ForkAttention execute batches=%d selected=%d graph_batches=%d",
                self.batches,
                self.selected,
                self.graph_batches,
            )


class ForkCapturedAttention:
    """One layer's FIA task, surrounded by fixed optional pack/merge kernels.

    The external event must precede pack as well as FIA: graph replay is
    launched before the host updates its metadata on vLLM's update stream.
    """

    def __init__(self, buffers, query, key, value, output, native_kwargs, native_lse, workspace):
        self.buffers = buffers
        self.query, self.key, self.value, self.output = weak_ref_tensors((query, key, value, output))
        self.native_lse = weak_ref_tensors(native_lse)
        self.workspace = weak_ref_tensors(workspace)
        self.native_kwargs = {
            name: weak_ref_tensors(value) if isinstance(value, torch.Tensor) else value
            for name, value in native_kwargs.items()
        }

    def attention(self, metadata=None):
        if metadata is not None and metadata.fork_plan is not None:
            self.buffers.attention(self.key, self.value, metadata.fork_plan, self.native_kwargs["scale"])
        else:
            kwargs = self.native_kwargs.copy()
            if metadata is not None:
                kwargs.update(
                    block_table=metadata.block_tables,
                    actual_seq_lengths=metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=metadata.seq_lens_list,
                )
            torch_npu.npu_fused_infer_attention_score.out(
                self.query,
                self.key,
                self.value,
                **kwargs,
                workspace=self.workspace,
                out=[self.output, self.native_lse],
            )

    def capture(self, stream):
        self.buffers.pack(self.query)
        torch.npu.graph_task_group_begin(stream)
        self.attention()
        handle = torch.npu.graph_task_group_end(stream)
        self.buffers.merge(self.output)
        return handle

    def update(self, metadata, stream, handle, event):
        self.buffers.record_execution(metadata, graph=True)
        self.buffers.prepare(metadata)
        torch.npu.graph_task_update_begin(stream, handle)
        self.attention(metadata)
        torch.npu.graph_task_update_end(stream)
        event.record(stream)
