# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vector kernels around the NPU's paged, grouped-query attention kernel."""

from vllm.triton_utils import tl, triton


@triton.jit
def pack_fork_queries(Q, PackedQ, BATCH: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q + (row % BATCH) * WIDTH + column, mask=column < WIDTH, other=0)
    tl.store(PackedQ + row * WIDTH + column, q, mask=column < WIDTH)


@triton.jit
def merge_fork_partials(
    PartialOut,
    PartialLSE,
    Out,
    ROWS: tl.constexpr,
    DIM: tl.constexpr,
    PARTS: tl.constexpr,
    PAD_PARTS: tl.constexpr,
):
    row = tl.program_id(0)
    parts = tl.arange(0, PAD_PARTS)
    dims = tl.arange(0, DIM)
    # FIA returns natural-log LSE. Accumulate in FP32 and cast only at store.
    lse = tl.load(PartialLSE + parts * ROWS + row, mask=parts < PARTS, other=-float("inf"))
    maximum = tl.max(lse, 0)
    weights = tl.exp(lse - maximum)
    denominator = tl.sum(weights, 0)
    values = tl.load(
        PartialOut + (parts[:, None] * ROWS + row) * DIM + dims[None, :],
        mask=(parts < PARTS)[:, None],
        other=0,
    ).to(tl.float32)
    result = tl.sum(values * weights[:, None], 0) / denominator
    tl.store(Out + row * DIM + dims, result)


@triton.jit
def pack_fork_batch(Q, PackedQ, QueryMap, Count, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    if row < tl.load(Count):
        source = tl.load(QueryMap + row)
        column = tile * BLOCK + tl.arange(0, BLOCK)
        q = tl.load(Q + source * WIDTH + column, mask=column < WIDTH, other=0)
        tl.store(PackedQ + row * WIDTH + column, q, mask=column < WIDTH)


@triton.jit
def merge_fork_batch(
    PartialOut,
    PartialLSE,
    Out,
    MergeMap,
    Count,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PARTS: tl.constexpr,
    PAD_PARTS: tl.constexpr,
):
    # Count == 0 is the native FIA path; native output must remain untouched.
    if tl.load(Count) > 0:
        query, head = tl.program_id(0), tl.program_id(1)
        parts = tl.arange(0, PAD_PARTS)
        indices = tl.load(MergeMap + query * PARTS + parts, mask=parts < PARTS, other=-1)
        valid = indices >= 0
        lse = tl.load(PartialLSE + indices * HEADS + head, mask=valid, other=-float("inf"))
        # Padded model graph queries have no partials. Write zero, not NaN.
        maximum = tl.maximum(tl.max(lse, 0), -1.0e30)
        weights = tl.exp(lse - maximum)
        denominator = tl.maximum(tl.sum(weights, 0), 1.0e-30)
        dims = tl.arange(0, DIM)
        values = tl.load(
            PartialOut + (indices[:, None] * HEADS + head) * DIM + dims[None, :],
            mask=valid[:, None],
            other=0,
        ).to(tl.float32)
        result = tl.sum(values * weights[:, None], 0) / denominator
        tl.store(Out + (query * HEADS + head) * DIM + dims, result)
