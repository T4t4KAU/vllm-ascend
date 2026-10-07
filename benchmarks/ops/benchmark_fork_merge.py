# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the installed LSE merge with independent layout references.

CPU interpreter checks cover arithmetic and masks only. Performance measurement
requires an NPU and includes both the merge kernel and pack/FIA/merge graph.
The serving path groups two heads; the references retain one-, two-, and
four-head grids and an alternate reduction layout for later comparisons.
"""

import argparse
import hashlib
import itertools
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def grouped_merge(
    PartialOut,
    PartialLSE,
    Out,
    MergeMap,
    Count,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PARTS: tl.constexpr,
    PAD_PARTS: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    DIM_MAJOR: tl.constexpr,
):
    # Keep the native fallback and padded-query behavior of merge_fork_batch.
    if tl.load(Count) > 0:
        query = tl.program_id(0)
        heads = tl.program_id(1) * HEAD_TILE + tl.arange(0, HEAD_TILE)
        parts = tl.arange(0, PAD_PARTS)
        dims = tl.arange(0, DIM)
        indices = tl.load(MergeMap + query * PARTS + parts, mask=parts < PARTS, other=-1)
        if DIM_MAJOR:
            valid = (heads[:, None] < HEADS) & (indices[None, :] >= 0)
            lse = tl.load(PartialLSE + indices[None, :] * HEADS + heads[:, None], valid, -float("inf"))
            maximum = tl.maximum(tl.max(lse, 1), -1.0e30)
            weights = tl.exp(lse - maximum[:, None])
            denominator = tl.maximum(tl.sum(weights, 1), 1.0e-30)
            values = tl.load(
                PartialOut + (indices[None, None, :] * HEADS + heads[:, None, None]) * DIM + dims[None, :, None],
                mask=valid[:, None, :],
                other=0,
            ).to(tl.float32)
            result = tl.sum(values * weights[:, None, :], 2) / denominator[:, None]
        else:
            valid = (indices[:, None] >= 0) & (heads[None, :] < HEADS)
            lse = tl.load(PartialLSE + indices[:, None] * HEADS + heads[None, :], valid, -float("inf"))
            maximum = tl.maximum(tl.max(lse, 0), -1.0e30)
            weights = tl.exp(lse - maximum[None, :])
            denominator = tl.maximum(tl.sum(weights, 0), 1.0e-30)
            values = tl.load(
                PartialOut + (indices[:, None, None] * HEADS + heads[None, :, None]) * DIM + dims[None, None, :],
                mask=valid[:, :, None],
                other=0,
            ).to(tl.float32)
            result = tl.sum(values * weights[:, :, None], 0) / denominator[:, None]
        tl.store(
            Out + (query * HEADS + heads[:, None]) * DIM + dims[None, :],
            result,
            mask=heads[:, None] < HEADS,
        )


def variants():
    return [(heads, layout) for heads, layout in itertools.product((1, 2, 4), (False, True))]


def launch(values, lse, out, mapping, count, head_tile, dim_major):
    _, heads, dim = out.shape
    grouped_merge[(out.shape[0], triton.cdiv(heads, head_tile))](
        values,
        lse,
        out,
        mapping,
        count,
        heads,
        dim,
        mapping.shape[1],
        triton.next_power_of_2(mapping.shape[1]),
        head_tile,
        dim_major,
    )


def check_numerics(device):
    """Test the kernel itself against FP64 softmax, including metadata changes."""
    if device == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU checks require TRITON_INTERPRET=1 before Python starts")
    dtypes = (torch.float32, torch.float16) if device == "cpu" else (torch.bfloat16, torch.float16)
    checks = 0
    for dtype, heads, seed in itertools.product(dtypes, (7, 16), (71, 79)):
        generator = torch.Generator().manual_seed(seed)
        rows, capacity, parts, dim = 8, 136, 17, 256
        values = torch.randn(capacity, heads, dim, generator=generator).to(dtype)
        lse = torch.randn(capacity, heads, generator=generator) * 10 + 1000
        # Unreferenced NaNs must never leak through masked descriptors.
        values[-1] = float("nan")
        lse[-1] = float("nan")
        value_device, lse_device = values.to(device), lse.to(device)
        mapping = torch.full((rows, parts), -1, dtype=torch.int32)
        for q, n in enumerate((1, 5, 17, 3, 0, 0, 0, 0)):
            slots = torch.randperm(parts, generator=generator)[:n]
            mapping[q, slots] = torch.randperm(capacity - 1, generator=generator)[:n].int()
        map_device = mapping.to(device)
        count = torch.tensor([int((mapping >= 0).sum())], dtype=torch.int32, device=device)
        # Reuse the same addresses with changed values and descriptors.
        for step in range(2):
            if step:
                values[:-1].neg_()
                lse[:-1].sub_(2000)
                mapping = mapping.flip(0).contiguous()
                value_device.copy_(values)
                lse_device.copy_(lse)
                map_device.copy_(mapping)
            expected = torch.zeros(rows, heads, dim, dtype=torch.float64)
            for q in range(rows):
                indices = mapping[q][mapping[q] >= 0].long()
                if indices.numel():
                    weights = lse[indices].double().softmax(0)
                    expected[q] = (values[indices].double() * weights[:, :, None]).sum(0)
            expected = expected.to(dtype)
            for head_tile, dim_major in variants():
                output = torch.full((rows, heads, dim), float("nan"), dtype=dtype, device=device)
                launch(value_device, lse_device, output, map_device, count, head_tile, dim_major)
                tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 2e-5
                torch.testing.assert_close(output.cpu(), expected, atol=tolerance, rtol=tolerance)
                # Count=0 must preserve native output exactly, including NaNs.
                sentinel = torch.randn(rows, heads, dim, generator=generator).to(dtype)
                sentinel[0, 0, 0] = float("nan")
                output.copy_(sentinel)
                count.zero_()
                launch(value_device, lse_device, output, map_device, count, head_tile, dim_major)
                torch.testing.assert_close(output.cpu(), sentinel, atol=0, rtol=0, equal_nan=True)
                count.fill_(int((mapping >= 0).sum()))
                checks += 1
    return {"checks": checks, "device": device, "dtypes": [str(dtype) for dtype in dtypes], "passed": True}


def capture(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fn()
    return graph


def timed_graphs(graphs, iterations):
    # Every variant occupies every order position, in both directions.
    names = list(graphs)
    samples = {name: [] for name in names}
    for reverse, offset in itertools.product((False, True), range(len(names))):
        order = names[offset:] + names[:offset]
        if reverse:
            order = order[::-1]
        for name in order:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / iterations)
    return samples


def benchmark_case(branches, prefix, tail, seed, dtype, iterations):
    # Import the installed serving implementation only on the NPU path.
    import torch_npu

    from vllm_ascend.attention.fork_batch import ForkAttentionConfig, ForkBatchPlanner
    from vllm_ascend.ops.fork_decode import ForkDecodeBuffers

    torch.manual_seed(seed)
    heads, kv_heads, dim, block = 16, 4, 256, 128
    shared, private = prefix // block, triton.cdiv(tail + branches, block)
    pages = shared + branches * private
    permutation = torch.randperm(pages).tolist()
    rows = np.asarray(
        [
            permutation[:shared] + permutation[shared + q * private : shared + (q + 1) * private]
            for q in range(branches)
        ],
        dtype=np.int32,
    )
    lengths = [prefix + tail + q for q in range(branches)]
    query = torch.randn(branches, heads, dim, dtype=dtype, device="npu")
    key = torch.randn(pages, block, kv_heads * dim, dtype=dtype, device="npu")
    value = torch.randn_like(key)
    table = torch.tensor(rows, device="npu")
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device="npu"), 1)
    native_kwargs = dict(
        block_table=table,
        block_size=block,
        actual_seq_lengths=list(range(1, branches + 1)),
        actual_seq_lengths_kv=lengths,
        num_heads=heads,
        num_key_value_heads=kv_heads,
        scale=dim**-0.5,
        atten_mask=mask,
        sparse_mode=3,
        input_layout="TND",
    )
    native, _ = torch_npu.npu_fused_infer_attention_score(query, key, value, **native_kwargs)
    plan = ForkBatchPlanner(ForkAttentionConfig(enabled=True)).build(rows, lengths)
    assert plan is not None
    buffers = ForkDecodeBuffers(query.device, dtype, heads, kv_heads, dim, max(lengths))
    buffers.prepare(SimpleNamespace(fork_plan=plan))
    outputs = {}
    full_graphs, merge_graphs = {}, {}
    methods = [("serving", None)] + [
        (f"{'dim' if layout else 'part'}_heads{h}", (h, layout)) for h, layout in variants()
    ]
    for name, variant in methods:
        output = outputs[name] = torch.empty((8, heads, dim), dtype=dtype, device="npu")

        def merge(output=output, variant=variant):
            if variant is None:
                buffers.merge(output)
            else:
                launch(buffers.partial, buffers.lse, output, buffers.merge_map, buffers.count, *variant)

        def full(merge=merge):
            buffers.pack(query)
            buffers.attention(key, value, plan, dim**-0.5)
            merge()

        full()
        torch.testing.assert_close(output[:branches], native, atol=2e-3 if dtype == torch.bfloat16 else 3e-4, rtol=2e-2)
        assert torch.count_nonzero(output[branches:]).item() == 0
        full_graphs[name], merge_graphs[name] = capture(full), capture(merge)
        full_graphs[name].replay()
        torch.testing.assert_close(output[:branches], native, atol=2e-3 if dtype == torch.bfloat16 else 3e-4, rtol=2e-2)
    full_samples = timed_graphs(full_graphs, iterations)
    merge_samples = timed_graphs(merge_graphs, iterations)
    # Replay must read fresh tensors from the original capture addresses.
    query.normal_()
    value.normal_()
    updated, _ = torch_npu.npu_fused_infer_attention_score(query, key, value, **native_kwargs)
    for name, graph in full_graphs.items():
        graph.replay()
        torch.testing.assert_close(
            outputs[name][:branches],
            updated,
            atol=2e-3 if dtype == torch.bfloat16 else 3e-4,
            rtol=2e-2,
        )
        assert torch.count_nonzero(outputs[name][branches:]).item() == 0
    # Also check dynamic Count on the captured candidate kernels.
    buffers.count.zero_()
    for name, graph in merge_graphs.items():
        outputs[name].fill_(7)
        graph.replay()
        torch.testing.assert_close(outputs[name], torch.full_like(outputs[name], 7), atol=0, rtol=0)
    return {
        "branches": branches,
        "prefix": prefix,
        "tail": tail,
        "seed": seed,
        "dtype": str(dtype),
        "padded_rows": 8,
        "packed_capacity": buffers.packed.shape[0],
        "workspace_bytes": buffers.workspace.numel() * buffers.workspace.element_size(),
        "full_us": {name: statistics.median(times) for name, times in full_samples.items()},
        "merge_us": {name: statistics.median(times) for name, times in merge_samples.items()},
        "full_samples_us": full_samples,
        "merge_samples_us": merge_samples,
        "correctness_passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-device", choices=("cpu", "cuda", "npu"), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--branches", type=int, nargs="+", choices=(4, 8), default=[4, 8])
    parser.add_argument("--prefix", type=int, nargs="+", choices=(32768, 65536), default=[32768, 65536])
    parser.add_argument("--tail", type=int, nargs="+", choices=(128, 1024), default=[128, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[71, 79])
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    if args.iterations < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("iterations must be positive and seeds nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if not args.check_only and args.output is None:
        parser.error("NPU measurements require --output in the server experiment directory")
    if not args.check_only and os.environ.get("TRITON_INTERPRET") == "1":
        parser.error("Interpreter mode cannot measure NPU performance")
    if not args.check_only or args.check_device == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
    elif args.check_device == "cuda":
        torch.cuda.set_device(args.device)
    if args.check_only:
        print(json.dumps(check_numerics(args.check_device)))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an earlier result, even before the first case finishes.
    with args.output.open("x") as stream:
        json.dump({"complete": False, "cases": []}, stream)
    result = {
        "complete": False,
        "scope": "Installed serving merge versus layout references; no model speedup claim",
        "cases": [],
        "iterations": args.iterations,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
    }
    try:
        from vllm_ascend.ops import fork_decode
        from vllm_ascend.ops.triton import fork_attention

        result["source_sha256"] = {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in {
                "candidate_and_benchmark": __file__,
                "serving_buffers": fork_decode.__file__,
                "serving_kernels": fork_attention.__file__,
            }.items()
        }
        result["serving_query_copy_tile"] = fork_decode.QUERY_COPY_TILE
        result["serving_merge_head_tile"] = fork_decode.MERGE_HEAD_TILE
        result["numeric_checks"] = check_numerics("npu")
        for seed, branches, prefix, tail in itertools.product(args.seeds, args.branches, args.prefix, args.tail):
            row = benchmark_case(branches, prefix, tail, seed, getattr(torch, args.dtype), args.iterations)
            result["cases"].append(row)
            print(
                json.dumps({k: row[k] for k in ("branches", "prefix", "tail", "seed", "full_us", "merge_us")}),
                flush=True,
            )
            temp = args.output.with_suffix(args.output.suffix + ".tmp")
            temp.write_text(json.dumps(result, indent=2) + "\n")
            temp.replace(args.output)
        result["complete"] = True
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.output)


if __name__ == "__main__":
    main()
