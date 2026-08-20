"""Microbenchmark unified versus phase-specialized FlashInfer attention.

The benchmark is model-free: it creates one reproducible mixed batch, a paged
NHD KV cache, and identical page metadata for both paths. Planning is performed
once outside the timed region, matching nano-vllm's per-batch/multi-layer use.
"""

from __future__ import annotations

import argparse
import os
from importlib.metadata import version
from math import ceil
from typing import Callable

import torch


WORKSPACE_MIB = 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-prefill", type=int, default=4)
    parser.add_argument("--prefill-q-len", type=int, default=128)
    parser.add_argument(
        "--prefill-kv-len",
        type=int,
        default=0,
        help="0 uses prefill-q-len (fresh prefill)",
    )
    parser.add_argument("--num-decode", type=int, default=64)
    parser.add_argument("--decode-kv-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--workspace-mib", type=int, default=WORKSPACE_MIB)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--backend",
        default="auto",
        help="FlashInfer prefill backend used by both unified and split paths",
    )
    parser.add_argument(
        "--flashinfer-cuda-arch-list",
        default=None,
        help=(
            "optional FLASHINFER_CUDA_ARCH_LIST override applied before "
            "FlashInfer import (for this host use 12.0f with the cu129 cache)"
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "block_size",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "num_prefill",
        "prefill_q_len",
        "num_decode",
        "decode_kv_len",
        "iters",
        "workspace_mib",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.num_q_heads % args.num_kv_heads:
        raise ValueError("--num-q-heads must be divisible by --num-kv-heads")
    if args.prefill_kv_len < 0:
        raise ValueError("--prefill-kv-len must be non-negative")
    if args.prefill_kv_len and args.prefill_kv_len < args.prefill_q_len:
        raise ValueError("--prefill-kv-len must be at least --prefill-q-len")


def _indptr(lengths: list[int], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _event_time_ms(
    operation: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> float:
    output = None
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        output = operation()
    end.record()
    end.synchronize()
    if output is None:
        raise RuntimeError("benchmark operation did not produce an output")
    return start.elapsed_time(end) / iters


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if args.flashinfer_cuda_arch_list:
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = (
            args.flashinfer_cuda_arch_list
        )
    if not torch.cuda.is_available():
        raise RuntimeError("bench_attention.py requires an NVIDIA CUDA GPU")

    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    prefill_kv_len = args.prefill_kv_len or args.prefill_q_len
    q_lens = [args.prefill_q_len] * args.num_prefill + [1] * args.num_decode
    kv_lens = [prefill_kv_len] * args.num_prefill + [
        args.decode_kv_len
    ] * args.num_decode
    page_counts = [ceil(length / args.block_size) for length in kv_lens]
    last_page_lens = [
        (length - 1) % args.block_size + 1 for length in kv_lens
    ]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    q_indptr = _indptr(q_lens, device)
    kv_indptr = _indptr(page_counts, device)
    page_indices = torch.arange(
        sum(page_counts),
        dtype=torch.int32,
        device=device,
    )
    last_page_len = torch.tensor(
        last_page_lens,
        dtype=torch.int32,
        device=device,
    )
    q = torch.randn(
        (sum(q_lens), args.num_q_heads, args.head_dim),
        dtype=dtype,
        device=device,
    )
    cache_shape = (
        sum(page_counts),
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
    )
    k_cache = torch.randn(cache_shape, dtype=dtype, device=device)
    v_cache = torch.randn(cache_shape, dtype=dtype, device=device)
    paged_kv_cache = (k_cache, v_cache)

    workspace_bytes = args.workspace_mib * 1024 * 1024
    unified_workspace = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device
    )
    split_workspace = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device
    )
    unified = BatchPrefillWithPagedKVCacheWrapper(
        unified_workspace,
        kv_layout="NHD",
        backend=args.backend,
    )
    split_prefill = BatchPrefillWithPagedKVCacheWrapper(
        split_workspace,
        kv_layout="NHD",
        backend=args.backend,
    )
    split_decode = BatchDecodeWithPagedKVCacheWrapper(
        split_workspace,
        kv_layout="NHD",
        backend=args.backend,
    )

    plan_kwargs = dict(
        num_qo_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim_qk=args.head_dim,
        page_size=args.block_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
    )
    unified.plan(
        q_indptr,
        kv_indptr,
        page_indices,
        last_page_len,
        **plan_kwargs,
    )
    prefill_page_end = int(kv_indptr[args.num_prefill].item())
    split_prefill.plan(
        q_indptr[: args.num_prefill + 1],
        kv_indptr[: args.num_prefill + 1],
        page_indices[:prefill_page_end],
        last_page_len[: args.num_prefill],
        **plan_kwargs,
    )
    split_decode.plan(
        kv_indptr[args.num_prefill :] - prefill_page_end,
        page_indices[prefill_page_end:],
        last_page_len[args.num_prefill :],
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
        args.block_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
    )
    num_prefill_tokens = args.num_prefill * args.prefill_q_len

    def run_unified() -> torch.Tensor:
        return unified.run(q, paged_kv_cache)

    def run_split() -> torch.Tensor:
        prefill_output = split_prefill.run(
            q[:num_prefill_tokens], paged_kv_cache
        )
        decode_output = split_decode.run(
            q[num_prefill_tokens:], paged_kv_cache
        )
        return torch.cat((prefill_output, decode_output), dim=0)

    with torch.inference_mode():
        unified_output = run_unified()
        split_output = run_split()
        atol = rtol = 3e-2 if dtype == torch.bfloat16 else 5e-3
        torch.testing.assert_close(
            split_output,
            unified_output,
            atol=atol,
            rtol=rtol,
        )
        max_abs_diff = float(
            (split_output.float() - unified_output.float()).abs().max().item()
        )
        unified_ms = _event_time_ms(run_unified, args.warmup, args.iters)
        split_ms = _event_time_ms(run_split, args.warmup, args.iters)

    speedup = unified_ms / split_ms
    gpu_name = torch.cuda.get_device_name(device)
    print(
        "config "
        f"gpu={gpu_name!r} flashinfer={version('flashinfer-python')} "
        f"dtype={args.dtype} B={args.block_size} "
        f"q_heads={args.num_q_heads} kv_heads={args.num_kv_heads} "
        f"head_dim={args.head_dim} prefill={args.num_prefill}x"
        f"{args.prefill_q_len} prefill_kv_len={prefill_kv_len} "
        f"decode={args.num_decode}x1 decode_kv_len={args.decode_kv_len} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(
        f"unified_ms={unified_ms:.4f} split_ms={split_ms:.4f} "
        f"speedup={speedup:.3f}x max_abs_diff={max_abs_diff:.6f}"
    )


if __name__ == "__main__":
    main()
