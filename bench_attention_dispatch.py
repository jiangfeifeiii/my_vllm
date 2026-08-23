"""Benchmark production mixed-attention dispatch against the retired route.

The default case is the Qwen3-0.6B BF16 attention shape used by this
repository: one 128-token prefill at KV length 4224 plus three one-token
decodes at KV length 4096.  Production measurements go through
``AttentionBackendRegistry`` selection plus ``build_metadata``,
``build_plan``, and ``forward`` using the same backend-neutral contract as
``ModelRunner``.  The reference is the
retired all-batch ``BatchPrefillWithPagedKVCacheWrapper`` route.

Planning, correctness checks, and output allocation are outside the timed
region.  Timed repeats alternate which method runs first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from pathlib import Path
from typing import Callable, Sequence

import torch


NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16
DTYPE_NAME = "bf16"
DEFAULT_WORKSPACE_MIB = 64
DEFAULT_WARMUP = 50
DEFAULT_ITERS = 500
DEFAULT_REPEATS = 6
DEFAULT_SEED = 2026
METHODS = (
    "production_dispatch",
    "retired_all_batch_paged_prefill",
)


@dataclass(frozen=True)
class CaseSpec:
    prefill_q_len: int = 128
    prefill_kv_len: int = 4224
    decode_batch: int = 3
    decode_kv_len: int = 4096

    @property
    def q_lens(self) -> tuple[int, ...]:
        return (self.prefill_q_len, *(1 for _ in range(self.decode_batch)))

    @property
    def kv_lens(self) -> tuple[int, ...]:
        return (
            self.prefill_kv_len,
            *(self.decode_kv_len for _ in range(self.decode_batch)),
        )


DEFAULT_CASE = CaseSpec()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON file receiving environment metadata and all raw repeats",
    )
    parser.add_argument(
        "--prefill-q-len", type=int, default=DEFAULT_CASE.prefill_q_len
    )
    parser.add_argument(
        "--prefill-kv-len", type=int, default=DEFAULT_CASE.prefill_kv_len
    )
    parser.add_argument(
        "--decode-batch", type=int, default=DEFAULT_CASE.decode_batch
    )
    parser.add_argument(
        "--decode-kv-len", type=int, default=DEFAULT_CASE.decode_kv_len
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--workspace-mib", type=int, default=DEFAULT_WORKSPACE_MIB)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--retired-prefill-backend",
        default="auto",
        help="FlashInfer backend used only by the retired reference wrapper",
    )
    parser.add_argument(
        "--expected-route",
        choices=("auto", "mixed_unified", "mixed_split"),
        default="auto",
        help=(
            "assert an exact production route; auto derives it from the "
            "backend capability gate"
        ),
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> CaseSpec:
    if args.output.suffix.lower() != ".json":
        raise ValueError("--output must name a .json file")
    positive = (
        "prefill_q_len",
        "prefill_kv_len",
        "decode_batch",
        "decode_kv_len",
        "iters",
        "repeats",
        "workspace_mib",
    )
    for name in positive:
        if isinstance(getattr(args, name), bool) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeats % len(METHODS):
        raise ValueError(
            f"--repeats must be divisible by {len(METHODS)} so each method "
            "runs first equally often"
        )
    if args.device < 0:
        raise ValueError("--device must be non-negative")
    if args.prefill_kv_len < args.prefill_q_len:
        raise ValueError("--prefill-kv-len must be at least --prefill-q-len")
    return CaseSpec(
        prefill_q_len=args.prefill_q_len,
        prefill_kv_len=args.prefill_kv_len,
        decode_batch=args.decode_batch,
        decode_kv_len=args.decode_kv_len,
    )


def _execution_orders(
    methods: Sequence[str], repeats: int
) -> list[list[str]]:
    if not methods:
        raise ValueError("at least one benchmark method is required")
    return [
        list(methods[offset:] + methods[:offset])
        for offset in (repeat % len(methods) for repeat in range(repeats))
    ]


def _summary(samples: Sequence[float]) -> dict[str, object]:
    if not samples:
        raise ValueError("at least one timing sample is required")
    values = [float(sample) for sample in samples]
    return {
        "raw_ms": values,
        "median_ms": float(statistics.median(values)),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _delta(
    production_median_ms: float,
    retired_median_ms: float,
) -> dict[str, float]:
    if production_median_ms <= 0 or retired_median_ms <= 0:
        raise ValueError("median timings must be positive")
    difference = production_median_ms - retired_median_ms
    return {
        "production_minus_retired_ms": difference,
        "production_minus_retired_percent": (
            difference / retired_median_ms * 100.0
        ),
        "production_speedup_vs_retired": (
            retired_median_ms / production_median_ms
        ),
    }


def _protocol_compliant(
    args: argparse.Namespace,
    case: CaseSpec,
) -> bool:
    return (
        case == DEFAULT_CASE
        and args.warmup == DEFAULT_WARMUP
        and args.iters == DEFAULT_ITERS
        and args.repeats == DEFAULT_REPEATS
        and args.workspace_mib == DEFAULT_WORKSPACE_MIB
        and args.seed == DEFAULT_SEED
        and args.retired_prefill_backend == "auto"
    )


def _validate_dispatch_route(
    *,
    batch_type: str,
    planned_route: str | None,
    mixed_attention_available: bool,
    expected_route: str,
) -> str:
    if batch_type != "mixed":
        raise AssertionError(f"expected BatchType.MIXED, got {batch_type!r}")
    capability_route = (
        "mixed_unified" if mixed_attention_available else "mixed_split"
    )
    required_route = (
        capability_route if expected_route == "auto" else expected_route
    )
    if planned_route != required_route:
        raise AssertionError(
            f"expected production route {required_route!r}, got "
            f"{planned_route!r}"
        )
    if planned_route != capability_route:
        raise AssertionError(
            "production route disagrees with the backend capability gate: "
            f"capability requires {capability_route!r}"
        )
    return required_route


def _indptr(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _measure_cuda_ms(
    operation: Callable[[], torch.Tensor],
    iters: int,
) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(iters):
        output = operation()
    end.record()
    end.synchronize()
    if output is None:
        raise RuntimeError("timed operation did not return an output")
    return float(start.elapsed_time(end) / iters)


def _time_alternating(
    operations: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[list[list[str]], dict[str, dict[str, object]]]:
    if tuple(operations) != METHODS:
        raise ValueError(f"operations must be ordered as {METHODS!r}")
    sink = None
    for operation in operations.values():
        for _ in range(warmup):
            sink = operation()
    torch.cuda.synchronize()
    if warmup and sink is None:
        raise RuntimeError("warmup did not return an output")

    samples = {method: [] for method in operations}
    orders = _execution_orders(tuple(operations), repeats)
    for order in orders:
        for method in order:
            samples[method].append(_measure_cuda_ms(operations[method], iters))
    return orders, {method: _summary(values) for method, values in samples.items()}


def _tensor_diff(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 3e-2,
    rtol: float = 3e-2,
) -> dict[str, object]:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    actual_float = actual.float()
    expected_float = expected.float()
    absolute = (actual_float - expected_float).abs()
    relative = absolute / expected_float.abs().clamp_min(1e-5)
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs_diff": float(absolute.max().item()),
        "max_rel_diff": float(relative.max().item()),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(device: torch.device) -> dict[str, object]:
    root = Path(__file__).resolve().parent
    script = Path(__file__).resolve()
    properties = torch.cuda.get_device_properties(device)
    status = _command_output(["git", "status", "--porcelain"], root)
    return {
        "timestamp_utc": _utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer_python": _distribution_version("flashinfer-python"),
        "flashinfer_cubin": _distribution_version("flashinfer-cubin"),
        "flashinfer_jit_cache": _distribution_version("flashinfer-jit-cache"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "flashinfer_env": {
            "FLASHINFER_CUDA_ARCH_LIST": os.environ.get(
                "FLASHINFER_CUDA_ARCH_LIST"
            ),
            "FLASHINFER_DISABLE_JIT": os.environ.get("FLASHINFER_DISABLE_JIT"),
        },
        "gpu": {
            "index": device.index,
            "name": properties.name,
            "compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "driver": _command_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(device.index),
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ]
            ),
        },
        "git": {
            "commit": _command_output(["git", "rev-parse", "HEAD"], root),
            "branch": _command_output(
                ["git", "branch", "--show-current"], root
            ),
            "dirty": bool(status),
            "status_porcelain": status.splitlines() if status else [],
        },
        "benchmark_script": {
            "path": str(script),
            "sha256": _file_sha256(script),
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


@torch.inference_mode()
def _run(args: argparse.Namespace, case: CaseSpec) -> dict[str, object]:
    started_at_utc = _utc_now()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bench_attention_dispatch.py requires an NVIDIA CUDA GPU"
        )
    if args.device >= torch.cuda.device_count():
        raise ValueError(
            f"--device {args.device} is unavailable; found "
            f"{torch.cuda.device_count()} GPUs"
        )
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    # Import after device selection so FlashInfer initializes on the intended
    # CUDA device. Keeping these imports local also makes helper tests CPU-only.
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    from nanovllm.layers.attention_backend import (
        AttentionBackendRegistry,
        FlashInferBackend,
    )
    from nanovllm.utils.context import BatchType, CommonAttentionMetadata

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    q_lens = case.q_lens
    kv_lens = case.kv_lens
    page_counts = tuple(ceil(length / BLOCK_SIZE) for length in kv_lens)
    last_page_lens = tuple((length - 1) % BLOCK_SIZE + 1 for length in kv_lens)
    num_pages = sum(page_counts)
    num_prefill_pages = page_counts[0]

    q_indptr = _indptr(q_lens, device)
    kv_indptr = _indptr(page_counts, device)
    page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        last_page_lens,
        dtype=torch.int32,
        device=device,
    )
    max_page_count = max(page_counts)
    block_table_rows = []
    page_offset = 0
    for page_count in page_counts:
        block_table_rows.append(
            list(range(page_offset, page_offset + page_count))
            + [-1] * (max_page_count - page_count)
        )
        page_offset += page_count
    block_tables = torch.tensor(
        block_table_rows,
        dtype=torch.int32,
        device=device,
    )
    common = CommonAttentionMetadata(
        num_prefill_seqs=1,
        num_decode_seqs=case.decode_batch,
        num_prefill_tokens=case.prefill_q_len,
        num_decode_tokens=case.decode_batch,
        query_start_loc=q_indptr,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        slot_mapping=torch.full(
            (sum(q_lens),), -1, dtype=torch.int32, device=device
        ),
        block_tables=block_tables,
        max_q_len=max(q_lens),
        max_kv_len=max(kv_lens),
        block_counts=page_counts,
        num_kv_blocks=num_pages,
        num_prefill_kv_blocks=num_prefill_pages,
        trusted=True,
    )

    q = torch.empty(
        (sum(q_lens), NUM_Q_HEADS, HEAD_DIM),
        dtype=DTYPE,
        device=device,
    ).normal_(mean=0.0, std=0.02)
    cache_shape = (num_pages, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k_cache = torch.empty(cache_shape, dtype=DTYPE, device=device).uniform_(
        -0.1, 0.1
    )
    v_cache = torch.empty(cache_shape, dtype=DTYPE, device=device).uniform_(
        -0.1, 0.1
    )
    cache = (k_cache, v_cache)
    unused_kv = q.new_empty((0, NUM_KV_HEADS, HEAD_DIM))

    production = AttentionBackendRegistry.create(
        "auto",
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        dtype=DTYPE,
        attention_mode="unified",
        device=device,
    )
    if not isinstance(production, FlashInferBackend):
        raise AssertionError("B16 auto selection must resolve to FlashInfer")
    backend_metadata = production.build_metadata(common)
    plan = production.build_plan(common, backend_metadata)
    if plan.batch_type is not BatchType.MIXED:
        raise AssertionError("benchmark plan must use BatchType.MIXED")
    asserted_route = _validate_dispatch_route(
        batch_type=plan.batch_type.value,
        planned_route=plan.route.value,
        mixed_attention_available=production.mixed_attention_available,
        expected_route=args.expected_route,
    )
    route_counts = dict(production.route_counts)
    expected_route_counts = dict.fromkeys(route_counts, 0)
    expected_route_counts[asserted_route] = 1
    if route_counts != expected_route_counts:
        raise AssertionError(
            "production planning must increment only its asserted route once: "
            f"expected {expected_route_counts}, got {route_counts}"
        )
    unavailable_reason = production.mixed_attention_unavailable_reason
    if production.mixed_attention_available:
        if unavailable_reason is not None:
            raise AssertionError(
                "available unified mixed attention must not report a "
                "fallback reason"
            )
    elif not unavailable_reason:
        raise AssertionError(
            "mixed split fallback must report why unified attention is "
            "unavailable"
        )

    retired_workspace = torch.empty(
        args.workspace_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    retired = BatchPrefillWithPagedKVCacheWrapper(
        retired_workspace,
        kv_layout="NHD",
        backend=args.retired_prefill_backend,
    )
    retired.plan(
        q_indptr,
        kv_indptr,
        page_indices,
        last_page_len,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_SIZE,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        o_data_type=DTYPE,
    )
    retired_output = torch.empty_like(q)

    def run_production() -> torch.Tensor:
        return production.forward(
            q,
            unused_kv,
            unused_kv,
            k_cache,
            v_cache,
            plan,
        )

    def run_retired() -> torch.Tensor:
        return retired.run(q, cache, out=retired_output)

    production_result = run_production()
    retired_result = run_retired()
    torch.cuda.synchronize()
    if production._output_buffer is None:
        raise AssertionError("production backend did not allocate reusable output")
    if production_result.data_ptr() != production._output_buffer.data_ptr():
        raise AssertionError("production backend did not reuse its output buffer")
    if retired_result.data_ptr() != retired_output.data_ptr():
        raise AssertionError("retired wrapper did not alias caller-owned out")
    diff = _tensor_diff(production_result, retired_result)

    orders, methods = _time_alternating(
        {
            METHODS[0]: run_production,
            METHODS[1]: run_retired,
        },
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    production_median = float(methods[METHODS[0]]["median_ms"])
    retired_median = float(methods[METHODS[1]]["median_ms"])
    element_size = torch.empty((), dtype=DTYPE).element_size()

    return {
        "schema_version": 2,
        "benchmark": "nanovllm_production_mixed_attention_dispatch",
        "status": "complete",
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "environment": _environment(device),
        "shape": {
            "model_shape": "Qwen3-0.6B",
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "block_size": BLOCK_SIZE,
            "dtype": DTYPE_NAME,
            "prefill_q_len": case.prefill_q_len,
            "prefill_kv_len": case.prefill_kv_len,
            "decode_batch": case.decode_batch,
            "decode_kv_len": case.decode_kv_len,
            "q_lens": list(q_lens),
            "kv_lens": list(kv_lens),
            "q_indptr": q_indptr.tolist(),
            "kv_indptr": kv_indptr.tolist(),
            "page_counts": list(page_counts),
            "last_page_lens": list(last_page_lens),
            "num_prefill_pages": num_prefill_pages,
            "total_q_tokens": sum(q_lens),
            "allocated_kv_pages": num_pages,
            "allocated_kv_cache_bytes": (
                num_pages
                * BLOCK_SIZE
                * NUM_KV_HEADS
                * HEAD_DIM
                * element_size
                * 2
            ),
        },
        "dispatch": {
            "backend_requested": "auto",
            "backend": production.backend_name,
            "batch_type": plan.batch_type.value,
            "attention_mode": production.attention_mode,
            "expected_route_argument": args.expected_route,
            "asserted_route": asserted_route,
            "actual_route": plan.route.value,
            "backend_metadata_type": type(backend_metadata).__name__,
            "mixed_attention_available": production.mixed_attention_available,
            "mixed_attention_unavailable_reason": (
                production.mixed_attention_unavailable_reason
            ),
            "route_counts": route_counts,
            "production_prefill_backend_resolved": getattr(
                production.prefill_wrapper, "_backend", None
            ),
            "production_decode_backend_resolved": getattr(
                production.decode_wrapper, "_backend", None
            ),
            "retired_prefill_backend_requested": (
                args.retired_prefill_backend
            ),
            "retired_prefill_backend_resolved": getattr(
                retired, "_backend", None
            ),
        },
        "protocol": {
            "planning_in_timed_region": False,
            "correctness_in_timed_region": False,
            "warmup": args.warmup,
            "iterations_per_repeat": args.iters,
            "independent_repeats": args.repeats,
            "timing": "CUDA events; per-call milliseconds",
            "execution_order": "alternating first method by repeat",
            "production_output": "backend-owned reusable output",
            "retired_output": "caller-owned preallocated output",
            "seed": args.seed,
            "protocol_compliant": _protocol_compliant(args, case),
        },
        "raw": {
            "execution_order_by_repeat": orders,
            METHODS[0]: methods[METHODS[0]]["raw_ms"],
            METHODS[1]: methods[METHODS[1]]["raw_ms"],
        },
        "median_ms": {
            METHODS[0]: production_median,
            METHODS[1]: retired_median,
        },
        "range_ms": {
            method: {
                "min": methods[method]["min_ms"],
                "max": methods[method]["max_ms"],
            }
            for method in METHODS
        },
        "delta": _delta(production_median, retired_median),
        "diff": diff,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    case = _validate_args(args)
    payload = _run(args, case)
    _write_json(args.output, payload)
    print(
        f"route={payload['dispatch']['actual_route']} "
        f"production={payload['median_ms'][METHODS[0]]:.6f}ms "
        f"retired={payload['median_ms'][METHODS[1]]:.6f}ms "
        f"delta={payload['delta']['production_minus_retired_percent']:+.2f}%",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
