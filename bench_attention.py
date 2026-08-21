"""Run the targeted FlashInfer attention experiment from the Stage 3 brief.

The two workloads use the Qwen3-8B attention shape and deliberately combine a
very small prefill with a large, long-context decode batch.  Planning is always
outside the timed region.  Unified and zero-copy split reuse caller-owned final
outputs; ``old_split_cat`` intentionally preserves the former production path,
including its phase-output allocations, final allocation, and concatenation.
"""

from __future__ import annotations

import argparse
import hashlib
import gc
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from pathlib import Path
from typing import Callable

import torch


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE_NAME = "bf16"
WORKSPACE_MIB = 64
MIXED_METHODS = ("unified", "old_split_cat", "zero_copy_split")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    prefill_q_len: int
    prefill_kv_len: int
    decode_batch: int
    decode_kv_len: int


STANDARD_CASES = {
    "case1": CaseSpec("case1", 16, 4096, 128, 8192),
    "case2": CaseSpec("case2", 32, 4096, 64, 16384),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON file receiving environment metadata, raw repeats, and summaries",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=tuple(STANDARD_CASES),
        help="case to run; repeat for both (default: case1 and case2)",
    )
    parser.add_argument(
        "--case1-decode-batch",
        type=int,
        default=STANDARD_CASES["case1"].decode_batch,
        help="may only lower Case 1 decode batch after an explicit OOM",
    )
    parser.add_argument(
        "--case2-decode-batch",
        type=int,
        default=STANDARD_CASES["case2"].decode_batch,
        help="may only lower Case 2 decode batch after an explicit OOM",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workspace-mib", type=int, default=WORKSPACE_MIB)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--backend",
        default="auto",
        help="FlashInfer prefill backend (recorded verbatim in the JSON)",
    )
    parser.add_argument(
        "--flashinfer-cuda-arch-list",
        default=None,
        help="optional FLASHINFER_CUDA_ARCH_LIST set before FlashInfer import",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> list[CaseSpec]:
    if args.output.suffix.lower() != ".json":
        raise ValueError("--output must name a .json file")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    for name in ("iters", "repeats", "workspace_mib"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.device < 0:
        raise ValueError("--device must be non-negative")

    requested = list(dict.fromkeys(args.cases or STANDARD_CASES))
    batches = {
        "case1": args.case1_decode_batch,
        "case2": args.case2_decode_batch,
    }
    cases: list[CaseSpec] = []
    for name in requested:
        standard = STANDARD_CASES[name]
        actual = batches[name]
        if not 1 <= actual <= standard.decode_batch:
            raise ValueError(
                f"--{name}-decode-batch must be in [1, {standard.decode_batch}]; "
                "decode KV length is intentionally not configurable"
            )
        cases.append(replace(standard, decode_batch=actual))
    return cases


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parent
    commit = _command_output(["git", "rev-parse", "HEAD"], root)
    branch = _command_output(["git", "branch", "--show-current"], root)
    status = _command_output(["git", "status", "--porcelain"], root)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment_metadata(device: torch.device) -> dict[str, object]:
    script = Path(__file__).resolve()
    properties = torch.cuda.get_device_properties(device)
    driver = _command_output(
        [
            "nvidia-smi",
            "-i",
            str(device.index),
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "timestamp_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "flashinfer_python": _distribution_version("flashinfer-python"),
        "flashinfer_cubin": _distribution_version("flashinfer-cubin"),
        "flashinfer_jit_cache": _distribution_version("flashinfer-jit-cache"),
        "benchmark_script": {
            "path": str(script),
            "sha256": _file_sha256(script),
        },
        "nvidia_driver": driver,
        "gpu": {
            "index": device.index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "uuid": str(properties.uuid),
        },
        "flashinfer_env": {
            "FLASHINFER_CUDA_ARCH_LIST": os.environ.get(
                "FLASHINFER_CUDA_ARCH_LIST"
            ),
            "FLASHINFER_DISABLE_JIT": os.environ.get("FLASHINFER_DISABLE_JIT"),
        },
        "git": _git_metadata(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _indptr(lengths: list[int], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _measure_cuda_ms(operation: Callable[[], torch.Tensor], iters: int) -> float:
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


def _summarize(samples: list[float]) -> dict[str, object]:
    return {
        "raw_ms": samples,
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def _time_rotated(
    operations: dict[str, Callable[[], torch.Tensor]],
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, object]:
    names = list(operations)
    sink = None
    for name in names:
        for _ in range(warmup):
            sink = operations[name]()
    torch.cuda.synchronize()
    if sink is None and warmup:
        raise RuntimeError("warmup did not return an output")

    samples = {name: [] for name in names}
    orders: list[list[str]] = []
    for repeat_index in range(repeats):
        offset = repeat_index % len(names)
        order = names[offset:] + names[:offset]
        orders.append(order)
        for name in order:
            samples[name].append(_measure_cuda_ms(operations[name], iters))
    return {
        "execution_order_by_repeat": orders,
        "methods": {name: _summarize(samples[name]) for name in names},
    }


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    relative = difference / expected_float.abs().clamp_min(1e-5)
    return {
        "max_abs_diff": float(difference.max().item()),
        "max_rel_diff": float(relative.max().item()),
    }


@torch.inference_mode()
def _run_case(
    spec: CaseSpec,
    args: argparse.Namespace,
    device: torch.device,
    prefill_wrapper_cls: type,
    decode_wrapper_cls: type,
) -> dict[str, object]:
    dtype = torch.bfloat16
    torch.manual_seed(args.seed + int(spec.name[-1]))
    torch.cuda.manual_seed_all(args.seed + int(spec.name[-1]))
    torch.cuda.reset_peak_memory_stats(device)

    q_lens = [spec.prefill_q_len] + [1] * spec.decode_batch
    kv_lens = [spec.prefill_kv_len] + [spec.decode_kv_len] * spec.decode_batch
    page_counts = [ceil(length / BLOCK_SIZE) for length in kv_lens]
    last_page_lens = [(length - 1) % BLOCK_SIZE + 1 for length in kv_lens]
    num_pages = sum(page_counts)
    num_prefill_tokens = spec.prefill_q_len

    q_indptr = _indptr(q_lens, device)
    kv_indptr = _indptr(page_counts, device)
    page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        last_page_lens, dtype=torch.int32, device=device
    )
    q = torch.empty(
        (sum(q_lens), NUM_Q_HEADS, HEAD_DIM), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.02)
    cache_shape = (num_pages, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k_cache = torch.empty(cache_shape, dtype=dtype, device=device).uniform_(
        -0.1, 0.1
    )
    v_cache = torch.empty(cache_shape, dtype=dtype, device=device).uniform_(
        -0.1, 0.1
    )
    paged_kv_cache = (k_cache, v_cache)

    workspace_bytes = args.workspace_mib * 1024 * 1024
    unified_workspace = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device
    )
    split_workspace = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device
    )
    diagnostic_workspace = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device
    )
    unified = prefill_wrapper_cls(
        unified_workspace, kv_layout="NHD", backend=args.backend
    )
    split_prefill = prefill_wrapper_cls(
        split_workspace, kv_layout="NHD", backend=args.backend
    )
    split_decode = decode_wrapper_cls(split_workspace, kv_layout="NHD")
    diagnostic_prefill = prefill_wrapper_cls(
        diagnostic_workspace, kv_layout="NHD", backend=args.backend
    )

    plan_kwargs = {
        "num_qo_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim_qk": HEAD_DIM,
        "page_size": BLOCK_SIZE,
        "causal": True,
        "q_data_type": dtype,
        "kv_data_type": dtype,
        "o_data_type": dtype,
    }
    unified.plan(
        q_indptr, kv_indptr, page_indices, last_page_len, **plan_kwargs
    )
    prefill_page_end = int(kv_indptr[1].item())
    split_prefill.plan(
        q_indptr[:2],
        kv_indptr[:2],
        page_indices[:prefill_page_end],
        last_page_len[:1],
        **plan_kwargs,
    )
    decode_kv_indptr = kv_indptr[1:] - prefill_page_end
    decode_page_indices = page_indices[prefill_page_end:]
    decode_last_page_len = last_page_len[1:]
    split_decode.plan(
        decode_kv_indptr,
        decode_page_indices,
        decode_last_page_len,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
    )
    diagnostic_q_indptr = torch.arange(
        spec.decode_batch + 1, dtype=torch.int32, device=device
    )
    diagnostic_prefill.plan(
        diagnostic_q_indptr,
        decode_kv_indptr,
        decode_page_indices,
        decode_last_page_len,
        **plan_kwargs,
    )

    unified_output = torch.empty_like(q)
    zero_copy_output = torch.empty_like(q)
    diagnostic_prefill_output = torch.empty_like(q[num_prefill_tokens:])
    diagnostic_decode_output = torch.empty_like(q[num_prefill_tokens:])

    def run_unified() -> torch.Tensor:
        return unified.run(q, paged_kv_cache, out=unified_output)

    def run_old_split_cat() -> torch.Tensor:
        prefill_output = split_prefill.run(
            q[:num_prefill_tokens], paged_kv_cache
        )
        decode_output = split_decode.run(
            q[num_prefill_tokens:], paged_kv_cache
        )
        return torch.cat((prefill_output, decode_output), dim=0)

    def run_zero_copy_split() -> torch.Tensor:
        split_prefill.run(
            q[:num_prefill_tokens],
            paged_kv_cache,
            out=zero_copy_output[:num_prefill_tokens],
        )
        decode_output = zero_copy_output[num_prefill_tokens:]
        decode_output.zero_()
        split_decode.run(
            q[num_prefill_tokens:], paged_kv_cache, out=decode_output
        )
        return zero_copy_output

    decode_q = q[num_prefill_tokens:]

    def run_decode_with_prefill_wrapper() -> torch.Tensor:
        return diagnostic_prefill.run(
            decode_q, paged_kv_cache, out=diagnostic_prefill_output
        )

    def run_decode_with_decode_wrapper() -> torch.Tensor:
        diagnostic_decode_output.zero_()
        return split_decode.run(
            decode_q, paged_kv_cache, out=diagnostic_decode_output
        )

    unified_result = run_unified()
    old_result = run_old_split_cat()
    zero_copy_result = run_zero_copy_split()
    diagnostic_prefill_result = run_decode_with_prefill_wrapper()
    diagnostic_decode_result = run_decode_with_decode_wrapper()
    torch.cuda.synchronize()
    if unified_result.data_ptr() != unified_output.data_ptr():
        raise AssertionError("unified wrapper did not alias its caller-owned out")
    if zero_copy_result.data_ptr() != zero_copy_output.data_ptr():
        raise AssertionError("zero-copy split did not reuse its final output")
    correctness = {
        "old_split_cat_vs_unified": _assert_close(old_result, unified_result),
        "zero_copy_split_vs_unified": _assert_close(
            zero_copy_result, unified_result
        ),
        "decode_wrapper_vs_prefill_wrapper": _assert_close(
            diagnostic_decode_result, diagnostic_prefill_result
        ),
    }
    del old_result

    mixed = _time_rotated(
        {
            "unified": run_unified,
            "old_split_cat": run_old_split_cat,
            "zero_copy_split": run_zero_copy_split,
        },
        args.warmup,
        args.iters,
        args.repeats,
    )
    pure_decode = _time_rotated(
        {
            "prefill_wrapper": run_decode_with_prefill_wrapper,
            "decode_wrapper": run_decode_with_decode_wrapper,
        },
        args.warmup,
        args.iters,
        args.repeats,
    )
    unified_median = mixed["methods"]["unified"]["median_ms"]
    for method in MIXED_METHODS:
        median = mixed["methods"][method]["median_ms"]
        mixed["methods"][method]["relative_speed_vs_unified"] = (
            unified_median / median
        )
    prefill_median = pure_decode["methods"]["prefill_wrapper"]["median_ms"]
    decode_median = pure_decode["methods"]["decode_wrapper"]["median_ms"]
    pure_decode["decode_wrapper_speedup"] = prefill_median / decode_median
    resolved_backends = {
        "requested_prefill_backend": args.backend,
        "unified_prefill": getattr(unified, "_backend", None),
        "split_prefill": getattr(split_prefill, "_backend", None),
        "split_decode": getattr(split_decode, "_backend", None),
        "diagnostic_prefill": getattr(diagnostic_prefill, "_backend", None),
    }

    element_size = torch.empty((), dtype=dtype).element_size()
    cache_bytes = num_pages * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM
    cache_bytes *= element_size * 2
    return {
        "case": asdict(spec),
        "attention_shape": {
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "block_size": BLOCK_SIZE,
            "dtype": DTYPE_NAME,
        },
        "allocated_kv_pages": num_pages,
        "allocated_kv_cache_bytes": cache_bytes,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "correctness": correctness,
        "mixed": mixed,
        "pure_decode_diagnostic": pure_decode,
        "resolved_wrapper_backends": resolved_backends,
    }


def _print_case_summary(result: dict[str, object]) -> None:
    name = result["case"]["name"]
    methods = result["mixed"]["methods"]
    details = " ".join(
        f"{method}={methods[method]['median_ms']:.6f}ms" for method in MIXED_METHODS
    )
    print(f"{name}: {details}", flush=True)
    diagnostic = result["pure_decode_diagnostic"]
    print(
        f"{name} pure_decode: prefill="
        f"{diagnostic['methods']['prefill_wrapper']['median_ms']:.6f}ms "
        f"decode={diagnostic['methods']['decode_wrapper']['median_ms']:.6f}ms",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    cases = _validate_args(args)
    if args.flashinfer_cuda_arch_list:
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = args.flashinfer_cuda_arch_list
    if not torch.cuda.is_available():
        raise RuntimeError("bench_attention.py requires an NVIDIA CUDA GPU")
    if args.device >= torch.cuda.device_count():
        raise ValueError(
            f"--device {args.device} is unavailable; found "
            f"{torch.cuda.device_count()} GPUs"
        )
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    decode_batch_adjustments = {
        spec.name: {
            "standard": STANDARD_CASES[spec.name].decode_batch,
            "actual": spec.decode_batch,
        }
        for spec in cases
        if spec.decode_batch != STANDARD_CASES[spec.name].decode_batch
    }
    compliant = (
        {spec.name for spec in cases} == set(STANDARD_CASES)
        and not decode_batch_adjustments
        and args.warmup == 50
        and 500 <= args.iters <= 1000
        and args.repeats == 5
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "targeted_flashinfer_attention",
        "status": "running",
        "started_at_utc": _utc_now(),
        "protocol": {
            "protocol_compliant": compliant,
            "warmup": args.warmup,
            "warmup_scope": "once per method before all timed repeats",
            "iterations_per_repeat": args.iters,
            "independent_repeats": args.repeats,
            "timing": "CUDA events; per-call milliseconds",
            "cuda_event_coverage": (
                "GPU work, stream gaps, and torch.cat copy are covered; "
                "host-only allocator latency is not directly measured"
            ),
            "planning_in_timed_region": False,
            "decode_batch_adjusted": bool(decode_batch_adjustments),
            "decode_batch_adjustments": decode_batch_adjustments,
            "requested_prefill_backend": args.backend,
            "workspace_mib_per_wrapper_group": args.workspace_mib,
            "seed": args.seed,
            "selected_cases": [asdict(spec) for spec in cases],
            "method_contracts": {
                "unified": {
                    "final_output_preallocated": True,
                    "allocation_in_timed_region": False,
                    "calls": "one BatchPrefill wrapper over packed [P|D]",
                },
                "old_split_cat": {
                    "final_output_preallocated": False,
                    "allocation_in_timed_region": True,
                    "calls": "phase outputs from run(), then torch.cat",
                    "timing_qualification": (
                        "tensor allocation APIs are invoked in the timed loop; "
                        "CUDA events capture device-side work, not host "
                        "allocator overhead"
                    ),
                },
                "zero_copy_split": {
                    "final_output_preallocated": True,
                    "allocation_in_timed_region": False,
                    "calls": "prefill/decode out slices; decode slice zeroed",
                },
            },
            "case_defaults": {
                name: asdict(spec) for name, spec in STANDARD_CASES.items()
            },
            "decode_batch_reduction_policy": (
                "only explicit --case1-decode-batch/--case2-decode-batch; "
                "KV lengths are fixed and never silently reduced"
            ),
        },
        "environment": _environment_metadata(device),
        "cases": [],
    }
    _write_json(args.output, payload)
    started = time.monotonic()
    current_case = "none"
    try:
        for spec in cases:
            current_case = spec.name
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            print(
                f"running {spec.name}: "
                f"P=1x(q{spec.prefill_q_len},kv{spec.prefill_kv_len}) "
                f"D={spec.decode_batch}x(q1,kv{spec.decode_kv_len}) "
                f"free={free_bytes / 2**30:.2f}GiB/"
                f"{total_bytes / 2**30:.2f}GiB",
                flush=True,
            )
            result = _run_case(
                spec,
                args,
                device,
                BatchPrefillWithPagedKVCacheWrapper,
                BatchDecodeWithPagedKVCacheWrapper,
            )
            result["memory_before_case"] = {
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
            }
            payload["cases"].append(result)
            _print_case_summary(result)
            _write_json(args.output, payload)
            del result
            gc.collect()
            torch.cuda.empty_cache()
        payload["status"] = "complete"
    except torch.OutOfMemoryError as error:
        message = (
            f"CUDA OOM in {current_case}. Re-run with the matching explicit "
            f"--{current_case}-decode-batch value lowered; decode KV length "
            "cannot be lowered and no automatic fallback was applied."
        )
        payload["status"] = "failed"
        payload["error"] = {"type": type(error).__name__, "message": message}
        raise RuntimeError(message) from error
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "case": current_case,
        }
        raise
    finally:
        payload["finished_at_utc"] = _utc_now()
        payload["elapsed_seconds"] = time.monotonic() - started
        _write_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
