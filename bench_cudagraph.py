"""Benchmark nano-vLLM eager versus full-decode CUDA Graph execution.

Run each policy in its own process.  A formal ablation uses the same command
twice with only ``--mode`` and ``--output`` changed::

    python bench_cudagraph.py --model /path/to/Qwen3-0.6B \
        --mode none --output benchmark_results/cudagraph_none.json
    python bench_cudagraph.py --model /path/to/Qwen3-0.6B \
        --mode full_decode_only \
        --output benchmark_results/cudagraph_full_decode_only.json

After both isolated runs complete, validate their provenance, configuration,
case/repeat matrix, and generated-token hashes and emit a README-ready table::

    python bench_cudagraph.py compare \
        --none benchmark_results/cudagraph_none.json \
        --full benchmark_results/cudagraph_full_decode_only.json \
        --output benchmark_results/cudagraph_comparison.json \
        --markdown-output benchmark_results/cudagraph_comparison.md

One engine runs the full 5 x 3 matrix.  Every case first primes a persistent
shared prefix, then admits an exact-size follower batch.  The follower prefill
and an independent warm-up batch are excluded from timing.  Measured repeats
contain only pure-decode steps; there is no graph-bucket padding.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import platform
import random
import shlex
import socket
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence as TypingSequence

import torch
import torch.distributed as dist

from nanovllm import LLM, SamplingParams
from nanovllm.config import CUDAGraphPolicy


BLOCK_SIZE = 16
DEFAULT_BATCH_SIZES = (1, 4, 8, 16, 32)
DEFAULT_KV_LENGTHS = (512, 2048, 4096)
RUNTIME_COUNTERS = (
    "full_graph_replay_steps",
    "eager_fallback_steps",
    "graph_bucket_hits",
    "graph_bucket_misses",
)
BENCHMARK_NAME = "nanovllm_internal_cudagraph_ablation"
COMPARISON_BENCHMARK_NAME = f"{BENCHMARK_NAME}_comparison"
FORMAL_DECODE_STEPS = 64
FORMAL_WARMUP_DECODE_STEPS = 8
FORMAL_REPEATS = 5


@dataclass(frozen=True, order=True)
class CaseSpec:
    batch_size: int
    kv_length: int

    @property
    def name(self) -> str:
        return f"bs{self.batch_size}_kv{self.kv_length}"

    @property
    def prompt_length(self) -> int:
        # The prefill samples one token.  Therefore the first measured decode
        # sees exactly kv_length tokens, rather than kv_length + 1.
        return self.kv_length - 1


class TokenFactory:
    """Generate deterministic valid token IDs without tokenizer round trips."""

    def __init__(self, vocab_size: int, seed: int, forbidden: set[int]):
        if vocab_size <= len(forbidden):
            raise ValueError("vocabulary is too small after excluding specials")
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)
        self.forbidden = forbidden

    def tokens(self, length: int) -> list[int]:
        result: list[int] = []
        while len(result) < length:
            token_id = self.rng.randrange(self.vocab_size)
            if token_id not in self.forbidden:
                result.append(token_id)
        return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parse_args(
    argv: TypingSequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local BF16 Qwen3 path")
    parser.add_argument(
        "--mode",
        choices=tuple(policy.value for policy in CUDAGraphPolicy),
        required=True,
        help="one process benchmarks exactly one runtime policy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON file receiving metadata and every raw repeat",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_sizes",
        action="append",
        type=_positive_int,
        help="exact batch bucket to run; repeat to select a subset",
    )
    parser.add_argument(
        "--kv-length",
        dest="kv_lengths",
        action="append",
        type=_positive_int,
        help="starting pure-decode KV length; repeat to select a subset",
    )
    parser.add_argument("--decode-steps", type=_positive_int, default=64)
    parser.add_argument(
        "--warmup-decode-steps",
        type=_positive_int,
        default=8,
        help="pure-decode steps in a separate, unmeasured follower batch",
    )
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1e-6,
        help=(
            "near-greedy positive temperature; NONE/FULL completion-token "
            "hashes are compared as a diagnostic"
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    return parser.parse_args(argv)


def _parse_compare_args(
    argv: TypingSequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} compare",
        description=(
            "Strictly validate and summarize one formal NONE result and one "
            "formal FULL_DECODE_ONLY result without starting a GPU engine."
        ),
    )
    parser.add_argument(
        "--none",
        type=Path,
        required=True,
        help="completed formal result produced with --mode none",
    )
    parser.add_argument(
        "--full",
        type=Path,
        required=True,
        help="completed formal result produced with --mode full_decode_only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON comparison artifact",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="optional README-ready Markdown table (also printed to stdout)",
    )
    return parser.parse_args(argv)


def _unique_sorted(
    requested: TypingSequence[int] | None,
    defaults: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(sorted(set(defaults if requested is None else requested)))


def _case_matrix(
    batch_sizes: TypingSequence[int] | None = None,
    kv_lengths: TypingSequence[int] | None = None,
) -> list[CaseSpec]:
    batches = _unique_sorted(batch_sizes, DEFAULT_BATCH_SIZES)
    lengths = _unique_sorted(kv_lengths, DEFAULT_KV_LENGTHS)
    return [
        CaseSpec(batch_size=batch_size, kv_length=kv_length)
        for batch_size in batches
        for kv_length in lengths
    ]


def _validate_args(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], tuple[int, ...], list[CaseSpec]]:
    batch_sizes = _unique_sorted(args.batch_sizes, DEFAULT_BATCH_SIZES)
    kv_lengths = _unique_sorted(args.kv_lengths, DEFAULT_KV_LENGTHS)
    if not batch_sizes:
        raise ValueError("at least one --batch-size is required")
    if not kv_lengths:
        raise ValueError("at least one --kv-length is required")
    invalid_lengths = [
        length
        for length in kv_lengths
        if length < 2 * BLOCK_SIZE or length % BLOCK_SIZE
    ]
    if invalid_lengths:
        raise ValueError(
            "KV lengths must be multiples of block_size=16 and at least 32; "
            f"got {invalid_lengths}"
        )
    if args.temperature <= 1e-10:
        raise ValueError("--temperature must be greater than 1e-10")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.output.suffix.lower() != ".json":
        raise ValueError("--output must name a .json file")
    return batch_sizes, kv_lengths, _case_matrix(batch_sizes, kv_lengths)


def _command_output(arguments: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _token_sha256(token_ids: TypingSequence[int]) -> str:
    encoded = ",".join(map(str, token_ids)).encode("ascii")
    return _sha256_bytes(encoded)


def _git_metadata() -> dict[str, Any]:
    repository = Path(__file__).resolve().parent
    status = _command_output(["git", "status", "--porcelain"], repository)
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"], repository),
        "branch": _command_output(
            ["git", "branch", "--show-current"], repository
        ),
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def _flashinfer_metadata() -> dict[str, Any]:
    return {
        "flashinfer_python": _package_version("flashinfer-python"),
        "flashinfer_cubin": _package_version("flashinfer-cubin"),
        "flashinfer_jit_cache": _package_version("flashinfer-jit-cache"),
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("FLASHINFER_")
        },
    }


def _environment_metadata() -> dict[str, Any]:
    script = Path(__file__).resolve()
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "timestamp_utc": _utc_now(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_driver": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": {
            "logical_index": device_index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "uuid": str(properties.uuid),
        },
        "transformers": _package_version("transformers"),
        "flashinfer": _flashinfer_metadata(),
        "git": _git_metadata(),
        "benchmark_script": {
            "path": str(script),
            "sha256": _sha256_bytes(script.read_bytes()),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark result {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark result {path} must contain a JSON object")
    return payload


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _formal_case_keys() -> list[tuple[int, int]]:
    return [
        (batch_size, kv_length)
        for batch_size in DEFAULT_BATCH_SIZES
        for kv_length in DEFAULT_KV_LENGTHS
    ]


def _result_case_key(
    case_result: dict[str, Any],
    location: str,
) -> tuple[int, int]:
    spec = _require_mapping(case_result.get("case"), f"{location}.case")
    batch_size = spec.get("batch_size")
    kv_length = spec.get("kv_length")
    if type(batch_size) is not int or type(kv_length) is not int:
        raise ValueError(
            f"{location}.case must contain integer batch_size and kv_length"
        )
    expected_name = CaseSpec(batch_size, kv_length).name
    if case_result.get("name") != expected_name:
        raise ValueError(
            f"{location}.name must be {expected_name!r}, got "
            f"{case_result.get('name')!r}"
        )
    return batch_size, kv_length


def _validate_derived_value(
    actual: Any,
    expected: Any,
    location: str,
) -> None:
    if isinstance(expected, dict):
        actual_mapping = _require_mapping(actual, location)
        for name, expected_value in expected.items():
            if name not in actual_mapping:
                raise ValueError(f"{location} is missing derived field {name!r}")
            _validate_derived_value(
                actual_mapping[name],
                expected_value,
                f"{location}.{name}",
            )
        return
    if isinstance(expected, float):
        actual_value = _finite_metric(actual, location)
        if not math.isclose(
            actual_value,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"{location} does not match raw steps: "
                f"{actual_value!r} != {expected!r}"
            )
        return
    if actual != expected:
        raise ValueError(
            f"{location} does not match raw steps: "
            f"{actual!r} != {expected!r}"
        )


def _validate_prefix_hits(
    value: Any,
    *,
    batch_size: int,
    kv_length: int,
    location: str,
) -> None:
    hits = _require_list(value, location)
    expected = [kv_length - BLOCK_SIZE] * batch_size
    if hits != expected:
        raise ValueError(
            f"{location} must prove the shared-prefix hit {expected!r}, "
            f"got {hits!r}"
        )


def _validate_formal_repeat(
    repeat: dict[str, Any],
    *,
    mode: str,
    batch_size: int,
    kv_length: int,
    location: str,
) -> None:
    if type(repeat.get("sampling_seed")) is not int:
        raise ValueError(f"{location}.sampling_seed must be an integer")
    _validate_prefix_hits(
        repeat.get("prefix_hit_tokens_per_request"),
        batch_size=batch_size,
        kv_length=kv_length,
        location=f"{location}.prefix_hit_tokens_per_request",
    )
    excluded = _require_mapping(
        repeat.get("excluded_prefill_runtime_counters"),
        f"{location}.excluded_prefill_runtime_counters",
    )
    try:
        _validate_excluded_prefill_stats(mode, excluded)
    except (AssertionError, ValueError) as error:
        raise ValueError(
            f"{location} has invalid excluded-prefill counters: {error}"
        ) from error

    steps = _require_list(repeat.get("steps"), f"{location}.steps")
    if len(steps) != FORMAL_DECODE_STEPS:
        raise ValueError(
            f"{location}.steps must contain {FORMAL_DECODE_STEPS} raw steps"
        )
    wall_samples: list[float] = []
    cuda_samples: list[float] = []
    final_counters: dict[str, int] | None = None
    for step_index, raw_step in enumerate(steps):
        step_location = f"{location}.steps[{step_index}]"
        step = _require_mapping(raw_step, step_location)
        expected_fields = {
            "step": step_index,
            "batch_size": batch_size,
            "kv_length": kv_length + step_index,
        }
        for name, expected in expected_fields.items():
            if step.get(name) != expected:
                raise ValueError(
                    f"{step_location}.{name} must be {expected!r}, got "
                    f"{step.get(name)!r}"
                )
        wall_samples.append(
            _finite_metric(
                step.get("wall_ms"),
                f"{step_location}.wall_ms",
                positive=True,
            )
        )
        cuda_samples.append(
            _finite_metric(
                step.get("cuda_event_ms"),
                f"{step_location}.cuda_event_ms",
                positive=True,
            )
        )
        counters = _require_mapping(
            step.get("runtime_counters_after_step"),
            f"{step_location}.runtime_counters_after_step",
        )
        try:
            final_counters = _validate_measured_stats(
                mode,
                step_index + 1,
                counters,
            )
        except (AssertionError, ValueError) as error:
            raise ValueError(
                f"{step_location} has invalid cumulative counters: {error}"
            ) from error

    assert final_counters is not None
    repeat_counters = _runtime_counter_view(
        _require_mapping(
            repeat.get("runtime_counters"),
            f"{location}.runtime_counters",
        )
    )
    if repeat_counters != final_counters:
        raise ValueError(
            f"{location}.runtime_counters do not match the final raw step: "
            f"{repeat_counters!r} != {final_counters!r}"
        )

    wall_elapsed_ms = sum(wall_samples)
    cuda_elapsed_ms = sum(cuda_samples)
    measured_output_tokens = batch_size * FORMAL_DECODE_STEPS
    expected_derived = {
        "measured_decode_steps": FORMAL_DECODE_STEPS,
        "measured_output_tokens": measured_output_tokens,
        "wall_elapsed_ms": wall_elapsed_ms,
        "cuda_event_elapsed_ms": cuda_elapsed_ms,
        "tpot_wall_ms": wall_elapsed_ms / FORMAL_DECODE_STEPS,
        "tpot_cuda_event_ms": cuda_elapsed_ms / FORMAL_DECODE_STEPS,
        "output_tokens_per_second_wall": (
            measured_output_tokens / (wall_elapsed_ms / 1000.0)
        ),
        "output_tokens_per_second_cuda_event": (
            measured_output_tokens / (cuda_elapsed_ms / 1000.0)
        ),
    }
    _validate_derived_value(repeat, expected_derived, location)


def _validate_formal_case_summary(
    case: dict[str, Any],
    *,
    mode: str,
    batch_size: int,
    kv_length: int,
    location: str,
) -> None:
    spec = _require_mapping(case.get("case"), f"{location}.case")
    expected_spec = {
        "batch_size": batch_size,
        "kv_length": kv_length,
        "prompt_length": kv_length - 1,
        "decode_steps_per_repeat": FORMAL_DECODE_STEPS,
        "first_measured_decode_kv_length": kv_length,
        "last_measured_decode_kv_length": (
            kv_length + FORMAL_DECODE_STEPS - 1
        ),
        "measured_output_tokens_per_repeat": (
            batch_size * FORMAL_DECODE_STEPS
        ),
    }
    for name, expected in expected_spec.items():
        if spec.get(name) != expected:
            raise ValueError(
                f"{location}.case.{name} must be {expected!r}, got "
                f"{spec.get(name)!r}"
            )

    invariants = _require_mapping(
        case.get("invariants"),
        f"{location}.invariants",
    )
    expected_invariants = {
        "timed_batch_type": "pure_decode",
        "exact_batch_bucket": batch_size,
        "runtime_padding_requests": 0,
        "prefill_excluded_from_timing": True,
        "shared_prefix_hit_verified_for_every_follower": True,
    }
    for name, expected in expected_invariants.items():
        if invariants.get(name) != expected:
            raise ValueError(
                f"{location}.invariants.{name} must be {expected!r}"
            )

    warmup = _require_mapping(case.get("warmup"), f"{location}.warmup")
    if warmup.get("decode_steps") != FORMAL_WARMUP_DECODE_STEPS:
        raise ValueError(f"{location}.warmup.decode_steps is not formal")
    _validate_prefix_hits(
        warmup.get("prefix_hit_tokens_per_request"),
        batch_size=batch_size,
        kv_length=kv_length,
        location=f"{location}.warmup.prefix_hit_tokens_per_request",
    )
    warmup_prefill = _require_mapping(
        warmup.get("excluded_prefill_runtime_counters"),
        f"{location}.warmup.excluded_prefill_runtime_counters",
    )
    try:
        _validate_excluded_prefill_stats(mode, warmup_prefill)
        _validate_measured_stats(
            mode,
            FORMAL_WARMUP_DECODE_STEPS,
            _require_mapping(
                warmup.get("runtime_counters"),
                f"{location}.warmup.runtime_counters",
            ),
        )
    except (AssertionError, ValueError) as error:
        raise ValueError(
            f"{location}.warmup has invalid runtime counters: {error}"
        ) from error

    repeats = _require_list(case.get("repeats"), f"{location}.repeats")
    expected_summary = _case_summary(mode, batch_size, repeats)
    summary = _require_mapping(case.get("summary"), f"{location}.summary")
    _validate_derived_value(summary, expected_summary, f"{location}.summary")


def _validate_formal_result(
    payload: dict[str, Any],
    label: str,
    expected_mode: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    prefix = f"{label} result"
    if payload.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(
            f"{prefix} has unexpected benchmark {payload.get('benchmark')!r}"
        )
    if payload.get("status") != "complete":
        raise ValueError(
            f"{prefix} status must be 'complete', got {payload.get('status')!r}"
        )
    if payload.get("mode") != expected_mode:
        raise ValueError(
            f"{prefix} mode must be {expected_mode!r}, got "
            f"{payload.get('mode')!r}"
        )
    protocol = _require_mapping(payload.get("protocol"), f"{prefix}.protocol")
    if protocol.get("formal_matrix") is not True:
        raise ValueError(f"{prefix} is not a formal matrix result")

    config = _require_mapping(payload.get("config"), f"{prefix}.config")
    required_config = {
        "cudagraph_mode": expected_mode,
        "cudagraph_batch_sizes": list(DEFAULT_BATCH_SIZES),
        "batch_sizes": list(DEFAULT_BATCH_SIZES),
        "kv_lengths": list(DEFAULT_KV_LENGTHS),
        "decode_steps": FORMAL_DECODE_STEPS,
        "warmup_decode_steps": FORMAL_WARMUP_DECODE_STEPS,
        "repeats": FORMAL_REPEATS,
    }
    for name, expected in required_config.items():
        if config.get(name) != expected:
            raise ValueError(
                f"{prefix}.config.{name} must be {expected!r}, got "
                f"{config.get(name)!r}"
            )
    if type(config.get("seed")) is not int:
        raise ValueError(f"{prefix}.config.seed must be an integer")

    raw_cases = _require_list(payload.get("cases"), f"{prefix}.cases")
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    ordered_keys: list[tuple[int, int]] = []
    for case_index, raw_case in enumerate(raw_cases):
        location = f"{prefix}.cases[{case_index}]"
        case = _require_mapping(raw_case, location)
        key = _result_case_key(case, location)
        if key in indexed:
            raise ValueError(f"{prefix} contains duplicate case {key}")
        ordered_keys.append(key)
        indexed[key] = case

        spec = _require_mapping(case["case"], f"{location}.case")
        if spec.get("decode_steps_per_repeat") != FORMAL_DECODE_STEPS:
            raise ValueError(
                f"{location}.case.decode_steps_per_repeat is not formal"
            )
        repeats = _require_list(case.get("repeats"), f"{location}.repeats")
        if len(repeats) != FORMAL_REPEATS:
            raise ValueError(
                f"{location} must contain {FORMAL_REPEATS} repeats"
            )
        for repeat_index, raw_repeat in enumerate(repeats):
            repeat_location = f"{location}.repeats[{repeat_index}]"
            repeat = _require_mapping(raw_repeat, repeat_location)
            _validate_formal_repeat(
                repeat,
                mode=expected_mode,
                batch_size=key[0],
                kv_length=key[1],
                location=repeat_location,
            )
            if repeat.get("repeat") != repeat_index:
                raise ValueError(
                    f"{repeat_location}.repeat must be {repeat_index}"
                )
            if repeat.get("measured_decode_steps") != FORMAL_DECODE_STEPS:
                raise ValueError(
                    f"{repeat_location}.measured_decode_steps is not formal"
                )
            hashes = _require_list(
                repeat.get("completion_token_sha256"),
                f"{repeat_location}.completion_token_sha256",
            )
            if len(hashes) != key[0] or not all(
                _is_sha256(value) for value in hashes
            ):
                raise ValueError(
                    f"{repeat_location}.completion_token_sha256 must contain "
                    f"{key[0]} SHA-256 values"
                )
            counters = _require_mapping(
                repeat.get("runtime_counters"),
                f"{repeat_location}.runtime_counters",
            )
            try:
                _validate_measured_stats(
                    expected_mode,
                    FORMAL_DECODE_STEPS,
                    counters,
                )
            except (AssertionError, ValueError) as error:
                raise ValueError(
                    f"{repeat_location} has invalid runtime counters: {error}"
                ) from error

        summary = _require_mapping(case.get("summary"), f"{location}.summary")
        summary_counters = _require_mapping(
            summary.get("runtime_counters"),
            f"{location}.summary.runtime_counters",
        )
        try:
            _validate_measured_stats(
                expected_mode,
                FORMAL_DECODE_STEPS * FORMAL_REPEATS,
                summary_counters,
            )
        except (AssertionError, ValueError) as error:
            raise ValueError(
                f"{location}.summary has invalid runtime counters: {error}"
            ) from error
        _validate_formal_case_summary(
            case,
            mode=expected_mode,
            batch_size=key[0],
            kv_length=key[1],
            location=location,
        )

    expected_keys = _formal_case_keys()
    if ordered_keys != expected_keys:
        raise ValueError(
            f"{prefix} case matrix/order must be {expected_keys}, got "
            f"{ordered_keys}"
        )
    return indexed


def _normalized_model_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return os.path.normcase(os.path.normpath(value))


def _validate_pair_provenance(
    none_payload: dict[str, Any],
    full_payload: dict[str, Any],
) -> dict[str, Any]:
    none_runtime = _require_mapping(
        none_payload.get("runtime"), "NONE result.runtime"
    )
    full_runtime = _require_mapping(
        full_payload.get("runtime"), "FULL result.runtime"
    )
    none_path = _normalized_model_path(none_runtime.get("model"))
    full_path = _normalized_model_path(full_runtime.get("model"))
    none_identity = none_runtime.get("model_identity")
    full_identity = full_runtime.get("model_identity")
    same_path = none_path is not None and none_path == full_path
    same_identity = (
        bool(none_identity)
        and bool(full_identity)
        and none_identity == full_identity
    )
    if not (same_path or same_identity):
        raise ValueError(
            "NONE and FULL results do not identify the same model: "
            f"paths={none_path!r}/{full_path!r}, "
            f"identities={none_identity!r}/{full_identity!r}"
        )
    for name in ("model_type", "model_dtype", "model_shape"):
        if none_runtime.get(name) != full_runtime.get(name):
            raise ValueError(
                f"NONE and FULL runtime.{name} differ: "
                f"{none_runtime.get(name)!r} != {full_runtime.get(name)!r}"
            )

    none_environment = _require_mapping(
        none_payload.get("environment"), "NONE result.environment"
    )
    full_environment = _require_mapping(
        full_payload.get("environment"), "FULL result.environment"
    )
    stable_environment_fields = (
        "cuda_driver",
        "cudnn",
        "python",
        "torch",
        "torch_cuda",
        "transformers",
    )
    for name in stable_environment_fields:
        none_value = none_environment.get(name)
        full_value = full_environment.get(name)
        if none_value is None or none_value != full_value:
            raise ValueError(
                f"NONE and FULL environment.{name} must match and be "
                f"non-null: {none_value!r} != {full_value!r}"
            )

    none_gpu = _require_mapping(
        none_environment.get("gpu"), "NONE result.environment.gpu"
    )
    full_gpu = _require_mapping(
        full_environment.get("gpu"), "FULL result.environment.gpu"
    )
    stable_gpu_fields = (
        "uuid",
        "name",
        "compute_capability",
        "total_memory_bytes",
        "multiprocessor_count",
    )
    for name in stable_gpu_fields:
        none_value = none_gpu.get(name)
        full_value = full_gpu.get(name)
        if none_value is None or none_value != full_value:
            raise ValueError(
                f"NONE and FULL environment.gpu.{name} must match and be "
                f"non-null: {none_value!r} != {full_value!r}"
            )

    none_flashinfer = _require_mapping(
        none_environment.get("flashinfer"),
        "NONE result.environment.flashinfer",
    )
    full_flashinfer = _require_mapping(
        full_environment.get("flashinfer"),
        "FULL result.environment.flashinfer",
    )
    for name in (
        "flashinfer_python",
        "flashinfer_cubin",
        "flashinfer_jit_cache",
    ):
        none_value = none_flashinfer.get(name)
        full_value = full_flashinfer.get(name)
        if not isinstance(none_value, str) or none_value != full_value:
            raise ValueError(
                f"NONE and FULL environment.flashinfer.{name} must match: "
                f"{none_value!r} != {full_value!r}"
            )
    if none_flashinfer.get("environment") != full_flashinfer.get("environment"):
        raise ValueError(
            "NONE and FULL FlashInfer environment settings must match"
        )

    none_git = _require_mapping(
        none_environment.get("git"), "NONE result.environment.git"
    )
    full_git = _require_mapping(
        full_environment.get("git"), "FULL result.environment.git"
    )
    none_commit = none_git.get("commit")
    full_commit = full_git.get("commit")
    if (
        not isinstance(none_commit, str)
        or not none_commit
        or none_commit != full_commit
    ):
        raise ValueError(
            "NONE and FULL git commits must be identical and non-empty: "
            f"{none_commit!r} != {full_commit!r}"
        )

    none_script = _require_mapping(
        none_environment.get("benchmark_script"),
        "NONE result.environment.benchmark_script",
    )
    full_script = _require_mapping(
        full_environment.get("benchmark_script"),
        "FULL result.environment.benchmark_script",
    )
    none_script_sha = none_script.get("sha256")
    full_script_sha = full_script.get("sha256")
    if (
        not _is_sha256(none_script_sha)
        or none_script_sha != full_script_sha
    ):
        raise ValueError(
            "NONE and FULL benchmark script SHA-256 values must match: "
            f"{none_script_sha!r} != {full_script_sha!r}"
        )

    return {
        "model": {
            "path": none_runtime.get("model") if same_path else None,
            "identity": none_identity if same_identity else None,
            "type": none_runtime.get("model_type"),
            "dtype": none_runtime.get("model_dtype"),
            "shape": none_runtime.get("model_shape"),
        },
        "git_commit": none_commit,
        "benchmark_script_sha256": none_script_sha,
        "environment": {
            name: none_environment[name]
            for name in stable_environment_fields
        },
        "gpu": {
            name: none_gpu[name]
            for name in stable_gpu_fields
        },
        "flashinfer": {
            name: none_flashinfer[name]
            for name in (
                "flashinfer_python",
                "flashinfer_cubin",
                "flashinfer_jit_cache",
                "environment",
            )
        },
    }


def _comparable_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = _require_mapping(payload.get("config"), "result.config")
    return {
        name: value
        for name, value in config.items()
        if name != "cudagraph_mode"
    }


def _finite_metric(
    value: Any,
    location: str,
    *,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value <= 0)
    ):
        qualifier = "positive finite" if positive else "finite non-negative"
        raise ValueError(f"{location} must be {qualifier}, got {value!r}")
    return float(value)


def _capture_metrics(payload: dict[str, Any], label: str) -> dict[str, float]:
    runtime = _require_mapping(payload.get("runtime"), f"{label}.runtime")
    startup = _require_mapping(
        runtime.get("cudagraph_startup"),
        f"{label}.runtime.cudagraph_startup",
    )
    capture_time_ms = _finite_metric(
        startup.get("capture_time_ms"),
        f"{label}.runtime.cudagraph_startup.capture_time_ms",
    )
    extra_memory_bytes = _finite_metric(
        startup.get("extra_memory_bytes"),
        f"{label}.runtime.cudagraph_startup.extra_memory_bytes",
    )
    return {
        "time_ms": capture_time_ms,
        "extra_memory_bytes": extra_memory_bytes,
        "extra_memory_mib": extra_memory_bytes / (1024.0 * 1024.0),
    }


def _case_metrics(case: dict[str, Any], location: str) -> dict[str, float]:
    summary = _require_mapping(case.get("summary"), f"{location}.summary")
    latency = _require_mapping(
        summary.get("decode_step_latency_wall_ms"),
        f"{location}.summary.decode_step_latency_wall_ms",
    )
    replay = _finite_metric(
        summary.get("graph_replay_hit_rate"),
        f"{location}.summary.graph_replay_hit_rate",
    )
    if replay > 1.0:
        raise ValueError(
            f"{location}.summary.graph_replay_hit_rate must be <= 1"
        )
    return {
        "latency_median_wall_ms": _finite_metric(
            latency.get("median"),
            f"{location}.summary.decode_step_latency_wall_ms.median",
            positive=True,
        ),
        "tpot_wall_ms": _finite_metric(
            summary.get("tpot_wall_ms"),
            f"{location}.summary.tpot_wall_ms",
            positive=True,
        ),
        "output_tokens_per_second_wall": _finite_metric(
            summary.get("output_tokens_per_second_wall"),
            f"{location}.summary.output_tokens_per_second_wall",
            positive=True,
        ),
        "graph_replay_hit_rate": replay,
    }


def _relative_delta_percent(full: float, none: float) -> float:
    if none == 0:
        raise ValueError("cannot compute a relative delta from a zero baseline")
    return (full - none) / none * 100.0


def _compare_formal_results(
    none_payload: dict[str, Any],
    full_payload: dict[str, Any],
    *,
    none_source: str | None = None,
    full_source: str | None = None,
) -> dict[str, Any]:
    none_cases = _validate_formal_result(
        none_payload,
        "NONE",
        CUDAGraphPolicy.NONE.value,
    )
    full_cases = _validate_formal_result(
        full_payload,
        "FULL",
        CUDAGraphPolicy.FULL_DECODE_ONLY.value,
    )
    provenance = _validate_pair_provenance(none_payload, full_payload)

    none_config = _comparable_config(none_payload)
    full_config = _comparable_config(full_payload)
    if none_config.get("seed") != full_config.get("seed"):
        raise ValueError(
            "NONE and FULL seeds differ: "
            f"{none_config.get('seed')!r} != {full_config.get('seed')!r}"
        )
    if none_config != full_config:
        differing = sorted(
            name
            for name in set(none_config) | set(full_config)
            if none_config.get(name) != full_config.get(name)
        )
        raise ValueError(
            "NONE and FULL configurations differ outside cudagraph_mode: "
            f"{differing}"
        )

    none_capture = _capture_metrics(none_payload, "NONE result")
    full_capture = _capture_metrics(full_payload, "FULL result")
    capture_delta = {
        name: full_capture[name] - none_capture[name]
        for name in none_capture
    }

    rows: list[dict[str, Any]] = []
    hash_matched = 0
    hash_total = 0
    hash_mismatches: list[dict[str, Any]] = []
    for key in _formal_case_keys():
        none_case = none_cases[key]
        full_case = full_cases[key]
        case_name = CaseSpec(*key).name
        if none_case.get("case") != full_case.get("case"):
            raise ValueError(f"{case_name} case specifications differ")
        none_prefix = _require_mapping(
            none_case.get("shared_prefix"), f"NONE {case_name}.shared_prefix"
        )
        full_prefix = _require_mapping(
            full_case.get("shared_prefix"), f"FULL {case_name}.shared_prefix"
        )
        for name in ("prompt_seed", "prompt_sha256"):
            if none_prefix.get(name) != full_prefix.get(name):
                raise ValueError(
                    f"{case_name} shared-prefix {name} differs between modes"
                )

        none_repeats = _require_list(
            none_case.get("repeats"), f"NONE {case_name}.repeats"
        )
        full_repeats = _require_list(
            full_case.get("repeats"), f"FULL {case_name}.repeats"
        )
        case_hash_matched = 0
        case_hash_total = 0
        case_hash_mismatches: list[dict[str, Any]] = []
        for repeat_index, (none_repeat, full_repeat) in enumerate(
            zip(none_repeats, full_repeats, strict=True)
        ):
            none_repeat = _require_mapping(
                none_repeat, f"NONE {case_name}.repeats[{repeat_index}]"
            )
            full_repeat = _require_mapping(
                full_repeat, f"FULL {case_name}.repeats[{repeat_index}]"
            )
            if none_repeat.get("sampling_seed") != full_repeat.get(
                "sampling_seed"
            ):
                raise ValueError(
                    f"{case_name} repeat {repeat_index} sampling seeds differ"
                )
            none_hashes = none_repeat["completion_token_sha256"]
            full_hashes = full_repeat["completion_token_sha256"]
            case_hash_total += 1
            hash_total += 1
            if none_hashes == full_hashes:
                case_hash_matched += 1
                hash_matched += 1
            else:
                mismatch = {
                    "case": case_name,
                    "repeat": repeat_index,
                    "mismatched_request_indices": [
                        request_index
                        for request_index, (
                            none_hash,
                            full_hash,
                        ) in enumerate(
                            zip(none_hashes, full_hashes, strict=True)
                        )
                        if none_hash != full_hash
                    ],
                }
                case_hash_mismatches.append(mismatch)
                hash_mismatches.append(mismatch)

        none_metrics = {
            **_case_metrics(none_case, f"NONE {case_name}"),
            "capture": none_capture,
        }
        full_metrics = {
            **_case_metrics(full_case, f"FULL {case_name}"),
            "capture": full_capture,
        }
        rows.append(
            {
                "name": case_name,
                "batch_size": key[0],
                "kv_length": key[1],
                "completion_token_sha256_match": {
                    "matched": case_hash_matched,
                    "total": case_hash_total,
                    "rate": case_hash_matched / case_hash_total,
                    "mismatches": case_hash_mismatches,
                },
                "none": none_metrics,
                "full_decode_only": full_metrics,
                "delta_full_minus_none": {
                    "latency_median_wall_percent": _relative_delta_percent(
                        full_metrics["latency_median_wall_ms"],
                        none_metrics["latency_median_wall_ms"],
                    ),
                    "tpot_wall_percent": _relative_delta_percent(
                        full_metrics["tpot_wall_ms"],
                        none_metrics["tpot_wall_ms"],
                    ),
                    "output_tokens_per_second_wall_percent": (
                        _relative_delta_percent(
                            full_metrics["output_tokens_per_second_wall"],
                            none_metrics["output_tokens_per_second_wall"],
                        )
                    ),
                    "graph_replay_hit_rate_percentage_points": (
                        (
                            full_metrics["graph_replay_hit_rate"]
                            - none_metrics["graph_replay_hit_rate"]
                        )
                        * 100.0
                    ),
                    "capture": capture_delta,
                },
            }
        )

    comparison: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": COMPARISON_BENCHMARK_NAME,
        "status": "complete",
        "generated_at_utc": _utc_now(),
        "sources": {
            "none": none_source,
            "full_decode_only": full_source,
        },
        "validation": {
            "status_complete": True,
            "formal_matrix": True,
            "same_model_identity_or_path": True,
            "same_git_commit": True,
            "same_benchmark_script_sha256": True,
            "same_seed_and_config_except_cudagraph_mode": True,
            "same_case_matrix": True,
            "same_gpu_and_software_stack": True,
            "raw_steps_recomputed": True,
            "summary_recomputed_from_raw_steps": True,
            "completion_token_sha256_compared": True,
            "same_completion_token_sha256_per_case_repeat": (
                hash_matched == hash_total
            ),
        },
        "provenance": provenance,
        "config": none_config,
        "capture": {
            "none": none_capture,
            "full_decode_only": full_capture,
            "delta_full_minus_none": capture_delta,
        },
        "completion_token_sha256_comparison": {
            "unit": "case_repeat",
            "matched": hash_matched,
            "total": hash_total,
            "rate": hash_matched / hash_total,
            "mismatches": hash_mismatches,
            "correctness_gate": False,
            "correctness_basis": (
                "CUDA Graph correctness is covered by hidden-state/logit "
                "tolerance GPU tests; autoregressive completion hashes are "
                "diagnostic because tolerance-level logit differences can "
                "cross sampling boundaries and accumulate"
            ),
        },
        "delta_definition": (
            "relative deltas are (FULL - NONE) / NONE * 100; replay delta "
            "is FULL - NONE in percentage points; capture deltas are absolute"
        ),
        "cases": rows,
    }
    comparison["readme_markdown"] = _format_comparison_markdown(comparison)
    return comparison


def _format_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _format_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _format_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "| Case | Hash repeats | NONE latency median (ms) | "
        "FULL latency median (ms) | "
        "Delta latency (%) | NONE TPOT (ms) | FULL TPOT (ms) | "
        "Delta TPOT (%) | NONE tok/s | FULL tok/s | Delta tok/s (%) | "
        "NONE replay (%) | FULL replay (%) | Delta replay (pp) | "
        "NONE capture (ms / MiB) | FULL capture (ms / MiB) | "
        "Delta capture (ms / MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---:|",
    ]
    for row in comparison["cases"]:
        none = row["none"]
        full = row["full_decode_only"]
        delta = row["delta_full_minus_none"]
        none_capture = none["capture"]
        full_capture = full["capture"]
        capture_delta = delta["capture"]
        hash_match = row["completion_token_sha256_match"]
        lines.append(
            "| {name} | {hash_match} | {none_latency} | {full_latency} | "
            "{latency_delta} | {none_tpot} | {full_tpot} | {tpot_delta} | "
            "{none_tps} | {full_tps} | {tps_delta} | {none_replay} | "
            "{full_replay} | {replay_delta} | {none_capture} | "
            "{full_capture} | {capture_delta} |".format(
                name=row["name"],
                hash_match=(
                    f"{hash_match['matched']}/{hash_match['total']} "
                    f"({_format_number(hash_match['rate'] * 100.0, 1)}%)"
                ),
                none_latency=_format_number(
                    none["latency_median_wall_ms"]
                ),
                full_latency=_format_number(
                    full["latency_median_wall_ms"]
                ),
                latency_delta=_format_signed(
                    delta["latency_median_wall_percent"]
                ),
                none_tpot=_format_number(none["tpot_wall_ms"]),
                full_tpot=_format_number(full["tpot_wall_ms"]),
                tpot_delta=_format_signed(delta["tpot_wall_percent"]),
                none_tps=_format_number(
                    none["output_tokens_per_second_wall"], 1
                ),
                full_tps=_format_number(
                    full["output_tokens_per_second_wall"], 1
                ),
                tps_delta=_format_signed(
                    delta["output_tokens_per_second_wall_percent"]
                ),
                none_replay=_format_number(
                    none["graph_replay_hit_rate"] * 100.0, 1
                ),
                full_replay=_format_number(
                    full["graph_replay_hit_rate"] * 100.0, 1
                ),
                replay_delta=_format_signed(
                    delta["graph_replay_hit_rate_percentage_points"], 1
                ),
                none_capture=(
                    f"{_format_number(none_capture['time_ms'])} / "
                    f"{_format_number(none_capture['extra_memory_mib'], 2)}"
                ),
                full_capture=(
                    f"{_format_number(full_capture['time_ms'])} / "
                    f"{_format_number(full_capture['extra_memory_mib'], 2)}"
                ),
                capture_delta=(
                    f"{_format_signed(capture_delta['time_ms'], 3)} / "
                    f"{_format_signed(capture_delta['extra_memory_mib'], 2)}"
                ),
            )
        )
    hash_summary = comparison["completion_token_sha256_comparison"]
    mismatch_locations = ", ".join(
        (
            f"{mismatch['case']} repeat {mismatch['repeat']} "
            f"(requests {mismatch['mismatched_request_indices']})"
        )
        for mismatch in hash_summary["mismatches"]
    )
    lines.extend(
        [
            "",
            (
                "Completion token SHA-256 diagnostic match: "
                f"{hash_summary['matched']}/{hash_summary['total']} "
                f"({_format_number(hash_summary['rate'] * 100.0, 2)}%)."
            ),
            (
                "Mismatched case/repeats: "
                f"{mismatch_locations or 'none'}."
            ),
            (
                "Completion-token SHA-256 equality is diagnostic only, not a "
                "correctness gate; CUDA Graph correctness is covered by the "
                "hidden-state/logit tolerance GPU tests."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _run_compare(argv: TypingSequence[str]) -> None:
    args = _parse_compare_args(argv)
    none_payload = _read_json_object(args.none)
    full_payload = _read_json_object(args.full)
    comparison = _compare_formal_results(
        none_payload,
        full_payload,
        none_source=str(args.none),
        full_source=str(args.full),
    )
    if args.output is not None:
        if args.output.suffix.lower() != ".json":
            raise ValueError("--output must name a .json file")
        _write_json(args.output, comparison)
    markdown = comparison["readme_markdown"]
    if args.markdown_output is not None:
        if args.markdown_output.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError("--markdown-output must name a Markdown file")
        _write_text(args.markdown_output, markdown)
    print(markdown, end="")


def _percentile(values: TypingSequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(values: TypingSequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": float(_percentile(values, 95.0)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _runtime_counter_view(stats: dict[str, Any]) -> dict[str, int]:
    counters: dict[str, int] = {}
    for name in RUNTIME_COUNTERS:
        value = stats.get(name)
        if type(value) is not int or value < 0:
            raise AssertionError(f"invalid CUDA Graph counter {name}={value!r}")
        counters[name] = value
    return counters


def _validate_measured_stats(
    mode: str,
    expected_decode_steps: int,
    stats: dict[str, Any],
) -> dict[str, int]:
    counters = _runtime_counter_view(stats)
    if mode == CUDAGraphPolicy.NONE.value:
        if any(counters.values()):
            raise AssertionError(
                "NONE must leave all CUDA Graph runtime counters at zero; "
                f"got {counters}"
            )
    elif mode == CUDAGraphPolicy.FULL_DECODE_ONLY.value:
        expected = {
            "full_graph_replay_steps": expected_decode_steps,
            "eager_fallback_steps": 0,
            "graph_bucket_hits": expected_decode_steps,
            "graph_bucket_misses": 0,
        }
        if counters != expected:
            raise AssertionError(
                "FULL_DECODE_ONLY measured phase was not an exact-bucket "
                f"pure-decode replay: expected {expected}, got {counters}"
            )
    else:
        raise ValueError(f"unknown mode: {mode}")
    return counters


def _validate_excluded_prefill_stats(
    mode: str,
    stats: dict[str, Any],
) -> dict[str, int]:
    counters = _runtime_counter_view(stats)
    expected = {name: 0 for name in RUNTIME_COUNTERS}
    if mode == CUDAGraphPolicy.FULL_DECODE_ONLY.value:
        expected["eager_fallback_steps"] = 1
    elif mode != CUDAGraphPolicy.NONE.value:
        raise ValueError(f"unknown mode: {mode}")
    if counters != expected:
        raise AssertionError(
            "excluded follower prefill did not use the expected eager path: "
            f"expected {expected}, got {counters}"
        )
    return counters


def _sum_counters(rows: TypingSequence[dict[str, int]]) -> dict[str, int]:
    return {
        name: sum(row[name] for row in rows)
        for name in RUNTIME_COUNTERS
    }


def _case_summary(
    mode: str,
    batch_size: int,
    repeats: TypingSequence[dict[str, Any]],
) -> dict[str, Any]:
    wall_samples = [
        step["wall_ms"]
        for repeat in repeats
        for step in repeat["steps"]
    ]
    cuda_samples = [
        step["cuda_event_ms"]
        for repeat in repeats
        for step in repeat["steps"]
    ]
    counter_total = _sum_counters(
        [repeat["runtime_counters"] for repeat in repeats]
    )
    measured_steps = len(wall_samples)
    measured_output_tokens = measured_steps * batch_size
    elapsed_wall_ms = sum(wall_samples)
    elapsed_cuda_ms = sum(cuda_samples)
    replay_hit_rate = (
        counter_total["full_graph_replay_steps"] / measured_steps
        if measured_steps
        else 0.0
    )
    bucket_decisions = (
        counter_total["graph_bucket_hits"]
        + counter_total["graph_bucket_misses"]
    )
    bucket_hit_rate = (
        counter_total["graph_bucket_hits"] / bucket_decisions
        if bucket_decisions
        else None
    )
    summary = {
        "measured_repeats": len(repeats),
        "measured_decode_steps": measured_steps,
        "measured_output_tokens": measured_output_tokens,
        "graph_replay_hit_rate": replay_hit_rate,
        "graph_bucket_hit_rate": bucket_hit_rate,
        "decode_step_latency_wall_ms": _latency_summary(wall_samples),
        "decode_step_latency_cuda_event_ms": _latency_summary(cuda_samples),
        "tpot_wall_ms": elapsed_wall_ms / measured_steps,
        "tpot_cuda_event_ms": elapsed_cuda_ms / measured_steps,
        "output_tokens_per_second_wall": (
            measured_output_tokens / (elapsed_wall_ms / 1000.0)
        ),
        "output_tokens_per_second_cuda_event": (
            measured_output_tokens / (elapsed_cuda_ms / 1000.0)
        ),
        "runtime_counters": counter_total,
    }
    _validate_measured_stats(mode, measured_steps, counter_total)
    return summary


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sampling_params(
    output_tokens: int,
    temperature: float,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        max_tokens=output_tokens,
        ignore_eos=True,
    )


def _add_followers(
    llm: LLM,
    prompt_token_ids: list[int],
    batch_size: int,
    output_tokens: int,
    temperature: float,
) -> tuple[list[Any], list[int]]:
    if not llm.scheduler.is_finished():
        raise AssertionError("a follower batch must start with an idle scheduler")
    sequences = []
    prefix_hits = []
    for _ in range(batch_size):
        llm.add_request(
            prompt_token_ids,
            _sampling_params(output_tokens, temperature),
        )
        sequence = llm.scheduler.waiting[-1]
        sequences.append(sequence)
        prefix_hits.append(
            len(llm.scheduler.block_manager.match_prefix(sequence))
            * BLOCK_SIZE
        )
    return sequences, prefix_hits


def _run_to_completion(llm: LLM) -> None:
    while not llm.is_finished():
        llm.step()


def _prime_prefix(
    llm: LLM,
    prompt_token_ids: list[int],
    temperature: float,
) -> None:
    llm.add_request(
        prompt_token_ids,
        _sampling_params(output_tokens=1, temperature=temperature),
    )
    _run_to_completion(llm)
    if not llm.scheduler.is_finished():
        raise AssertionError("prefix primer did not drain the scheduler")


def _run_follower_prefill(
    llm: LLM,
    sequences: TypingSequence[Any],
) -> dict[str, Any]:
    outputs, _ = llm.step()
    if outputs:
        raise AssertionError("followers completed during the excluded prefill")
    if llm.scheduler.num_scheduled_prefill_seqs != len(sequences):
        raise AssertionError(
            "follower prefill was not admitted as one exact batch: "
            f"expected {len(sequences)}, got "
            f"{llm.scheduler.num_scheduled_prefill_seqs}"
        )
    if llm.scheduler.waiting or llm.scheduler.chunked_req is not None:
        raise AssertionError("follower prefill unexpectedly chunked or queued")
    if len(llm.scheduler.running) != len(sequences):
        raise AssertionError("not all followers entered decode together")
    if any(sequence.num_completion_tokens != 1 for sequence in sequences):
        raise AssertionError("prefill must produce exactly one excluded token")
    return llm.model_runner.call("get_cudagraph_stats")


def _run_unmeasured_warmup(
    llm: LLM,
    args: argparse.Namespace,
    spec: CaseSpec,
    prompt_token_ids: list[int],
    expected_prefix_hit_tokens: int,
    seed: int,
) -> dict[str, Any]:
    _seed_all(seed)
    sequences, prefix_hits = _add_followers(
        llm,
        prompt_token_ids,
        spec.batch_size,
        args.warmup_decode_steps + 1,
        args.temperature,
    )
    if prefix_hits != [expected_prefix_hit_tokens] * spec.batch_size:
        raise AssertionError(
            "warm-up followers did not share the primed prefix: "
            f"expected {expected_prefix_hit_tokens}, got {prefix_hits}"
        )
    llm.model_runner.call("reset_cudagraph_stats")
    prefill_stats = _validate_excluded_prefill_stats(
        args.mode,
        _run_follower_prefill(llm, sequences),
    )
    llm.model_runner.call("reset_cudagraph_stats")
    _run_to_completion(llm)
    stats = llm.model_runner.call("get_cudagraph_stats")
    counters = _validate_measured_stats(
        args.mode,
        args.warmup_decode_steps,
        stats,
    )
    return {
        "decode_steps": args.warmup_decode_steps,
        "prefix_hit_tokens_per_request": prefix_hits,
        "excluded_prefill_runtime_counters": prefill_stats,
        "runtime_counters": counters,
    }


def _run_measured_repeat(
    llm: LLM,
    args: argparse.Namespace,
    spec: CaseSpec,
    prompt_token_ids: list[int],
    expected_prefix_hit_tokens: int,
    repeat_index: int,
    seed: int,
) -> dict[str, Any]:
    _seed_all(seed)
    sequences, prefix_hits = _add_followers(
        llm,
        prompt_token_ids,
        spec.batch_size,
        args.decode_steps + 1,
        args.temperature,
    )
    if prefix_hits != [expected_prefix_hit_tokens] * spec.batch_size:
        raise AssertionError(
            "measured followers did not share the primed prefix: "
            f"expected {expected_prefix_hit_tokens}, got {prefix_hits}"
        )
    llm.model_runner.call("reset_cudagraph_stats")
    prefill_stats = _validate_excluded_prefill_stats(
        args.mode,
        _run_follower_prefill(llm, sequences),
    )
    llm.model_runner.call("reset_cudagraph_stats")
    torch.cuda.synchronize()

    raw_steps: list[dict[str, Any]] = []
    timing_events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(args.decode_steps)
    ]
    for step_index in range(args.decode_steps):
        context_lengths = sorted({len(sequence) for sequence in sequences})
        expected_context_length = spec.kv_length + step_index
        if context_lengths != [expected_context_length]:
            raise AssertionError(
                "unexpected decode KV lengths: "
                f"expected {[expected_context_length]}, got {context_lengths}"
            )
        if len(llm.scheduler.running) != spec.batch_size:
            raise AssertionError("decode batch size changed before the timed step")

        start_event, end_event = timing_events[step_index]
        start_event.record()
        wall_start = perf_counter()
        outputs, _ = llm.step()
        end_event.record()
        end_event.synchronize()
        wall_ms = (perf_counter() - wall_start) * 1000.0
        cuda_event_ms = float(start_event.elapsed_time(end_event))

        is_final_step = step_index == args.decode_steps - 1
        if bool(outputs) != is_final_step:
            raise AssertionError(
                "followers must finish together on the final measured step"
            )
        if llm.scheduler.num_scheduled_prefill_seqs != 0:
            raise AssertionError("timed region included a prefill sequence")
        counters_after = _runtime_counter_view(
            llm.model_runner.call("get_cudagraph_stats")
        )
        _validate_measured_stats(
            args.mode,
            step_index + 1,
            counters_after,
        )
        raw_steps.append(
            {
                "step": step_index,
                "batch_size": spec.batch_size,
                "kv_length": expected_context_length,
                "wall_ms": wall_ms,
                "cuda_event_ms": cuda_event_ms,
                "runtime_counters_after_step": counters_after,
            }
        )

    if not llm.scheduler.is_finished():
        raise AssertionError("measured follower batch did not drain")
    if any(
        sequence.num_completion_tokens != args.decode_steps + 1
        for sequence in sequences
    ):
        raise AssertionError("a follower produced an unexpected token count")

    stats = llm.model_runner.call("get_cudagraph_stats")
    counters = _validate_measured_stats(args.mode, args.decode_steps, stats)
    wall_total_ms = sum(step["wall_ms"] for step in raw_steps)
    cuda_total_ms = sum(step["cuda_event_ms"] for step in raw_steps)
    measured_output_tokens = spec.batch_size * args.decode_steps
    return {
        "repeat": repeat_index,
        "sampling_seed": seed,
        "prefix_hit_tokens_per_request": prefix_hits,
        "excluded_prefill_runtime_counters": prefill_stats,
        "runtime_counters": counters,
        "measured_decode_steps": args.decode_steps,
        "measured_output_tokens": measured_output_tokens,
        "wall_elapsed_ms": wall_total_ms,
        "cuda_event_elapsed_ms": cuda_total_ms,
        "tpot_wall_ms": wall_total_ms / args.decode_steps,
        "tpot_cuda_event_ms": cuda_total_ms / args.decode_steps,
        "output_tokens_per_second_wall": (
            measured_output_tokens / (wall_total_ms / 1000.0)
        ),
        "output_tokens_per_second_cuda_event": (
            measured_output_tokens / (cuda_total_ms / 1000.0)
        ),
        "completion_token_sha256": [
            _token_sha256(sequence.completion_token_ids)
            for sequence in sequences
        ],
        "steps": raw_steps,
    }


def _case_seed(base_seed: int, spec: CaseSpec, offset: int) -> int:
    return base_seed + spec.batch_size * 100_003 + spec.kv_length * 101 + offset


def _run_case(
    llm: LLM,
    args: argparse.Namespace,
    spec: CaseSpec,
    forbidden_token_ids: set[int],
) -> dict[str, Any]:
    prompt_seed = _case_seed(args.seed, spec, 0)
    prompt = TokenFactory(
        llm.config.hf_config.vocab_size,
        prompt_seed,
        forbidden_token_ids,
    ).tokens(spec.prompt_length)
    _seed_all(_case_seed(args.seed, spec, 1))
    _prime_prefix(llm, prompt, args.temperature)

    expected_prefix_hit_tokens = spec.kv_length - BLOCK_SIZE
    warmup = _run_unmeasured_warmup(
        llm,
        args,
        spec,
        prompt,
        expected_prefix_hit_tokens,
        _case_seed(args.seed, spec, 2),
    )
    repeats = [
        _run_measured_repeat(
            llm,
            args,
            spec,
            prompt,
            expected_prefix_hit_tokens,
            repeat_index,
            _case_seed(args.seed, spec, 100 + repeat_index),
        )
        for repeat_index in range(args.repeats)
    ]
    return {
        "name": spec.name,
        "case": {
            **asdict(spec),
            "prompt_length": spec.prompt_length,
            "first_measured_decode_kv_length": spec.kv_length,
            "last_measured_decode_kv_length": (
                spec.kv_length + args.decode_steps - 1
            ),
            "decode_steps_per_repeat": args.decode_steps,
            "measured_output_tokens_per_repeat": (
                spec.batch_size * args.decode_steps
            ),
        },
        "shared_prefix": {
            "prompt_seed": prompt_seed,
            "prompt_sha256": _token_sha256(prompt),
            "prompt_token_ids_head": prompt[:8],
            "prompt_token_ids_tail": prompt[-8:],
            "expected_persistent_hit_tokens_per_follower": (
                expected_prefix_hit_tokens
            ),
            "physical_kv_strategy": (
                "all followers reference one persistent full-block prefix; "
                "only the final prompt page and decode tail are per request"
            ),
        },
        "warmup": warmup,
        "repeats": repeats,
        "summary": _case_summary(args.mode, spec.batch_size, repeats),
        "invariants": {
            "prefill_excluded_from_timing": True,
            "timed_batch_type": "pure_decode",
            "exact_batch_bucket": spec.batch_size,
            "runtime_padding_requests": 0,
            "shared_prefix_hit_verified_for_every_follower": True,
        },
    }


def _required_resident_blocks(
    spec: CaseSpec,
    decode_steps: int,
) -> int:
    shared_blocks = spec.kv_length // BLOCK_SIZE - 1
    final_tokens = spec.kv_length + decode_steps
    blocks_per_request = (final_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    return shared_blocks + spec.batch_size * (
        blocks_per_request - shared_blocks
    )


def _validate_runtime(
    llm: LLM,
    args: argparse.Namespace,
    batch_sizes: tuple[int, ...],
    cases: TypingSequence[CaseSpec],
) -> dict[str, Any]:
    config = llm.config
    runner = llm.model_runner
    policy = CUDAGraphPolicy(args.mode)
    if config.hf_config.model_type != "qwen3":
        raise ValueError("bench_cudagraph.py requires a Qwen3 model")
    if runner.dtype != torch.bfloat16:
        raise ValueError(
            "bench_cudagraph.py requires BF16 weights; "
            f"resolved dtype is {runner.dtype}"
        )
    expected = {
        "policy": policy,
        "attention_backend": "flashinfer",
        "attention_mode": "unified",
        "tensor_parallel_size": 1,
        "block_size": BLOCK_SIZE,
        "chunked_prefill": True,
    }
    actual = {
        "policy": config.cudagraph_mode,
        "attention_backend": config.attention_backend,
        "attention_mode": config.attention_mode,
        "tensor_parallel_size": config.tensor_parallel_size,
        "block_size": config.kvcache_block_size,
        "chunked_prefill": config.chunked_prefill,
    }
    if actual != expected:
        raise AssertionError(f"runtime contract mismatch: {actual} != {expected}")

    startup_stats = runner.call("get_cudagraph_stats")
    if startup_stats.get("policy") != args.mode:
        raise AssertionError(
            "runtime reported an unexpected CUDA Graph policy: "
            f"expected {args.mode}, got {startup_stats.get('policy')!r}"
        )
    expected_captured = (
        list(batch_sizes)
        if policy is CUDAGraphPolicy.FULL_DECODE_ONLY
        else []
    )
    if startup_stats["captured_batch_sizes"] != expected_captured:
        raise AssertionError(
            "captured batch sizes do not equal the requested exact buckets: "
            f"expected {expected_captured}, got "
            f"{startup_stats['captured_batch_sizes']}"
        )
    _validate_measured_stats(args.mode, 0, startup_stats)
    capture_time_ms = startup_stats.get("capture_time_ms")
    if (
        isinstance(capture_time_ms, bool)
        or not isinstance(capture_time_ms, (int, float))
        or not math.isfinite(capture_time_ms)
        or capture_time_ms < 0
    ):
        raise AssertionError(
            f"invalid CUDA Graph capture time: {capture_time_ms!r}"
        )
    extra_memory_bytes = startup_stats.get("extra_memory_bytes")
    if type(extra_memory_bytes) is not int or extra_memory_bytes < 0:
        raise AssertionError(
            f"invalid CUDA Graph extra memory: {extra_memory_bytes!r}"
        )
    if policy is CUDAGraphPolicy.NONE and (
        capture_time_ms != 0 or extra_memory_bytes != 0
    ):
        raise AssertionError(
            "NONE unexpectedly allocated or captured CUDA Graph state: "
            f"capture_time_ms={capture_time_ms}, "
            f"extra_memory_bytes={extra_memory_bytes}"
        )

    required_blocks = max(
        _required_resident_blocks(spec, args.decode_steps)
        for spec in cases
    )
    if config.num_kvcache_blocks < required_blocks:
        raise ValueError(
            "physical KV cache cannot keep the largest shared-prefix case "
            f"resident: need {required_blocks} blocks, have "
            f"{config.num_kvcache_blocks}"
        )
    return {
        "model": os.path.realpath(config.model),
        "model_type": config.hf_config.model_type,
        "model_dtype": str(runner.dtype),
        "model_shape": {
            name: getattr(config.hf_config, name, None)
            for name in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
            )
        },
        "physical_kv_blocks": config.num_kvcache_blocks,
        "largest_case_required_resident_blocks": required_blocks,
        "cudagraph_startup": startup_stats,
    }


def _shutdown_llm(llm: LLM | None) -> None:
    if llm is not None:
        try:
            atexit.unregister(llm.exit)
        except Exception:
            pass
        try:
            llm.exit()
        except Exception as error:
            print(f"warning: LLM cleanup failed: {error}", file=sys.stderr)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run(
    args: argparse.Namespace,
    batch_sizes: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    cases: list[CaseSpec],
    payload: dict[str, Any],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("bench_cudagraph.py requires an NVIDIA CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "expose exactly one GPU per benchmark process with "
            "CUDA_VISIBLE_DEVICES; tensor parallelism is intentionally disabled"
        )

    _seed_all(args.seed)
    max_model_len = max(kv_lengths) + args.decode_steps + 1
    max_num_seqs = max(batch_sizes)
    max_num_batched_tokens = max_num_seqs * BLOCK_SIZE
    llm: LLM | None = None
    try:
        llm = LLM(
            args.model,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=(args.mode == CUDAGraphPolicy.NONE.value),
            cudagraph_mode=args.mode,
            cudagraph_batch_sizes=batch_sizes,
            kvcache_block_size=BLOCK_SIZE,
            chunked_prefill=True,
            enable_lpm=True,
            enable_in_batch_prefix_deprioritization=True,
            attention_backend="flashinfer",
            attention_mode="unified",
        )
        payload["environment"] = _environment_metadata()
        payload["runtime"] = _validate_runtime(
            llm,
            args,
            batch_sizes,
            cases,
        )
        payload["config"].update(
            {
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
            }
        )
        forbidden = {
            token_id
            for token_id in (
                llm.tokenizer.eos_token_id,
                llm.tokenizer.pad_token_id,
                llm.tokenizer.bos_token_id,
            )
            if token_id is not None
        }
        # Engine initialization samples during warmup.  Case-local seeds below
        # make formal NONE/FULL runs independent of that implementation detail.
        for case_index, spec in enumerate(cases):
            payload["active_case"] = spec.name
            _write_json(args.output, payload)
            print(
                f"[{case_index + 1}/{len(cases)}] {spec.name}: "
                f"mode={args.mode} repeats={args.repeats} "
                f"decode_steps={args.decode_steps}",
                flush=True,
            )
            result = _run_case(llm, args, spec, forbidden)
            payload["cases"].append(result)
            payload.pop("active_case", None)
            _write_json(args.output, payload)
            summary = result["summary"]
            print(
                f"  median={summary['decode_step_latency_wall_ms']['median']:.3f}ms "
                f"tpot={summary['tpot_wall_ms']:.3f}ms "
                f"throughput={summary['output_tokens_per_second_wall']:.1f}tok/s "
                f"replay_hit={summary['graph_replay_hit_rate']:.3f}",
                flush=True,
            )
    finally:
        _shutdown_llm(llm)


def main(argv: TypingSequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"compare", "summary"}:
        _run_compare(arguments[1:])
        return
    args = _parse_args(arguments)
    batch_sizes, kv_lengths, cases = _validate_args(args)
    formal_matrix = (
        batch_sizes == DEFAULT_BATCH_SIZES
        and kv_lengths == DEFAULT_KV_LENGTHS
        and args.decode_steps == FORMAL_DECODE_STEPS
        and args.warmup_decode_steps == FORMAL_WARMUP_DECODE_STEPS
        and args.repeats == FORMAL_REPEATS
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "status": "running",
        "started_at_utc": _utc_now(),
        "mode": args.mode,
        "protocol": {
            "formal_matrix": formal_matrix,
            "comparison_scope": (
                "runtime feature validation only; never a cross-framework "
                "CUDA Graph comparison"
            ),
            "process_isolation": "one policy and one LLM engine per process",
            "timed_region": (
                "exact-size pure-decode LLMEngine.step calls only; follower "
                "prefill and independent warm-up batch excluded"
            ),
            "primary_timing": (
                "wall clock around each synchronized decode step"
            ),
            "secondary_timing": "CUDA events around the same full step",
            "tpot_definition": (
                "elapsed pure-decode batch time / decode steps; one inter-token "
                "interval per request per step"
            ),
            "throughput_definition": (
                "batch_size * measured decode steps / elapsed seconds"
            ),
            "kv_length_definition": (
                "attention KV length at the first measured decode step; it "
                "increases by one on each subsequent step"
            ),
            "prefix_memory_strategy": (
                "one persistent block-aligned prefix shared by every follower; "
                "no synthetic graph padding"
            ),
            "raw_repeat_retention": True,
        },
        "config": {
            "seed": args.seed,
            "temperature": args.temperature,
            "ignore_eos": True,
            "dtype": "bfloat16",
            "block_size": BLOCK_SIZE,
            "attention_backend": "flashinfer",
            "attention_mode": "unified",
            "tensor_parallel_size": 1,
            "chunked_prefill": True,
            "prefix_cache": True,
            "cudagraph_mode": args.mode,
            "cudagraph_batch_sizes": list(batch_sizes),
            "batch_sizes": list(batch_sizes),
            "kv_lengths": list(kv_lengths),
            "decode_steps": args.decode_steps,
            "warmup_decode_steps": args.warmup_decode_steps,
            "repeats": args.repeats,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "cases": [],
    }
    _write_json(args.output, payload)
    started = perf_counter()
    try:
        _run(args, batch_sizes, kv_lengths, cases, payload)
        payload["status"] = "complete"
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "case": payload.get("active_case"),
        }
        raise
    finally:
        payload["finished_at_utc"] = _utc_now()
        payload["elapsed_seconds"] = perf_counter() - started
        _write_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
