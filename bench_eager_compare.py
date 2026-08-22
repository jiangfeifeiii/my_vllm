"""Reproducible eager-only nano-vllm versus vLLM benchmarks.

The two frameworks intentionally run in separate processes and consume the
same immutable request trace.  A typical workflow is::

    python bench_eager_compare.py generate-trace lpm \
        --model /workspace/aiinfra/models/Qwen3-0.6B \
        --output benchmark_results/eager_compare/lpm.trace.json

    python bench_eager_compare.py run-nano \
        --trace benchmark_results/eager_compare/lpm.trace.json \
        --output benchmark_results/eager_compare/lpm.nano.json

    /path/to/vllm/python bench_eager_compare.py run-vllm \
        --trace benchmark_results/eager_compare/lpm.trace.json \
        --output benchmark_results/eager_compare/lpm.vllm.json

    python bench_eager_compare.py compare \
        --trace benchmark_results/eager_compare/lpm.trace.json \
        --nano-result benchmark_results/eager_compare/lpm.nano.json \
        --vllm-result benchmark_results/eager_compare/lpm.vllm.json \
        --output benchmark_results/eager_compare/lpm.comparison.json

The LPM comparison reports five definition-aligned metrics: locally computed
prompt tokens, allocation-pressure cached-block evictions, P95 TTFT, request
throughput, and total batch completion time.  A benchmark-only read-only vLLM
scheduler observer supplies the two counters without changing FCFS ordering or
admission behavior.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Sequence as TypingSequence


SCHEMA_VERSION = 3
BLOCK_SIZE = 16
CROSS_FRAMEWORK_WORKLOAD_KEY = "lpm"
VLLM_SCHEDULER_CLASS = "bench_vllm_metrics.InstrumentedScheduler"
VLLM_METRICS_HELPER_PATH = Path(__file__).with_name("bench_vllm_metrics.py")
VLLM_NULL_BLOCK_COUNT = 1
COMPARABLE_METRIC_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("computed_prompt_tokens", "tokens", True),
    ("cached_block_eviction_count", "blocks", True),
    ("p95_ttft_ms", "ms", True),
    ("request_throughput_rps", "requests/s", False),
    ("total_batch_completion_s", "s", True),
)
COMPARABLE_METRIC_NAMES = tuple(
    name for name, _unit, _lower_is_better in COMPARABLE_METRIC_SPECS
)
TARGET_MODEL_SHAPE = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
}
WORKLOAD_CONFIGS: dict[str, dict[str, Any]] = {
    "lpm": {
        "name": "shared_long_prefix_kv_pressure",
        "logical_kv_blocks": 896,
        "max_model_len": 4352,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 4,
        "prefix_groups": 3,
        "shared_prefix_len": 4096,
    },
    "in-batch": {
        "name": "in_batch_prefix_burst",
        "logical_kv_blocks": 2240,
        "max_model_len": 2304,
        "max_num_batched_tokens": 10240,
        "max_num_seqs": 16,
        "prefix_groups": 4,
        "shared_prefix_len": 2048,
    },
}


def _expected_workload_contract(workload_key: str) -> dict[str, Any]:
    profile = WORKLOAD_CONFIGS[workload_key]
    contract: dict[str, Any] = {
        "prefix_groups": profile["prefix_groups"],
        "shared_prefix_len": profile["shared_prefix_len"],
    }
    if workload_key == "in-batch":
        requests_per_group = 4
        unique_suffix_len = 128
        output_len = 64
        request_count = profile["prefix_groups"] * requests_per_group
        same_step_minimum_blocks = (
            profile["prefix_groups"]
            * (profile["shared_prefix_len"] // BLOCK_SIZE)
            + request_count
            * ((unique_suffix_len + output_len) // BLOCK_SIZE)
        )
        cold_worst_case_blocks = request_count * (
            (profile["shared_prefix_len"] + unique_suffix_len + output_len)
            // BLOCK_SIZE
        )
        if same_step_minimum_blocks != 704:
            raise AssertionError(
                "in-batch same-step KV capacity contract drifted: "
                f"expected 704, got {same_step_minimum_blocks}"
            )
        if cold_worst_case_blocks != 2240:
            raise AssertionError(
                "in-batch cold KV capacity contract drifted: "
                f"expected 2240, got {cold_worst_case_blocks}"
            )
        if profile["logical_kv_blocks"] < cold_worst_case_blocks:
            raise AssertionError(
                "in-batch logical KV blocks do not cover sixteen cold "
                "requests without preemption"
            )
        contract.update(
            {
                "requests_per_group": requests_per_group,
                "unique_suffix_len": unique_suffix_len,
                "output_len": output_len,
                "arrival_layout": "grouped_by_prefix",
                "expected_off_first_step_admissions": 4,
                "expected_on_first_step_admissions": request_count,
                "same_step_minimum_blocks": same_step_minimum_blocks,
                "cold_worst_case_blocks": cold_worst_case_blocks,
                "comparison_logical_kv_blocks": profile["logical_kv_blocks"],
            }
        )
    return contract


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _command_output(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        resolved = resolved.parent

    def git(*arguments: str) -> str | None:
        return _command_output(["git", "-C", str(resolved), *arguments])

    root = git("rev-parse", "--show-toplevel")
    status = git("status", "--porcelain")
    return {
        "root": root,
        "head": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": None if status is None else status.splitlines(),
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _percentile(values: TypingSequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _prompt_sha256(token_ids: list[int]) -> str:
    return _canonical_sha256(token_ids)


def _model_file_hashes(model_path: Path) -> dict[str, str]:
    identity_files = {
        model_path / name
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    }
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        identity_files.update(model_path.glob(pattern))
    return {
        candidate.name: _sha256_file(candidate)
        for candidate in sorted(identity_files)
        if candidate.is_file()
    }


def _validate_model_files(trace: dict[str, Any]) -> Path:
    """Prove that a runner is loading the artifact recorded in the trace."""
    model_path = Path(trace["model"]["path"]).resolve()
    if not model_path.is_dir():
        raise ValueError(f"trace model path is not a local directory: {model_path}")
    expected = trace["model"]["files_sha256"]
    actual = _model_file_hashes(model_path)
    if actual != expected:
        raise ValueError(
            "model/tokenizer files changed after trace generation; "
            f"expected {expected}, got {actual}"
        )
    return model_path


def _load_model_identity(model: str) -> tuple[dict[str, Any], int, set[int]]:
    from transformers import AutoConfig, AutoTokenizer

    model_path = Path(model).resolve()
    if not model_path.is_dir():
        raise ValueError(f"model must be a local directory, got {model}")
    config = AutoConfig.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    shape = {
        name: getattr(config, name, None)
        for name in TARGET_MODEL_SHAPE
    }
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("eager comparison traces require a Qwen3 model")
    if shape != TARGET_MODEL_SHAPE:
        raise ValueError(
            "targeted eager comparisons require Qwen3-0.6B shape; "
            f"expected {TARGET_MODEL_SHAPE}, got {shape}"
        )
    vocab_size = int(config.vocab_size)
    special_ids = {
        int(token_id)
        for token_id in (
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
        )
        if token_id is not None
    }
    identity = {
        "path": str(model_path),
        "model_type": config.model_type,
        "shape": shape,
        "vocab_size": vocab_size,
        "configured_dtype": str(getattr(config, "torch_dtype", None)),
        "tokenizer_class": type(tokenizer).__name__,
        "special_token_ids": sorted(special_ids),
        "files_sha256": _model_file_hashes(model_path),
    }
    return identity, vocab_size, special_ids


def _request_from_spec(spec: Any, arrival_order: int) -> dict[str, Any]:
    token_ids = list(spec.prompt_token_ids)
    return {
        "request_id": spec.name,
        "input_token_ids": token_ids,
        "prompt_len": len(token_ids),
        "output_len": spec.output_len,
        "arrival_order": arrival_order,
        "arrival_time_ms": 0.0,
        "prefix_group": spec.group,
        "kind": spec.kind,
        "shared_prefix_len": spec.shared_prefix_len,
    }


def _build_trace_data(
    workload: str,
    model_identity: dict[str, Any],
    vocab_size: int,
    forbidden_token_ids: set[int],
    seed: int,
) -> dict[str, Any]:
    # Importing these definitions guarantees that cross-framework workloads do
    # not silently drift from the already validated scheduler benchmarks.
    from bench_scheduler import (
        TokenFactory,
        _build_in_batch_workload,
        _build_lpm_workload,
    )

    if workload not in WORKLOAD_CONFIGS:
        raise ValueError(f"unknown workload: {workload}")
    factory = TokenFactory(vocab_size, seed, forbidden_token_ids)
    if workload == "lpm":
        leaders, measured = _build_lpm_workload(factory)
        phases = [
            {
                "name": "prefix_priming",
                "measured": False,
                "submission": "sequential_to_completion",
                "requests": [
                    _request_from_spec(spec, index)
                    for index, spec in enumerate(leaders)
                ],
            },
            {
                "name": "measured",
                "measured": True,
                "submission": "simultaneous_batch",
                "requests": [
                    _request_from_spec(spec, index)
                    for index, spec in enumerate(measured)
                ],
            },
        ]
    else:
        measured = _build_in_batch_workload(factory)
        phases = [
            {
                "name": "measured",
                "measured": True,
                "submission": "simultaneous_batch",
                "requests": [
                    _request_from_spec(spec, index)
                    for index, spec in enumerate(measured)
                ],
            }
        ]

    profile = WORKLOAD_CONFIGS[workload]
    trace = {
        "schema_version": SCHEMA_VERSION,
        "trace_kind": "eager_only_cross_framework",
        "workload": profile["name"],
        "workload_key": workload,
        "seed": seed,
        "model": model_identity,
        "execution_contract": {
            "dtype": "bfloat16",
            "temperature": 1.0,
            "ignore_eos": True,
            "prefix_caching": True,
            "chunked_prefill": True,
            "block_size": BLOCK_SIZE,
            "logical_kv_blocks": profile["logical_kv_blocks"],
            "max_model_len": profile["max_model_len"],
            "max_num_batched_tokens": profile["max_num_batched_tokens"],
            "max_num_seqs": profile["max_num_seqs"],
            "tensor_parallel_size": 1,
            "cuda_graphs": "none",
        },
        "workload_contract": _expected_workload_contract(workload),
        "phases": phases,
    }
    _validate_trace(trace)
    return trace


def _iter_requests(trace: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for phase in trace["phases"]:
        yield from phase["requests"]


def _measured_phase(trace: dict[str, Any]) -> dict[str, Any]:
    measured = [phase for phase in trace["phases"] if phase["measured"]]
    if len(measured) != 1:
        raise ValueError("trace must contain exactly one measured phase")
    return measured[0]


def _priming_phases(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [phase for phase in trace["phases"] if not phase["measured"]]


def _validate_request(
    request: dict[str, Any],
    expected_order: int,
    vocab_size: int,
) -> None:
    required = {
        "request_id",
        "input_token_ids",
        "prompt_len",
        "output_len",
        "arrival_order",
        "arrival_time_ms",
        "prefix_group",
        "kind",
        "shared_prefix_len",
    }
    missing = required - request.keys()
    if missing:
        raise ValueError(f"request is missing fields: {sorted(missing)}")
    token_ids = request["input_token_ids"]
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("input_token_ids must be a non-empty list")
    if request["prompt_len"] != len(token_ids):
        raise ValueError(
            f"request {request['request_id']} prompt_len does not match tokens"
        )
    if request["arrival_order"] != expected_order:
        raise ValueError("arrival_order must be contiguous within each phase")
    if request["arrival_time_ms"] != 0.0:
        raise ValueError("target workloads require simultaneous zero-time arrivals")
    if type(request["output_len"]) is not int or request["output_len"] <= 0:
        raise ValueError("output_len must be a positive integer")
    if type(request["shared_prefix_len"]) is not int:
        raise ValueError("shared_prefix_len must be an integer")
    if not 0 <= request["shared_prefix_len"] <= request["prompt_len"]:
        raise ValueError("shared_prefix_len is outside the prompt")
    if any(
        type(token_id) is not int or not 0 <= token_id < vocab_size
        for token_id in token_ids
    ):
        raise ValueError("input_token_ids contain an invalid vocabulary ID")


def _validate_prefix_groups(trace: dict[str, Any]) -> None:
    prefixes: dict[str, list[int]] = {}
    for request in _iter_requests(trace):
        group = request["prefix_group"]
        shared_len = request["shared_prefix_len"]
        if group is None:
            if shared_len != 0:
                raise ValueError("ungrouped requests cannot declare a shared prefix")
            continue
        prefix = request["input_token_ids"][:shared_len]
        if group in prefixes and prefixes[group] != prefix:
            raise ValueError(f"prefix group {group} does not share identical tokens")
        prefixes.setdefault(group, prefix)
    fingerprints = {_prompt_sha256(prefix) for prefix in prefixes.values()}
    if len(fingerprints) != len(prefixes):
        raise ValueError("different prefix groups must use different tokens")


def _validate_target_shape(trace: dict[str, Any]) -> None:
    measured = _measured_phase(trace)["requests"]
    if trace["workload_key"] == "lpm":
        priming = _priming_phases(trace)
        if len(priming) != 1:
            raise ValueError("LPM trace requires one prefix-priming phase")
        if [phase["name"] for phase in trace["phases"]] != [
            "prefix_priming",
            "measured",
        ]:
            raise ValueError("LPM phase names or order changed")
        if priming[0]["submission"] != "sequential_to_completion":
            raise ValueError("LPM priming submission mode changed")
        if _measured_phase(trace)["submission"] != "simultaneous_batch":
            raise ValueError("LPM measured submission mode changed")
        if [row["request_id"] for row in priming[0]["requests"]] != [
            "A0",
            "B0",
            "C0",
        ]:
            raise ValueError("LPM priming order changed")
        if any(
            row["kind"] != "leader"
            or row["prefix_group"] not in ("A", "B", "C")
            or row["shared_prefix_len"] != 4096
            or row["prompt_len"] != 4224
            or row["output_len"] != 64
            for row in priming[0]["requests"]
        ):
            raise ValueError("LPM priming request shape changed")
        expected = [f"Cold{index}" for index in range(1, 13)]
        expected.extend(
            f"{group}{index}"
            for index in range(1, 5)
            for group in ("A", "B", "C")
        )
        if [row["request_id"] for row in measured] != expected:
            raise ValueError("LPM measured arrival order changed")
        if any(
            row["prompt_len"] != 4224 or row["output_len"] != 64
            for row in measured
            if row["kind"] == "follower"
        ):
            raise ValueError("LPM follower shape changed")
        cold = [row for row in measured if row["kind"] == "cold"]
        followers = [row for row in measured if row["kind"] == "follower"]
        if len(cold) != 12 or len(followers) != 12:
            raise ValueError("LPM request kinds changed")
        if any(
            row["prefix_group"] is not None
            or row["shared_prefix_len"] != 0
            or not 512 <= row["prompt_len"] <= 1024
            or row["output_len"] != 64
            for row in cold
        ):
            raise ValueError("LPM cold request shape changed")
        if any(
            row["prefix_group"] not in ("A", "B", "C")
            or row["shared_prefix_len"] != 4096
            for row in followers
        ):
            raise ValueError("LPM follower group metadata changed")
    elif trace["workload_key"] == "in-batch":
        if _priming_phases(trace):
            raise ValueError("in-batch trace must not have a priming phase")
        if len(trace["phases"]) != 1 or trace["phases"][0]["name"] != "measured":
            raise ValueError("in-batch phase layout changed")
        if trace["phases"][0]["submission"] != "simultaneous_batch":
            raise ValueError("in-batch submission mode changed")
        expected = [
            f"{group}{index}"
            for group in ("A", "B", "C", "D")
            for index in range(1, 5)
        ]
        if [row["request_id"] for row in measured] != expected:
            raise ValueError("in-batch measured arrival order changed")
        if any(
            row["kind"] != "burst"
            or row["prefix_group"] not in ("A", "B", "C", "D")
            or row["shared_prefix_len"] != 2048
            or row["prompt_len"] != 2176
            or row["output_len"] != 64
            for row in measured
        ):
            raise ValueError("in-batch request shape changed")
    else:
        raise ValueError(f"unknown workload_key: {trace['workload_key']}")


def _validate_trace(trace: dict[str, Any]) -> None:
    if trace.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported trace schema_version")
    if trace.get("trace_kind") != "eager_only_cross_framework":
        raise ValueError("not an eager-only comparison trace")
    workload_key = trace.get("workload_key")
    if workload_key not in WORKLOAD_CONFIGS:
        raise ValueError(f"unknown workload_key: {workload_key}")
    profile = WORKLOAD_CONFIGS[workload_key]
    if trace.get("workload") != profile["name"]:
        raise ValueError("workload name and key disagree")
    if type(trace.get("seed")) is not int:
        raise ValueError("trace seed must be an integer")
    model = trace.get("model", {})
    if model.get("model_type") != "qwen3":
        raise ValueError("trace model must be Qwen3")
    if model.get("shape") != TARGET_MODEL_SHAPE:
        raise ValueError("trace model must have Qwen3-0.6B shape")
    vocab_size = model.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("trace model vocab_size must be positive")
    model_path = model.get("path")
    if not isinstance(model_path, str) or not Path(model_path).is_absolute():
        raise ValueError("trace model path must be an absolute local path")
    file_hashes = model.get("files_sha256")
    if not isinstance(file_hashes, dict) or "config.json" not in file_hashes:
        raise ValueError("trace must identify the model config by SHA256")
    if not ({"tokenizer.json", "tokenizer_config.json"} & file_hashes.keys()):
        raise ValueError("trace must identify tokenizer files by SHA256")
    if not any(
        name.endswith(".safetensors") or name.endswith(".bin")
        for name in file_hashes
    ):
        raise ValueError("trace must identify model weights by SHA256")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in file_hashes.items()
    ):
        raise ValueError("trace model file hashes must be lowercase SHA256 values")
    contract = trace.get("execution_contract", {})
    expected_contract = {
        "dtype": "bfloat16",
        "temperature": 1.0,
        "ignore_eos": True,
        "prefix_caching": True,
        "chunked_prefill": True,
        "block_size": BLOCK_SIZE,
        "logical_kv_blocks": profile["logical_kv_blocks"],
        "max_model_len": profile["max_model_len"],
        "max_num_batched_tokens": profile["max_num_batched_tokens"],
        "max_num_seqs": profile["max_num_seqs"],
        "tensor_parallel_size": 1,
        "cuda_graphs": "none",
    }
    if contract != expected_contract:
        raise ValueError(
            "trace execution contract changed; expected "
            f"{expected_contract}, got {contract}"
        )
    workload_contract = trace.get("workload_contract", {})
    if workload_contract != _expected_workload_contract(workload_key):
        raise ValueError("trace workload contract changed")
    phases = trace.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("trace phases must be a non-empty list")
    seen_ids: set[str] = set()
    for phase in phases:
        if not isinstance(phase, dict):
            raise ValueError("every trace phase must be an object")
        required_phase_fields = {"name", "measured", "submission", "requests"}
        if required_phase_fields - phase.keys():
            raise ValueError("trace phase is missing required metadata")
        if not isinstance(phase["name"], str) or not phase["name"]:
            raise ValueError("trace phase name must be a non-empty string")
        if type(phase["measured"]) is not bool:
            raise ValueError("trace phase measured must be a boolean")
        if not isinstance(phase["submission"], str):
            raise ValueError("trace phase submission must be a string")
        requests = phase.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("every trace phase must contain requests")
        for index, request in enumerate(requests):
            _validate_request(request, index, vocab_size)
            request_id = request["request_id"]
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a non-empty string")
            if request_id in seen_ids:
                raise ValueError(f"duplicate request_id: {request_id}")
            seen_ids.add(request_id)
    _measured_phase(trace)
    _validate_prefix_groups(trace)
    _validate_target_shape(trace)


def _load_trace(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    trace = json.loads(raw)
    if not isinstance(trace, dict):
        raise ValueError("trace root must be a JSON object")
    _validate_trace(trace)
    return trace, _sha256_bytes(raw)


def _require_cross_framework_lpm(trace: dict[str, Any]) -> None:
    if trace.get("workload_key") != CROSS_FRAMEWORK_WORKLOAD_KEY:
        raise ValueError(
            "the five-metric cross-framework comparison is intentionally "
            "limited to the LPM workload; same-step prefix reuse remains an "
            "internal functional ablation"
        )


def _request_manifest(trace: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = []
    for phase in trace["phases"]:
        for request in phase["requests"]:
            manifest.append(
                {
                    "phase": phase["name"],
                    "measured": phase["measured"],
                    "request_id": request["request_id"],
                    "arrival_order": request["arrival_order"],
                    "arrival_time_ms": request["arrival_time_ms"],
                    "prompt_len": request["prompt_len"],
                    "prompt_sha256": _prompt_sha256(
                        request["input_token_ids"]
                    ),
                    "output_len": request["output_len"],
                    "prefix_group": request["prefix_group"],
                }
            )
    return manifest


def _nano_engine_kwargs(
    trace: dict[str, Any],
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    contract = trace["execution_contract"]
    return {
        "max_model_len": contract["max_model_len"],
        "max_num_batched_tokens": contract["max_num_batched_tokens"],
        "max_num_seqs": contract["max_num_seqs"],
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": contract["tensor_parallel_size"],
        "enforce_eager": True,
        "cudagraph_mode": "none",
        "kvcache_block_size": contract["block_size"],
        "chunked_prefill": True,
        "enable_lpm": True,
        "enable_same_step_prefix_reuse": True,
        "attention_backend": "flashinfer",
        "attention_mode": "unified",
    }


def _validate_nano_runtime_config(
    llm: Any,
    trace: dict[str, Any],
) -> tuple[dict[str, bool], str]:
    config = llm.config
    scheduler = llm.scheduler
    contract = trace["execution_contract"]
    mode = getattr(config.cudagraph_mode, "value", config.cudagraph_mode)
    actual_shape = {
        name: getattr(config.hf_config, name, None)
        for name in TARGET_MODEL_SHAPE
    }
    checks = {
        "model_path": (
            Path(config.model).resolve() == Path(trace["model"]["path"]).resolve()
        ),
        "model_shape": actual_shape == TARGET_MODEL_SHAPE,
        "dtype_bfloat16": str(llm.model_runner.dtype) == "torch.bfloat16",
        "enforce_eager": config.enforce_eager is True,
        "cudagraph_none": mode == "none",
        "prefix_caching": hasattr(scheduler.block_manager, "match_prefix"),
        "chunked_prefill": (
            config.chunked_prefill is True and scheduler.enable_chunked is True
        ),
        "block_size": (
            config.kvcache_block_size == contract["block_size"]
            and scheduler.block_manager.block_size == contract["block_size"]
        ),
        "max_model_len": config.max_model_len == contract["max_model_len"],
        "max_num_batched_tokens": (
            config.max_num_batched_tokens == contract["max_num_batched_tokens"]
            and scheduler.max_num_batched_tokens
            == contract["max_num_batched_tokens"]
        ),
        "max_num_seqs": (
            config.max_num_seqs == contract["max_num_seqs"]
            and scheduler.max_num_seqs == contract["max_num_seqs"]
        ),
        "tensor_parallel_size": (
            config.tensor_parallel_size == contract["tensor_parallel_size"]
        ),
        "cache_aware_scheduler": (
            scheduler.enable_lpm is True
            and scheduler.enable_same_step_prefix_reuse is True
        ),
        "flashinfer_unified": (
            config.attention_backend == "flashinfer"
            and config.attention_mode == "unified"
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"nano-vllm fairness configuration failed: {checks}")
    return checks, mode


def _vllm_engine_kwargs(
    trace: dict[str, Any],
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    contract = trace["execution_contract"]
    return {
        "dtype": "bfloat16",
        "seed": trace["seed"],
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": contract["tensor_parallel_size"],
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "block_size": contract["block_size"],
        # vLLM permanently reserves block 0 as a null block.  Add that
        # implementation-only block so both engines expose the same usable
        # logical capacity from the trace.
        "num_gpu_blocks_override": (
            contract["logical_kv_blocks"] + VLLM_NULL_BLOCK_COUNT
        ),
        "max_model_len": contract["max_model_len"],
        "max_num_batched_tokens": contract["max_num_batched_tokens"],
        "max_num_seqs": contract["max_num_seqs"],
        "attention_config": {"backend": "FLASHINFER"},
        # The default V1 configuration selects AsyncScheduler.  The observer
        # subclasses that implementation and keeps its default FCFS policy.
        "async_scheduling": True,
        "scheduler_cls": VLLM_SCHEDULER_CLASS,
        "disable_log_stats": False,
    }


def _nvidia_device_selector(device_index: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",")]
        if device_index < len(devices) and devices[device_index]:
            return devices[device_index]
    return str(device_index)


def _nvidia_smi_field(field: str, selector: str) -> str | None:
    output = _command_output(
        [
            "nvidia-smi",
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
            f"--id={selector}",
        ]
    )
    if output is None:
        return None
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    return rows[0] if len(rows) == 1 else None


def _runtime_metadata(
    torch_module: Any,
    framework: str,
    framework_path: Path,
) -> dict[str, Any]:
    device_index = torch_module.cuda.current_device()
    properties = torch_module.cuda.get_device_properties(device_index)
    selector = _nvidia_device_selector(device_index)
    gpu_uuid = _nvidia_smi_field("uuid", selector)
    driver_version = _nvidia_smi_field("driver_version", selector)
    if gpu_uuid is None or driver_version is None:
        raise RuntimeError(
            "nvidia-smi must expose GPU UUID and driver version to prove "
            "cross-framework hardware identity"
        )
    nvidia_row = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
            f"--id={selector}",
        ]
    )
    framework_version = (
        _package_version("vllm")
        if framework == "vllm"
        else _package_version("nano-vllm")
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": [sys.executable, *sys.argv],
        "environment": {
            name: os.environ[name]
            for name in (
                "CUDA_HOME",
                "CUDA_VISIBLE_DEVICES",
                "FLASHINFER_CUDA_ARCH_LIST",
                "FLASHINFER_DISABLE_JIT",
                "LD_LIBRARY_PATH",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_ALLOW_INSECURE_SERIALIZATION",
            )
            if name in os.environ
        },
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_module.__version__,
        "torch_cuda": torch_module.version.cuda,
        "cuda_driver_and_gpu": nvidia_row,
        "gpu": {
            "name": properties.name,
            "uuid": gpu_uuid,
            "driver_version": driver_version,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "torch_device_index": device_index,
            "nvidia_smi_selector": selector,
        },
        "flashinfer": {
            "version": _package_version("flashinfer-python"),
            "cubin_package": _package_version("flashinfer-cubin"),
            "jit_cache_package": _package_version("flashinfer-jit-cache"),
        },
        "framework": {
            "name": framework,
            "version": framework_version,
            "module_path": str(framework_path.resolve()),
            "git": _git_metadata(framework_path),
        },
        "benchmark_script_sha256": _sha256_file(Path(__file__)),
    }


def _sampling_spec(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": 1.0,
        "max_tokens": request["output_len"],
        "ignore_eos": True,
    }


def _trace_proof(
    trace_path: Path,
    trace_sha256: str,
    trace: dict[str, Any],
) -> dict[str, Any]:
    manifest = _request_manifest(trace)
    return {
        "path": str(trace_path.resolve()),
        "sha256": trace_sha256,
        "request_manifest_sha256": _canonical_sha256(manifest),
        "request_manifest": manifest,
    }


def _base_result(
    framework: str,
    trace_path: Path,
    trace_sha256: str,
    trace: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "result_kind": "eager_only_cross_framework",
        "framework": framework,
        "workload": trace["workload"],
        "seed": trace["seed"],
        "trace": _trace_proof(trace_path, trace_sha256, trace),
        "execution_contract": trace["execution_contract"],
        "interpretation_boundary": (
            "This is a targeted shared-prefix eager-only scheduler workload, "
            "not a claim about general framework performance."
        ),
    }


def _request_spec_for_nano(request: dict[str, Any], request_spec_cls: Any) -> Any:
    return request_spec_cls(
        name=request["request_id"],
        prompt_token_ids=request["input_token_ids"],
        output_len=request["output_len"],
        kind=request["kind"],
        group=request["prefix_group"],
        shared_prefix_len=request["shared_prefix_len"],
    )


def _run_nano(
    trace_path: Path,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    import nanovllm
    from bench_scheduler import (
        RequestSpec,
        SchedulerMetrics,
        _add_request,
        _metric_summary,
        _request_rows,
        _run_measured_to_completion,
        _run_unmeasured_to_completion,
    )
    from nanovllm import LLM
    from nanovllm.engine.block_manager import BlockManager

    trace, trace_sha256 = _load_trace(trace_path)
    _require_cross_framework_lpm(trace)
    model_path = _validate_model_files(trace)
    if not torch.cuda.is_available():
        raise RuntimeError("run-nano requires an NVIDIA CUDA GPU")
    torch.manual_seed(trace["seed"])
    torch.cuda.manual_seed_all(trace["seed"])
    llm = None
    try:
        kwargs = _nano_engine_kwargs(trace, gpu_memory_utilization)
        llm = LLM(str(model_path), **kwargs)
        fairness_checks, mode = _validate_nano_runtime_config(llm, trace)
        contract = trace["execution_contract"]
        physical_blocks = llm.config.num_kvcache_blocks
        logical_blocks = contract["logical_kv_blocks"]
        if logical_blocks > physical_blocks:
            raise ValueError(
                f"trace requires {logical_blocks} logical blocks but nano-vllm "
                f"allocated only {physical_blocks} physical blocks"
            )
        if not llm.scheduler.is_finished():
            raise AssertionError("logical KV limit must be installed while idle")
        llm.scheduler.block_manager = BlockManager(logical_blocks, BLOCK_SIZE)

        priming_rows = []
        for phase in _priming_phases(trace):
            for request in phase["requests"]:
                spec = _request_spec_for_nano(request, RequestSpec)
                observation = _add_request(llm, spec)
                _run_unmeasured_to_completion(llm)
                if observation.seq.num_completion_tokens != request["output_len"]:
                    raise AssertionError(
                        f"priming request {request['request_id']} output length changed"
                    )
                priming_rows.append(
                    {
                        "request_id": request["request_id"],
                        "prompt_sha256": _prompt_sha256(
                            request["input_token_ids"]
                        ),
                        "output_tokens": observation.seq.num_completion_tokens,
                    }
                )

        # Initialization and prefix priming consume random values.  Reset the
        # documented benchmark seed immediately before the measured phase.
        torch.manual_seed(trace["seed"])
        torch.cuda.manual_seed_all(trace["seed"])
        measured = _measured_phase(trace)["requests"]
        observations: dict[int, Any] = {}
        for request in measured:
            spec = _request_spec_for_nano(request, RequestSpec)
            params = _sampling_spec(request)
            if params["temperature"] != 1.0:
                raise AssertionError("trace sampling contract changed")
            observation = _add_request(llm, spec)
            observations[observation.seq.seq_id] = observation

        torch.cuda.synchronize()
        phase_started_at = time.perf_counter()
        for observation in observations.values():
            observation.submitted_at = phase_started_at
        collector = SchedulerMetrics(llm, observations, phase_started_at)
        collector.install()
        try:
            phase_ended_at = _run_measured_to_completion(
                llm,
                observations,
                collector,
            )
        finally:
            collector.uninstall()

        internal_metrics = _metric_summary(
            observations,
            collector,
            phase_started_at,
            phase_ended_at,
            prefix_groups=trace["workload_contract"]["prefix_groups"],
            shared_prefix_len=trace["workload_contract"]["shared_prefix_len"],
        )
        raw_rows = _request_rows(observations, phase_started_at)
        trace_by_id = {request["request_id"]: request for request in measured}
        request_rows = []
        for row in raw_rows:
            request = trace_by_id[row["name"]]
            if row["output_tokens"] != request["output_len"]:
                raise AssertionError(
                    f"request {row['name']} generated the wrong output length"
                )
            request_rows.append(
                {
                    "request_id": row["name"],
                    "arrival_order": request["arrival_order"],
                    "prompt_len": request["prompt_len"],
                    "prompt_sha256": _prompt_sha256(
                        request["input_token_ids"]
                    ),
                    "requested_output_len": request["output_len"],
                    "output_tokens": row["output_tokens"],
                    "ttft_ms": row["ttft_ms"],
                    "completion_ms": row["completion_ms"],
                    "prefix_group": request["prefix_group"],
                    "initial_persistent_hit_tokens": (
                        row["initial_persistent_hit_tokens"]
                    ),
                    "same_step_hit_tokens": row["same_step_hit_tokens"],
                    "computed_prompt_tokens": row["computed_prompt_tokens"],
                }
            )
        request_rows.sort(key=lambda row: row["arrival_order"])
        elapsed_s = phase_ended_at - phase_started_at
        result = _base_result(
            "nano-vllm",
            trace_path,
            trace_sha256,
            trace,
        )
        result.update(
            {
                "runtime": _runtime_metadata(
                    torch,
                    "nano-vllm",
                    Path(nanovllm.__file__),
                ),
                "engine": {
                    "requested": {"model": str(model_path), **kwargs},
                    "fairness_checks": fairness_checks,
                    "actual_model_path": str(Path(llm.config.model).resolve()),
                    "actual_tokenizer_path": str(Path(llm.config.model).resolve()),
                    "actual_dtype": str(llm.model_runner.dtype),
                    "seed": trace["seed"],
                    "seed_reset_before_measured": True,
                    "physical_kv_blocks": physical_blocks,
                    "usable_logical_kv_blocks": logical_blocks,
                    "logical_kv_blocks": logical_blocks,
                    "scheduler": (
                        "nano-vllm cache-aware "
                        "(LPM + same-step prefix reuse)"
                    ),
                    "cudagraph_mode": mode,
                    "enforce_eager": llm.config.enforce_eager,
                },
                "metrics": {
                    "comparable": {
                        "computed_prompt_tokens": internal_metrics[
                            "computed_prompt_tokens"
                        ],
                        "cached_block_eviction_count": internal_metrics[
                            "cached_block_eviction_count"
                        ],
                        "p95_ttft_ms": _percentile(
                            [row["ttft_ms"] for row in request_rows],
                            95.0,
                        ),
                        "request_throughput_rps": len(request_rows) / elapsed_s,
                        "total_batch_completion_s": elapsed_s,
                    },
                    "backend_specific": {
                        "definition": (
                            "nano-vLLM SchedulerMetrics. Computed prompt tokens "
                            "and allocation-pressure cached-block evictions use "
                            "the same definitions as the vLLM benchmark observer; "
                            "the initial/same-step hit breakdown remains native."
                        ),
                        **internal_metrics,
                    },
                },
                "priming_requests": priming_rows,
                "requests": request_rows,
                "scheduler_steps": collector.steps,
            }
        )
        return result
    finally:
        if llm is not None:
            try:
                atexit.unregister(llm.exit)
            except Exception:
                pass
            try:
                llm.exit()
            except Exception as error:
                print(f"warning: nano-vllm cleanup failed: {error}", file=sys.stderr)
        llm = None
        gc.collect()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _vllm_prompt(request: dict[str, Any]) -> dict[str, Any]:
    return {"prompt_token_ids": request["input_token_ids"]}


def _vllm_output_request_id(engine_request_id: str) -> str:
    external_id, separator, random_suffix = engine_request_id.rpartition("-")
    if separator and external_id and len(random_suffix) == 8:
        return external_id
    return engine_request_id


def _vllm_sampling(request: dict[str, Any], sampling_cls: Any) -> Any:
    return sampling_cls(
        temperature=1.0,
        max_tokens=request["output_len"],
        ignore_eos=True,
        detokenize=False,
    )


def _validate_vllm_runtime_config(llm: Any, trace: dict[str, Any]) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    model = config.model_config
    cache = config.cache_config
    scheduler = config.scheduler_config
    parallel = config.parallel_config
    attention = config.attention_config
    contract = trace["execution_contract"]
    expected_model = Path(trace["model"]["path"]).resolve()
    backend_name = getattr(attention.backend, "name", None)
    configured_blocks = cache.num_gpu_blocks
    expected_configured_blocks = (
        contract["logical_kv_blocks"] + VLLM_NULL_BLOCK_COUNT
    )
    checks = {
        "model_path": Path(model.model).resolve() == expected_model,
        "tokenizer_path": Path(model.tokenizer).resolve() == expected_model,
        "model_shape": {
            name: getattr(model.hf_config, name, None)
            for name in TARGET_MODEL_SHAPE
        }
        == TARGET_MODEL_SHAPE,
        "seed": model.seed == trace["seed"],
        "enforce_eager": model.enforce_eager is True,
        "dtype_bfloat16": str(model.dtype) == "torch.bfloat16",
        "prefix_caching": cache.enable_prefix_caching is True,
        "block_size": cache.block_size == contract["block_size"],
        "configured_kv_blocks_override": (
            cache.num_gpu_blocks_override == expected_configured_blocks
        ),
        "equal_usable_kv_capacity": (
            configured_blocks == expected_configured_blocks
            and configured_blocks - VLLM_NULL_BLOCK_COUNT
            == contract["logical_kv_blocks"]
        ),
        "chunked_prefill": scheduler.enable_chunked_prefill is True,
        "max_model_len": model.max_model_len == contract["max_model_len"],
        "max_num_batched_tokens": (
            scheduler.max_num_batched_tokens
            == contract["max_num_batched_tokens"]
        ),
        "max_num_seqs": scheduler.max_num_seqs == contract["max_num_seqs"],
        "tensor_parallel_size": (
            parallel.tensor_parallel_size == contract["tensor_parallel_size"]
        ),
        "single_data_parallel_engine": parallel.data_parallel_size == 1,
        "single_pipeline_stage": parallel.pipeline_parallel_size == 1,
        "default_fcfs_policy": scheduler.policy == "fcfs",
        "default_async_scheduling": scheduler.async_scheduling is True,
        "read_only_scheduler_observer": (
            scheduler.scheduler_cls == VLLM_SCHEDULER_CLASS
        ),
        "flashinfer_attention": backend_name == "FLASHINFER",
    }
    if not all(checks.values()):
        raise AssertionError(f"vLLM fairness configuration failed: {checks}")
    return checks


def _verify_vllm_output(output: Any, request: dict[str, Any]) -> int:
    if list(output.prompt_token_ids) != request["input_token_ids"]:
        raise AssertionError(
            f"vLLM changed direct token IDs for {request['request_id']}"
        )
    if len(output.outputs) != 1:
        raise AssertionError("benchmark requires exactly one completion per request")
    output_tokens = len(output.outputs[0].token_ids)
    if output_tokens != request["output_len"]:
        raise AssertionError(
            f"vLLM request {request['request_id']} generated {output_tokens} "
            f"tokens, expected {request['output_len']}"
        )
    return output_tokens


def _reset_vllm_worker_seed(_worker: Any, seed: int) -> None:
    from vllm.utils.torch_utils import set_random_seed

    set_random_seed(seed)


def _shutdown_vllm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if not callable(shutdown):
        raise RuntimeError("vLLM engine exposes no shutdown method")
    shutdown()


def _vllm_counter_value(
    llm: Any,
    name: str,
    required_labels: dict[str, str],
) -> int:
    matches = []
    for metric in llm.get_metrics():
        labels = getattr(metric, "labels", {})
        if (
            getattr(metric, "name", None) == name
            and all(labels.get(key) == value for key, value in required_labels.items())
        ):
            matches.append(metric)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one vLLM counter {name} with labels "
            f"{required_labels}, got {len(matches)}"
        )
    value = getattr(matches[0], "value", None)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"vLLM counter {name} has invalid value {value!r}")
    return value


def _vllm_counter_window(
    name: str,
    labels: dict[str, str],
    before: int,
    after: int,
) -> dict[str, Any]:
    if type(before) is not int or type(after) is not int or after < before:
        raise RuntimeError(f"vLLM counter {name} is not monotonic")
    return {
        "metric_name": name,
        "labels": labels,
        "before": before,
        "after": after,
        "delta": after - before,
    }


def _vllm_observer_snapshot(llm: Any) -> dict[str, Any]:
    from bench_vllm_metrics import (
        PROTOCOL_VERSION,
        SCHEDULER_CLASS,
        SNAPSHOT_METHOD,
    )

    if SCHEDULER_CLASS != VLLM_SCHEDULER_CLASS:
        raise RuntimeError("vLLM observer scheduler class constant drifted")
    raw = llm.llm_engine.engine_core.call_utility(SNAPSHOT_METHOD)
    if not isinstance(raw, dict):
        raise RuntimeError("vLLM observer returned a non-object snapshot")
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("vLLM observer protocol mismatch")
    if raw.get("async_scheduler") is not True:
        raise RuntimeError("vLLM observer changed the default async scheduler")
    if raw.get("block_pool_observer_installed") is not True:
        raise RuntimeError("vLLM block-pool observer is not installed")
    for name in (
        "observer_pid",
        "computed_prompt_tokens",
        "cached_block_eviction_count",
        "allocated_block_count",
        "preemption_count",
        "scheduler_step_count",
    ):
        value = raw.get(name)
        if type(value) is not int or value < 0:
            raise RuntimeError(f"vLLM observer returned invalid {name}")
    if raw["observer_pid"] <= 0:
        raise RuntimeError("vLLM observer did not report its process")
    return raw


def _vllm_observer_window(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    for name in (
        "protocol_version",
        "observer_pid",
        "async_scheduler",
        "block_pool_observer_installed",
    ):
        if before.get(name) != after.get(name):
            raise RuntimeError(f"vLLM observer identity changed during {name}")
    deltas = {}
    for name in (
        "computed_prompt_tokens",
        "cached_block_eviction_count",
        "allocated_block_count",
        "preemption_count",
        "scheduler_step_count",
    ):
        before_value = before[name]
        after_value = after[name]
        if after_value < before_value:
            raise RuntimeError(f"vLLM observer counter {name} is not monotonic")
        deltas[name] = after_value - before_value
    if deltas["scheduler_step_count"] <= 0:
        raise RuntimeError("vLLM observer recorded no measured scheduler steps")
    if (
        deltas["cached_block_eviction_count"]
        > deltas["allocated_block_count"]
    ):
        raise RuntimeError("vLLM observed more evictions than allocations")
    return {
        "before": before,
        "after": after,
        "delta": deltas,
    }


def _run_vllm(
    trace_path: Path,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    # These must be set before vLLM imports its model runner and sampler.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # collective_rpc must transport the local, top-level seed-reset callback to
    # the single-host engine process. vLLM deliberately gates callable
    # serialization; this benchmark opts in only for that trusted local RPC.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    import torch
    import torch.distributed as dist
    import vllm
    from vllm import LLM, SamplingParams

    trace, trace_sha256 = _load_trace(trace_path)
    _require_cross_framework_lpm(trace)
    model_path = _validate_model_files(trace)
    if not torch.cuda.is_available():
        raise RuntimeError("run-vllm requires an NVIDIA CUDA GPU")
    torch.manual_seed(trace["seed"])
    torch.cuda.manual_seed_all(trace["seed"])
    llm = None
    try:
        kwargs = _vllm_engine_kwargs(trace, gpu_memory_utilization)
        llm = LLM(model=str(model_path), **kwargs)
        fairness_checks = _validate_vllm_runtime_config(llm, trace)
        cache_config = llm.llm_engine.vllm_config.cache_config
        configured_blocks = int(cache_config.num_gpu_blocks)
        usable_blocks = configured_blocks - VLLM_NULL_BLOCK_COUNT
        logical_blocks = trace["execution_contract"]["logical_kv_blocks"]
        if usable_blocks != logical_blocks:
            raise AssertionError(
                "vLLM usable KV capacity differs from the trace: "
                f"configured={configured_blocks}, usable={usable_blocks}, "
                f"required={logical_blocks}"
            )

        priming_rows = []
        for phase in _priming_phases(trace):
            for request in phase["requests"]:
                outputs = llm.generate(
                    [_vllm_prompt(request)],
                    [_vllm_sampling(request, SamplingParams)],
                    use_tqdm=False,
                )
                if len(outputs) != 1:
                    raise AssertionError("vLLM priming request did not finish")
                output_tokens = _verify_vllm_output(outputs[0], request)
                priming_rows.append(
                    {
                        "request_id": request["request_id"],
                        "prompt_sha256": _prompt_sha256(
                            request["input_token_ids"]
                        ),
                        "output_tokens": output_tokens,
                        "reported_num_cached_tokens": (
                            outputs[0].num_cached_tokens
                        ),
                    }
                )

        # Priming advances random generators. Reset every vLLM worker and the
        # host before measured arrivals, matching nano-vLLM's seed reset.
        llm.collective_rpc(_reset_vllm_worker_seed, args=(trace["seed"],))
        torch.manual_seed(trace["seed"])
        torch.cuda.manual_seed_all(trace["seed"])
        measured = _measured_phase(trace)["requests"]
        # Level-0 sleep keeps model/KV allocations and the primed prefix cache,
        # but pauses scheduling while every measured request is enqueued.
        llm.sleep(level=0)
        engine_request_ids = llm.enqueue(
            [_vllm_prompt(request) for request in measured],
            [_vllm_sampling(request, SamplingParams) for request in measured],
            use_tqdm=False,
        )
        if len(engine_request_ids) != len(measured):
            raise AssertionError("vLLM did not enqueue every measured request")
        output_request_ids = [
            _vllm_output_request_id(request_id)
            for request_id in engine_request_ids
        ]
        trace_by_output_id = dict(zip(output_request_ids, measured))
        if len(trace_by_output_id) != len(measured):
            raise AssertionError("vLLM generated duplicate external request IDs")

        local_compute_labels = {"engine": "0", "source": "local_compute"}
        preemption_labels = {"engine": "0"}
        observer_before = _vllm_observer_snapshot(llm)
        local_compute_before = _vllm_counter_value(
            llm,
            "vllm:prompt_tokens_by_source",
            local_compute_labels,
        )
        preemptions_before = _vllm_counter_value(
            llm,
            "vllm:num_preemptions",
            preemption_labels,
        )

        torch.cuda.synchronize()
        phase_started_at = time.monotonic()
        llm.wake_up(tags=["scheduling"])
        outputs = llm.wait_for_completion(use_tqdm=False)
        torch.cuda.synchronize()
        phase_ended_at = time.monotonic()

        # Utility and Prometheus snapshots are intentionally outside the timed
        # interval. The direct observer delta is authoritative; vLLM's native
        # counters provide an independent no-preemption cross-check.
        observer_after = _vllm_observer_snapshot(llm)
        observer_window = _vllm_observer_window(
            observer_before,
            observer_after,
        )
        local_compute_after = _vllm_counter_value(
            llm,
            "vllm:prompt_tokens_by_source",
            local_compute_labels,
        )
        preemptions_after = _vllm_counter_value(
            llm,
            "vllm:num_preemptions",
            preemption_labels,
        )
        local_compute_window = _vllm_counter_window(
            "vllm:prompt_tokens_by_source",
            local_compute_labels,
            local_compute_before,
            local_compute_after,
        )
        preemption_window = _vllm_counter_window(
            "vllm:num_preemptions",
            preemption_labels,
            preemptions_before,
            preemptions_after,
        )
        observed = observer_window["delta"]
        computed_prompt_tokens = observed["computed_prompt_tokens"]
        cached_block_evictions = observed["cached_block_eviction_count"]
        preemption_count = observed["preemption_count"]
        if preemption_window["delta"] != preemption_count:
            raise RuntimeError(
                "vLLM direct and native preemption counters disagree"
            )
        if local_compute_window["delta"] > computed_prompt_tokens:
            raise RuntimeError(
                "vLLM native local-compute counter exceeds direct prompt work"
            )
        if (
            preemption_count == 0
            and local_compute_window["delta"] != computed_prompt_tokens
        ):
            raise RuntimeError(
                "vLLM direct and native prompt-compute counters disagree "
                "without preemption"
            )

        output_by_id = {output.request_id: output for output in outputs}
        if output_by_id.keys() != trace_by_output_id.keys():
            raise AssertionError(
                "vLLM completed external request IDs differ from enqueued "
                f"IDs: completed={sorted(output_by_id)}, "
                f"enqueued={sorted(trace_by_output_id)}"
            )
        request_rows = []
        reported_cached_tokens = 0
        derived_prompt_tokens = 0
        for engine_id, output_id, request in zip(
            engine_request_ids, output_request_ids, measured
        ):
            output = output_by_id[output_id]
            output_tokens = _verify_vllm_output(output, request)
            metrics = output.metrics
            if metrics is None:
                raise RuntimeError(
                    "vLLM returned no RequestStateStats; "
                    "disable_log_stats must be False"
                )
            if metrics.first_token_ts <= 0.0 or metrics.last_token_ts <= 0.0:
                raise RuntimeError("vLLM returned incomplete request timestamps")
            ttft_ms = (metrics.first_token_ts - phase_started_at) * 1e3
            completion_ms = (metrics.last_token_ts - phase_started_at) * 1e3
            if ttft_ms < 0.0 or completion_ms < ttft_ms:
                raise RuntimeError(
                    "vLLM request timestamps are incompatible with host "
                    "monotonic time"
                )
            cached_tokens = int(output.num_cached_tokens or 0)
            if cached_tokens < 0 or cached_tokens > request["prompt_len"]:
                raise RuntimeError("vLLM reported an invalid cached-token count")
            reported_cached_tokens += cached_tokens
            derived_prompt_tokens += request["prompt_len"] - cached_tokens
            request_rows.append(
                {
                    "request_id": request["request_id"],
                    "engine_request_id": engine_id,
                    "vllm_output_request_id": output_id,
                    "arrival_order": request["arrival_order"],
                    "prompt_len": request["prompt_len"],
                    "prompt_sha256": _prompt_sha256(
                        request["input_token_ids"]
                    ),
                    "requested_output_len": request["output_len"],
                    "output_tokens": output_tokens,
                    "ttft_ms": ttft_ms,
                    "completion_ms": completion_ms,
                    "prefix_group": request["prefix_group"],
                    "reported_num_cached_tokens": cached_tokens,
                }
            )
        total_prompt_tokens = sum(
            request["prompt_len"] for request in measured
        )
        if reported_cached_tokens + derived_prompt_tokens != total_prompt_tokens:
            raise RuntimeError("vLLM output prompt-token accounting is invalid")
        if (
            preemption_count == 0
            and derived_prompt_tokens != computed_prompt_tokens
        ):
            raise RuntimeError(
                "vLLM prompt-minus-cache diagnostic disagrees with exact "
                "computed work without preemption"
            )

        elapsed_s = phase_ended_at - phase_started_at
        result = _base_result("vllm", trace_path, trace_sha256, trace)
        result.update(
            {
                "runtime": _runtime_metadata(
                    torch,
                    "vllm",
                    Path(vllm.__file__),
                ),
                "engine": {
                    "requested": {"model": str(model_path), **kwargs},
                    "fairness_checks": fairness_checks,
                    "actual_model_path": str(
                        Path(
                            llm.llm_engine.vllm_config.model_config.model
                        ).resolve()
                    ),
                    "actual_tokenizer_path": str(
                        Path(
                            llm.llm_engine.vllm_config.model_config.tokenizer
                        ).resolve()
                    ),
                    "actual_dtype": str(
                        llm.llm_engine.vllm_config.model_config.dtype
                    ),
                    "seed": trace["seed"],
                    "seed_reset_before_measured": True,
                    "seed_reset_transport": (
                        "trusted local collective_rpc callback"
                    ),
                    "configured_physical_kv_blocks": configured_blocks,
                    "reserved_null_blocks": VLLM_NULL_BLOCK_COUNT,
                    "usable_logical_kv_blocks": usable_blocks,
                    "scheduler": (
                        "vLLM default async FCFS policy with a read-only "
                        "benchmark counter observer"
                    ),
                    "async_scheduling": True,
                    "counter_instrumentation": {
                        "benchmark_only": True,
                        "scheduler_class": VLLM_SCHEDULER_CLASS,
                        "inherits_default_async_scheduler": True,
                        "engine_core_snapshot_method": (
                            "_nanovllm_benchmark_metrics_snapshot"
                        ),
                        "source_path": str(
                            VLLM_METRICS_HELPER_PATH.resolve()
                        ),
                        "source_sha256": _sha256_file(
                            VLLM_METRICS_HELPER_PATH
                        ),
                        "protocol_version": observer_after[
                            "protocol_version"
                        ],
                        "changes_scheduling_policy": False,
                    },
                    "cudagraph_mode": "none",
                    "enforce_eager": True,
                },
                "metrics": {
                    "comparable": {
                        "computed_prompt_tokens": computed_prompt_tokens,
                        "cached_block_eviction_count": (
                            cached_block_evictions
                        ),
                        "p95_ttft_ms": _percentile(
                            [row["ttft_ms"] for row in request_rows],
                            95.0,
                        ),
                        "request_throughput_rps": (
                            len(request_rows) / elapsed_s
                        ),
                        "total_batch_completion_s": elapsed_s,
                    },
                    "backend_specific": {
                        "definition": (
                            "Direct read-only scheduler observation. Computed "
                            "prompt tokens sum scheduled prompt intervals, "
                            "including recomputation. Cached-block evictions "
                            "count physical allocation reuse that removes at "
                            "least one live prefix-cache mapping."
                        ),
                        "computed_prompt_tokens": computed_prompt_tokens,
                        "cached_block_eviction_count": (
                            cached_block_evictions
                        ),
                        "preemption_count": preemption_count,
                        "allocated_block_count": observed[
                            "allocated_block_count"
                        ],
                        "scheduler_step_count": observed[
                            "scheduler_step_count"
                        ],
                        "observer_pid": observer_after["observer_pid"],
                        "total_prompt_tokens": total_prompt_tokens,
                        "reported_cached_prompt_tokens": (
                            reported_cached_tokens
                        ),
                        "derived_computed_prompt_tokens": (
                            derived_prompt_tokens
                        ),
                        "observer_window": observer_window,
                        "native_counter_windows": {
                            "local_compute": local_compute_window,
                            "preemptions": preemption_window,
                        },
                    },
                },
                "priming_requests": priming_rows,
                "requests": request_rows,
            }
        )
        return result
    finally:
        if llm is not None:
            try:
                _shutdown_vllm(llm)
            except Exception as error:
                print(f"warning: vLLM cleanup failed: {error}", file=sys.stderr)
        llm = None
        gc.collect()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _validate_recorded_runtime(result: dict[str, Any], framework: str) -> None:
    runtime = result.get("runtime", {})
    for name in (
        "command",
        "torch",
        "torch_cuda",
        "cuda_driver_and_gpu",
        "benchmark_script_sha256",
    ):
        if not isinstance(runtime.get(name), str) or not runtime[name]:
            raise ValueError(f"{framework} result is missing runtime metadata {name}")
    gpu = runtime.get("gpu", {})
    required_gpu = {
        "name": str,
        "uuid": str,
        "driver_version": str,
        "compute_capability": str,
        "total_memory_bytes": int,
    }
    for name, expected_type in required_gpu.items():
        value = gpu.get(name)
        if type(value) is not expected_type or not value:
            raise ValueError(f"{framework} result has invalid GPU metadata {name}")
    flashinfer = runtime.get("flashinfer", {})
    if not isinstance(flashinfer.get("version"), str) or not flashinfer["version"]:
        raise ValueError(f"{framework} result did not record FlashInfer version")
    if runtime["benchmark_script_sha256"] != _sha256_file(Path(__file__)):
        raise ValueError(
            f"{framework} result benchmark script SHA does not match this source"
        )
    framework_metadata = runtime.get("framework", {})
    git = framework_metadata.get("git", {})
    if not isinstance(git.get("head"), str) or not git["head"]:
        raise ValueError(f"{framework} result did not record a framework commit")
    if framework == "vllm" and not framework_metadata.get("version"):
        raise ValueError("vLLM result did not record its package version")


def _validate_nano_backend_metrics(
    result: dict[str, Any],
    measured_requests: list[dict[str, Any]],
) -> None:
    metrics = result.get("metrics", {}).get("backend_specific", {})
    required_counts = (
        "initial_persistent_hit_tokens",
        "same_step_hit_tokens",
        "claimed_prefix_tokens",
        "computed_prompt_tokens",
        "cached_block_eviction_count",
        "preemption_count",
        "total_prompt_tokens",
        "same_step_reused_request_count",
        "same_step_reused_blocks",
        "first_step_prefill_admission_count",
        "max_step_prefill_admission_count",
    )
    for name in required_counts:
        value = metrics.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(
                f"nano-vllm backend metric {name} must be a non-negative integer"
            )

    initial_tokens = metrics["initial_persistent_hit_tokens"]
    same_step_tokens = metrics["same_step_hit_tokens"]
    claimed_tokens = metrics["claimed_prefix_tokens"]
    computed_tokens = metrics["computed_prompt_tokens"]
    total_prompt_tokens = sum(
        request["prompt_len"] for request in measured_requests
    )
    comparable = result["metrics"]["comparable"]
    if comparable["computed_prompt_tokens"] != computed_tokens:
        raise ValueError(
            "nano-vllm comparable computed prompt tokens differ from "
            "scheduler instrumentation"
        )
    if (
        comparable["cached_block_eviction_count"]
        != metrics["cached_block_eviction_count"]
    ):
        raise ValueError(
            "nano-vllm comparable cached-block evictions differ from "
            "scheduler instrumentation"
        )
    if metrics["total_prompt_tokens"] != total_prompt_tokens:
        raise ValueError("nano-vllm total_prompt_tokens differs from the trace")
    if initial_tokens + same_step_tokens != claimed_tokens:
        raise ValueError(
            "nano-vllm initial + same-step hits do not equal claimed prefix tokens"
        )
    if claimed_tokens + computed_tokens != total_prompt_tokens:
        raise ValueError("nano-vllm prompt-token conservation failed")
    if same_step_tokens % BLOCK_SIZE != 0:
        raise ValueError("nano-vllm same-step hit tokens are not block aligned")
    if metrics["same_step_reused_blocks"] * BLOCK_SIZE != same_step_tokens:
        raise ValueError("nano-vllm same-step block/token counts disagree")

    conservation = metrics.get("prompt_token_conservation")
    if not isinstance(conservation, dict):
        raise ValueError("nano-vllm result has no prompt-token conservation proof")
    expected_conservation = {
        "initial_persistent_hit_tokens": initial_tokens,
        "same_step_hit_tokens": same_step_tokens,
        "computed_prompt_tokens": computed_tokens,
        "accounted_prompt_tokens": total_prompt_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "delta_tokens": 0,
        "balanced": True,
    }
    if conservation != expected_conservation:
        raise ValueError("nano-vllm prompt-token conservation proof is invalid")

    rows = result.get("requests", [])
    if len(rows) != len(measured_requests):
        raise ValueError("nano-vllm request metric row count differs from trace")
    row_sums = {
        "initial_persistent_hit_tokens": 0,
        "same_step_hit_tokens": 0,
        "computed_prompt_tokens": 0,
    }
    same_step_reused_requests = 0
    for row in rows:
        for name in row_sums:
            value = row.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"nano-vllm request metric {name} must be non-negative"
                )
            row_sums[name] += value
        if (
            row["initial_persistent_hit_tokens"]
            + row["same_step_hit_tokens"]
            + row["computed_prompt_tokens"]
            != row["prompt_len"]
        ):
            raise ValueError(
                f"nano-vllm request {row.get('request_id')} violates "
                "prompt-token conservation"
            )
        if row["same_step_hit_tokens"] > 0:
            same_step_reused_requests += 1

    if row_sums["initial_persistent_hit_tokens"] != initial_tokens:
        raise ValueError("nano-vllm request initial-hit rows do not match summary")
    if row_sums["same_step_hit_tokens"] != same_step_tokens:
        raise ValueError("nano-vllm request same-step rows do not match summary")
    if row_sums["computed_prompt_tokens"] != computed_tokens:
        raise ValueError("nano-vllm request compute rows do not match summary")
    if (
        metrics["same_step_reused_request_count"]
        != same_step_reused_requests
    ):
        raise ValueError("nano-vllm same-step reused request count is invalid")
    if not (
        0
        < metrics["first_step_prefill_admission_count"]
        <= metrics["max_step_prefill_admission_count"]
        <= len(measured_requests)
    ):
        raise ValueError("nano-vllm prefill admission counts are invalid")


def _validate_vllm_backend_metrics(
    result: dict[str, Any],
    measured_requests: list[dict[str, Any]],
) -> None:
    backend = result.get("metrics", {}).get("backend_specific", {})
    comparable = result.get("metrics", {}).get("comparable", {})
    required_counts = (
        "computed_prompt_tokens",
        "cached_block_eviction_count",
        "preemption_count",
        "allocated_block_count",
        "scheduler_step_count",
        "observer_pid",
        "total_prompt_tokens",
        "reported_cached_prompt_tokens",
        "derived_computed_prompt_tokens",
    )
    for name in required_counts:
        value = backend.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(
                f"vLLM backend metric {name} must be a non-negative integer"
            )
    if backend["observer_pid"] <= 0 or backend["scheduler_step_count"] <= 0:
        raise ValueError("vLLM observer did not run in the measured phase")
    if (
        backend["cached_block_eviction_count"]
        > backend["allocated_block_count"]
    ):
        raise ValueError("vLLM evictions exceed measured physical allocations")
    for name in ("computed_prompt_tokens", "cached_block_eviction_count"):
        if comparable.get(name) != backend[name]:
            raise ValueError(
                f"vLLM comparable {name} differs from exact observer evidence"
            )

    total_prompt_tokens = sum(
        request["prompt_len"] for request in measured_requests
    )
    if backend["total_prompt_tokens"] != total_prompt_tokens:
        raise ValueError("vLLM total prompt tokens differ from the trace")
    rows = result.get("requests", [])
    reported_cached_tokens = 0
    for row in rows:
        cached_tokens = row.get("reported_num_cached_tokens")
        if (
            type(cached_tokens) is not int
            or cached_tokens < 0
            or cached_tokens > row["prompt_len"]
            or cached_tokens % BLOCK_SIZE != 0
        ):
            raise ValueError("vLLM request has invalid reported cached tokens")
        reported_cached_tokens += cached_tokens
    derived_prompt_tokens = sum(
        row["prompt_len"] - row["reported_num_cached_tokens"]
        for row in rows
    )
    if reported_cached_tokens != backend["reported_cached_prompt_tokens"]:
        raise ValueError("vLLM request cached-token rows do not match summary")
    if derived_prompt_tokens != backend["derived_computed_prompt_tokens"]:
        raise ValueError("vLLM prompt-minus-cache rows do not match summary")
    if reported_cached_tokens + derived_prompt_tokens != total_prompt_tokens:
        raise ValueError("vLLM output prompt-token conservation failed")

    observer_window = backend.get("observer_window")
    if not isinstance(observer_window, dict):
        raise ValueError("vLLM result has no exact observer window")
    before = observer_window.get("before")
    after = observer_window.get("after")
    delta = observer_window.get("delta")
    if not all(isinstance(item, dict) for item in (before, after, delta)):
        raise ValueError("vLLM observer window is malformed")
    identity_fields = (
        "protocol_version",
        "observer_pid",
        "async_scheduler",
        "block_pool_observer_installed",
    )
    if any(before.get(name) != after.get(name) for name in identity_fields):
        raise ValueError("vLLM observer identity changed during measurement")
    if (
        after.get("protocol_version") != 1
        or after.get("observer_pid") != backend["observer_pid"]
        or after.get("async_scheduler") is not True
        or after.get("block_pool_observer_installed") is not True
    ):
        raise ValueError("vLLM observer identity evidence is invalid")
    observer_counts = (
        "computed_prompt_tokens",
        "cached_block_eviction_count",
        "allocated_block_count",
        "preemption_count",
        "scheduler_step_count",
    )
    for name in observer_counts:
        before_value = before.get(name)
        after_value = after.get(name)
        delta_value = delta.get(name)
        if (
            type(before_value) is not int
            or type(after_value) is not int
            or type(delta_value) is not int
            or before_value < 0
            or after_value < before_value
            or delta_value != after_value - before_value
            or delta_value != backend[name]
        ):
            raise ValueError(f"vLLM observer window disagrees for {name}")

    native_windows = backend.get("native_counter_windows")
    if not isinstance(native_windows, dict):
        raise ValueError("vLLM result has no native counter cross-check")
    expected_native = {
        "local_compute": (
            "vllm:prompt_tokens_by_source",
            {"engine": "0", "source": "local_compute"},
        ),
        "preemptions": ("vllm:num_preemptions", {"engine": "0"}),
    }
    native_deltas: dict[str, int] = {}
    for key, (metric_name, labels) in expected_native.items():
        window = native_windows.get(key)
        if not isinstance(window, dict):
            raise ValueError(f"vLLM native counter window {key} is missing")
        before_value = window.get("before")
        after_value = window.get("after")
        delta_value = window.get("delta")
        if (
            window.get("metric_name") != metric_name
            or window.get("labels") != labels
            or type(before_value) is not int
            or type(after_value) is not int
            or type(delta_value) is not int
            or before_value < 0
            or after_value < before_value
            or delta_value != after_value - before_value
        ):
            raise ValueError(f"vLLM native counter window {key} is invalid")
        native_deltas[key] = delta_value
    if native_deltas["preemptions"] != backend["preemption_count"]:
        raise ValueError("vLLM native and direct preemption counts disagree")
    if native_deltas["local_compute"] > backend["computed_prompt_tokens"]:
        raise ValueError("vLLM native local compute exceeds direct prompt work")
    if backend["preemption_count"] == 0:
        if not (
            native_deltas["local_compute"]
            == backend["computed_prompt_tokens"]
            == derived_prompt_tokens
        ):
            raise ValueError(
                "vLLM no-preemption prompt-compute conservation failed"
            )

    engine = result.get("engine", {})
    logical_blocks = result["execution_contract"]["logical_kv_blocks"]
    if (
        engine.get("configured_physical_kv_blocks")
        != logical_blocks + VLLM_NULL_BLOCK_COUNT
        or engine.get("reserved_null_blocks") != VLLM_NULL_BLOCK_COUNT
        or engine.get("usable_logical_kv_blocks") != logical_blocks
        or engine.get("async_scheduling") is not True
    ):
        raise ValueError("vLLM effective KV capacity or scheduler mode is invalid")
    instrumentation = engine.get("counter_instrumentation")
    if not isinstance(instrumentation, dict):
        raise ValueError("vLLM result has no counter instrumentation provenance")
    expected_instrumentation = {
        "benchmark_only": True,
        "scheduler_class": VLLM_SCHEDULER_CLASS,
        "inherits_default_async_scheduler": True,
        "engine_core_snapshot_method": (
            "_nanovllm_benchmark_metrics_snapshot"
        ),
        "source_path": str(VLLM_METRICS_HELPER_PATH.resolve()),
        "source_sha256": _sha256_file(VLLM_METRICS_HELPER_PATH),
        "protocol_version": 1,
        "changes_scheduling_policy": False,
    }
    if instrumentation != expected_instrumentation:
        raise ValueError("vLLM counter instrumentation provenance is invalid")


def _validate_result(
    result: dict[str, Any],
    framework: str,
    trace: dict[str, Any],
    trace_sha256: str,
) -> None:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{framework} result has an unsupported schema")
    if result.get("result_kind") != "eager_only_cross_framework":
        raise ValueError(f"{framework} result kind is invalid")
    if result.get("framework") != framework:
        raise ValueError(f"expected {framework} result")
    if result.get("workload") != trace["workload"]:
        raise ValueError(f"{framework} result workload differs from trace")
    if result.get("seed") != trace["seed"]:
        raise ValueError(f"{framework} result seed differs from trace")
    proof = result.get("trace", {})
    manifest = _request_manifest(trace)
    manifest_sha = _canonical_sha256(manifest)
    if proof.get("sha256") != trace_sha256:
        raise ValueError(f"{framework} result used a different trace SHA")
    if proof.get("request_manifest_sha256") != manifest_sha:
        raise ValueError(f"{framework} result used a different request manifest")
    if proof.get("request_manifest") != manifest:
        raise ValueError(f"{framework} result request manifest content changed")
    if result.get("execution_contract") != trace["execution_contract"]:
        raise ValueError(f"{framework} execution contract differs from trace")

    engine = result.get("engine", {})
    if engine.get("enforce_eager") is not True:
        raise ValueError(f"{framework} result is not enforce-eager")
    if engine.get("cudagraph_mode") != "none":
        raise ValueError(f"{framework} result enabled CUDA Graph")
    if engine.get("seed") != trace["seed"]:
        raise ValueError(f"{framework} engine seed differs from trace")
    if engine.get("seed_reset_before_measured") is not True:
        raise ValueError(f"{framework} did not reset its measured-phase seed")
    expected_model = Path(trace["model"]["path"]).resolve()
    for name in ("actual_model_path", "actual_tokenizer_path"):
        value = engine.get(name)
        if not isinstance(value, str) or Path(value).resolve() != expected_model:
            raise ValueError(f"{framework} {name} differs from trace")
    if engine.get("actual_dtype") != "torch.bfloat16":
        raise ValueError(f"{framework} did not run with BF16")
    fairness_checks = engine.get("fairness_checks")
    if not isinstance(fairness_checks, dict) or not fairness_checks:
        raise ValueError(f"{framework} result has no runtime fairness checks")
    if any(value is not True for value in fairness_checks.values()):
        raise ValueError(f"{framework} runtime fairness checks failed")
    logical_blocks = trace["execution_contract"]["logical_kv_blocks"]
    if engine.get("usable_logical_kv_blocks") != logical_blocks:
        raise ValueError(f"{framework} usable KV capacity differs from trace")
    if (
        framework == "nano-vllm"
        and engine.get("logical_kv_blocks") != logical_blocks
    ):
        raise ValueError("nano-vllm logical KV capacity differs from trace")

    requested = engine.get("requested")
    if not isinstance(requested, dict):
        raise ValueError(f"{framework} result has no requested engine config")
    utilization = requested.get("gpu_memory_utilization")
    if not isinstance(utilization, (int, float)):
        raise ValueError(f"{framework} GPU memory utilization is missing")
    _validate_gpu_utilization(float(utilization))
    expected_requested = (
        _nano_engine_kwargs(trace, float(utilization))
        if framework == "nano-vllm"
        else _vllm_engine_kwargs(trace, float(utilization))
    )
    expected_requested = {
        "model": str(expected_model),
        **expected_requested,
    }
    if requested != expected_requested:
        raise ValueError(f"{framework} requested engine config differs from trace")

    expected_priming = [
        request
        for phase in _priming_phases(trace)
        for request in phase["requests"]
    ]
    priming_rows = result.get("priming_requests", [])
    if [row.get("request_id") for row in priming_rows] != [
        request["request_id"] for request in expected_priming
    ]:
        raise ValueError(f"{framework} priming request order differs from trace")
    for row, request in zip(priming_rows, expected_priming):
        if row.get("prompt_sha256") != _prompt_sha256(request["input_token_ids"]):
            raise ValueError(f"{framework} priming prompt differs from trace")
        if row.get("output_tokens") != request["output_len"]:
            raise ValueError(f"{framework} priming output length differs from trace")

    measured = _measured_phase(trace)["requests"]
    result_rows = result.get("requests", [])
    expected_ids = [request["request_id"] for request in measured]
    if [row.get("request_id") for row in result_rows] != expected_ids:
        raise ValueError(f"{framework} request order differs from trace")
    for row, request in zip(result_rows, measured):
        if row.get("prompt_sha256") != _prompt_sha256(request["input_token_ids"]):
            raise ValueError(f"{framework} prompt tokens differ from trace")
        expected_fields = {
            "arrival_order": request["arrival_order"],
            "prompt_len": request["prompt_len"],
            "requested_output_len": request["output_len"],
            "output_tokens": request["output_len"],
            "prefix_group": request["prefix_group"],
        }
        if any(row.get(name) != value for name, value in expected_fields.items()):
            raise ValueError(f"{framework} request metadata differs from trace")
        ttft = row.get("ttft_ms")
        completion = row.get("completion_ms")
        if (
            type(ttft) not in (int, float)
            or type(completion) not in (int, float)
            or not math.isfinite(ttft)
            or not math.isfinite(completion)
            or ttft < 0.0
            or completion < ttft
        ):
            raise ValueError(f"{framework} request has invalid timing")

    comparable = result.get("metrics", {}).get("comparable", {})
    if not isinstance(comparable, dict) or set(comparable) != set(
        COMPARABLE_METRIC_NAMES
    ):
        raise ValueError(
            f"{framework} result must contain exactly the five comparable metrics"
        )
    for name in ("computed_prompt_tokens", "cached_block_eviction_count"):
        value = comparable[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"{framework} result has invalid metric {name}")
    if comparable["computed_prompt_tokens"] <= 0:
        raise ValueError(f"{framework} computed no measured prompt tokens")
    for name in (
        "p95_ttft_ms",
        "request_throughput_rps",
        "total_batch_completion_s",
    ):
        value = comparable[name]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{framework} result has invalid metric {name}")
    elapsed = comparable["total_batch_completion_s"]
    expected_p95 = _percentile([row["ttft_ms"] for row in result_rows], 95.0)
    expected_throughput = len(result_rows) / elapsed
    if not math.isclose(comparable["p95_ttft_ms"], expected_p95, rel_tol=1e-9):
        raise ValueError(f"{framework} P95 TTFT does not match request rows")
    if not math.isclose(
        comparable["request_throughput_rps"],
        expected_throughput,
        rel_tol=1e-9,
    ):
        raise ValueError(f"{framework} throughput does not match completion time")
    if max(row["completion_ms"] for row in result_rows) > elapsed * 1e3 + 1e-6:
        raise ValueError(f"{framework} completion time ends before a request")
    if framework == "nano-vllm":
        _validate_nano_backend_metrics(result, measured)
    else:
        _validate_vllm_backend_metrics(result, measured)
    _validate_recorded_runtime(result, framework)


def _metric_comparison(
    name: str,
    unit: str,
    lower_is_better: bool,
    nano_value: int | float,
    vllm_value: int | float,
) -> dict[str, Any]:
    if vllm_value == 0:
        ratio = None
        relative_delta = None
        improvement = None
        baseline_status = (
            "both_zero" if nano_value == 0 else "undefined_zero_vllm_baseline"
        )
    else:
        ratio = nano_value / vllm_value
        relative_delta = (nano_value - vllm_value) / vllm_value * 100.0
        improvement = (
            -relative_delta if lower_is_better else relative_delta
        )
        baseline_status = "defined"
    return {
        "metric": name,
        "unit": unit,
        "lower_is_better": lower_is_better,
        "nano_vllm": nano_value,
        "vllm": vllm_value,
        "absolute_delta": nano_value - vllm_value,
        "nano_over_vllm_ratio": ratio,
        "nano_relative_delta_percent": relative_delta,
        "nano_improvement_percent": improvement,
        "relative_baseline_status": baseline_status,
    }


def _compare_results(
    trace_path: Path,
    nano_result_path: Path,
    vllm_result_path: Path,
) -> dict[str, Any]:
    trace, trace_sha256 = _load_trace(trace_path)
    _require_cross_framework_lpm(trace)
    nano = json.loads(nano_result_path.read_text(encoding="utf-8"))
    vllm = json.loads(vllm_result_path.read_text(encoding="utf-8"))
    _validate_result(nano, "nano-vllm", trace, trace_sha256)
    _validate_result(vllm, "vllm", trace, trace_sha256)
    if (
        nano["runtime"]["benchmark_script_sha256"]
        != vllm["runtime"]["benchmark_script_sha256"]
    ):
        raise ValueError("cross-framework results used different benchmark scripts")
    nano_gpu = nano["runtime"]["gpu"]
    vllm_gpu = vllm["runtime"]["gpu"]
    stable_gpu_fields = (
        "uuid",
        "name",
        "driver_version",
        "compute_capability",
        "total_memory_bytes",
    )
    nano_gpu_identity = {name: nano_gpu[name] for name in stable_gpu_fields}
    vllm_gpu_identity = {name: vllm_gpu[name] for name in stable_gpu_fields}
    if nano_gpu_identity != vllm_gpu_identity:
        raise ValueError(
            "cross-framework results were not recorded on identical GPU hardware"
        )
    for name in ("torch", "torch_cuda"):
        if nano["runtime"][name] != vllm["runtime"][name]:
            raise ValueError(f"cross-framework results used different {name}")
    nano_utilization = nano["engine"]["requested"][
        "gpu_memory_utilization"
    ]
    vllm_utilization = vllm["engine"]["requested"][
        "gpu_memory_utilization"
    ]
    if nano_utilization != vllm_utilization:
        raise ValueError(
            "cross-framework results used different GPU memory utilization"
        )
    logical_blocks = trace["execution_contract"]["logical_kv_blocks"]
    if not (
        nano["engine"]["usable_logical_kv_blocks"]
        == vllm["engine"]["usable_logical_kv_blocks"]
        == logical_blocks
    ):
        raise ValueError("cross-framework usable KV capacity is not identical")

    nano_metrics = nano["metrics"]["comparable"]
    vllm_metrics = vllm["metrics"]["comparable"]
    table = [
        _metric_comparison(
            name,
            unit,
            lower_is_better,
            nano_metrics[name],
            vllm_metrics[name],
        )
        for name, unit, lower_is_better in COMPARABLE_METRIC_SPECS
    ]
    nano_flashinfer = nano["runtime"]["flashinfer"]
    vllm_flashinfer = vllm["runtime"]["flashinfer"]
    same_flashinfer_version = (
        nano_flashinfer.get("version") == vllm_flashinfer.get("version")
        and nano_flashinfer.get("cubin_package")
        == vllm_flashinfer.get("cubin_package")
        and nano_flashinfer.get("jit_cache_package")
        == vllm_flashinfer.get("jit_cache_package")
    )
    nano_backend = nano["metrics"]["backend_specific"]
    vllm_backend = vllm["metrics"]["backend_specific"]
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_kind": "eager_only_cross_framework_five_metric",
        "workload": trace["workload"],
        "workload_key": trace["workload_key"],
        "trace": _trace_proof(trace_path, trace_sha256, trace),
        "fairness": {
            "same_trace": True,
            "same_request_manifest": True,
            "same_input_token_ids": True,
            "same_arrival_order_and_time": True,
            "same_output_lengths": True,
            "same_ignore_eos": True,
            "same_seed": True,
            "same_gpu": True,
            "gpu_uuid": nano_gpu["uuid"],
            "same_model_path": True,
            "same_model_and_tokenizer_file_hashes": True,
            "same_benchmark_script_sha256": True,
            "same_torch": True,
            "same_torch_cuda": True,
            "same_gpu_memory_utilization": True,
            "gpu_memory_utilization": nano_utilization,
            "same_effective_usable_kv_capacity": True,
            "nano_usable_kv_blocks": logical_blocks,
            "vllm_configured_kv_blocks": (
                vllm["engine"]["configured_physical_kv_blocks"]
            ),
            "vllm_reserved_null_blocks": VLLM_NULL_BLOCK_COUNT,
            "vllm_usable_kv_blocks": logical_blocks,
            "definition_aligned_prompt_compute_counter": True,
            "definition_aligned_cached_block_eviction_counter": True,
            "vllm_observer_changes_scheduling_policy": False,
            "dtype": trace["execution_contract"]["dtype"],
            "both_eager_only": True,
            "prefix_caching_enabled": True,
            "chunked_prefill_enabled": True,
            "block_size": BLOCK_SIZE,
            "logical_kv_blocks": logical_blocks,
            "max_num_seqs": trace["execution_contract"]["max_num_seqs"],
            "max_num_batched_tokens": trace["execution_contract"][
                "max_num_batched_tokens"
            ],
            "same_flashinfer_package_stack": same_flashinfer_version,
            "nano_flashinfer": nano_flashinfer,
            "vllm_flashinfer": vllm_flashinfer,
        },
        "comparable_metrics": table,
        "control_metrics": {
            "preemption_count": {
                "nano_vllm": nano_backend["preemption_count"],
                "vllm": vllm_backend["preemption_count"],
                "headline_metric": False,
                "purpose": (
                    "diagnoses recomputation and validates prompt-token "
                    "conservation"
                ),
            }
        },
        "backend_specific_metrics": {
            "nano_vllm": nano_backend,
            "vllm": vllm_backend,
            "headline_comparable_fields": list(COMPARABLE_METRIC_NAMES),
            "native_breakdowns_are_diagnostics": True,
        },
        "interpretation_boundary": (
            "This compares complete eager-only engines on one fixed "
            "long-prefix KV-pressure trace: nano-vLLM with cache-aware LPM "
            "versus vLLM's default async FCFS policy. The two exact mechanism "
            "counters are definition-aligned. Timing remains an end-to-end "
            "framework result; if FlashInfer package stacks differ, it must not "
            "be attributed solely to the scheduling algorithm. Use the internal "
            "nano-vLLM LPM-versus-FCFS ablation for causal LPM claims, and do "
            "not generalize this single workload or attribute it to CUDA Graph."
        ),
        "inputs": {
            "nano_result": str(nano_result_path.resolve()),
            "vllm_result": str(vllm_result_path.resolve()),
        },
    }

def _parse_args(argv: TypingSequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-trace",
        help="generate one deterministic request trace",
    )
    generate.add_argument("workload", choices=tuple(WORKLOAD_CONFIGS))
    generate.add_argument("--model", required=True)
    generate.add_argument("--seed", type=int, default=2026)
    generate.add_argument("--output", type=Path, required=True)

    for command in ("run-nano", "run-vllm"):
        run = subparsers.add_parser(
            command,
            help=f"run the {command.removeprefix('run-')} backend",
        )
        run.add_argument("--trace", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    compare = subparsers.add_parser(
        "compare",
        help="validate fairness and combine two eager-only results",
    )
    compare.add_argument("--trace", type=Path, required=True)
    compare.add_argument("--nano-result", type=Path, required=True)
    compare.add_argument("--vllm-result", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _validate_gpu_utilization(value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")


def main(argv: TypingSequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "generate-trace":
        identity, vocab_size, forbidden = _load_model_identity(args.model)
        payload = _build_trace_data(
            args.workload,
            identity,
            vocab_size,
            forbidden,
            args.seed,
        )
    elif args.command == "run-nano":
        _validate_gpu_utilization(args.gpu_memory_utilization)
        payload = _run_nano(args.trace, args.gpu_memory_utilization)
    elif args.command == "run-vllm":
        _validate_gpu_utilization(args.gpu_memory_utilization)
        payload = _run_vllm(args.trace, args.gpu_memory_utilization)
    else:
        payload = _compare_results(
            args.trace,
            args.nano_result,
            args.vllm_result,
        )
    _write_json(args.output, payload)
    summary = {
        "command": args.command,
        "output": str(args.output),
        "workload": payload["workload"],
    }
    if args.command == "generate-trace":
        summary["trace_sha256"] = _sha256_file(args.output)
        summary["request_manifest_sha256"] = _canonical_sha256(
            _request_manifest(payload)
        )
    elif args.command == "compare":
        summary["comparable_metrics"] = payload["comparable_metrics"]
    else:
        summary["trace_sha256"] = payload["trace"]["sha256"]
        summary["metrics"] = payload["metrics"]["comparable"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
