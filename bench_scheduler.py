"""Targeted real-model benchmarks for nano-vllm scheduler policies.

Each invocation runs exactly one benchmark variant so CUDA/distributed state is
never reused across configurations::

    python bench_scheduler.py lpm --mode fcfs --model /path/to/Qwen3-0.6B
    python bench_scheduler.py lpm --mode lpm --model /path/to/Qwen3-0.6B
    python bench_scheduler.py in-batch --mode off --model /path/to/Qwen3-0.6B
    python bench_scheduler.py in-batch --mode on --model /path/to/Qwen3-0.6B

The workloads use deterministic, valid token IDs directly.  The physical KV
tensor is allocated normally, then the scheduler is given a smaller logical
BlockManager after verifying that every logical block maps into that tensor.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Any, Sequence as TypingSequence

import torch
import torch.distributed as dist

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


BLOCK_SIZE = 16
GROUP_NAMES = ("A", "B", "C", "D")
QWEN3_06B_SHAPE = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
}


@dataclass(frozen=True)
class RequestSpec:
    name: str
    prompt_token_ids: list[int]
    output_len: int
    kind: str
    group: str | None = None
    shared_prefix_len: int = 0


@dataclass
class RequestObservation:
    spec: RequestSpec
    seq: Sequence
    submitted_at: float = 0.0
    first_token_at: float | None = None
    completed_at: float | None = None
    persistent_hit_tokens: int = 0
    computed_prompt_tokens: int = 0
    computed_shared_prefix_tokens: int = 0


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _percentile(values: TypingSequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _prompt_fingerprint(token_ids: list[int]) -> str:
    encoded = ",".join(map(str, token_ids)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _git_metadata() -> dict[str, Any]:
    repository = Path(__file__).resolve().parent

    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "head": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": None if status is None else status.splitlines(),
        "bench_scheduler_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


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


def _flashinfer_metadata() -> dict[str, Any]:
    paths: dict[str, str] = {}
    try:
        from flashinfer.jit import env as flashinfer_env

        for name in (
            "FLASHINFER_AOT_DIR",
            "FLASHINFER_CACHE_DIR",
            "FLASHINFER_CUBIN_DIR",
            "FLASHINFER_GEN_SRC_DIR",
            "FLASHINFER_JIT_DIR",
            "FLASHINFER_WORKSPACE_DIR",
        ):
            value = getattr(flashinfer_env, name, None)
            if value is not None:
                paths[name] = str(value)
    except ImportError:
        pass
    module = sys.modules.get("flashinfer")
    return {
        "version": _package_version("flashinfer-python"),
        "cubin_package": _package_version("flashinfer-cubin"),
        "jit_cache_package": _package_version("flashinfer-jit-cache"),
        "module_path": getattr(module, "__file__", None),
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("FLASHINFER_")
        },
        "paths": paths,
    }


class TokenFactory:
    """Create deterministic valid token IDs while avoiding configured specials."""

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


class SchedulerMetrics:
    """Instance-local instrumentation for one measured scheduler phase."""

    def __init__(
        self,
        llm: LLM,
        observations: dict[int, RequestObservation],
        phase_started_at: float,
    ) -> None:
        self.scheduler = llm.scheduler
        self.block_manager = self.scheduler.block_manager
        self.observations = observations
        self.phase_started_at = phase_started_at
        self.steps: list[dict[str, Any]] = []
        self.persistent_hit_tokens = 0
        self.computed_prompt_tokens = 0
        self.computed_shared_prefix_tokens = 0
        self.cached_block_evictions = 0
        self.preemptions = 0
        self.temporary_deprioritized_ids: set[int] = set()
        self._claim_depth = 0
        self._active_step: dict[str, Any] | None = None
        self._pending_step: dict[str, Any] | None = None
        self._original_schedule = self.scheduler.schedule
        self._original_claim_prefix = self.block_manager.claim_prefix
        self._original_allocate_block = self.block_manager._allocate_block
        self._original_preempt = self.scheduler.preempt
        self._installed = False

    def _request_name(self, seq_id: int) -> str:
        observation = self.observations.get(seq_id)
        return observation.spec.name if observation is not None else str(seq_id)

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("scheduler metrics hooks are already installed")
        collector = self

        def claim_prefix_hook(
            manager: BlockManager,
            seq: Sequence,
            block_ids: list[int],
        ) -> None:
            collector._claim_depth += 1
            try:
                result = collector._original_claim_prefix(seq, block_ids)
            finally:
                collector._claim_depth -= 1
            hit_tokens = len(block_ids) * manager.block_size
            collector.persistent_hit_tokens += hit_tokens
            observation = collector.observations.get(seq.seq_id)
            if observation is not None:
                observation.persistent_hit_tokens += hit_tokens
            if collector._active_step is not None and hit_tokens:
                collector._active_step["persistent_claims"].append(
                    {
                        "seq_id": seq.seq_id,
                        "name": collector._request_name(seq.seq_id),
                        "block_count": len(block_ids),
                        "hit_tokens": hit_tokens,
                    }
                )
                collector._active_step["persistent_hit_tokens"] += hit_tokens
            return result

        def allocate_block_hook(
            manager: BlockManager,
            block_id: int,
        ):
            block = manager.blocks[block_id]
            is_eviction = (
                collector._claim_depth == 0
                and block.ref_count == 0
                and block.hash != -1
                and manager.hash_to_block_id.get(block.hash) == block_id
            )
            old_hash = block.hash
            result = collector._original_allocate_block(block_id)
            if is_eviction:
                collector.cached_block_evictions += 1
                if collector._active_step is not None:
                    collector._active_step["cached_block_evictions"] += 1
                    collector._active_step["evicted_blocks"].append(
                        {"block_id": block_id, "hash": old_hash}
                    )
            return result

        def preempt_hook(scheduler, seq: Sequence) -> None:
            result = collector._original_preempt(seq)
            collector.preemptions += 1
            if collector._active_step is not None:
                collector._active_step["preemptions"].append(
                    {
                        "seq_id": seq.seq_id,
                        "name": collector._request_name(seq.seq_id),
                    }
                )
            return result

        def schedule_hook(scheduler) -> list[Sequence]:
            if collector._pending_step is not None:
                raise RuntimeError("previous scheduler step was not finalized")
            step = {
                "step": len(collector.steps),
                "start_ms": (perf_counter() - collector.phase_started_at) * 1e3,
                "persistent_hit_tokens": 0,
                "persistent_claims": [],
                "cached_block_evictions": 0,
                "evicted_blocks": [],
                "preemptions": [],
            }
            collector._active_step = step
            try:
                scheduled = collector._original_schedule()
            except BaseException:
                collector._active_step = None
                raise

            scheduled_rows = []
            step_computed_prompt_tokens = 0
            step_computed_shared_tokens = 0
            for seq in scheduled:
                observation = collector.observations.get(seq.seq_id)
                prompt_remaining = max(
                    seq.num_prompt_tokens - seq.num_cached_tokens,
                    0,
                )
                computed_prompt = min(seq.num_new_tokens, prompt_remaining)
                shared_computed = 0
                if observation is not None and computed_prompt:
                    shared_len = observation.spec.shared_prefix_len
                    start = seq.num_cached_tokens
                    end = start + computed_prompt
                    shared_computed = max(
                        0,
                        min(end, shared_len) - min(start, shared_len),
                    )
                    observation.computed_prompt_tokens += computed_prompt
                    observation.computed_shared_prefix_tokens += shared_computed
                collector.computed_prompt_tokens += computed_prompt
                collector.computed_shared_prefix_tokens += shared_computed
                step_computed_prompt_tokens += computed_prompt
                step_computed_shared_tokens += shared_computed
                scheduled_rows.append(
                    {
                        "seq_id": seq.seq_id,
                        "name": collector._request_name(seq.seq_id),
                        "status": seq.status.name,
                        "cached_tokens_before_forward": seq.num_cached_tokens,
                        "new_tokens": seq.num_new_tokens,
                        "computed_prompt_tokens": computed_prompt,
                        "computed_shared_prefix_tokens": shared_computed,
                    }
                )

            temporary_ids = sorted(scheduler.temporary_deprioritized)
            collector.temporary_deprioritized_ids.update(temporary_ids)
            step.update(
                {
                    "scheduled": scheduled_rows,
                    "scheduled_names": [row["name"] for row in scheduled_rows],
                    "num_scheduled_prefill_seqs": (
                        scheduler.num_scheduled_prefill_seqs
                    ),
                    "num_scheduled_prefill_tokens": (
                        scheduler.num_scheduled_prefill_tokens
                    ),
                    "computed_prompt_tokens": step_computed_prompt_tokens,
                    "computed_shared_prefix_tokens": step_computed_shared_tokens,
                    "temporary_deprioritized_seq_ids": temporary_ids,
                    "temporary_deprioritized_names": [
                        collector._request_name(seq_id)
                        for seq_id in temporary_ids
                    ],
                    "temporary_prefix_index_size": len(
                        scheduler.temporary_prefix_index
                    ),
                    "waiting_after_schedule": [
                        collector._request_name(seq.seq_id)
                        for seq in scheduler.waiting
                    ],
                    "running_after_schedule": [
                        collector._request_name(seq.seq_id)
                        for seq in scheduler.running
                    ],
                    "free_blocks_after_schedule": len(
                        collector.block_manager.free_block_ids
                    ),
                    "used_blocks_after_schedule": len(
                        collector.block_manager.used_block_ids
                    ),
                }
            )
            collector._active_step = None
            collector._pending_step = step
            return scheduled

        self.scheduler.schedule = MethodType(schedule_hook, self.scheduler)
        self.block_manager.claim_prefix = MethodType(
            claim_prefix_hook,
            self.block_manager,
        )
        self.block_manager._allocate_block = MethodType(
            allocate_block_hook,
            self.block_manager,
        )
        self.scheduler.preempt = MethodType(preempt_hook, self.scheduler)
        self._installed = True

    def finish_step(self, ended_at: float) -> None:
        if self._pending_step is None:
            raise RuntimeError("scheduler step hook did not create a record")
        step = self._pending_step
        step["end_ms"] = (ended_at - self.phase_started_at) * 1e3
        step["duration_ms"] = step["end_ms"] - step["start_ms"]
        step["new_first_tokens"] = [
            observation.spec.name
            for observation in self.observations.values()
            if observation.first_token_at == ended_at
        ]
        step["new_completions"] = [
            observation.spec.name
            for observation in self.observations.values()
            if observation.completed_at == ended_at
        ]
        self.steps.append(step)
        self._pending_step = None

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.scheduler.schedule = self._original_schedule
        self.block_manager.claim_prefix = self._original_claim_prefix
        self.block_manager._allocate_block = self._original_allocate_block
        self.scheduler.preempt = self._original_preempt
        self._installed = False


def _add_request(llm: LLM, spec: RequestSpec) -> RequestObservation:
    params = SamplingParams(
        temperature=1.0,
        max_tokens=spec.output_len,
        ignore_eos=True,
    )
    llm.add_request(spec.prompt_token_ids, params)
    seq = llm.scheduler.waiting[-1]
    if seq.prompt_token_ids != spec.prompt_token_ids:
        raise AssertionError("LLM changed direct prompt token IDs")
    return RequestObservation(spec=spec, seq=seq)


def _run_unmeasured_to_completion(llm: LLM) -> None:
    while not llm.is_finished():
        llm.step()


def _run_measured_to_completion(
    llm: LLM,
    observations: dict[int, RequestObservation],
    collector: SchedulerMetrics,
) -> float:
    while not llm.is_finished():
        llm.step()
        ended_at = perf_counter()
        for observation in observations.values():
            if (
                observation.first_token_at is None
                and observation.seq.num_completion_tokens > 0
            ):
                observation.first_token_at = ended_at
            if observation.completed_at is None and observation.seq.is_finished:
                observation.completed_at = ended_at
        collector.finish_step(ended_at)
    return perf_counter()


def _common_specs(
    factory: TokenFactory,
    shared_prefix_len: int,
    unique_suffix_len: int,
    output_len: int,
    groups: TypingSequence[str],
) -> tuple[dict[str, list[int]], dict[str, RequestSpec]]:
    prefixes = {
        group: factory.tokens(shared_prefix_len)
        for group in groups
    }
    first_blocks = {
        tuple(prefix[:BLOCK_SIZE]) for prefix in prefixes.values()
    }
    if len(first_blocks) != len(groups):
        raise AssertionError("generated group prefixes are not distinct")
    leaders = {
        group: RequestSpec(
            name=f"{group}0",
            prompt_token_ids=(
                prefixes[group] + factory.tokens(unique_suffix_len)
            ),
            output_len=output_len,
            kind="leader",
            group=group,
            shared_prefix_len=shared_prefix_len,
        )
        for group in groups
    }
    return prefixes, leaders


def _build_lpm_workload(
    factory: TokenFactory,
) -> tuple[list[RequestSpec], list[RequestSpec]]:
    shared_prefix_len = 4096
    unique_suffix_len = 128
    output_len = 64
    groups = GROUP_NAMES[:3]
    prefixes, leader_map = _common_specs(
        factory,
        shared_prefix_len,
        unique_suffix_len,
        output_len,
        groups,
    )
    leaders = [leader_map[group] for group in groups]
    cold: list[RequestSpec] = []
    for index in range(1, 13):
        prompt_len = factory.rng.randint(512, 1024)
        cold.append(
            RequestSpec(
                name=f"Cold{index}",
                prompt_token_ids=factory.tokens(prompt_len),
                output_len=output_len,
                kind="cold",
            )
        )
    followers: list[RequestSpec] = []
    for follower_index in range(1, 5):
        for group in groups:
            followers.append(
                RequestSpec(
                    name=f"{group}{follower_index}",
                    prompt_token_ids=(
                        prefixes[group] + factory.tokens(unique_suffix_len)
                    ),
                    output_len=output_len,
                    kind="follower",
                    group=group,
                    shared_prefix_len=shared_prefix_len,
                )
            )
    return leaders, cold + followers


def _build_in_batch_workload(factory: TokenFactory) -> list[RequestSpec]:
    shared_prefix_len = 2048
    unique_suffix_len = 128
    output_len = 64
    prefixes, _ = _common_specs(
        factory,
        shared_prefix_len,
        unique_suffix_len,
        output_len,
        GROUP_NAMES,
    )
    requests: list[RequestSpec] = []
    for group in GROUP_NAMES:
        for request_index in range(1, 5):
            requests.append(
                RequestSpec(
                    name=f"{group}{request_index}",
                    prompt_token_ids=(
                        prefixes[group] + factory.tokens(unique_suffix_len)
                    ),
                    output_len=output_len,
                    kind="burst",
                    group=group,
                    shared_prefix_len=shared_prefix_len,
                )
            )
    return requests


def _request_rows(
    observations: dict[int, RequestObservation],
    phase_started_at: float,
) -> list[dict[str, Any]]:
    rows = []
    for observation in observations.values():
        if observation.first_token_at is None or observation.completed_at is None:
            raise AssertionError(f"request {observation.spec.name} did not finish")
        spec = observation.spec
        rows.append(
            {
                "seq_id": observation.seq.seq_id,
                "name": spec.name,
                "kind": spec.kind,
                "group": spec.group,
                "prompt_tokens": len(spec.prompt_token_ids),
                "output_tokens": observation.seq.num_completion_tokens,
                "shared_prefix_tokens": spec.shared_prefix_len,
                "prompt_sha256": _prompt_fingerprint(spec.prompt_token_ids),
                "prompt_token_ids_head": spec.prompt_token_ids[:8],
                "prompt_token_ids_tail": spec.prompt_token_ids[-8:],
                "submitted_ms": (
                    observation.submitted_at - phase_started_at
                ) * 1e3,
                "ttft_ms": (
                    observation.first_token_at - observation.submitted_at
                ) * 1e3,
                "completion_ms": (
                    observation.completed_at - observation.submitted_at
                ) * 1e3,
                "persistent_hit_tokens": observation.persistent_hit_tokens,
                "computed_prompt_tokens": observation.computed_prompt_tokens,
                "computed_shared_prefix_tokens": (
                    observation.computed_shared_prefix_tokens
                ),
            }
        )
    return rows


def _metric_summary(
    observations: dict[int, RequestObservation],
    collector: SchedulerMetrics,
    phase_started_at: float,
    phase_ended_at: float,
    prefix_groups: int,
    shared_prefix_len: int,
) -> dict[str, Any]:
    request_rows = _request_rows(observations, phase_started_at)
    elapsed_s = phase_ended_at - phase_started_at
    ttfts = [row["ttft_ms"] for row in request_rows]
    total_output_tokens = sum(row["output_tokens"] for row in request_rows)
    duplicate_prefill_tokens = max(
        collector.computed_shared_prefix_tokens
        - prefix_groups * shared_prefix_len,
        0,
    )
    return {
        "persistent_cache_hit_tokens": collector.persistent_hit_tokens,
        "persistent_hit_tokens_after_first_step": sum(
            step["persistent_hit_tokens"] for step in collector.steps[1:]
        ),
        "actually_computed_prompt_tokens": collector.computed_prompt_tokens,
        "computed_shared_prefix_tokens": (
            collector.computed_shared_prefix_tokens
        ),
        "duplicate_prefill_tokens": duplicate_prefill_tokens,
        "cached_block_eviction_count": collector.cached_block_evictions,
        "preemption_count": collector.preemptions,
        "temporary_deprioritized_request_count": len(
            collector.temporary_deprioritized_ids
        ),
        "temporary_deprioritized_seq_ids": sorted(
            collector.temporary_deprioritized_ids
        ),
        "p95_ttft_ms": _percentile(ttfts, 95.0),
        "total_batch_completion_s": elapsed_s,
        "request_throughput_rps": len(observations) / elapsed_s,
        "output_token_throughput_tps": total_output_tokens / elapsed_s,
        "request_count": len(observations),
        "output_token_count": total_output_tokens,
        "scheduler_step_count": len(collector.steps),
    }


def _runtime_metadata(llm: LLM) -> dict[str, Any]:
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    hf_config = llm.config.hf_config
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": [sys.executable, *sys.argv],
        "git": _git_metadata(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_driver": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "transformers": _package_version("transformers"),
        "flashinfer": _flashinfer_metadata(),
        "gpu": properties.name,
        "gpu_compute_capability": (
            f"{properties.major}.{properties.minor}"
        ),
        "gpu_total_memory_bytes": properties.total_memory,
        "model": os.path.realpath(llm.config.model),
        "model_type": hf_config.model_type,
        "model_dtype": str(llm.model_runner.dtype),
        "model_shape": {
            name: getattr(hf_config, name)
            for name in QWEN3_06B_SHAPE
        },
        "enforce_eager": llm.config.enforce_eager,
    }


def _validate_runtime(llm: LLM, logical_blocks: int) -> int:
    if not llm.scheduler.is_finished():
        raise AssertionError("logical BlockManager must be installed while idle")
    if llm.config.kvcache_block_size != BLOCK_SIZE:
        raise ValueError("targeted scheduler benchmarks require block_size=16")
    if llm.config.hf_config.model_type != "qwen3":
        raise ValueError("targeted scheduler benchmarks require a Qwen3 model")
    actual_shape = {
        name: getattr(llm.config.hf_config, name, None)
        for name in QWEN3_06B_SHAPE
    }
    if actual_shape != QWEN3_06B_SHAPE:
        raise ValueError(
            "targeted scheduler benchmarks require Qwen3-0.6B shape; "
            f"expected {QWEN3_06B_SHAPE}, got {actual_shape}"
        )
    if llm.model_runner.dtype != torch.bfloat16:
        raise ValueError("targeted scheduler benchmarks require BF16 weights")
    physical_blocks = llm.config.num_kvcache_blocks
    if logical_blocks > physical_blocks:
        raise ValueError(
            f"logical KV cap {logical_blocks} exceeds physical cache "
            f"({physical_blocks} blocks); increase --gpu-memory-utilization"
        )
    llm.scheduler.block_manager = BlockManager(logical_blocks, BLOCK_SIZE)
    return physical_blocks


def _benchmark_lpm(
    llm: LLM,
    args: argparse.Namespace,
    factory: TokenFactory,
) -> dict[str, Any]:
    leaders, measured_specs = _build_lpm_workload(factory)
    phase1_resident_blocks = 3 * (4096 + 128 + 64) // BLOCK_SIZE
    first_lpm_batch_blocks = (
        3 * (4096 // BLOCK_SIZE)
        + 4 * ((128 + 64) // BLOCK_SIZE)
    )
    minimum_logical_blocks = max(
        phase1_resident_blocks,
        first_lpm_batch_blocks,
    )
    if args.logical_kv_blocks < minimum_logical_blocks:
        raise ValueError(
            "LPM workload needs at least "
            f"{minimum_logical_blocks} logical blocks for phase-1 residency "
            "and the four-request priority batch"
        )

    for leader in leaders:
        _add_request(llm, leader)
        _run_unmeasured_to_completion(llm)
    if not llm.scheduler.is_finished():
        raise AssertionError("leader preparation did not drain the scheduler")

    observations: dict[int, RequestObservation] = {}
    for spec in measured_specs:
        observation = _add_request(llm, spec)
        observations[observation.seq.seq_id] = observation

    followers = [
        observation
        for observation in observations.values()
        if observation.spec.kind == "follower"
    ]
    phase1_cache_hits = {
        observation.spec.name: len(
            llm.scheduler.block_manager.match_prefix(observation.seq)
        ) * BLOCK_SIZE
        for observation in followers
    }
    if min(phase1_cache_hits.values()) < 4096:
        raise AssertionError(
            "phase 1 did not leave all three 4096-token prefixes resident"
        )

    torch.cuda.synchronize()
    phase_started_at = perf_counter()
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

    metrics = _metric_summary(
        observations,
        collector,
        phase_started_at,
        phase_ended_at,
        prefix_groups=3,
        shared_prefix_len=4096,
    )
    first_names = collector.steps[0]["scheduled_names"]
    assertions = {
        "all_phase1_prefixes_resident": min(phase1_cache_hits.values()) >= 4096,
        "temporary_deprioritization_disabled": (
            metrics["temporary_deprioritized_request_count"] == 0
        ),
        "cold_requests_triggered_cached_eviction": (
            metrics["cached_block_eviction_count"] > 0
        ),
    }
    if args.mode == "fcfs":
        assertions["first_batch_is_fcfs_cold"] = first_names == [
            "Cold1", "Cold2", "Cold3", "Cold4"
        ]
    else:
        assertions["first_batch_prioritizes_resident_prefixes"] = (
            first_names == ["A1", "B1", "C1", "A2"]
            and collector.steps[0]["persistent_hit_tokens"] >= 4 * 4096
        )
    if not all(assertions.values()):
        raise AssertionError(f"LPM causal assertion failed: {assertions}")

    return {
        "benchmark": "lpm_cache_eviction_stress",
        "variant": args.mode,
        "workload": {
            "shared_prefix_len": 4096,
            "unique_suffix_len": 128,
            "output_len": 64,
            "prefix_groups": 3,
            "followers_per_group": 4,
            "cold_request_count": 12,
            "cold_prompt_len_range": [512, 1024],
            "arrival_order": [spec.name for spec in measured_specs],
            "leader_order": [spec.name for spec in leaders],
        },
        "phase1_cache_hit_tokens_by_follower": phase1_cache_hits,
        "causal_assertions": assertions,
        "metrics": metrics,
        "raw": {
            "requests": _request_rows(observations, phase_started_at),
            "steps": collector.steps,
        },
    }


def _benchmark_in_batch(
    llm: LLM,
    args: argparse.Namespace,
    factory: TokenFactory,
) -> dict[str, Any]:
    minimum_logical_blocks = 4 * (
        (2048 + 128 + 64) // BLOCK_SIZE
    )
    if args.logical_kv_blocks < minimum_logical_blocks:
        raise ValueError(
            "in-batch workload needs at least "
            f"{minimum_logical_blocks} logical blocks for four active requests"
        )
    specs = _build_in_batch_workload(factory)
    observations: dict[int, RequestObservation] = {}
    for spec in specs:
        observation = _add_request(llm, spec)
        observations[observation.seq.seq_id] = observation

    torch.cuda.synchronize()
    phase_started_at = perf_counter()
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

    metrics = _metric_summary(
        observations,
        collector,
        phase_started_at,
        phase_ended_at,
        prefix_groups=4,
        shared_prefix_len=2048,
    )
    first_step = collector.steps[0]
    first_names = first_step["scheduled_names"]
    assertions: dict[str, bool]
    if args.mode == "off":
        assertions = {
            "first_batch_repeats_group_a": first_names
            == ["A1", "A2", "A3", "A4"],
            "no_temporary_deprioritization": (
                metrics["temporary_deprioritized_request_count"] == 0
            ),
            "duplicate_shared_prefill_observed": (
                metrics["duplicate_prefill_tokens"] == 4 * 3 * 2048
            ),
            "no_later_persistent_hits": (
                metrics["persistent_hit_tokens_after_first_step"] == 0
            ),
        }
    else:
        expected_followers = {
            f"{group}{index}"
            for group in GROUP_NAMES
            for index in range(2, 5)
        }
        first_temporary = set(first_step["temporary_deprioritized_names"])
        assertions = {
            "first_batch_has_one_leader_per_group": first_names
            == ["A1", "B1", "C1", "D1"],
            "first_detection_marks_all_followers": (
                first_temporary == expected_followers
            ),
            "followers_hit_persistent_prefix_later": (
                metrics["persistent_hit_tokens_after_first_step"]
                >= 12 * 2048
            ),
            "no_duplicate_shared_prefill": (
                metrics["duplicate_prefill_tokens"] == 0
            ),
        }
    if not all(assertions.values()):
        raise AssertionError(
            f"in-batch causal assertion failed: {assertions}"
        )

    return {
        "benchmark": "in_batch_prefix_burst",
        "variant": args.mode,
        "workload": {
            "shared_prefix_len": 2048,
            "unique_suffix_len": 128,
            "output_len": 64,
            "prefix_groups": 4,
            "requests_per_group": 4,
            "arrival_order": [spec.name for spec in specs],
        },
        "causal_assertions": assertions,
        "metrics": metrics,
        "raw": {
            "requests": _request_rows(observations, phase_started_at),
            "steps": collector.steps,
        },
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


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="local Qwen3-0.6B path")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--attention-mode", choices=("unified", "split"), default="unified")
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use eager execution (recommended for scheduler timing)",
    )


def _parse_args(argv: TypingSequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    lpm = subparsers.add_parser("lpm", help="cache-eviction stress")
    lpm.add_argument("--mode", choices=("fcfs", "lpm"), required=True)
    lpm.add_argument("--logical-kv-blocks", type=int, default=896)
    _add_common_arguments(lpm)

    in_batch = subparsers.add_parser(
        "in-batch",
        help="simultaneous shared-prefix burst",
    )
    in_batch.add_argument("--mode", choices=("off", "on"), required=True)
    in_batch.add_argument("--logical-kv-blocks", type=int, default=640)
    _add_common_arguments(in_batch)
    return parser.parse_args(argv)


def _default_output_path(args: argparse.Namespace) -> Path:
    return Path("benchmark_results") / f"scheduler_{args.experiment}_{args.mode}.json"


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.logical_kv_blocks <= 0:
        raise ValueError("--logical-kv-blocks must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    minimum_logical_blocks = (
        816 if args.experiment == "lpm" else 560
    )
    if args.logical_kv_blocks < minimum_logical_blocks:
        raise ValueError(
            f"{args.experiment} workload needs at least "
            f"{minimum_logical_blocks} logical KV blocks, got "
            f"{args.logical_kv_blocks}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("bench_scheduler.py requires an NVIDIA CUDA GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.experiment == "lpm":
        max_model_len = 4352
        max_num_batched_tokens = 16384
        enable_lpm = args.mode == "lpm"
        enable_temporary = False
    else:
        max_model_len = 2304
        max_num_batched_tokens = 4 * (2048 + 128)
        enable_lpm = True
        enable_temporary = args.mode == "on"

    llm: LLM | None = None
    try:
        llm = LLM(
            args.model,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=4,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=args.enforce_eager,
            kvcache_block_size=BLOCK_SIZE,
            chunked_prefill=False,
            enable_lpm=enable_lpm,
            enable_in_batch_prefix_deprioritization=enable_temporary,
            attention_backend="flashinfer",
            attention_mode=args.attention_mode,
        )
        metadata = _runtime_metadata(llm)
        physical_blocks = _validate_runtime(llm, args.logical_kv_blocks)
        # Model warmup samples tokens. Reset here so the measured workload's
        # sampling stream is independent of initialization internals.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        forbidden = {
            token_id
            for token_id in (
                llm.tokenizer.eos_token_id,
                llm.tokenizer.pad_token_id,
                llm.tokenizer.bos_token_id,
            )
            if token_id is not None
        }
        factory = TokenFactory(
            llm.config.hf_config.vocab_size,
            args.seed,
            forbidden,
        )
        if args.experiment == "lpm":
            payload = _benchmark_lpm(llm, args, factory)
        else:
            payload = _benchmark_in_batch(llm, args, factory)
        payload["schema_version"] = 1
        payload["metadata"] = metadata
        payload["config"] = {
            "seed": args.seed,
            "block_size": BLOCK_SIZE,
            "logical_kv_blocks": args.logical_kv_blocks,
            "physical_kv_blocks": physical_blocks,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": 4,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_lpm": enable_lpm,
            "enable_in_batch_prefix_deprioritization": enable_temporary,
            "chunked_prefill": False,
            "attention_backend": "flashinfer",
            "attention_mode": args.attention_mode,
            "enforce_eager": args.enforce_eager,
            "argv": [sys.executable, *sys.argv],
        }
        return payload
    finally:
        _shutdown_llm(llm)


def main(argv: TypingSequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = _run(args)
    output_path = args.output or _default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "benchmark": payload["benchmark"],
                "variant": payload["variant"],
                "output": str(output_path),
                "causal_assertions": payload["causal_assertions"],
                "metrics": payload["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
