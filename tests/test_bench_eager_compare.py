from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench_eager_compare.py"
if str(_SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("bench_eager_compare", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


@pytest.fixture
def model_identity() -> dict:
    return {
        "path": "/tmp/Qwen3-0.6B",
        "model_type": "qwen3",
        "shape": dict(benchmark.TARGET_MODEL_SHAPE),
        "vocab_size": 512,
        "configured_dtype": "torch.bfloat16",
        "tokenizer_class": "FakeTokenizer",
        "special_token_ids": [0, 1, 2],
        "files_sha256": {
            "config.json": "0" * 64,
            "tokenizer.json": "1" * 64,
            "model.safetensors": "2" * 64,
        },
    }


def _trace(workload: str, model_identity: dict) -> dict:
    return benchmark._build_trace_data(
        workload,
        copy.deepcopy(model_identity),
        vocab_size=512,
        forbidden_token_ids={0, 1, 2},
        seed=2026,
    )


def test_vllm_internal_request_id_maps_to_external_output_id():
    assert benchmark._vllm_output_request_id("17-deadbeef") == "17"
    assert benchmark._vllm_output_request_id("17") == "17"


def test_lpm_trace_reuses_target_workload(model_identity):
    trace = _trace("lpm", model_identity)

    assert trace["schema_version"] == 2
    assert trace["workload"] == "shared_long_prefix_kv_pressure"
    assert trace["execution_contract"] == {
        "dtype": "bfloat16",
        "temperature": 1.0,
        "ignore_eos": True,
        "prefix_caching": True,
        "chunked_prefill": True,
        "block_size": 16,
        "logical_kv_blocks": 896,
        "max_model_len": 4352,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 4,
        "tensor_parallel_size": 1,
        "cuda_graphs": "none",
    }
    priming, measured = trace["phases"]
    assert priming["submission"] == "sequential_to_completion"
    assert [request["request_id"] for request in priming["requests"]] == [
        "A0",
        "B0",
        "C0",
    ]
    expected_measured = [f"Cold{index}" for index in range(1, 13)]
    expected_measured.extend(
        f"{group}{index}"
        for index in range(1, 5)
        for group in ("A", "B", "C")
    )
    assert [
        request["request_id"] for request in measured["requests"]
    ] == expected_measured
    assert len(measured["requests"]) == 24

    group_prefixes = {
        request["prefix_group"]: request["input_token_ids"][:4096]
        for request in priming["requests"]
    }
    for request in measured["requests"]:
        if request["kind"] == "follower":
            assert request["prompt_len"] == 4224
            assert request["output_len"] == 64
            assert (
                request["input_token_ids"][:4096]
                == group_prefixes[request["prefix_group"]]
            )


def test_in_batch_trace_preserves_grouped_arrival_order(model_identity):
    trace = _trace("in-batch", model_identity)
    requests = benchmark._measured_phase(trace)["requests"]

    assert len(trace["phases"]) == 1
    assert trace["execution_contract"]["logical_kv_blocks"] == 2240
    assert trace["execution_contract"]["max_num_batched_tokens"] == 10240
    assert trace["execution_contract"]["max_num_seqs"] == 16
    assert trace["workload_contract"] == {
        "prefix_groups": 4,
        "shared_prefix_len": 2048,
        "requests_per_group": 4,
        "unique_suffix_len": 128,
        "output_len": 64,
        "arrival_layout": "grouped_by_prefix",
        "expected_off_first_step_admissions": 4,
        "expected_on_first_step_admissions": 16,
        "same_step_minimum_blocks": 704,
        "cold_worst_case_blocks": 2240,
        "comparison_logical_kv_blocks": 2240,
    }
    assert [request["request_id"] for request in requests] == [
        f"{group}{index}"
        for group in ("A", "B", "C", "D")
        for index in range(1, 5)
    ]
    assert [request["arrival_order"] for request in requests] == list(range(16))
    assert {request["arrival_time_ms"] for request in requests} == {0.0}
    assert {request["prompt_len"] for request in requests} == {2176}
    assert {request["output_len"] for request in requests} == {64}
    for group in ("A", "B", "C", "D"):
        group_requests = [
            request for request in requests if request["prefix_group"] == group
        ]
        prefixes = {
            tuple(request["input_token_ids"][:2048])
            for request in group_requests
        }
        assert len(prefixes) == 1


def test_trace_generation_is_deterministic(model_identity):
    first = _trace("lpm", model_identity)
    second = _trace("lpm", model_identity)

    assert first == second
    assert benchmark._canonical_sha256(first) == benchmark._canonical_sha256(second)


def test_trace_file_sha_and_request_manifest_are_stable(tmp_path, model_identity):
    trace = _trace("in-batch", model_identity)
    path = tmp_path / "trace.json"
    benchmark._write_json(path, trace)

    loaded, digest = benchmark._load_trace(path)

    assert loaded == trace
    assert digest == benchmark._sha256_file(path)
    assert benchmark._canonical_sha256(
        benchmark._request_manifest(loaded)
    ) == benchmark._canonical_sha256(benchmark._request_manifest(trace))


def test_trace_validation_rejects_token_or_length_drift(model_identity):
    trace = _trace("in-batch", model_identity)
    broken_length = copy.deepcopy(trace)
    broken_length["phases"][0]["requests"][0]["prompt_len"] -= 1
    with pytest.raises(ValueError, match="prompt_len"):
        benchmark._validate_trace(broken_length)

    broken_prefix = copy.deepcopy(trace)
    broken_prefix["phases"][0]["requests"][1]["input_token_ids"][0] = 0
    with pytest.raises(ValueError, match="prefix group"):
        benchmark._validate_trace(broken_prefix)


def test_backend_configs_force_eager_and_match_budgets(model_identity):
    trace = _trace("lpm", model_identity)
    contract = trace["execution_contract"]
    nano = benchmark._nano_engine_kwargs(trace, 0.85)
    vllm = benchmark._vllm_engine_kwargs(trace, 0.85)

    assert nano["enforce_eager"] is True
    assert nano["cudagraph_mode"] == "none"
    assert nano["chunked_prefill"] is True
    assert nano["enable_same_step_prefix_reuse"] is True
    assert "enable_in_batch_prefix_deprioritization" not in nano
    assert vllm["enforce_eager"] is True
    assert vllm["enable_prefix_caching"] is True
    assert vllm["enable_chunked_prefill"] is True
    assert nano["kvcache_block_size"] == vllm["block_size"] == 16
    assert vllm["num_gpu_blocks_override"] == contract["logical_kv_blocks"]
    for key in ("max_model_len", "max_num_batched_tokens", "max_num_seqs"):
        assert nano[key] == vllm[key] == contract[key]


def _fake_result(
    framework: str,
    trace_path: Path,
    trace: dict,
    trace_sha256: str,
) -> dict:
    result = benchmark._base_result(
        framework,
        trace_path,
        trace_sha256,
        trace,
    )
    scale = 1.0 if framework == "nano-vllm" else 1.25
    requests = []
    for request in benchmark._measured_phase(trace)["requests"]:
        row = {
            "request_id": request["request_id"],
            "arrival_order": request["arrival_order"],
            "prompt_len": request["prompt_len"],
            "prompt_sha256": benchmark._prompt_sha256(
                request["input_token_ids"]
            ),
            "requested_output_len": request["output_len"],
            "output_tokens": request["output_len"],
            "ttft_ms": (10.0 + request["arrival_order"]) * scale,
            "completion_ms": (100.0 + request["arrival_order"]) * scale,
            "prefix_group": request["prefix_group"],
        }
        if framework == "nano-vllm":
            row.update(
                {
                    "initial_persistent_hit_tokens": 0,
                    "same_step_hit_tokens": 0,
                    "computed_prompt_tokens": request["prompt_len"],
                }
            )
        requests.append(row)
    priming = [
        {
            "request_id": request["request_id"],
            "prompt_sha256": benchmark._prompt_sha256(
                request["input_token_ids"]
            ),
            "output_tokens": request["output_len"],
        }
        for phase in benchmark._priming_phases(trace)
        for request in phase["requests"]
    ]
    elapsed = 2.0 if framework == "nano-vllm" else 2.4
    gpu_utilization = 0.85
    requested = (
        benchmark._nano_engine_kwargs(trace, gpu_utilization)
        if framework == "nano-vllm"
        else benchmark._vllm_engine_kwargs(trace, gpu_utilization)
    )
    requested = {"model": trace["model"]["path"], **requested}
    return {
        **result,
        "runtime": {
            "command": f"python bench_eager_compare.py run-{framework}",
            "torch": "2.0.0",
            "torch_cuda": "12.0",
            "cuda_driver_and_gpu": "Fake GPU, GPU-fake, 999.0, 123 MiB",
            "benchmark_script_sha256": "a" * 64,
            "gpu": {
                "name": "Fake GPU",
                "uuid": "GPU-fake",
                "driver_version": "999.0",
                "compute_capability": "9.9",
                "total_memory_bytes": 123,
            },
            "flashinfer": {"version": "0.0.fake"},
            "framework": {
                "version": "0.0.fake",
                "git": {"head": "f" * 40},
            },
        },
        "engine": {
            "requested": requested,
            "fairness_checks": {"all": True},
            "actual_model_path": trace["model"]["path"],
            "actual_tokenizer_path": trace["model"]["path"],
            "actual_dtype": "torch.bfloat16",
            "seed": trace["seed"],
            "seed_reset_before_measured": True,
            "enforce_eager": True,
            "cudagraph_mode": "none",
        },
        "metrics": {
            "comparable": {
                "p95_ttft_ms": benchmark._percentile(
                    [row["ttft_ms"] for row in requests],
                    95.0,
                ),
                "request_throughput_rps": len(requests) / elapsed,
                "total_batch_completion_s": elapsed,
            },
            "backend_specific": (
                {
                    "definition": framework,
                    "initial_persistent_hit_tokens": 0,
                    "same_step_hit_tokens": 0,
                    "claimed_prefix_tokens": 0,
                    "computed_prompt_tokens": sum(
                        row["prompt_len"] for row in requests
                    ),
                    "total_prompt_tokens": sum(
                        row["prompt_len"] for row in requests
                    ),
                    "prompt_token_conservation": {
                        "initial_persistent_hit_tokens": 0,
                        "same_step_hit_tokens": 0,
                        "computed_prompt_tokens": sum(
                            row["prompt_len"] for row in requests
                        ),
                        "accounted_prompt_tokens": sum(
                            row["prompt_len"] for row in requests
                        ),
                        "total_prompt_tokens": sum(
                            row["prompt_len"] for row in requests
                        ),
                        "delta_tokens": 0,
                        "balanced": True,
                    },
                    "same_step_reused_request_count": 0,
                    "same_step_reused_blocks": 0,
                    "first_step_prefill_admission_count": min(
                        4, len(requests)
                    ),
                    "max_step_prefill_admission_count": min(
                        4, len(requests)
                    ),
                }
                if framework == "nano-vllm"
                else {"definition": framework}
            ),
        },
        "priming_requests": priming,
        "requests": requests,
    }


def test_compare_accepts_only_same_trace_and_eager_results(
    tmp_path,
    model_identity,
):
    trace = _trace("in-batch", model_identity)
    trace_path = tmp_path / "trace.json"
    nano_path = tmp_path / "nano.json"
    vllm_path = tmp_path / "vllm.json"
    benchmark._write_json(trace_path, trace)
    trace_sha256 = benchmark._sha256_file(trace_path)
    benchmark._write_json(
        nano_path,
        _fake_result("nano-vllm", trace_path, trace, trace_sha256),
    )
    benchmark._write_json(
        vllm_path,
        _fake_result("vllm", trace_path, trace, trace_sha256),
    )

    comparison = benchmark._compare_results(trace_path, nano_path, vllm_path)

    assert comparison["fairness"]["same_trace"] is True
    assert comparison["fairness"]["both_eager_only"] is True
    assert comparison["fairness"]["same_benchmark_script_sha256"] is True
    assert [row["metric"] for row in comparison["comparable_metrics"]] == [
        "p95_ttft_ms",
        "request_throughput_rps",
        "total_batch_completion_s",
    ]
    assert comparison["backend_specific_metrics"][
        "cross_framework_comparable"
    ] is False


def test_compare_rejects_graph_or_trace_mismatch(tmp_path, model_identity):
    trace = _trace("in-batch", model_identity)
    trace_path = tmp_path / "trace.json"
    nano_path = tmp_path / "nano.json"
    vllm_path = tmp_path / "vllm.json"
    benchmark._write_json(trace_path, trace)
    trace_sha256 = benchmark._sha256_file(trace_path)
    nano = _fake_result("nano-vllm", trace_path, trace, trace_sha256)
    vllm = _fake_result("vllm", trace_path, trace, trace_sha256)
    nano["engine"]["cudagraph_mode"] = "full_decode_only"
    benchmark._write_json(nano_path, nano)
    benchmark._write_json(vllm_path, vllm)
    with pytest.raises(ValueError, match="CUDA Graph"):
        benchmark._compare_results(trace_path, nano_path, vllm_path)

    nano["engine"]["cudagraph_mode"] = "none"
    vllm["trace"]["sha256"] = "0" * 64
    benchmark._write_json(nano_path, nano)
    benchmark._write_json(vllm_path, vllm)
    with pytest.raises(ValueError, match="different trace SHA"):
        benchmark._compare_results(trace_path, nano_path, vllm_path)

    vllm["trace"]["sha256"] = trace_sha256
    vllm["runtime"]["benchmark_script_sha256"] = "b" * 64
    benchmark._write_json(vllm_path, vllm)
    with pytest.raises(ValueError, match="different benchmark scripts"):
        benchmark._compare_results(trace_path, nano_path, vllm_path)


def test_cli_requires_separate_backend_commands():
    nano = benchmark._parse_args(
        ["run-nano", "--trace", "trace.json", "--output", "nano.json"]
    )
    vllm = benchmark._parse_args(
        ["run-vllm", "--trace", "trace.json", "--output", "vllm.json"]
    )

    assert nano.command == "run-nano"
    assert vllm.command == "run-vllm"
    assert nano.trace == vllm.trace



def test_trace_validation_rejects_phase_or_workload_kind_drift(model_identity):
    trace = _trace("in-batch", model_identity)
    broken_submission = copy.deepcopy(trace)
    broken_submission["phases"][0]["submission"] = "sequential_to_completion"
    with pytest.raises(ValueError, match="submission mode"):
        benchmark._validate_trace(broken_submission)

    broken_kind = copy.deepcopy(trace)
    broken_kind["phases"][0]["requests"][0]["kind"] = "cold"
    with pytest.raises(ValueError, match="request shape"):
        benchmark._validate_trace(broken_kind)


def test_model_file_proof_detects_post_trace_drift(
    tmp_path,
    model_identity,
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("config", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
    (model_path / "model.safetensors").write_text("weights", encoding="utf-8")
    trace = _trace("in-batch", model_identity)
    trace["model"]["path"] = str(model_path.resolve())
    trace["model"]["files_sha256"] = benchmark._model_file_hashes(model_path)

    assert benchmark._validate_model_files(trace) == model_path.resolve()

    (model_path / "tokenizer.json").write_text("drifted", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after trace generation"):
        benchmark._validate_model_files(trace)


def test_compare_rejects_runtime_config_manifest_or_gpu_drift(
    tmp_path,
    model_identity,
):
    trace = _trace("in-batch", model_identity)
    trace_path = tmp_path / "trace.json"
    nano_path = tmp_path / "nano.json"
    vllm_path = tmp_path / "vllm.json"
    benchmark._write_json(trace_path, trace)
    trace_sha256 = benchmark._sha256_file(trace_path)
    nano = _fake_result("nano-vllm", trace_path, trace, trace_sha256)
    vllm = _fake_result("vllm", trace_path, trace, trace_sha256)

    nano["engine"]["requested"]["max_num_seqs"] += 1
    benchmark._write_json(nano_path, nano)
    benchmark._write_json(vllm_path, vllm)
    with pytest.raises(ValueError, match="requested engine config"):
        benchmark._compare_results(trace_path, nano_path, vllm_path)

    nano = _fake_result("nano-vllm", trace_path, trace, trace_sha256)
    nano["trace"]["request_manifest"][0]["prompt_len"] += 1
    benchmark._write_json(nano_path, nano)
    with pytest.raises(ValueError, match="manifest content"):
        benchmark._compare_results(trace_path, nano_path, vllm_path)

    nano = _fake_result("nano-vllm", trace_path, trace, trace_sha256)
    vllm["runtime"]["gpu"]["uuid"] = "GPU-other"
    benchmark._write_json(nano_path, nano)
    benchmark._write_json(vllm_path, vllm)
    with pytest.raises(ValueError, match="identical GPU"):
        benchmark._compare_results(trace_path, nano_path, vllm_path)


def test_compare_rejects_nano_prompt_accounting_drift(
    tmp_path,
    model_identity,
):
    trace = _trace("in-batch", model_identity)
    trace_path = tmp_path / "trace.json"
    nano_path = tmp_path / "nano.json"
    vllm_path = tmp_path / "vllm.json"
    benchmark._write_json(trace_path, trace)
    trace_sha256 = benchmark._sha256_file(trace_path)
    nano = _fake_result("nano-vllm", trace_path, trace, trace_sha256)
    vllm = _fake_result("vllm", trace_path, trace, trace_sha256)
    nano["metrics"]["backend_specific"]["computed_prompt_tokens"] -= 1
    benchmark._write_json(nano_path, nano)
    benchmark._write_json(vllm_path, vllm)

    with pytest.raises(ValueError, match="prompt-token conservation"):
        benchmark._compare_results(trace_path, nano_path, vllm_path)


def test_vllm_shutdown_uses_engine_core_lifecycle():
    calls = []
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
        )
    )

    benchmark._shutdown_vllm(llm)

    assert calls == ["shutdown"]
