from argparse import Namespace
import inspect
import json
from pathlib import Path
import sys

import pytest


_REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY))
import bench_cudagraph as benchmark  # noqa: E402


def _args(**overrides):
    values = {
        "batch_sizes": None,
        "kv_lengths": None,
        "temperature": 1.0,
        "gpu_memory_utilization": 0.9,
        "output": Path("result.json"),
    }
    values.update(overrides)
    return Namespace(**values)


def test_default_case_matrix_is_the_required_five_by_three_grid():
    cases = benchmark._case_matrix()

    assert len(cases) == 15
    assert {case.batch_size for case in cases} == {1, 4, 8, 16, 32}
    assert {case.kv_length for case in cases} == {512, 2048, 4096}
    assert len({case.name for case in cases}) == len(cases)


def test_case_prompt_makes_first_decode_kv_length_exact():
    case = benchmark.CaseSpec(batch_size=32, kv_length=4096)

    assert case.prompt_length == 4095
    assert case.prompt_length + 1 == case.kv_length


def test_case_matrix_deduplicates_and_sorts_explicit_exact_buckets():
    cases = benchmark._case_matrix(
        batch_sizes=[16, 1, 16, 4],
        kv_lengths=[2048, 512, 2048],
    )

    assert [(case.batch_size, case.kv_length) for case in cases] == [
        (1, 512),
        (1, 2048),
        (4, 512),
        (4, 2048),
        (16, 512),
        (16, 2048),
    ]


@pytest.mark.parametrize("kv_length", [1, 16, 511, 513])
def test_validate_args_rejects_kv_lengths_that_break_block_protocol(kv_length):
    with pytest.raises(ValueError, match="multiples of block_size=16"):
        benchmark._validate_args(_args(kv_lengths=[kv_length]))


def test_token_factory_is_reproducible_and_excludes_special_tokens():
    first = benchmark.TokenFactory(32, 2026, {0, 1, 31}).tokens(256)
    second = benchmark.TokenFactory(32, 2026, {0, 1, 31}).tokens(256)

    assert first == second
    assert not ({0, 1, 31} & set(first))


def test_none_requires_all_graph_counters_to_remain_zero():
    zero = {name: 0 for name in benchmark.RUNTIME_COUNTERS}

    assert benchmark._validate_measured_stats("none", 64, zero) == zero
    with pytest.raises(AssertionError, match="NONE"):
        benchmark._validate_measured_stats(
            "none",
            64,
            {**zero, "full_graph_replay_steps": 1},
        )


def test_full_decode_requires_every_timed_step_to_replay_exact_bucket():
    exact = {
        "full_graph_replay_steps": 64,
        "eager_fallback_steps": 0,
        "graph_bucket_hits": 64,
        "graph_bucket_misses": 0,
    }

    assert (
        benchmark._validate_measured_stats(
            "full_decode_only",
            64,
            exact,
        )
        == exact
    )
    with pytest.raises(AssertionError, match="exact-bucket pure-decode"):
        benchmark._validate_measured_stats(
            "full_decode_only",
            64,
            {**exact, "eager_fallback_steps": 1},
        )


@pytest.mark.parametrize(
    ("mode", "expected_fallbacks"),
    [("none", 0), ("full_decode_only", 1)],
)
def test_excluded_prefill_stats_are_isolated_and_eager(
    mode,
    expected_fallbacks,
):
    counters = {name: 0 for name in benchmark.RUNTIME_COUNTERS}
    counters["eager_fallback_steps"] = expected_fallbacks

    assert benchmark._validate_excluded_prefill_stats(mode, counters) == counters

    contaminated = {**counters, "full_graph_replay_steps": 1}
    with pytest.raises(AssertionError, match="excluded follower prefill"):
        benchmark._validate_excluded_prefill_stats(mode, contaminated)


def test_case_summary_keeps_batch_latency_tpot_and_throughput_distinct():
    repeats = [
        {
            "steps": [
                {"wall_ms": 2.0, "cuda_event_ms": 1.5},
                {"wall_ms": 4.0, "cuda_event_ms": 2.5},
            ],
            "runtime_counters": {
                "full_graph_replay_steps": 2,
                "eager_fallback_steps": 0,
                "graph_bucket_hits": 2,
                "graph_bucket_misses": 0,
            },
        },
        {
            "steps": [
                {"wall_ms": 3.0, "cuda_event_ms": 2.0},
                {"wall_ms": 3.0, "cuda_event_ms": 2.0},
            ],
            "runtime_counters": {
                "full_graph_replay_steps": 2,
                "eager_fallback_steps": 0,
                "graph_bucket_hits": 2,
                "graph_bucket_misses": 0,
            },
        },
    ]

    summary = benchmark._case_summary(
        "full_decode_only",
        batch_size=4,
        repeats=repeats,
    )

    assert summary["measured_decode_steps"] == 4
    assert summary["measured_output_tokens"] == 16
    assert summary["decode_step_latency_wall_ms"]["median"] == 3.0
    assert summary["tpot_wall_ms"] == 3.0
    assert summary["output_tokens_per_second_wall"] == pytest.approx(
        16 / 0.012
    )
    assert summary["graph_replay_hit_rate"] == 1.0
    assert summary["graph_bucket_hit_rate"] == 1.0


def test_shared_prefix_capacity_scales_with_tail_not_batch_times_context():
    case = benchmark.CaseSpec(batch_size=32, kv_length=4096)

    required = benchmark._required_resident_blocks(case, decode_steps=64)

    assert required == 415
    assert required < case.batch_size * (case.kv_length // benchmark.BLOCK_SIZE)


def _formal_result(mode):
    is_full = mode == "full_decode_only"
    repeats_per_case = benchmark.FORMAL_REPEATS
    steps_per_repeat = benchmark.FORMAL_DECODE_STEPS
    wall_ms = 2.0 if is_full else 2.5
    cuda_event_ms = 1.5 if is_full else 2.0

    def counters(step_count):
        return {
            "full_graph_replay_steps": step_count if is_full else 0,
            "eager_fallback_steps": 0,
            "graph_bucket_hits": step_count if is_full else 0,
            "graph_bucket_misses": 0,
        }

    excluded_prefill = {
        "full_graph_replay_steps": 0,
        "eager_fallback_steps": 1 if is_full else 0,
        "graph_bucket_hits": 0,
        "graph_bucket_misses": 0,
    }
    cases = []
    for batch_size, kv_length in benchmark._formal_case_keys():
        spec = benchmark.CaseSpec(batch_size, kv_length)
        prefix_hits = [kv_length - benchmark.BLOCK_SIZE] * batch_size
        repeats = []
        for repeat_index in range(repeats_per_case):
            steps = [
                {
                    "step": step_index,
                    "batch_size": batch_size,
                    "kv_length": kv_length + step_index,
                    "wall_ms": wall_ms,
                    "cuda_event_ms": cuda_event_ms,
                    "runtime_counters_after_step": counters(step_index + 1),
                }
                for step_index in range(steps_per_repeat)
            ]
            wall_elapsed_ms = sum(step["wall_ms"] for step in steps)
            cuda_elapsed_ms = sum(step["cuda_event_ms"] for step in steps)
            measured_output_tokens = batch_size * steps_per_repeat
            repeats.append(
                {
                    "repeat": repeat_index,
                    "sampling_seed": 10_000 + repeat_index,
                    "prefix_hit_tokens_per_request": list(prefix_hits),
                    "excluded_prefill_runtime_counters": dict(
                        excluded_prefill
                    ),
                    "measured_decode_steps": steps_per_repeat,
                    "measured_output_tokens": measured_output_tokens,
                    "wall_elapsed_ms": wall_elapsed_ms,
                    "cuda_event_elapsed_ms": cuda_elapsed_ms,
                    "tpot_wall_ms": wall_elapsed_ms / steps_per_repeat,
                    "tpot_cuda_event_ms": (
                        cuda_elapsed_ms / steps_per_repeat
                    ),
                    "output_tokens_per_second_wall": (
                        measured_output_tokens
                        / (wall_elapsed_ms / 1000.0)
                    ),
                    "output_tokens_per_second_cuda_event": (
                        measured_output_tokens
                        / (cuda_elapsed_ms / 1000.0)
                    ),
                    "runtime_counters": counters(steps_per_repeat),
                    "completion_token_sha256": [
                        (
                            f"{batch_size * 1000 + repeat_index * 100 + request:064x}"
                        )
                        for request in range(batch_size)
                    ],
                    "steps": steps,
                }
            )
        cases.append(
            {
                "name": spec.name,
                "case": {
                    "batch_size": batch_size,
                    "kv_length": kv_length,
                    "prompt_length": kv_length - 1,
                    "decode_steps_per_repeat": steps_per_repeat,
                    "first_measured_decode_kv_length": kv_length,
                    "last_measured_decode_kv_length": (
                        kv_length + steps_per_repeat - 1
                    ),
                    "measured_output_tokens_per_repeat": (
                        batch_size * steps_per_repeat
                    ),
                },
                "invariants": {
                    "timed_batch_type": "pure_decode",
                    "exact_batch_bucket": batch_size,
                    "runtime_padding_requests": 0,
                    "prefill_excluded_from_timing": True,
                    "shared_prefix_hit_verified_for_every_follower": True,
                },
                "shared_prefix": {
                    "prompt_seed": batch_size * 100_000 + kv_length,
                    "prompt_sha256": f"{batch_size * 10_000 + kv_length:064x}",
                },
                "warmup": {
                    "decode_steps": benchmark.FORMAL_WARMUP_DECODE_STEPS,
                    "prefix_hit_tokens_per_request": list(prefix_hits),
                    "excluded_prefill_runtime_counters": dict(
                        excluded_prefill
                    ),
                    "runtime_counters": counters(
                        benchmark.FORMAL_WARMUP_DECODE_STEPS
                    ),
                },
                "repeats": repeats,
                "summary": benchmark._case_summary(
                    mode,
                    batch_size,
                    repeats,
                ),
            }
        )
    capture_time_ms = 123.0 if is_full else 0.0
    extra_memory_bytes = 16 * 1024 * 1024 if is_full else 0
    return {
        "schema_version": 1,
        "benchmark": benchmark.BENCHMARK_NAME,
        "status": "complete",
        "mode": mode,
        "protocol": {"formal_matrix": True},
        "config": {
            "seed": 2026,
            "temperature": 1.0,
            "cudagraph_mode": mode,
            "cudagraph_batch_sizes": list(benchmark.DEFAULT_BATCH_SIZES),
            "batch_sizes": list(benchmark.DEFAULT_BATCH_SIZES),
            "kv_lengths": list(benchmark.DEFAULT_KV_LENGTHS),
            "decode_steps": steps_per_repeat,
            "warmup_decode_steps": benchmark.FORMAL_WARMUP_DECODE_STEPS,
            "repeats": repeats_per_case,
        },
        "environment": {
            "cuda_driver": "596.49",
            "cudnn": 91900,
            "python": "3.10.12",
            "torch": "2.11.0+cu128",
            "torch_cuda": "12.8",
            "transformers": "5.14.1",
            "gpu": {
                "uuid": "GPU-1234",
                "name": "NVIDIA Test GPU",
                "compute_capability": "12.0",
                "total_memory_bytes": 12_820_480_000,
                "multiprocessor_count": 48,
            },
            "flashinfer": {
                "flashinfer_python": "0.6.17",
                "flashinfer_cubin": "0.6.17",
                "flashinfer_jit_cache": "0.6.17+cu129",
                "environment": {
                    "FLASHINFER_CUDA_ARCH_LIST": "12.0f",
                    "FLASHINFER_DISABLE_JIT": "1",
                },
            },
            "git": {"commit": "deadbeef"},
            "benchmark_script": {"sha256": "a" * 64},
        },
        "runtime": {
            "model": "/models/Qwen3-0.6B",
            "model_type": "qwen3",
            "model_dtype": "torch.bfloat16",
            "model_shape": {"hidden_size": 1024, "num_hidden_layers": 28},
            "cudagraph_startup": {
                "capture_time_ms": capture_time_ms,
                "extra_memory_bytes": extra_memory_bytes,
            },
        },
        "cases": cases,
    }


def test_compare_formal_results_validates_and_emits_readme_table():
    comparison = benchmark._compare_formal_results(
        _formal_result("none"),
        _formal_result("full_decode_only"),
        none_source="none.json",
        full_source="full.json",
    )

    assert comparison["status"] == "complete"
    assert all(comparison["validation"].values())
    assert len(comparison["cases"]) == 15
    first = comparison["cases"][0]
    assert first["name"] == "bs1_kv512"
    assert first["delta_full_minus_none"][
        "latency_median_wall_percent"
    ] == pytest.approx(-20.0)
    assert first["delta_full_minus_none"][
        "tpot_wall_percent"
    ] == pytest.approx(-20.0)
    assert first["delta_full_minus_none"][
        "output_tokens_per_second_wall_percent"
    ] == pytest.approx(25.0)
    assert first["delta_full_minus_none"][
        "graph_replay_hit_rate_percentage_points"
    ] == pytest.approx(100.0)
    assert first["completion_token_sha256_match"] == {
        "matched": 5,
        "total": 5,
        "rate": 1.0,
        "mismatches": [],
    }
    assert comparison["completion_token_sha256_comparison"]["matched"] == 75
    assert comparison["completion_token_sha256_comparison"]["total"] == 75
    assert comparison["completion_token_sha256_comparison"]["rate"] == 1.0
    assert "NONE capture (ms / MiB)" in comparison["readme_markdown"]
    assert "| bs32_kv4096 |" in comparison["readme_markdown"]
    assert "not a correctness gate" in comparison["readme_markdown"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda result: result.update(status="failed"),
            "status must be 'complete'",
        ),
        (
            lambda result: result["protocol"].update(formal_matrix=False),
            "not a formal matrix",
        ),
        (
            lambda result: result["runtime"].update(model="/models/other"),
            "same model",
        ),
        (
            lambda result: result["environment"]["git"].update(
                commit="cafebabe"
            ),
            "git commits",
        ),
        (
            lambda result: result["environment"][
                "benchmark_script"
            ].update(sha256="b" * 64),
            "script SHA-256",
        ),
        (
            lambda result: result["config"].update(seed=2027),
            "seeds differ",
        ),
        (
            lambda result: result["config"].update(temperature=0.5),
            "configurations differ",
        ),
        (
            lambda result: result["cases"].pop(),
            "case matrix/order",
        ),
        (
            lambda result: result["cases"][0]["summary"].update(
                tpot_wall_ms=999.0
            ),
            "does not match raw steps",
        ),
        (
            lambda result: result["cases"][0]["repeats"][0]["steps"][
                1
            ].update(kv_length=999),
            "kv_length must be",
        ),
        (
            lambda result: result["cases"][0]["repeats"][0]["steps"][
                0
            ].update(wall_ms=-1.0),
            "positive finite",
        ),
        (
            lambda result: result["cases"][0]["repeats"][0]["steps"][0][
                "runtime_counters_after_step"
            ].update(graph_bucket_hits=0),
            "invalid cumulative counters",
        ),
        (
            lambda result: result["environment"]["gpu"].update(
                uuid="GPU-other"
            ),
            "environment.gpu.uuid",
        ),
        (
            lambda result: result["environment"].update(
                torch="2.12.0+cu130"
            ),
            "environment.torch",
        ),
    ],
)
def test_compare_rejects_incomparable_formal_results(mutation, match):
    none = _formal_result("none")
    full = _formal_result("full_decode_only")
    mutation(full)

    with pytest.raises(ValueError, match=match):
        benchmark._compare_formal_results(none, full)


def test_compare_accepts_same_explicit_model_identity_with_different_paths():
    none = _formal_result("none")
    full = _formal_result("full_decode_only")
    none["runtime"]["model_identity"] = {"repository": "Qwen/Qwen3-0.6B"}
    full["runtime"]["model_identity"] = {"repository": "Qwen/Qwen3-0.6B"}
    full["runtime"]["model"] = "/a/different/mount/Qwen3-0.6B"

    comparison = benchmark._compare_formal_results(none, full)

    assert comparison["provenance"]["model"]["path"] is None
    assert comparison["provenance"]["model"]["identity"] == {
        "repository": "Qwen/Qwen3-0.6B"
    }


def test_compare_records_completion_hash_mismatch_at_exact_case_and_repeat():
    none = _formal_result("none")
    full = _formal_result("full_decode_only")
    full["cases"][7]["repeats"][3]["completion_token_sha256"][0] = "f" * 64

    comparison = benchmark._compare_formal_results(none, full)

    hash_summary = comparison["completion_token_sha256_comparison"]
    assert hash_summary["matched"] == 74
    assert hash_summary["total"] == 75
    assert hash_summary["rate"] == pytest.approx(74 / 75)
    assert hash_summary["mismatches"] == [
        {
            "case": "bs8_kv2048",
            "repeat": 3,
            "mismatched_request_indices": [0],
        }
    ]
    assert hash_summary["correctness_gate"] is False
    assert comparison["validation"]["completion_token_sha256_compared"] is True
    assert (
        comparison["validation"][
            "same_completion_token_sha256_per_case_repeat"
        ]
        is False
    )
    case_summary = comparison["cases"][7][
        "completion_token_sha256_match"
    ]
    assert case_summary["matched"] == 4
    assert case_summary["total"] == 5
    assert case_summary["rate"] == pytest.approx(0.8)
    assert case_summary["mismatches"] == hash_summary["mismatches"]
    assert "74/75 (98.67%)" in comparison["readme_markdown"]
    assert "bs8_kv2048 repeat 3" in comparison["readme_markdown"]
    assert "hidden-state/logit tolerance GPU tests" in (
        comparison["readme_markdown"]
    )


def test_compare_still_rejects_repeat_sampling_seed_mismatch():
    full = _formal_result("full_decode_only")
    full["cases"][7]["repeats"][3]["sampling_seed"] += 1

    with pytest.raises(
        ValueError,
        match=r"bs8_kv2048 repeat 3 sampling seeds differ",
    ):
        benchmark._compare_formal_results(_formal_result("none"), full)


def test_compare_rejects_malformed_completion_hash_before_pairing():
    full = _formal_result("full_decode_only")
    full["cases"][0]["repeats"][0]["completion_token_sha256"][0] = "not-sha"

    with pytest.raises(ValueError, match="must contain 1 SHA-256"):
        benchmark._compare_formal_results(_formal_result("none"), full)


def test_summary_cli_alias_writes_json_and_markdown_without_gpu(
    tmp_path,
    capsys,
):
    none_path = tmp_path / "none.json"
    full_path = tmp_path / "full.json"
    output_path = tmp_path / "comparison.json"
    markdown_path = tmp_path / "comparison.md"
    none_path.write_text(json.dumps(_formal_result("none")), encoding="utf-8")
    full_path.write_text(
        json.dumps(_formal_result("full_decode_only")),
        encoding="utf-8",
    )

    benchmark.main(
        [
            "summary",
            "--none",
            str(none_path),
            "--full",
            str(full_path),
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["benchmark"] == benchmark.COMPARISON_BENCHMARK_NAME
    assert markdown_path.read_text(encoding="utf-8") == capsys.readouterr().out


def test_existing_run_cli_remains_backward_compatible():
    args = benchmark._parse_args(
        [
            "--model",
            "/models/Qwen3-0.6B",
            "--mode",
            "none",
            "--output",
            "result.json",
        ]
    )

    assert args.model == "/models/Qwen3-0.6B"
    assert args.mode == "none"
    assert args.output == Path("result.json")
    assert args.temperature == 1e-6


def test_timing_events_are_precreated_and_start_record_precedes_wall_start():
    source = inspect.getsource(benchmark._run_measured_repeat)

    events_position = source.index("timing_events =")
    loop_position = source.index("for step_index in range(args.decode_steps):")
    record_position = source.index("start_event.record()", loop_position)
    wall_position = source.index("wall_start = perf_counter()", loop_position)

    assert events_position < loop_position
    assert source.count("torch.cuda.Event(enable_timing=True)") == 2
    assert record_position < wall_position
