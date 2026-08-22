from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest


_REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY))
import bench_attention_dispatch as benchmark  # noqa: E402


def _args(**overrides):
    values = {
        "output": Path("result.json"),
        "prefill_q_len": 128,
        "prefill_kv_len": 4224,
        "decode_batch": 3,
        "decode_kv_len": 4096,
        "warmup": 50,
        "iters": 500,
        "repeats": benchmark.DEFAULT_REPEATS,
        "workspace_mib": 64,
        "seed": 2026,
        "device": 0,
        "retired_prefill_backend": "auto",
        "expected_route": "auto",
    }
    values.update(overrides)
    return Namespace(**values)


def test_default_case_is_exact_qwen3_06b_p128_d3_workload():
    case = benchmark._validate_args(_args())

    assert case == benchmark.CaseSpec(128, 4224, 3, 4096)
    assert case.q_lens == (128, 1, 1, 1)
    assert case.kv_lens == (4224, 4096, 4096, 4096)
    assert benchmark.NUM_Q_HEADS == 16
    assert benchmark.NUM_KV_HEADS == 8
    assert benchmark.HEAD_DIM == 128
    assert benchmark.BLOCK_SIZE == 16
    assert benchmark.DTYPE_NAME == "bf16"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"output": Path("result.txt")}, "json"),
        ({"decode_batch": 0}, "decode-batch"),
        ({"repeats": 0}, "repeats"),
        ({"repeats": 5}, "divisible"),
        ({"warmup": -1}, "warmup"),
        (
            {"prefill_q_len": 129, "prefill_kv_len": 128},
            "prefill-kv-len",
        ),
    ],
)
def test_validate_args_rejects_protocol_drift(override, message):
    with pytest.raises(ValueError, match=message):
        benchmark._validate_args(_args(**override))


def test_execution_orders_balance_first_method_without_dropping_methods():
    orders = benchmark._execution_orders(benchmark.METHODS, 6)

    assert orders == [
        ["production_dispatch", "retired_all_batch_paged_prefill"],
        ["retired_all_batch_paged_prefill", "production_dispatch"],
        ["production_dispatch", "retired_all_batch_paged_prefill"],
        ["retired_all_batch_paged_prefill", "production_dispatch"],
        ["production_dispatch", "retired_all_batch_paged_prefill"],
        ["retired_all_batch_paged_prefill", "production_dispatch"],
    ]


def test_summary_and_delta_keep_raw_median_and_direction_explicit():
    summary = benchmark._summary([0.20, 0.18, 0.19])
    delta = benchmark._delta(0.18, 0.20)

    assert summary == {
        "raw_ms": [0.20, 0.18, 0.19],
        "median_ms": 0.19,
        "min_ms": 0.18,
        "max_ms": 0.20,
    }
    assert delta["production_minus_retired_ms"] == pytest.approx(-0.02)
    assert delta["production_minus_retired_percent"] == pytest.approx(-10.0)
    assert delta["production_speedup_vs_retired"] == pytest.approx(1.0 / 0.9)


def test_formal_protocol_allows_explicit_route_assertion_only():
    explicit_route = _args(expected_route="mixed_split")
    case = benchmark._validate_args(explicit_route)

    assert benchmark._protocol_compliant(explicit_route, case)
    assert not benchmark._protocol_compliant(
        _args(iters=100),
        case,
    )


@pytest.mark.parametrize(
    ("available", "route"),
    [(True, "mixed_holistic"), (False, "mixed_split")],
)
def test_dispatch_route_assertion_follows_capability(available, route):
    assert (
        benchmark._validate_dispatch_route(
            batch_type="mixed",
            planned_route=route,
            mixed_attention_available=available,
            expected_route="auto",
        )
        == route
    )


def test_dispatch_route_assertion_rejects_non_mixed_and_capability_drift():
    with pytest.raises(AssertionError, match="BatchType.MIXED"):
        benchmark._validate_dispatch_route(
            batch_type="pure_prefill",
            planned_route="mixed_split",
            mixed_attention_available=False,
            expected_route="auto",
        )
    with pytest.raises(AssertionError, match="expected production route"):
        benchmark._validate_dispatch_route(
            batch_type="mixed",
            planned_route="mixed_split",
            mixed_attention_available=True,
            expected_route="auto",
        )


def test_json_writer_is_atomic_and_round_trips(tmp_path):
    output = tmp_path / "dispatch.json"
    payload = {"status": "complete", "raw": {"production_dispatch": [0.1]}}

    benchmark._write_json(output, payload)

    assert json.loads(output.read_text()) == payload
    assert not output.with_name("dispatch.json.tmp").exists()
