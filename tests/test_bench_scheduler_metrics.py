from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import pytest


_REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY))
import bench_scheduler as benchmark  # noqa: E402
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def _run_grouped_first_step(enable_same_step: bool):
    config = SimpleNamespace(
        chunked_prefill=False,
        max_model_len=128,
        max_num_seqs=16,
        max_num_batched_tokens=144,
        eos=-1,
        num_kvcache_blocks=24,
        kvcache_block_size=benchmark.BLOCK_SIZE,
        enable_lpm=True,
        enable_same_step_prefix_reuse=enable_same_step,
    )
    scheduler = Scheduler(config)
    observations = {}
    names = []
    for group_index, group in enumerate(benchmark.GROUP_NAMES):
        prefix = list(
            range(
                1000 * (group_index + 1),
                1000 * (group_index + 1) + 2 * benchmark.BLOCK_SIZE,
            )
        )
        for request_index in range(1, 5):
            name = f"{group}{request_index}"
            names.append(name)
            spec = benchmark.RequestSpec(
                name=name,
                prompt_token_ids=prefix + [10_000 + len(names)],
                output_len=1,
                kind="burst",
                group=group,
                shared_prefix_len=len(prefix),
            )
            seq = Sequence(
                spec.prompt_token_ids,
                block_size=benchmark.BLOCK_SIZE,
            )
            scheduler.add(seq)
            observations[seq.seq_id] = benchmark.RequestObservation(spec, seq)

    phase_started_at = perf_counter()
    for observation in observations.values():
        observation.submitted_at = phase_started_at
    llm = SimpleNamespace(scheduler=scheduler)
    collector = benchmark.SchedulerMetrics(
        llm,
        observations,
        phase_started_at,
    )
    collector.install()
    try:
        scheduled = scheduler.schedule()
        ended_at = perf_counter()
        collector.finish_step(ended_at)
    finally:
        collector.uninstall()
    return names, scheduled, observations, collector, phase_started_at


def test_grouped_same_step_metrics_and_admission_are_exact():
    names, scheduled, observations, collector, started_at = (
        _run_grouped_first_step(True)
    )

    assert [seq.seq_id for seq in scheduled] == [
        observation.seq.seq_id for observation in observations.values()
    ]
    assert collector.steps[0]["scheduled_names"] == names
    assert collector.steps[0]["num_scheduled_prefill_seqs"] == 16
    assert collector.initial_persistent_hit_tokens == 0
    assert collector.same_step_hit_tokens == 12 * 2 * benchmark.BLOCK_SIZE
    assert collector.claimed_prefix_tokens == collector.same_step_hit_tokens
    assert collector.same_step_reused_blocks == 12 * 2
    assert len(collector.same_step_reused_ids) == 12
    assert collector.computed_prompt_tokens == 4 * 33 + 12
    assert (
        collector.initial_persistent_hit_tokens
        + collector.same_step_hit_tokens
        + collector.computed_prompt_tokens
        == 16 * 33
    )

    for observation in observations.values():
        observation.first_token_at = started_at + 0.01
        observation.completed_at = started_at + 0.02
    metrics = benchmark._metric_summary(
        observations,
        collector,
        started_at,
        started_at + 0.03,
        prefix_groups=4,
        shared_prefix_len=2 * benchmark.BLOCK_SIZE,
    )
    assert metrics["prompt_token_conservation"]["balanced"] is True
    assert metrics["first_step_prefill_admission_count"] == 16
    assert metrics["same_step_reused_request_count"] == 12
    assert metrics["duplicate_prefill_tokens"] == 0


def test_same_step_off_freezes_misses_and_admits_only_four_cold_requests():
    names, scheduled, _, collector, _ = _run_grouped_first_step(False)

    assert collector.steps[0]["scheduled_names"] == names[:4]
    assert collector.steps[0]["num_scheduled_prefill_seqs"] == 4
    assert collector.initial_persistent_hit_tokens == 0
    assert collector.same_step_hit_tokens == 0
    assert collector.same_step_reused_blocks == 0
    assert collector.computed_prompt_tokens == 4 * 33


def test_metric_summary_rejects_prompt_token_accounting_drift():
    _, _, observations, collector, started_at = _run_grouped_first_step(True)
    for observation in observations.values():
        observation.first_token_at = started_at + 0.01
        observation.completed_at = started_at + 0.02
    collector.computed_prompt_tokens -= 1

    with pytest.raises(AssertionError, match="prompt-token conservation"):
        benchmark._metric_summary(
            observations,
            collector,
            started_at,
            started_at + 0.03,
            prefix_groups=4,
            shared_prefix_len=2 * benchmark.BLOCK_SIZE,
        )
