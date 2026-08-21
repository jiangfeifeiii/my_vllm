from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import nanovllm.engine.model_runner as model_runner_module
from nanovllm.config import CUDAGraphPolicy, Config
from nanovllm.engine.model_runner import DecodeGraphState, ModelRunner
from nanovllm.utils.context import BatchType, RuntimeExecutionMode


class _FakeCudaIntTensor(torch.Tensor):
    """CPU-backed tensor that exposes CUDA metadata for dispatcher unit tests."""

    @staticmethod
    def __new__(cls, values):
        tensor = torch.as_tensor(values, dtype=torch.int32)
        return torch.Tensor._make_subclass(cls, tensor, require_grad=False)

    @property
    def device(self):
        return torch.device("cuda")


class _FakeGraph:

    def __init__(self):
        self.replay = Mock()
        self.reset = Mock()


def _config(tmp_path, **kwargs) -> Config:
    hf_config = SimpleNamespace(max_position_embeddings=8192)
    with patch(
        "nanovllm.config.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        return Config(str(tmp_path), **kwargs)


def _graph_state(
    batch_size: int,
    *,
    page_indices_capacity: int | None = None,
    captured: bool = True,
) -> DecodeGraphState:
    capacity = (
        batch_size
        if page_indices_capacity is None
        else page_indices_capacity
    )
    return DecodeGraphState(
        batch_size=batch_size,
        page_indices_capacity=capacity,
        wrapper=object(),
        cuda_graph=_FakeGraph() if captured else None,
        static_input_ids=torch.full((batch_size,), -1, dtype=torch.int64),
        static_positions=torch.full((batch_size,), -1, dtype=torch.int64),
        static_slot_mapping=torch.full(
            (batch_size,), -1, dtype=torch.int32
        ),
        static_page_q_indptr=torch.empty(
            batch_size + 1, dtype=torch.int32
        ),
        static_page_kv_indptr=torch.empty(
            batch_size + 1, dtype=torch.int32
        ),
        static_page_indices=torch.empty(capacity, dtype=torch.int32),
        static_page_last_page_len=torch.empty(
            batch_size, dtype=torch.int32
        ),
        static_attention_output=torch.empty(batch_size, 1, 1),
        static_hidden_output=torch.arange(
            batch_size * 3, dtype=torch.float32
        ).reshape(batch_size, 3),
    )


def _runner(
    *,
    policy: CUDAGraphPolicy = CUDAGraphPolicy.FULL_DECODE_ONLY,
    attention_mode: str = "unified",
    supports_full_decode_graph: bool = True,
    world_size: int = 1,
    states: dict[int, DecodeGraphState] | None = None,
) -> ModelRunner:
    runner = object.__new__(ModelRunner)
    runner.cudagraph_policy = policy
    runner.config = SimpleNamespace(
        attention_mode=attention_mode,
        num_kvcache_blocks=4096,
    )
    runner.attention_backend = SimpleNamespace(
        supports_full_decode_graph=supports_full_decode_graph,
        plan_full_decode_graph=Mock(),
    )
    runner.world_size = world_size
    runner.block_size = 16
    runner.decode_graph_states = dict(states or {})
    runner.cudagraph_capture_time_ms = 7.5
    runner.cudagraph_extra_memory_bytes = 2048
    runner._cudagraph_stats = {
        "full_graph_replay_steps": 0,
        "eager_fallback_steps": 0,
        "graph_bucket_hits": 0,
        "graph_bucket_misses": 0,
    }
    return runner


def _decode_context(
    batch_size: int,
    *,
    pages_per_request: int = 1,
    batch_type: BatchType = BatchType.PURE_DECODE,
):
    num_pages = batch_size * pages_per_request
    return SimpleNamespace(
        batch_type=batch_type,
        num_prefill_seqs=0,
        num_prefill_tokens=0,
        num_decode_tokens=batch_size,
        page_q_indptr=_FakeCudaIntTensor(range(batch_size + 1)),
        page_kv_indptr=_FakeCudaIntTensor(
            range(0, num_pages + 1, pages_per_request)
        ),
        page_indices=_FakeCudaIntTensor(range(num_pages)),
        page_last_page_len=_FakeCudaIntTensor([16] * batch_size),
        slot_mapping=_FakeCudaIntTensor(range(batch_size)),
    )


def _select(runner: ModelRunner, context, num_input_tokens: int):
    input_ids = torch.zeros(num_input_tokens, dtype=torch.int64)
    return runner.select_runtime_mode(context, input_ids)


def test_config_normalizes_policy_and_sorts_exact_buckets(tmp_path):
    config = _config(
        tmp_path,
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_batch_sizes=(16, 1, 8, 4),
    )

    assert config.cudagraph_mode is CUDAGraphPolicy.FULL_DECODE_ONLY
    assert config.enforce_eager is False
    assert config.cudagraph_batch_sizes == (1, 4, 8, 16)


@pytest.mark.parametrize(
    ("mode", "enforce_eager"),
    [
        (CUDAGraphPolicy.NONE, False),
        (CUDAGraphPolicy.FULL_DECODE_ONLY, True),
    ],
)
def test_config_none_and_legacy_enforce_eager_select_none(
    tmp_path,
    mode,
    enforce_eager,
):
    config = _config(
        tmp_path,
        cudagraph_mode=mode,
        enforce_eager=enforce_eager,
    )

    assert config.cudagraph_mode is CUDAGraphPolicy.NONE
    assert config.enforce_eager is True


@pytest.mark.parametrize(
    "buckets",
    [(), (0,), (1, 1), (True,)],
)
def test_config_rejects_invalid_graph_buckets(tmp_path, buckets):
    with pytest.raises(ValueError, match="cudagraph_batch_sizes"):
        _config(tmp_path, cudagraph_batch_sizes=buckets)

def test_full_decode_graph_capture_session_can_be_claimed_once(monkeypatch):
    monkeypatch.setattr(
        model_runner_module,
        "_FULL_DECODE_GRAPH_CAPTURE_ATTEMPTED",
        False,
    )
    assert model_runner_module._claim_full_decode_graph_capture_session()
    assert not model_runner_module._claim_full_decode_graph_capture_session()


def test_policy_none_is_eager_without_counting_a_fallback():
    state = _graph_state(4, page_indices_capacity=8)
    runner = _runner(
        policy=CUDAGraphPolicy.NONE,
        states={4: state},
    )

    mode, selected = _select(runner, _decode_context(4), 4)

    assert mode is RuntimeExecutionMode.EAGER
    assert selected is None
    assert runner._cudagraph_stats == {
        "full_graph_replay_steps": 0,
        "eager_fallback_steps": 0,
        "graph_bucket_hits": 0,
        "graph_bucket_misses": 0,
    }


@pytest.mark.parametrize(
    "batch_type",
    [BatchType.PURE_PREFILL, BatchType.MIXED],
)
def test_prefill_and_mixed_batches_fall_back_to_eager(batch_type):
    runner = _runner(states={4: _graph_state(4)})
    context = _decode_context(4, batch_type=batch_type)

    mode, selected = _select(runner, context, 4)

    assert mode is RuntimeExecutionMode.EAGER
    assert selected is None
    assert runner._cudagraph_stats["eager_fallback_steps"] == 1
    assert runner._cudagraph_stats["graph_bucket_hits"] == 0


@pytest.mark.parametrize(
    ("attention_mode", "supports_graph", "world_size"),
    [
        ("split", True, 1),
        ("unified", False, 1),
        ("unified", True, 2),
    ],
    ids=("split", "backend-capability-false", "tensor-parallel"),
)
def test_unsupported_attention_or_tp_falls_back(
    attention_mode,
    supports_graph,
    world_size,
):
    runner = _runner(
        attention_mode=attention_mode,
        supports_full_decode_graph=supports_graph,
        world_size=world_size,
        states={4: _graph_state(4)},
    )

    mode, selected = _select(runner, _decode_context(4), 4)

    assert mode is RuntimeExecutionMode.EAGER
    assert selected is None
    assert runner._cudagraph_stats["eager_fallback_steps"] == 1
    assert runner._cudagraph_stats["graph_bucket_hits"] == 0


@pytest.mark.parametrize("malformation", ["q_indptr_size", "q_len", "input"])
def test_non_decode_query_shape_falls_back(malformation):
    runner = _runner(states={4: _graph_state(4)})
    context = _decode_context(4)
    num_input_tokens = 4
    if malformation == "q_indptr_size":
        context.page_q_indptr = _FakeCudaIntTensor([0, 1, 2, 4])
    elif malformation == "q_len":
        context.page_q_indptr = _FakeCudaIntTensor([0, 2, 2, 3, 4])
    else:
        num_input_tokens = 5

    mode, selected = _select(runner, context, num_input_tokens)

    assert mode is RuntimeExecutionMode.EAGER
    assert selected is None
    assert runner._cudagraph_stats["eager_fallback_steps"] == 1
    assert runner._cudagraph_stats["graph_bucket_hits"] == 0


def test_batch_size_must_hit_an_exact_bucket_without_padding():
    larger_state = _graph_state(16, page_indices_capacity=32)
    runner = _runner(states={16: larger_state})

    mode, selected = _select(runner, _decode_context(12), 12)

    assert mode is RuntimeExecutionMode.EAGER
    assert selected is None
    assert runner._cudagraph_stats["graph_bucket_misses"] == 1
    assert runner._cudagraph_stats["graph_bucket_hits"] == 0
    assert runner._cudagraph_stats["eager_fallback_steps"] == 1


def test_page_indices_capacity_overflow_falls_back_after_bucket_hit():
    state = _graph_state(4, page_indices_capacity=4)
    runner = _runner(states={4: state})
    context = _decode_context(4, pages_per_request=2)

    assert context.page_indices.numel() == 8
    assert runner.graph_metadata_fits(state, context) is False
    mode, selected = _select(runner, context, 4)

    assert mode is RuntimeExecutionMode.EAGER
    assert selected is None
    assert runner._cudagraph_stats["graph_bucket_hits"] == 1
    assert runner._cudagraph_stats["graph_bucket_misses"] == 0
    assert runner._cudagraph_stats["eager_fallback_steps"] == 1


def test_exact_bucket_with_valid_metadata_selects_full_graph():
    state = _graph_state(4, page_indices_capacity=8)
    runner = _runner(states={4: state})
    context = _decode_context(4, pages_per_request=2)

    assert runner.graph_metadata_fits(state, context) is True
    mode, selected = _select(runner, context, 4)

    assert mode is RuntimeExecutionMode.FULL_GRAPH
    assert selected is state
    assert runner._cudagraph_stats["graph_bucket_hits"] == 1
    assert runner._cudagraph_stats["graph_bucket_misses"] == 0
    assert runner._cudagraph_stats["eager_fallback_steps"] == 0


def test_uncaptured_exact_bucket_falls_back_but_counts_bucket_hit():
    state = _graph_state(4, captured=False)
    runner = _runner(states={4: state})

    mode, selected = _select(runner, _decode_context(4), 4)

    assert mode is RuntimeExecutionMode.EAGER
    assert selected is None
    assert runner._cudagraph_stats["graph_bucket_hits"] == 1
    assert runner._cudagraph_stats["graph_bucket_misses"] == 0
    assert runner._cudagraph_stats["eager_fallback_steps"] == 1


def test_replay_updates_static_inputs_and_replay_statistics():
    state = _graph_state(2, page_indices_capacity=4)
    runner = _runner(states={2: state})
    context = _decode_context(2, pages_per_request=2)
    input_ids = torch.tensor([11, 12], dtype=torch.int64)
    positions = torch.tensor([31, 47], dtype=torch.int64)

    output = runner.replay_full_decode_graph(
        state,
        context,
        input_ids,
        positions,
    )

    runner.attention_backend.plan_full_decode_graph.assert_called_once_with(
        state.wrapper,
        context,
    )
    assert state.static_input_ids.tolist() == [11, 12]
    assert state.static_positions.tolist() == [31, 47]
    assert state.static_slot_mapping.tolist() == [0, 1]
    state.cuda_graph.replay.assert_called_once_with()
    assert output is state.static_hidden_output
    assert runner._cudagraph_stats["full_graph_replay_steps"] == 1


@pytest.mark.parametrize(
    ("capture_reserve_bytes", "expected_blocks"),
    [(0, 187), (100, 162)],
)
def test_kv_cache_budget_reserves_graph_capture_memory(
    capture_reserve_bytes,
    expected_blocks,
):
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        gpu_memory_utilization=1.0,
        hf_config=SimpleNamespace(
            num_key_value_heads=1,
            num_attention_heads=1,
            hidden_size=1,
            num_hidden_layers=1,
        ),
    )
    runner.world_size = 1
    runner.block_size = 1
    runner.dtype = torch.float16
    runner.model = SimpleNamespace(modules=lambda: [])

    with (
        patch("torch.cuda.mem_get_info", return_value=(900, 1000)),
        patch(
            "torch.cuda.memory_stats",
            return_value={
                "allocated_bytes.all.peak": 200,
                "allocated_bytes.all.current": 50,
            },
        ),
        patch("torch.empty", return_value=Mock()),
    ):
        runner.allocate_kv_cache(
            capture_reserve_bytes=capture_reserve_bytes,
        )

    assert runner.config.num_kvcache_blocks == expected_blocks


def test_kv_cache_budget_rejects_negative_capture_reserve():
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(hf_config=SimpleNamespace())

    with pytest.raises(ValueError, match="non-negative"):
        runner.allocate_kv_cache(capture_reserve_bytes=-1)


def test_exit_resets_graph_execs_before_clearing_states():
    state = _graph_state(1)
    graph = state.cuda_graph
    runner = object.__new__(ModelRunner)
    runner.world_size = 1
    runner.decode_graph_states = {1: state}
    runner._graph_pool = object()

    with (
        patch("torch.cuda.synchronize"),
        patch("torch.cuda.empty_cache") as empty_cache,
        patch("torch.distributed.destroy_process_group") as destroy_group,
    ):
        runner.exit()

    graph.reset.assert_called_once_with()
    assert state.cuda_graph is None
    assert runner.decode_graph_states == {}
    assert runner._graph_pool is None
    empty_cache.assert_called_once_with()
    destroy_group.assert_called_once_with()


def test_stats_report_and_reset_are_stable():
    states = {
        4: _graph_state(4),
        1: _graph_state(1),
    }
    runner = _runner(states=states)
    runner._cudagraph_stats.update(
        full_graph_replay_steps=3,
        eager_fallback_steps=5,
        graph_bucket_hits=7,
        graph_bucket_misses=2,
    )

    stats = runner.get_cudagraph_stats()

    assert stats == {
        "full_graph_replay_steps": 3,
        "eager_fallback_steps": 5,
        "graph_bucket_hits": 7,
        "graph_bucket_misses": 2,
        "policy": "full_decode_only",
        "captured_batch_sizes": [1, 4],
        "capture_time_ms": 7.5,
        "extra_memory_bytes": 2048,
    }

    runner.reset_cudagraph_stats()
    assert runner._cudagraph_stats == {
        "full_graph_replay_steps": 0,
        "eager_fallback_steps": 0,
        "graph_bucket_hits": 0,
        "graph_bucket_misses": 0,
    }
