"""Opt-in real-GPU coverage for FlashInfer full-decode CUDA Graphs.

Run the backend-only coverage with, for example::

    NANOVLLM_RUN_CUDAGRAPH_TESTS=1 pytest -q \
        tests/test_cudagraph_flashinfer.py -k metadata_freshness

Set ``NANOVLLM_TEST_MODEL`` to a local Qwen3 directory to additionally run
the full ModelRunner test.  These tests intentionally bypass Scheduler: they
exercise one already-created execution engine with explicit pure-decode
metadata, so no scheduling policy or request priority is changed by the test.
"""

import atexit
from contextlib import contextmanager
from math import ceil
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from nanovllm.config import CUDAGraphPolicy
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import (
    BatchType,
    RuntimeExecutionMode,
    get_context,
    reset_context,
)


RUN_CUDAGRAPH_TESTS = os.environ.get(
    "NANOVLLM_RUN_CUDAGRAPH_TESTS", ""
).lower() in {"1", "true", "yes", "on"}
TEST_MODEL = os.environ.get("NANOVLLM_TEST_MODEL")
GPU_MEMORY_UTILIZATION = float(
    os.environ.get(
        "NANOVLLM_CUDAGRAPH_GPU_MEMORY_UTILIZATION",
        os.environ.get("NANOVLLM_E2E_GPU_MEMORY_UTILIZATION", "0.35"),
    )
)
GRAPH_BUCKETS = (1, 2, 4, 8, 16)

try:
    from nanovllm.layers.attention_backend import (
        FLASHINFER_ATTENTION_AVAILABLE,
        FlashInferAttentionBackend,
    )
except Exception as exc:
    FLASHINFER_ATTENTION_AVAILABLE = False
    FlashInferAttentionBackend = None
    _FLASHINFER_GATE_ERROR = exc
else:
    _FLASHINFER_GATE_ERROR = None


pytestmark = [
    pytest.mark.skipif(
        not RUN_CUDAGRAPH_TESTS,
        reason="set NANOVLLM_RUN_CUDAGRAPH_TESTS=1 for real-GPU graph tests",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="FlashInfer CUDA Graph tests require an NVIDIA GPU",
    ),
    pytest.mark.skipif(
        not FLASHINFER_ATTENTION_AVAILABLE,
        reason=f"FlashInfer attention is unavailable: {_FLASHINFER_GATE_ERROR}",
    ),
]


def _local_model_path() -> Path:
    assert TEST_MODEL is not None
    path = Path(TEST_MODEL).expanduser()
    if not path.is_dir():
        pytest.fail(f"NANOVLLM_TEST_MODEL is not a directory: {path}")
    return path


@contextmanager
def _owned_graph_llm():
    from nanovllm import LLM

    original_dtype = torch.get_default_dtype()
    original_device = torch.get_default_device()
    llm = None
    try:
        llm = LLM(
            str(_local_model_path()),
            enforce_eager=False,
            cudagraph_mode=CUDAGraphPolicy.FULL_DECODE_ONLY,
            cudagraph_batch_sizes=GRAPH_BUCKETS,
            tensor_parallel_size=1,
            max_model_len=64,
            max_num_batched_tokens=64,
            max_num_seqs=max(GRAPH_BUCKETS),
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            attention_mode="unified",
            attention_backend="flashinfer",
            kvcache_block_size=16,
            chunked_prefill=False,
        )
        # LLMEngine registered this exact bound method; this context owns exit.
        atexit.unregister(llm.exit)
        yield llm
    finally:
        try:
            if llm is not None:
                llm.exit()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            reset_context()
            torch.set_default_device(original_device)
            torch.set_default_dtype(original_dtype)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _pure_decode_sequences(
    batch_size: int,
    *,
    block_size: int,
    vocab_size: int,
) -> list[Sequence]:
    # All requests share one fully cached prefix page and have independent
    # writable tail pages.  This is a real prefix-sharing layout, not padding.
    prefix = [1 + token % (vocab_size - 1) for token in range(block_size)]
    sequences = []
    for request_index in range(batch_size):
        next_token = 1 + (1000 + request_index) % (vocab_size - 1)
        sequence = Sequence(prefix + [next_token], block_size=block_size)
        sequence.num_cached_tokens = block_size
        sequence.num_new_tokens = 1
        sequence.block_table = [1, request_index + 2]
        sequences.append(sequence)
    return sequences


def _run_model_runner_exact_buckets_eager_vs_full_graph():
    """One graph engine matches forced eager for every requested bucket."""

    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)

    with _owned_graph_llm() as llm:
        runner = llm.model_runner
        runner.model.eval()
        assert sorted(runner.decode_graph_states) == list(GRAPH_BUCKETS)

        initial_stats = runner.get_cudagraph_stats()
        assert initial_stats["full_graph_replay_steps"] == 0
        assert initial_stats["eager_fallback_steps"] == 0
        assert initial_stats["graph_bucket_hits"] == 0
        assert initial_stats["graph_bucket_misses"] == 0
        assert initial_stats["captured_batch_sizes"] == list(GRAPH_BUCKETS)

        # Capture uses shared page zero with slot_mapping=-1.  The KV-store
        # kernel must therefore leave both K and V zero for every model layer.
        torch.cuda.synchronize()
        assert torch.count_nonzero(runner.kv_cache[:, :, 0]).item() == 0

        wrappers = [
            runner.decode_graph_states[size].wrapper
            for size in GRAPH_BUCKETS
        ]
        assert len({id(wrapper) for wrapper in wrappers}) == len(GRAPH_BUCKETS)
        graph_workspace = runner.attention_backend.graph_workspace
        assert graph_workspace is not None
        assert {
            wrapper._float_workspace_buffer.data_ptr()
            for wrapper in wrappers
        } == {graph_workspace.data_ptr()}
        assert (
            runner.attention_backend.workspace.data_ptr()
            != graph_workspace.data_ptr()
        )

        max_private_page = max(GRAPH_BUCKETS) + 1
        if runner.config.num_kvcache_blocks <= max_private_page:
            pytest.skip(
                "the configured KV pool is too small for 16 independent "
                "decode tail pages"
            )
        runner.kv_cache[:, :, 1 : max_private_page + 1].normal_()

        original_policy = runner.cudagraph_policy
        try:
            for batch_size in GRAPH_BUCKETS:
                sequences = _pure_decode_sequences(
                    batch_size,
                    block_size=runner.block_size,
                    vocab_size=runner.config.hf_config.vocab_size,
                )
                input_ids, positions = runner.prepare_model_input(
                    sequences,
                    num_prefill_seqs=0,
                )
                context = get_context()
                assert context.batch_type is BatchType.PURE_DECODE
                assert context.num_decode_tokens == batch_size
                assert input_ids.shape == (batch_size,)

                with torch.inference_mode():
                    runner.cudagraph_policy = CUDAGraphPolicy.NONE
                    eager_mode, eager_state = runner.select_runtime_mode(
                        context,
                        input_ids,
                    )
                    assert eager_mode is RuntimeExecutionMode.EAGER
                    assert eager_state is None
                    context.runtime_mode = eager_mode
                    runner.attention_backend.plan(context)
                    eager_hidden = runner.model(input_ids, positions).clone()
                    eager_logits = runner.model.compute_logits(
                        eager_hidden
                    ).clone()

                    runner.cudagraph_policy = (
                        CUDAGraphPolicy.FULL_DECODE_ONLY
                    )
                    graph_mode, graph_state = runner.select_runtime_mode(
                        context,
                        input_ids,
                    )
                    assert graph_mode is RuntimeExecutionMode.FULL_GRAPH
                    assert graph_state is not None
                    assert graph_state.batch_size == batch_size
                    context.runtime_mode = graph_mode
                    graph_hidden = runner.replay_full_decode_graph(
                        graph_state,
                        context,
                        input_ids,
                        positions,
                    ).clone()
                    graph_logits = runner.model.compute_logits(
                        graph_hidden
                    ).clone()

                tolerance = 3e-2 if runner.dtype == torch.bfloat16 else 1e-2
                torch.testing.assert_close(
                    graph_hidden,
                    eager_hidden,
                    atol=tolerance,
                    rtol=tolerance,
                )
                torch.testing.assert_close(
                    graph_logits,
                    eager_logits,
                    atol=tolerance,
                    rtol=tolerance,
                )
                reset_context()
        finally:
            runner.cudagraph_policy = original_policy
            reset_context()

        final_stats = runner.get_cudagraph_stats()
        assert final_stats["full_graph_replay_steps"] == len(GRAPH_BUCKETS)
        assert final_stats["graph_bucket_hits"] == len(GRAPH_BUCKETS)
        assert final_stats["graph_bucket_misses"] == 0
        assert final_stats["eager_fallback_steps"] == 0


def _backend_decode_context(
    kv_length: int,
    *,
    case_index: int,
    batch_size: int,
    block_size: int,
    max_pages_per_request: int,
):
    pages_per_request = ceil(kv_length / block_size)
    shift = (case_index * 17) % max_pages_per_request
    local_pages = (
        torch.arange(pages_per_request, dtype=torch.int32, device="cuda")
        + shift
    ) % max_pages_per_request
    if case_index % 2:
        local_pages = local_pages.flip(0)

    request_indices = [
        local_pages + request_index * max_pages_per_request
        for request_index in range(batch_size)
    ]
    page_indices = torch.cat(request_indices)
    page_q_indptr = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cuda"
    )
    page_kv_indptr = torch.arange(
        0,
        (batch_size + 1) * pages_per_request,
        pages_per_request,
        dtype=torch.int32,
        device="cuda",
    )
    last_page_len = kv_length % block_size or block_size
    page_last_page_len = torch.full(
        (batch_size,),
        last_page_len,
        dtype=torch.int32,
        device="cuda",
    )
    return SimpleNamespace(
        page_q_indptr=page_q_indptr,
        page_kv_indptr=page_kv_indptr,
        page_indices=page_indices,
        page_last_page_len=page_last_page_len,
        num_prefill_seqs=0,
        num_prefill_tokens=0,
        num_decode_tokens=batch_size,
        batch_type=BatchType.PURE_DECODE,
    )


def _run_graph_wrapper_replan_metadata_freshness_across_kv_boundaries():
    """One captured wrapper consumes fresh CSR metadata at every replay."""

    assert FlashInferAttentionBackend is not None
    torch.manual_seed(1907)
    torch.cuda.manual_seed_all(1907)

    batch_size = 2
    block_size = 16
    max_kv_length = 4096
    max_pages_per_request = ceil(max_kv_length / block_size)
    num_q_heads = 16
    num_kv_heads = 4
    head_dim = 128
    dtype = torch.bfloat16
    device = torch.device("cuda", torch.cuda.current_device())

    backend = FlashInferAttentionBackend(
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        dtype,
        attention_mode="unified",
    )
    fixed_q_indptr = torch.empty(
        batch_size + 1, dtype=torch.int32, device=device
    )
    fixed_kv_indptr = torch.empty_like(fixed_q_indptr)
    fixed_indices = torch.empty(
        batch_size * max_pages_per_request,
        dtype=torch.int32,
        device=device,
    )
    fixed_last_page_len = torch.empty(
        batch_size, dtype=torch.int32, device=device
    )
    wrapper = backend.create_full_decode_graph_wrapper(
        fixed_q_indptr,
        fixed_kv_indptr,
        fixed_indices,
        fixed_last_page_len,
    )

    # Creating another exact-batch wrapper must not alias its wrapper state,
    # although serial graph wrappers deliberately share one stable workspace.
    second_batch_size = 4
    second_wrapper = backend.create_full_decode_graph_wrapper(
        torch.empty(second_batch_size + 1, dtype=torch.int32, device=device),
        torch.empty(second_batch_size + 1, dtype=torch.int32, device=device),
        torch.empty(
            second_batch_size * max_pages_per_request,
            dtype=torch.int32,
            device=device,
        ),
        torch.empty(second_batch_size, dtype=torch.int32, device=device),
    )
    assert wrapper is not second_wrapper
    assert backend.graph_workspace is not None
    assert {
        wrapper._float_workspace_buffer.data_ptr(),
        second_wrapper._float_workspace_buffer.data_ptr(),
    } == {backend.graph_workspace.data_ptr()}
    assert backend.workspace.data_ptr() != backend.graph_workspace.data_ptr()

    num_cache_pages = batch_size * max_pages_per_request
    k_cache = torch.randn(
        num_cache_pages,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    unused_k = torch.empty(
        0, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    unused_v = torch.empty_like(unused_k)
    static_q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    static_output = torch.empty_like(static_q)

    capture_context = _backend_decode_context(
        15,
        case_index=0,
        batch_size=batch_size,
        block_size=block_size,
        max_pages_per_request=max_pages_per_request,
    )
    backend.plan_full_decode_graph(wrapper, capture_context)
    backend.activate_full_decode_graph(wrapper, static_output)
    try:
        # Warm all dispatch and kernel paths before entering stream capture.
        backend.forward(
            static_q,
            unused_k,
            unused_v,
            k_cache,
            v_cache,
            capture_context,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            backend.forward(
                static_q,
                unused_k,
                unused_v,
                k_cache,
                v_cache,
                capture_context,
            )
        torch.cuda.synchronize()
    finally:
        backend.deactivate_full_decode_graph()

    # Start long, shrink to one page, then repeatedly cross boundaries.  The
    # final long replay also catches wrappers that retained an intermediate
    # plan instead of the latest fixed-buffer contents.
    kv_lengths = (4096, 15, 2048, 16, 512, 17, 4096)
    try:
        for case_index, kv_length in enumerate(kv_lengths, start=1):
            context = _backend_decode_context(
                kv_length,
                case_index=case_index,
                batch_size=batch_size,
                block_size=block_size,
                max_pages_per_request=max_pages_per_request,
            )
            query = torch.randn_like(static_q)

            # plan() is outside the graph and copies only active metadata into
            # the wrapper's fixed-capacity buffers.
            backend.plan_full_decode_graph(wrapper, context)
            active_pages = context.page_indices.numel()
            assert torch.equal(fixed_q_indptr, context.page_q_indptr)
            assert torch.equal(fixed_kv_indptr, context.page_kv_indptr)
            assert torch.equal(
                fixed_indices[:active_pages], context.page_indices
            )
            assert torch.equal(
                fixed_last_page_len, context.page_last_page_len
            )

            static_q.copy_(query)
            graph.replay()
            graph_output = static_output.clone()

            backend.plan(context)
            eager_output = backend.forward(
                query,
                unused_k,
                unused_v,
                k_cache,
                v_cache,
                context,
            )
            assert torch.isfinite(graph_output).all()
            torch.testing.assert_close(
                graph_output,
                eager_output,
                atol=3e-2,
                rtol=3e-2,
            )
    finally:
        torch.cuda.synchronize()
        del graph
        del second_wrapper
        del wrapper
        del backend
        torch.cuda.empty_cache()


WORKER_ARGUMENT = "--cudagraph-worker"


def _run_worker_in_subprocess(worker_name: str) -> None:
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        repo_root
        if not current_pythonpath
        else repo_root + os.pathsep + current_pythonpath
    )
    timeout_seconds = int(
        env.get("NANOVLLM_CUDAGRAPH_TEST_TIMEOUT", "300")
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            WORKER_ARGUMENT,
            worker_name,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            f"CUDA Graph worker {worker_name!r} failed "
            f"with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


@pytest.mark.skipif(
    not TEST_MODEL,
    reason="set NANOVLLM_TEST_MODEL to a local Qwen3 model directory",
)
def test_model_runner_exact_buckets_eager_vs_full_graph():
    _run_worker_in_subprocess("model_runner")


def test_graph_wrapper_replan_metadata_freshness_across_kv_boundaries():
    _run_worker_in_subprocess("metadata_freshness")
if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == WORKER_ARGUMENT:
        workers = {
            "model_runner": _run_model_runner_exact_buckets_eager_vs_full_graph,
            "metadata_freshness": (
                _run_graph_wrapper_replan_metadata_freshness_across_kv_boundaries
            ),
        }
        worker = workers.get(sys.argv[2])
        if worker is None:
            raise SystemExit(f"unknown CUDA Graph worker: {sys.argv[2]}")
        worker()
