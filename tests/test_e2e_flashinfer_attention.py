"""Opt-in GPU E2E coverage for paged FlashInfer serving paths."""

import atexit
from contextlib import contextmanager
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


TEST_MODEL = os.environ.get("NANOVLLM_TEST_MODEL")
BATCH_TOKENS = int(os.environ.get("NANOVLLM_E2E_BATCH_TOKENS", "96"))
GPU_MEMORY_UTILIZATION = float(
    os.environ.get("NANOVLLM_E2E_GPU_MEMORY_UTILIZATION", "0.35")
)

try:
    from nanovllm.layers.attention_backend import (
        FLASHINFER_ATTENTION_AVAILABLE,
    )
except Exception as exc:
    FLASHINFER_ATTENTION_AVAILABLE = False
    _FLASHINFER_GATE_ERROR = exc
else:
    _FLASHINFER_GATE_ERROR = None


pytestmark = [
    pytest.mark.skipif(
        not TEST_MODEL,
        reason="set NANOVLLM_TEST_MODEL to a local Qwen3 model directory",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="FlashInfer E2E tests require an NVIDIA GPU",
    ),
    pytest.mark.skipif(
        not FLASHINFER_ATTENTION_AVAILABLE,
        reason=f"FlashInfer attention is unavailable: {_FLASHINFER_GATE_ERROR}",
    ),
]


def _model_path() -> Path:
    assert TEST_MODEL is not None
    path = Path(TEST_MODEL).expanduser()
    if not path.is_dir():
        pytest.fail(f"NANOVLLM_TEST_MODEL is not a directory: {path}")
    return path


@contextmanager
def _owned_llm(*, batch_tokens: int, chunked_prefill: bool):
    from nanovllm import LLM

    original_dtype = torch.get_default_dtype()
    original_device = torch.get_default_device()
    llm = None
    try:
        llm = LLM(
            str(_model_path()),
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=96,
            max_num_batched_tokens=batch_tokens,
            max_num_seqs=4,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            attention_backend="flashinfer",
            kvcache_block_size=16,
            chunked_prefill=chunked_prefill,
        )
        # LLMEngine owns this registration; this context owns exactly one exit.
        atexit.unregister(llm.exit)
        yield llm
    finally:
        try:
            if llm is not None:
                llm.exit()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            torch.set_default_device(original_device)
            torch.set_default_dtype(original_dtype)
            torch.cuda.empty_cache()


def _sampling_params(max_tokens: int):
    from nanovllm import SamplingParams

    return SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )


def _assert_generation(outputs, expected_lengths):
    assert isinstance(outputs, list)
    assert len(outputs) == len(expected_lengths)
    for output, expected_length in zip(outputs, expected_lengths):
        assert set(output) == {"text", "token_ids"}
        assert isinstance(output["text"], str)
        assert len(output["token_ids"]) == expected_length
        assert all(type(token_id) is int for token_id in output["token_ids"])


@pytest.mark.skipif(
    BATCH_TOKENS < 34,
    reason="NANOVLLM_E2E_BATCH_TOKENS must be at least 34 for full prefills",
)
def test_flashinfer_prefill_decode_prefix_reuse_and_mixed_batch():
    torch.manual_seed(211)
    torch.cuda.manual_seed_all(211)

    with _owned_llm(
        batch_tokens=BATCH_TOKENS,
        chunked_prefill=False,
    ) as llm:
        # A 15-token prefill followed by three decode steps crosses a page.
        boundary_prompt = list(range(100, 115))
        boundary_outputs = llm.generate(
            [boundary_prompt],
            _sampling_params(3),
            use_tqdm=False,
        )
        _assert_generation(boundary_outputs, [3])
        boundary_tokens = boundary_prompt + boundary_outputs[0]["token_ids"][:1]
        boundary_hash = llm.scheduler.block_manager.compute_hash(boundary_tokens)
        assert boundary_hash in llm.scheduler.block_manager.hash_to_block_id

        # Finish a request with two complete pages, leaving both cached-free.
        shared_prefix = list(range(1000, 1032))
        first_outputs = llm.generate(
            [shared_prefix + [2000]],
            _sampling_params(1),
            use_tqdm=False,
        )
        _assert_generation(first_outputs, [1])

        block_manager = llm.scheduler.block_manager
        prefix_hash = -1
        cached_free_ids = []
        for offset in range(0, 32, 16):
            prefix_hash = block_manager.compute_hash(
                shared_prefix[offset : offset + 16],
                prefix_hash,
            )
            block_id = block_manager.hash_to_block_id[prefix_hash]
            assert block_manager.blocks[block_id].ref_count == 0
            assert block_id in block_manager.free_block_ids
            cached_free_ids.append(block_id)
        assert len(set(cached_free_ids)) == 2

        second_outputs = llm.generate(
            [shared_prefix + [2001, 2002]],
            _sampling_params(2),
            use_tqdm=False,
        )
        _assert_generation(second_outputs, [2])

        # Keep one request in decode, then admit a fresh prefill in the same step.
        decode_prompt = list(range(3000, 3012))
        llm.add_request(decode_prompt, _sampling_params(3))
        initial_outputs, _ = llm.step()
        assert initial_outputs == []
        assert len(llm.scheduler.running) == 1
        decode_id = llm.scheduler.running[0].seq_id
        assert llm.scheduler.running[0].num_completion_tokens == 1

        prefill_prompt = list(range(4000, 4007))
        llm.add_request(prefill_prompt, _sampling_params(2))
        prefill_id = llm.scheduler.waiting[-1].seq_id

        scheduled_batches = []
        original_schedule = llm.scheduler.schedule

        def recording_schedule():
            sequences = original_schedule()
            scheduled_batches.append(
                [
                    (seq.seq_id, seq.num_new_tokens, seq.num_cached_tokens)
                    for seq in sequences
                ]
            )
            return sequences

        llm.scheduler.schedule = recording_schedule
        step_outputs, _ = llm.step()
        mixed_work = [
            (seq_id, num_new_tokens)
            for seq_id, num_new_tokens, _ in scheduled_batches[0]
        ]
        assert mixed_work == [
            (prefill_id, len(prefill_prompt)), (decode_id, 1)
        ]

        finished = dict(step_outputs)
        for _ in range(16):
            if llm.is_finished():
                break
            outputs, _ = llm.step()
            finished.update(outputs)
        assert llm.is_finished()
        assert set(finished) == {decode_id, prefill_id}
        assert len(finished[decode_id]) == 3
        assert len(finished[prefill_id]) == 2


def test_flashinfer_chunked_prefill_completes_partial_page():
    torch.manual_seed(223)
    torch.cuda.manual_seed_all(223)

    with _owned_llm(batch_tokens=8, chunked_prefill=True) as llm:
        # 21 is deliberately unaligned: chunks are 8, 8, and 5 tokens.
        prompt = list(range(5000, 5021))
        outputs = llm.generate(
            [prompt],
            _sampling_params(2),
            use_tqdm=False,
        )
        _assert_generation(outputs, [2])
        assert llm.scheduler.is_finished()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
