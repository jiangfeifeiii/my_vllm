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
def _owned_llm(
    *,
    batch_tokens: int,
    chunked_prefill: bool,
    attention_mode: str,
    same_step_prefix_reuse: bool = True,
):
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
            attention_mode=attention_mode,
            attention_backend="flashinfer",
            kvcache_block_size=16,
            chunked_prefill=chunked_prefill,
            enable_same_step_prefix_reuse=same_step_prefix_reuse,
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


def _same_step_logits(attention_mode: str, enabled: bool):
    from nanovllm.utils.context import reset_context

    shared = list(range(7000, 7032))
    prompts = [shared + [7100], shared + [7200]]
    with _owned_llm(
        batch_tokens=BATCH_TOKENS,
        chunked_prefill=False,
        attention_mode=attention_mode,
        same_step_prefix_reuse=enabled,
    ) as llm:
        for prompt in prompts:
            llm.add_request(prompt, _sampling_params(1))
        scheduled = llm.scheduler.schedule()
        cached_tokens = [seq.num_cached_tokens for seq in scheduled]
        new_tokens = [seq.num_new_tokens for seq in scheduled]
        prefix_tables = [seq.block_table[:2] for seq in scheduled]
        try:
            input_ids, positions = llm.model_runner.prepare_model_input(
                scheduled,
                llm.scheduler.num_scheduled_prefill_seqs,
            )
            logits = llm.model_runner.run_model(input_ids, positions)
            torch.cuda.synchronize()
            logits = logits.float().cpu()
        finally:
            reset_context()
    return logits, cached_tokens, new_tokens, prefix_tables


def _same_step_alias_and_private_logits(attention_mode: str):
    from nanovllm.utils.context import get_context, reset_context

    shared = list(range(7000, 7032))
    prompts = [shared + [7100], shared + [7200]]
    with _owned_llm(
        batch_tokens=BATCH_TOKENS,
        chunked_prefill=False,
        attention_mode=attention_mode,
    ) as llm:
        for prompt in prompts:
            llm.add_request(prompt, _sampling_params(1))
        leader, follower = llm.scheduler.schedule()
        assert [leader.num_cached_tokens, follower.num_cached_tokens] == [0, 32]
        assert [leader.num_new_tokens, follower.num_new_tokens] == [33, 1]
        assert follower.block_table[:2] == leader.block_table[:2]

        used_pages = set(leader.block_table + follower.block_table)
        private_pages = [
            block_id
            for block_id in llm.scheduler.block_manager.free_block_ids
            if block_id not in used_pages
        ][:2]
        assert len(private_pages) == 2
        device = llm.model_runner.kv_cache.device
        shared_page_tensor = torch.tensor(
            leader.block_table[:2], dtype=torch.long, device=device
        )
        private_page_tensor = torch.tensor(
            private_pages, dtype=torch.long, device=device
        )

        expected_input_ids = prompts[0] + [prompts[1][-1]]
        expected_positions = list(range(33)) + [32]
        expected_slots = [
            leader.block_table[index // 16] * 16 + index % 16
            for index in range(33)
        ] + [follower.block_table[2] * 16]

        def run_once():
            try:
                input_ids, positions = llm.model_runner.prepare_model_input(
                    [leader, follower],
                    llm.scheduler.num_scheduled_prefill_seqs,
                )
                context = get_context()
                metadata = {
                    "input_ids": input_ids.cpu().tolist(),
                    "positions": positions.cpu().tolist(),
                    "q_indptr": context.page_q_indptr.cpu().tolist(),
                    "kv_indptr": context.page_kv_indptr.cpu().tolist(),
                    "page_indices": context.page_indices.cpu().tolist(),
                    "last_page_len": context.page_last_page_len.cpu().tolist(),
                    "slot_mapping": context.slot_mapping.cpu().tolist(),
                    "logits_indices": (
                        context.seq_need_compute_logits.cpu().tolist()
                    ),
                }
                logits = llm.model_runner.run_model(input_ids, positions)
                torch.cuda.synchronize()
                return logits.float().cpu(), metadata
            finally:
                reset_context()

        # Stale reads become NaNs unless every layer's leader store is visible
        # before the follower reads the two aliased pages.
        llm.model_runner.kv_cache.index_fill_(
            2, shared_page_tensor, float("nan")
        )
        torch.cuda.synchronize()
        alias_logits, alias_metadata = run_once()
        assert torch.isfinite(alias_logits).all()
        assert torch.isfinite(
            llm.model_runner.kv_cache.index_select(2, shared_page_tensor)
        ).all()

        llm.model_runner.kv_cache.index_copy_(
            2,
            private_page_tensor,
            llm.model_runner.kv_cache.index_select(2, shared_page_tensor),
        )
        torch.cuda.synchronize()
        follower.block_table[:2] = private_pages
        private_logits, private_metadata = run_once()

    assert alias_metadata["input_ids"] == expected_input_ids
    assert alias_metadata["positions"] == expected_positions
    assert alias_metadata["q_indptr"] == [0, 33, 34]
    assert alias_metadata["kv_indptr"] == [0, 3, 6]
    assert alias_metadata["last_page_len"] == [1, 1]
    assert alias_metadata["slot_mapping"] == expected_slots
    assert alias_metadata["logits_indices"] == [0, 1]
    for key in alias_metadata.keys() - {"page_indices"}:
        assert private_metadata[key] == alias_metadata[key]
    assert (
        private_metadata["page_indices"][:3]
        == alias_metadata["page_indices"][:3]
    )
    assert private_metadata["page_indices"][3:5] == private_pages
    assert (
        private_metadata["page_indices"][5]
        == alias_metadata["page_indices"][5]
    )
    return alias_logits, private_logits


@pytest.mark.skipif(
    BATCH_TOKENS < 34,
    reason="NANOVLLM_E2E_BATCH_TOKENS must be at least 34",
)
@pytest.mark.parametrize("attention_mode", ["unified", "split"])
def test_same_step_alias_matches_private_prefix_pages(attention_mode):
    alias_logits, private_logits = _same_step_alias_and_private_logits(
        attention_mode
    )
    assert torch.isfinite(private_logits).all()
    torch.testing.assert_close(
        alias_logits,
        private_logits,
        rtol=1e-3,
        atol=1e-3,
    )

@pytest.mark.skipif(
    BATCH_TOKENS < 66,
    reason="NANOVLLM_E2E_BATCH_TOKENS must be at least 66 for cold baseline",
)
@pytest.mark.parametrize("attention_mode", ["unified", "split"])
def test_same_step_logits_match_independent_cold_prefill(attention_mode):
    cold_logits, cold_cached, cold_new, cold_tables = _same_step_logits(
        attention_mode, False
    )
    reuse_logits, reuse_cached, reuse_new, reuse_tables = _same_step_logits(
        attention_mode, True
    )

    assert cold_cached == [0, 0]
    assert cold_new == [33, 33]
    assert set(cold_tables[0]).isdisjoint(cold_tables[1])
    assert reuse_cached == [0, 32]
    assert reuse_new == [33, 1]
    assert reuse_tables[0] == reuse_tables[1]

    # OFF packs 66 queries while ON packs 34, so BF16 GEMMs have a different
    # deterministic rounding path. The tight alias/private test above isolates
    # KV correctness; this envelope checks both rows against the full cold run.
    cosine = torch.nn.functional.cosine_similarity(
        reuse_logits, cold_logits, dim=-1
    )
    assert torch.all(cosine > 0.9995)
    rms = (reuse_logits - cold_logits).square().mean(dim=-1).sqrt()
    assert rms[1] <= 2 * rms[0]
    torch.testing.assert_close(
        reuse_logits,
        cold_logits,
        rtol=2e-2,
        atol=2e-1,
    )



@pytest.mark.skipif(
    BATCH_TOKENS < 66,
    reason="NANOVLLM_E2E_BATCH_TOKENS must be at least 66 for full prefills",
)
@pytest.mark.parametrize("attention_mode", ["unified", "split"])
def test_flashinfer_prefill_decode_prefix_reuse_and_mixed_batch(attention_mode):
    torch.manual_seed(211)
    torch.cuda.manual_seed_all(211)

    with _owned_llm(
        batch_tokens=BATCH_TOKENS,
        chunked_prefill=False,
        attention_mode=attention_mode,
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

        # The follower reuses full pages published by the leader while both
        # requests remain in the same packed GPU prefill batch.
        in_batch_prefix = list(range(6000, 6032))
        llm.add_request(
            in_batch_prefix + [6100],
            _sampling_params(1),
        )
        llm.add_request(
            in_batch_prefix + [6200],
            _sampling_params(1),
        )
        leader, follower = list(llm.scheduler.waiting)
        in_batch = llm.scheduler.schedule()
        assert in_batch == [leader, follower]
        leader_prefix_ids = leader.block_table[:2]
        follower_prefix_ids = follower.block_table[:2]
        tail_ids = [leader.block_table[2], follower.block_table[2]]
        assert follower_prefix_ids == leader_prefix_ids
        assert leader.num_cached_tokens == 0
        assert follower.num_cached_tokens == 32
        assert [leader.num_new_tokens, follower.num_new_tokens] == [33, 1]
        assert all(
            block_manager.blocks[block_id].ref_count == 2
            for block_id in leader_prefix_ids
        )
        assert all(block_manager.blocks[block_id].ref_count == 1 for block_id in tail_ids)
        assert llm.scheduler.same_step_hit_blocks_by_seq == {follower.seq_id: 2}

        token_ids, logits_indices = llm.model_runner.call(
            "run",
            in_batch,
            llm.scheduler.num_scheduled_prefill_seqs,
        )
        llm.scheduler.postprocess(
            in_batch,
            token_ids,
            logits_indices,
        )
        assert leader.is_finished and follower.is_finished
        assert all(
            block_manager.blocks[block_id].ref_count == 0
            for block_id in leader_prefix_ids + tail_ids
        )
        assert all(
            block_id in block_manager.free_block_ids
            for block_id in leader_prefix_ids + tail_ids
        )

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
        backend = llm.model_runner.attention_backend
        assert backend.attention_mode == attention_mode
        assert backend._num_prefill_seqs == 1
        assert backend._num_prefill_tokens == len(prefill_prompt)
        assert backend._num_decode_seqs == 1
        assert backend._num_decode_tokens == 1

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


@pytest.mark.parametrize("attention_mode", ["unified", "split"])
def test_flashinfer_chunked_prefill_completes_partial_page(attention_mode):
    torch.manual_seed(223)
    torch.cuda.manual_seed_all(223)

    with _owned_llm(
        batch_tokens=8,
        chunked_prefill=True,
        attention_mode=attention_mode,
    ) as llm:
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
