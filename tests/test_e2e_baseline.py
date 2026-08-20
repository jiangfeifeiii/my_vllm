"""Opt-in GPU end-to-end smoke test for the unmodified baseline."""

import atexit
import os
from pathlib import Path

import pytest
import torch


TEST_MODEL = os.environ.get("NANOVLLM_TEST_MODEL")

pytestmark = [
    pytest.mark.skipif(
        not TEST_MODEL,
        reason="set NANOVLLM_TEST_MODEL to a local Qwen3 model directory",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="nano-vllm baseline requires an NVIDIA GPU",
    ),
]


def test_gpu_generation_baseline():
    from nanovllm import LLM, SamplingParams

    assert TEST_MODEL is not None
    model_path = Path(TEST_MODEL).expanduser()
    if not model_path.is_dir():
        pytest.fail(f"NANOVLLM_TEST_MODEL is not a directory: {model_path}")

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    llm = None
    try:
        llm = LLM(
            str(model_path),
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=64,
            max_num_batched_tokens=64,
            max_num_seqs=2,
            gpu_memory_utilization=0.35,
        )
        # The engine registers this bound method itself; this test owns cleanup.
        atexit.unregister(llm.exit)

        outputs = llm.generate(
            [[1, 2], [1, 2, 3]],
            SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True),
            use_tqdm=False,
        )

        assert isinstance(outputs, list)
        assert len(outputs) == 2
        for output in outputs:
            assert set(output) == {"text", "token_ids"}
            assert isinstance(output["text"], str)
            assert isinstance(output["token_ids"], list)
            assert len(output["token_ids"]) == 2
            assert all(type(token_id) is int for token_id in output["token_ids"])
    finally:
        if llm is not None:
            try:
                llm.exit()
            finally:
                torch.cuda.empty_cache()
