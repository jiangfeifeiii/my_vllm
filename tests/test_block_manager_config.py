from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams


def _config(tmp_path, **kwargs) -> Config:
    hf_config = SimpleNamespace(max_position_embeddings=8192)
    with patch(
        "nanovllm.config.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        return Config(str(tmp_path), **kwargs)


def test_flashinfer_defaults_to_block_size_16(tmp_path):
    config = _config(tmp_path)

    assert config.attention_backend == "flashinfer"
    assert config.enable_lpm is True
    assert config.enable_in_batch_prefix_deprioritization is True
    assert config.kvcache_block_size == 16


@pytest.mark.parametrize(
    ("attention_backend", "block_size"),
    [
        ("flashinfer", 16),
        ("flashinfer", 7),
        ("legacy", 256),
        ("legacy", 512),
    ],
)
def test_attention_backend_accepts_supported_block_sizes(
    tmp_path, attention_backend: str, block_size: int
):
    config = _config(
        tmp_path,
        attention_backend=attention_backend,
        kvcache_block_size=block_size,
    )

    assert config.attention_backend == attention_backend
    assert config.kvcache_block_size == block_size


@pytest.mark.parametrize(
    ("attention_backend", "block_size"),
    [
        ("unknown", 16),
        ("flashinfer", 0),
        ("flashinfer", -16),
        ("legacy", 16),
        ("legacy", 257),
    ],
)
def test_attention_backend_rejects_invalid_block_sizes(
    tmp_path, attention_backend: str, block_size: int
):
    with pytest.raises(AssertionError):
        _config(
            tmp_path,
            attention_backend=attention_backend,
            kvcache_block_size=block_size,
        )


def test_llm_engine_add_request_propagates_configured_block_size():
    from nanovllm.engine.llm_engine import LLMEngine

    class CapturingScheduler:
        def __init__(self):
            self.sequence = None

        def add(self, sequence):
            self.sequence = sequence

    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(kvcache_block_size=256)
    engine.scheduler = CapturingScheduler()
    params = SamplingParams(max_tokens=3, ignore_eos=True)

    engine.add_request([1, 2, 3], params)

    assert engine.scheduler.sequence is not None
    assert engine.scheduler.sequence.block_size == 256
    assert engine.scheduler.sequence.max_tokens == 3
    assert engine.scheduler.sequence.ignore_eos is True
