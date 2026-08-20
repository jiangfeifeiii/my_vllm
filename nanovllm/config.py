import os
from dataclasses import dataclass, field
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 40960
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    chunked_prefill: bool = False
    attention_backend: str = "flashinfer"
    operator_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size > 0
        assert self.attention_backend in ("flashinfer", "legacy")
        if self.attention_backend == "legacy":
            assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # assert self.max_num_batched_tokens >= self.max_model_len
