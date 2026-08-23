import os
from dataclasses import dataclass, field
from enum import Enum

from transformers import AutoConfig


class CUDAGraphPolicy(str, Enum):
    NONE = "none"
    FULL_DECODE_ONLY = "full_decode_only"


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 40960
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    cudagraph_mode: CUDAGraphPolicy | str = (
        CUDAGraphPolicy.FULL_DECODE_ONLY
    )
    cudagraph_batch_sizes: tuple[int, ...] = (
        1, 2, 4, 8, 16, 32, 64
    )
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    chunked_prefill: bool = False
    enable_lpm: bool = True
    enable_same_step_prefix_reuse: bool = True
    attention_backend: str = "auto"
    attention_mode: str = "unified"
    operator_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        if isinstance(self.cudagraph_mode, str):
            try:
                self.cudagraph_mode = CUDAGraphPolicy(
                    self.cudagraph_mode.lower()
                )
            except ValueError as exc:
                raise ValueError(
                    "cudagraph_mode must be 'none' or "
                    "'full_decode_only'"
                ) from exc
        elif not isinstance(self.cudagraph_mode, CUDAGraphPolicy):
            raise TypeError("cudagraph_mode must be a CUDAGraphPolicy or str")
        if self.enforce_eager:
            self.cudagraph_mode = CUDAGraphPolicy.NONE
        self.enforce_eager = self.cudagraph_mode is CUDAGraphPolicy.NONE
        self.cudagraph_batch_sizes = tuple(self.cudagraph_batch_sizes)
        if not self.cudagraph_batch_sizes or any(
            type(size) is not int or size <= 0
            for size in self.cudagraph_batch_sizes
        ):
            raise ValueError("cudagraph_batch_sizes must contain positive ints")
        if len(set(self.cudagraph_batch_sizes)) != len(self.cudagraph_batch_sizes):
            raise ValueError("cudagraph_batch_sizes must not contain duplicates")
        self.cudagraph_batch_sizes = tuple(sorted(self.cudagraph_batch_sizes))
        if (
            type(self.kvcache_block_size) is not int
            or self.kvcache_block_size <= 0
        ):
            raise ValueError("kvcache_block_size must be a positive int")
        self.attention_backend = self.attention_backend.lower()
        if self.attention_backend not in (
            "auto",
            "flashattention",
            "flashinfer",
            "legacy",
        ):
            raise ValueError(
                "attention_backend must be 'auto', 'flashattention', "
                "'flashinfer', or the compatibility alias 'legacy'"
            )
        self.attention_mode = self.attention_mode.lower()
        if self.attention_mode not in ("unified", "split"):
            raise ValueError("attention_mode must be 'unified' or 'split'")
        if self.attention_backend in ("flashattention", "legacy"):
            if self.kvcache_block_size % 256:
                raise ValueError(
                    "FlashAttention requires kvcache_block_size divisible by 256"
                )
            if self.attention_mode != "unified":
                raise ValueError(
                    "FlashAttention supports only attention_mode='unified'"
                )
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # assert self.max_num_batched_tokens >= self.max_model_len
