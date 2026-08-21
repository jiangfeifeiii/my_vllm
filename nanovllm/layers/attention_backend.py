from __future__ import annotations

from importlib.util import find_spec
from math import sqrt
from typing import Any

import torch
from flash_attn import flash_attn_varlen_func

FLASHINFER_WORKSPACE_BYTES = 64 * 1024 * 1024

_FLASHINFER_ATTENTION_IMPORT_ERROR: Exception | None = None
_BatchPrefillWithPagedKVCacheWrapper = None
_BatchDecodeWithPagedKVCacheWrapper = None
FLASHINFER_ATTENTION_AVAILABLE = find_spec("flashinfer") is not None
FLASHINFER_IMPORT_ERROR: Exception | None = None
if not FLASHINFER_ATTENTION_AVAILABLE:
    FLASHINFER_IMPORT_ERROR = ModuleNotFoundError(
        "flashinfer-python==0.6.17 is not installed"
    )


def _load_flashinfer_attention() -> None:
    """Import FlashInfer only after ModelRunner selected its CUDA device."""
    global FLASHINFER_ATTENTION_AVAILABLE
    global FLASHINFER_IMPORT_ERROR
    global _FLASHINFER_ATTENTION_IMPORT_ERROR
    global _BatchDecodeWithPagedKVCacheWrapper
    global _BatchPrefillWithPagedKVCacheWrapper

    if (
        not FLASHINFER_ATTENTION_AVAILABLE
        or (
            _BatchPrefillWithPagedKVCacheWrapper is not None
            and _BatchDecodeWithPagedKVCacheWrapper is not None
        )
    ):
        return
    try:
        from nanovllm.layers import flashinfer_ops

        if not flashinfer_ops.FLASHINFER_AVAILABLE:
            raise RuntimeError(
                "FlashInfer operator initialization failed: "
                f"{flashinfer_ops.FLASHINFER_IMPORT_ERROR}"
            )
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        _BatchPrefillWithPagedKVCacheWrapper = (
            BatchPrefillWithPagedKVCacheWrapper
        )
        _BatchDecodeWithPagedKVCacheWrapper = (
            BatchDecodeWithPagedKVCacheWrapper
        )
    except Exception as exc:  # FlashInfer remains an optional CUDA dependency.
        _FLASHINFER_ATTENTION_IMPORT_ERROR = exc
        FLASHINFER_IMPORT_ERROR = exc
        FLASHINFER_ATTENTION_AVAILABLE = False


class AttentionBackend:
    """Small batch-planning interface shared by attention implementations."""
    supports_cudagraph = False

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
    ) -> None:
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if num_q_heads % num_kv_heads:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if head_dim <= 0 or block_size <= 0:
            raise ValueError("head_dim and block_size must be positive")

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.softmax_scale = 1.0 / sqrt(head_dim)

    def plan(self, context: Any) -> None:
        """Plan one batch. Implementations may reuse the plan across layers."""
        raise NotImplementedError

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        raise NotImplementedError


class LegacyFlashAttentionBackend(AttentionBackend):
    """The original FlashAttention varlen/paged implementation."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            num_q_heads, num_kv_heads, head_dim, block_size, dtype
        )
        if block_size % 256:
            raise ValueError(
                "Legacy FlashAttention paged KV cache requires block_size "
                "to be divisible by 256"
            )

    def plan(self, context: Any) -> None:
        # FlashAttention consumes the context metadata directly at each layer.
        return None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        block_tables = getattr(context, "block_tables", None)
        if block_tables is not None:
            _validate_cache_pair(k_cache, v_cache)
            k, v = k_cache, v_cache

        return _flash_attention_varlen(q, k, v, context, block_tables)


class FlashInferAttentionBackend(AttentionBackend):
    """Configurable unified/split paged backend shared by all layers."""

    _PAGE_FIELDS = (
        "page_q_indptr",
        "page_kv_indptr",
        "page_indices",
        "page_last_page_len",
    )

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
        attention_mode: str = "unified",
    ) -> None:
        super().__init__(
            num_q_heads, num_kv_heads, head_dim, block_size, dtype
        )
        _load_flashinfer_attention()
        if attention_mode not in ("unified", "split"):
            raise ValueError("attention_mode must be 'unified' or 'split'")
        self.attention_mode = attention_mode

        if dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "FlashInfer attention supports only float16 and bfloat16"
            )
        if not FLASHINFER_ATTENTION_AVAILABLE:
            error = (
                _FLASHINFER_ATTENTION_IMPORT_ERROR
                or FLASHINFER_IMPORT_ERROR
            )
            detail = f": {error}" if error is not None else ""
            raise RuntimeError(
                "FlashInfer attention is unavailable" + detail
            )
        if not torch.cuda.is_available():
            raise RuntimeError("FlashInfer attention requires a CUDA device")

        self.workspace = torch.empty(
            FLASHINFER_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        self.prefill_wrapper = _BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            kv_layout="NHD",
            backend="auto",
        )
        self.decode_wrapper = _BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            kv_layout="NHD",
            backend="auto",
        )
        self._planned = False
        self._num_prefill_seqs = 0
        self._num_prefill_tokens = 0
        self._num_decode_seqs = 0
        self._num_decode_tokens = 0

        self._output_buffer: torch.Tensor | None = None

    def plan(self, context: Any) -> None:
        self._planned = False
        self._num_prefill_seqs = 0
        self._num_prefill_tokens = 0
        self._num_decode_seqs = 0
        self._num_decode_tokens = 0
        metadata = [getattr(context, name, None) for name in self._PAGE_FIELDS]
        if all(item is None for item in metadata):
            # Model-memory profiling has no allocated KV cache. Its forward
            # call takes the explicitly warmup-only FlashAttention fallback.
            self._planned = False
            return
        missing = [
            name
            for name, item in zip(self._PAGE_FIELDS, metadata)
            if item is None
        ]
        if missing:
            raise ValueError(
                "incomplete FlashInfer page metadata; missing "
                + ", ".join(missing)
            )

        for name, item in zip(self._PAGE_FIELDS, metadata):
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"context.{name} must be a torch.Tensor")
            if item.dtype != torch.int32:
                raise TypeError(f"context.{name} must have dtype torch.int32")
            if item.device.type != "cuda":
                raise ValueError(f"context.{name} must be on CUDA")

        if metadata[2].numel() == 0:
            self._planned = False
            return

        q_indptr, kv_indptr, indices, last_page_len = metadata
        batch_size = last_page_len.numel()
        if (
            q_indptr.ndim != 1
            or kv_indptr.ndim != 1
            or indices.ndim != 1
            or last_page_len.ndim != 1
        ):
            raise ValueError("FlashInfer page metadata must be one-dimensional")
        if (
            q_indptr.numel() != batch_size + 1
            or kv_indptr.numel() != batch_size + 1
        ):
            raise ValueError("page indptr lengths must equal batch_size + 1")
        if int(q_indptr[0].item()) != 0 or int(kv_indptr[0].item()) != 0:
            raise ValueError("page indptr tensors must start at zero")
        if int(kv_indptr[-1].item()) != indices.numel():
            raise ValueError("page_kv_indptr does not match page_indices")

        num_prefill_seqs = getattr(context, "num_prefill_seqs", None)
        if num_prefill_seqs is None:
            num_prefill_seqs = batch_size
        if (
            isinstance(num_prefill_seqs, bool)
            or not isinstance(num_prefill_seqs, int)
        ):
            raise TypeError("context.num_prefill_seqs must be an int")
        if not 0 <= num_prefill_seqs <= batch_size:
            raise ValueError("context.num_prefill_seqs is outside the batch")

        expected_prefill_tokens = int(q_indptr[num_prefill_seqs].item())
        num_prefill_tokens = getattr(context, "num_prefill_tokens", None)
        if num_prefill_tokens is None:
            num_prefill_tokens = expected_prefill_tokens
        if (
            isinstance(num_prefill_tokens, bool)
            or not isinstance(num_prefill_tokens, int)
        ):
            raise TypeError("context.num_prefill_tokens must be an int")
        if num_prefill_tokens != expected_prefill_tokens:
            raise ValueError(
                "context.num_prefill_tokens must equal "
                "page_q_indptr[num_prefill_seqs]"
            )

        num_decode_seqs = batch_size - num_prefill_seqs
        num_decode_tokens = getattr(context, "num_decode_tokens", None)
        if num_decode_tokens is None:
            num_decode_tokens = int(q_indptr[-1].item()) - num_prefill_tokens
        if (
            isinstance(num_decode_tokens, bool)
            or not isinstance(num_decode_tokens, int)
        ):
            raise TypeError("context.num_decode_tokens must be an int")
        if num_decode_tokens != num_decode_seqs:
            raise ValueError(
                "decode suffix must contain one query token per sequence"
            )
        if num_decode_seqs:
            decode_q_lens = (
                q_indptr[num_prefill_seqs + 1 :]
                - q_indptr[num_prefill_seqs:-1]
            )
            if not bool(torch.all(decode_q_lens == 1).item()):
                raise ValueError(
                    "every sequence in the decode suffix must have q_len == 1"
                )

        if self.attention_mode == "unified":
            self.prefill_wrapper.plan(
                q_indptr,
                kv_indptr,
                indices,
                last_page_len,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                o_data_type=self.dtype,
            )
        else:
            prefill_page_end = int(kv_indptr[num_prefill_seqs].item())
            if num_prefill_seqs:
                self.prefill_wrapper.plan(
                    q_indptr[: num_prefill_seqs + 1],
                    kv_indptr[: num_prefill_seqs + 1],
                    indices[:prefill_page_end],
                    last_page_len[:num_prefill_seqs],
                    self.num_q_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.block_size,
                    causal=True,
                    q_data_type=self.dtype,
                    kv_data_type=self.dtype,
                    o_data_type=self.dtype,
                )
            if num_decode_seqs:
                decode_kv_indptr = (
                    kv_indptr[num_prefill_seqs:] - prefill_page_end
                )
                self.decode_wrapper.plan(
                    decode_kv_indptr,
                    indices[prefill_page_end:],
                    last_page_len[num_prefill_seqs:],
                    self.num_q_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.block_size,
                    pos_encoding_mode="NONE",
                    q_data_type=self.dtype,
                    kv_data_type=self.dtype,
                    o_data_type=self.dtype,
                )
        self._num_prefill_seqs = num_prefill_seqs
        self._num_prefill_tokens = num_prefill_tokens
        self._num_decode_seqs = num_decode_seqs
        self._num_decode_tokens = num_decode_tokens
        self._planned = True

    def _get_reusable_attention_output(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        required_tokens = q.size(0)
        buffer = self._output_buffer
        if (
            buffer is None
            or buffer.size(0) < required_tokens
            or buffer.device != q.device
            or buffer.dtype != q.dtype
        ):
            buffer = torch.empty_like(q)
            self._output_buffer = buffer
        return buffer[:required_tokens]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        has_k_cache = _has_cache(k_cache)
        has_v_cache = _has_cache(v_cache)
        if has_k_cache != has_v_cache:
            raise ValueError("key and value caches must both be present or absent")

        if not has_k_cache:
            # This path exists only for the model-runner memory warmup before
            # cache allocation. Serving batches always use the paged wrapper.
            output = _flash_attention_varlen(q, k, v, context, None)
            if self.attention_mode == "split":
                scratch = self._get_reusable_attention_output(q)
                scratch.copy_(output)
                return scratch
            return output

        if not self._planned:
            raise RuntimeError(
                "FlashInferAttentionBackend.plan(context) must be called "
                "once before running a paged batch"
            )
        _validate_flashinfer_inputs(self, q, k_cache, v_cache)
        expected_tokens = self._num_prefill_tokens + self._num_decode_tokens
        if q.size(0) != expected_tokens:
            raise ValueError(
                "query token count does not match the planned phase split"
            )

        cache = (k_cache, v_cache)
        if self.attention_mode == "unified":
            return self.prefill_wrapper.run(q, cache)

        output = self._get_reusable_attention_output(q)
        if self._num_prefill_seqs:
            self.prefill_wrapper.run(
                q[: self._num_prefill_tokens],
                cache,
                out=output[: self._num_prefill_tokens],
            )
        if self._num_decode_seqs:
            decode_output = output[self._num_prefill_tokens :]
            # FlashInfer's cute-dsl decode backend requires caller-provided
            # output to be zero-initialized; doing so is safe for AOT backends.
            decode_output.zero_()
            self.decode_wrapper.run(
                q[self._num_prefill_tokens :],
                cache,
                out=decode_output,
            )
        return output


def _has_cache(cache: torch.Tensor | None) -> bool:
    return isinstance(cache, torch.Tensor) and cache.numel() > 0


def _validate_cache_pair(
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
) -> None:
    if not _has_cache(k_cache) or not _has_cache(v_cache):
        raise ValueError("paged attention requires both key and value caches")


def _validate_flashinfer_inputs(
    backend: FlashInferAttentionBackend,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    if q.device.type != "cuda" or q.dtype != backend.dtype:
        raise ValueError("query must be a CUDA tensor with the backend dtype")
    if q.ndim != 3 or q.shape[1:] != (
        backend.num_q_heads,
        backend.head_dim,
    ):
        raise ValueError(
            "query must have shape [tokens, num_q_heads, head_dim]"
        )
    expected_tail = (
        backend.block_size,
        backend.num_kv_heads,
        backend.head_dim,
    )
    for name, cache in (("key", k_cache), ("value", v_cache)):
        if cache.device != q.device or cache.dtype != backend.dtype:
            raise ValueError(
                f"{name} cache must share the query device and dtype"
            )
        if cache.ndim != 4 or cache.shape[1:] != expected_tail:
            raise ValueError(
                f"{name} cache must have shape "
                "[pages, block_size, num_kv_heads, head_dim]"
            )


def _flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context: Any,
    block_tables: torch.Tensor | None,
) -> torch.Tensor:
    return flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=getattr(context, "max_seqlen_q"),
        cu_seqlens_q=getattr(context, "cu_seqlens_q"),
        max_seqlen_k=getattr(context, "max_seqlen_k"),
        cu_seqlens_k=getattr(context, "cu_seqlens_k"),
        softmax_scale=1.0 / sqrt(q.shape[-1]),
        causal=True,
        block_table=block_tables,
    )


__all__ = [
    "AttentionBackend",
    "LegacyFlashAttentionBackend",
    "FlashInferAttentionBackend",
    "FLASHINFER_ATTENTION_AVAILABLE",
    "FLASHINFER_WORKSPACE_BYTES",
]
