from __future__ import annotations

from importlib.util import find_spec
from math import sqrt
from typing import Any

import torch
from flash_attn import flash_attn_varlen_func

FLASHINFER_WORKSPACE_BYTES = 64 * 1024 * 1024

_FLASHINFER_ATTENTION_IMPORT_ERROR: Exception | None = None
_BatchPrefillWithPagedKVCacheWrapper = None
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
    global _BatchPrefillWithPagedKVCacheWrapper

    if (
        not FLASHINFER_ATTENTION_AVAILABLE
        or _BatchPrefillWithPagedKVCacheWrapper is not None
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

        _BatchPrefillWithPagedKVCacheWrapper = (
            BatchPrefillWithPagedKVCacheWrapper
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
    """Unified paged prefill/decode backend using one plan for all layers."""

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
    ) -> None:
        super().__init__(
            num_q_heads, num_kv_heads, head_dim, block_size, dtype
        )
        _load_flashinfer_attention()
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
        self._planned = False

    def plan(self, context: Any) -> None:
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
        self._planned = True

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
            return _flash_attention_varlen(q, k, v, context, None)

        if not self._planned:
            raise RuntimeError(
                "FlashInferAttentionBackend.plan(context) must be called "
                "once before running a paged batch"
            )
        _validate_flashinfer_inputs(self, q, k_cache, v_cache)
        # K/V are stored into the cache by the layer before this call. One
        # wrapper plan is therefore reusable for every transformer layer.
        return self.prefill_wrapper.run(q, (k_cache, v_cache))


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
