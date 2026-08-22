from __future__ import annotations

from importlib.util import find_spec
from math import sqrt
from types import MappingProxyType
from typing import Any

import torch
from flash_attn import flash_attn_varlen_func

from nanovllm.utils.context import BatchType

FLASHINFER_WORKSPACE_BYTES = 64 * 1024 * 1024

_FLASHINFER_ATTENTION_IMPORT_ERROR: Exception | None = None
_FLASHINFER_MIXED_ATTENTION_IMPORT_ERROR: Exception | None = None
_BatchPrefillWithPagedKVCacheWrapper = None
_BatchDecodeWithPagedKVCacheWrapper = None
_BatchAttention = None
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
    global _FLASHINFER_MIXED_ATTENTION_IMPORT_ERROR
    global _BatchAttention
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
        try:
            from flashinfer.attention import BatchAttention

            _BatchAttention = BatchAttention
        except Exception as exc:
            # Holistic mixed attention is optional. The serving backend can
            # always fall back to its zero-copy prefill/decode composition.
            _FLASHINFER_MIXED_ATTENTION_IMPORT_ERROR = exc
    except Exception as exc:  # FlashInfer remains an optional CUDA dependency.
        _FLASHINFER_ATTENTION_IMPORT_ERROR = exc
        FLASHINFER_IMPORT_ERROR = exc
        FLASHINFER_ATTENTION_AVAILABLE = False


class AttentionBackend:
    """Small batch-planning interface shared by attention implementations."""
    supports_full_decode_graph = False

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
        # FlashInfer requires the decode float workspace to be zero before its
        # first use. Both paged wrappers intentionally share this scratch.
        self.workspace.zero_()

        self.mixed_wrapper = None
        self._mixed_attention_unavailable_reason: str | None = None
        device_capability = torch.cuda.get_device_capability(
            self.workspace.device
        )
        if self.attention_mode == "split":
            self._mixed_attention_unavailable_reason = (
                "holistic mixed attention is disabled in split mode"
            )
        elif _BatchAttention is None:
            error = _FLASHINFER_MIXED_ATTENTION_IMPORT_ERROR
            detail = f": {error}" if error is not None else ""
            self._mixed_attention_unavailable_reason = (
                "FlashInfer BatchAttention is unavailable" + detail
            )
        elif self.head_dim > 256:
            self._mixed_attention_unavailable_reason = (
                "FlashInfer 0.6.17 BatchAttention requires head_dim <= 256"
            )
        elif device_capability == (12, 0):
            # FlashInfer 0.6.17's holistic planner hard-codes two CTAs per SM
            # for head_dim < 256. SM120 permits one for this kernel, so every
            # launch fails with "too many blocks in cooperative launch".
            self._mixed_attention_unavailable_reason = (
                "FlashInfer 0.6.17 BatchAttention is disabled on SM120"
            )
        else:
            try:
                self.mixed_wrapper = _BatchAttention(
                    kv_layout="NHD",
                    device=str(self.workspace.device),
                )
            except Exception as exc:
                self._mixed_attention_unavailable_reason = (
                    "FlashInfer BatchAttention initialization failed: "
                    f"{exc}"
                )

        self._planned = False
        self._planned_route: str | None = None
        self._route_counts = {
            "prefill": 0,
            "decode": 0,
            "mixed_holistic": 0,
            "mixed_split": 0,
        }
        self._num_prefill_seqs = 0
        self._num_prefill_tokens = 0
        self._num_decode_seqs = 0
        self._num_decode_tokens = 0

        self._output_buffer: torch.Tensor | None = None
        self._lse_buffer: torch.Tensor | None = None
        self._decode_graph_wrapper = None
        self._decode_graph_output: torch.Tensor | None = None
        self.graph_workspace: torch.Tensor | None = None

    @property
    def supports_full_decode_graph(self) -> bool:
        return self.attention_mode == "unified"

    @property
    def planned_route(self) -> str | None:
        return self._planned_route

    @property
    def mixed_attention_available(self) -> bool:
        return self.mixed_wrapper is not None

    @property
    def mixed_attention_unavailable_reason(self) -> str | None:
        return self._mixed_attention_unavailable_reason

    @property
    def route_counts(self):
        return MappingProxyType(dict(self._route_counts))

    def create_full_decode_graph_wrapper(
        self,
        qo_indptr_buf: torch.Tensor,
        paged_kv_indptr_buf: torch.Tensor,
        paged_kv_indices_buf: torch.Tensor,
        paged_kv_last_page_len_buf: torch.Tensor,
    ):
        if not self.supports_full_decode_graph:
            raise RuntimeError(
                "full decode CUDA Graph requires unified FlashInfer attention"
            )
        if self.graph_workspace is None:
            self.graph_workspace = torch.zeros(
                FLASHINFER_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=self.workspace.device,
            )
        wrapper = _BatchDecodeWithPagedKVCacheWrapper(
            self.graph_workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=paged_kv_indptr_buf,
            paged_kv_indices_buffer=paged_kv_indices_buf,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buf,
            backend="auto",
        )
        # Decode attention does not consume q indptr, but ModelRunner owns and
        # reuses this fixed-address buffer as part of its graph metadata.
        wrapper._nanovllm_qo_indptr_buffer = qo_indptr_buf
        return wrapper

    def plan_full_decode_graph(self, wrapper, context: Any) -> None:
        if not self.supports_full_decode_graph:
            raise RuntimeError(
                "full decode CUDA Graph requires unified FlashInfer attention"
            )
        if getattr(context, "batch_type", None) is not BatchType.PURE_DECODE:
            raise ValueError("full decode graph planning requires PURE_DECODE")
        if getattr(context, "num_prefill_seqs", None) != 0:
            raise ValueError("full decode graph planning requires zero prefills")

        metadata = [getattr(context, name, None) for name in self._PAGE_FIELDS]
        if any(item is None for item in metadata):
            raise ValueError("full decode graph planning requires page metadata")
        q_indptr, kv_indptr, indices, last_page_len = metadata
        batch_size = last_page_len.numel()
        if q_indptr.numel() != batch_size + 1:
            raise ValueError("decode q indptr length must equal batch size + 1")
        if kv_indptr.numel() != batch_size + 1:
            raise ValueError("decode KV indptr length must equal batch size + 1")
        trusted_metadata = getattr(
            context, "page_metadata_trusted", False
        )
        if not isinstance(trusted_metadata, bool):
            raise TypeError("context.page_metadata_trusted must be a bool")
        if trusted_metadata:
            num_pages = getattr(context, "num_pages", None)
            num_prefill_pages = getattr(
                context, "num_prefill_pages", None
            )
            if type(num_pages) is not int or num_pages != indices.numel():
                raise ValueError(
                    "trusted decode num_pages must match page indices"
                )
            if type(num_prefill_pages) is not int or num_prefill_pages != 0:
                raise ValueError(
                    "trusted decode metadata must have zero prefill pages"
                )
        else:
            if int(kv_indptr[-1].item()) != indices.numel():
                raise ValueError(
                    "decode KV indptr does not match page indices"
                )
            query_lengths = q_indptr[1:] - q_indptr[:-1]
            if not bool(torch.all(query_lengths == 1).item()):
                raise ValueError(
                    "full decode graph requires one query per request"
                )
        if getattr(context, "num_decode_tokens", None) != batch_size:
            raise ValueError("decode token count must equal decode batch size")

        wrapper._nanovllm_qo_indptr_buffer.copy_(q_indptr)
        wrapper.plan(
            kv_indptr,
            indices,
            last_page_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            o_data_type=self.dtype,
        )

    def activate_full_decode_graph(
        self,
        wrapper,
        output: torch.Tensor,
    ) -> None:
        if self._decode_graph_wrapper is not None:
            raise RuntimeError("a decode graph wrapper is already active")
        self._decode_graph_wrapper = wrapper
        self._decode_graph_output = output

    def deactivate_full_decode_graph(self) -> None:
        self._decode_graph_wrapper = None
        self._decode_graph_output = None

    def plan(self, context: Any) -> None:
        self._planned = False
        self._planned_route = None
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

        trusted_metadata = getattr(
            context, "page_metadata_trusted", False
        )
        if not isinstance(trusted_metadata, bool):
            raise TypeError("context.page_metadata_trusted must be a bool")
        if trusted_metadata:
            num_pages = getattr(context, "num_pages", None)
            num_prefill_pages = getattr(
                context, "num_prefill_pages", None
            )
            if isinstance(num_pages, bool) or not isinstance(num_pages, int):
                raise TypeError("context.num_pages must be an int")
            if num_pages != indices.numel():
                raise ValueError(
                    "context.num_pages must equal page_indices.numel()"
                )
            if (
                isinstance(num_prefill_pages, bool)
                or not isinstance(num_prefill_pages, int)
            ):
                raise TypeError("context.num_prefill_pages must be an int")
            if not 0 <= num_prefill_pages <= num_pages:
                raise ValueError(
                    "context.num_prefill_pages is outside page_indices"
                )
        else:
            num_pages = indices.numel()
            num_prefill_pages = None
            if (
                int(q_indptr[0].item()) != 0
                or int(kv_indptr[0].item()) != 0
            ):
                raise ValueError("page indptr tensors must start at zero")
            if int(kv_indptr[-1].item()) != indices.numel():
                raise ValueError(
                    "page_kv_indptr does not match page_indices"
                )

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
        if trusted_metadata:
            if num_prefill_seqs == 0 and num_prefill_pages != 0:
                raise ValueError(
                    "pure decode metadata must have zero prefill pages"
                )
            if (
                num_prefill_seqs == batch_size
                and num_prefill_pages != num_pages
            ):
                raise ValueError(
                    "pure prefill metadata must own all batch pages"
                )
            if (
                0 < num_prefill_seqs < batch_size
                and not 0 < num_prefill_pages < num_pages
            ):
                raise ValueError(
                    "mixed metadata must split prefill and decode pages"
                )

        num_prefill_tokens = getattr(context, "num_prefill_tokens", None)
        if num_prefill_tokens is None and not trusted_metadata:
            num_prefill_tokens = int(
                q_indptr[num_prefill_seqs].item()
            )
        if (
            isinstance(num_prefill_tokens, bool)
            or not isinstance(num_prefill_tokens, int)
        ):
            raise TypeError("context.num_prefill_tokens must be an int")
        if not trusted_metadata:
            expected_prefill_tokens = int(
                q_indptr[num_prefill_seqs].item()
            )
            if num_prefill_tokens != expected_prefill_tokens:
                raise ValueError(
                    "context.num_prefill_tokens must equal "
                    "page_q_indptr[num_prefill_seqs]"
                )

        num_decode_seqs = batch_size - num_prefill_seqs
        num_decode_tokens = getattr(context, "num_decode_tokens", None)
        if num_decode_tokens is None and not trusted_metadata:
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
        if num_decode_seqs and not trusted_metadata:
            decode_q_lens = (
                q_indptr[num_prefill_seqs + 1 :]
                - q_indptr[num_prefill_seqs:-1]
            )
            if not bool(torch.all(decode_q_lens == 1).item()):
                raise ValueError(
                    "every sequence in the decode suffix must have q_len == 1"
                )

        if num_prefill_seqs == 0:
            expected_batch_type = BatchType.PURE_DECODE
        elif num_prefill_seqs == batch_size:
            expected_batch_type = BatchType.PURE_PREFILL
        else:
            expected_batch_type = BatchType.MIXED
        batch_type = getattr(context, "batch_type", None)
        if batch_type is not None:
            if not isinstance(batch_type, BatchType):
                raise TypeError("context.batch_type must be a BatchType")
            if batch_type is not expected_batch_type:
                raise ValueError(
                    "context.batch_type does not match num_prefill_seqs"
                )

        if expected_batch_type is BatchType.PURE_PREFILL:
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
            planned_route = "prefill"
        elif expected_batch_type is BatchType.PURE_DECODE:
            self.decode_wrapper.plan(
                kv_indptr,
                indices,
                last_page_len,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                pos_encoding_mode="NONE",
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                o_data_type=self.dtype,
            )
            planned_route = "decode"
        elif (
            self.attention_mode == "unified"
            and self.mixed_wrapper is not None
        ):
            kv_lengths = (
                (kv_indptr[1:] - kv_indptr[:-1] - 1) * self.block_size
                + last_page_len
            )
            self.mixed_wrapper.plan(
                q_indptr,
                kv_indptr,
                indices,
                kv_lengths,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.head_dim,
                self.block_size,
                causal=True,
                sm_scale=self.softmax_scale,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            planned_route = "mixed_holistic"
        else:
            prefill_page_end = (
                num_prefill_pages
                if trusted_metadata
                else int(kv_indptr[num_prefill_seqs].item())
            )
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
            planned_route = "mixed_split"
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
        self._planned_route = planned_route
        self._route_counts[planned_route] += 1
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

    def _get_reusable_lse(self, q: torch.Tensor) -> torch.Tensor:
        required_tokens = q.size(0)
        buffer = self._lse_buffer
        if (
            buffer is None
            or buffer.size(0) < required_tokens
            or buffer.device != q.device
        ):
            buffer = torch.empty(
                required_tokens,
                self.num_q_heads,
                dtype=torch.float32,
                device=q.device,
            )
            self._lse_buffer = buffer
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

        graph_wrapper = self._decode_graph_wrapper
        if graph_wrapper is not None:
            if not has_k_cache:
                raise RuntimeError("full decode graph requires allocated KV cache")
            _validate_flashinfer_inputs(self, q, k_cache, v_cache)
            output = self._decode_graph_output
            if output is None or output.shape != q.shape:
                raise RuntimeError(
                    "decode graph attention output does not match query shape"
                )
            if output.device != q.device or output.dtype != q.dtype:
                raise RuntimeError(
                    "decode graph attention output device/dtype mismatch"
                )
            return graph_wrapper.run(
                q,
                (k_cache, v_cache),
                out=output,
            )

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
        output = self._get_reusable_attention_output(q)
        if self._planned_route == "prefill":
            self.prefill_wrapper.run(q, cache, out=output)
            return output
        if self._planned_route == "decode":
            self.decode_wrapper.run(q, cache, out=output)
            return output
        if self._planned_route == "mixed_holistic":
            if self.mixed_wrapper is None:
                raise RuntimeError(
                    "planned holistic mixed attention wrapper is unavailable"
                )
            result, _ = self.mixed_wrapper.run(
                q,
                cache,
                out=output,
                lse=self._get_reusable_lse(q),
            )
            return result
        if self._planned_route != "mixed_split":
            raise RuntimeError("invalid planned FlashInfer attention route")

        if self._num_prefill_seqs:
            self.prefill_wrapper.run(
                q[: self._num_prefill_tokens],
                cache,
                out=output[: self._num_prefill_tokens],
            )
        if self._num_decode_seqs:
            decode_output = output[self._num_prefill_tokens :]
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
