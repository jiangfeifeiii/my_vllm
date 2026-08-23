from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from math import sqrt
from types import MappingProxyType
from typing import Any

import torch
import torch.nn.functional as F

from nanovllm.utils.context import BatchType, CommonAttentionMetadata


FLASHINFER_WORKSPACE_BYTES = 64 * 1024 * 1024


def _module_is_present(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_flash_attn_varlen_func = None
_FLASH_ATTENTION_IMPORT_ERROR: Exception | None = None
FLASH_ATTENTION_AVAILABLE = _module_is_present("flash_attn")
if not FLASH_ATTENTION_AVAILABLE:
    _FLASH_ATTENTION_IMPORT_ERROR = ModuleNotFoundError(
        "Dao FlashAttention is not installed"
    )


def _load_flash_attention() -> None:
    """Lazily import Dao FlashAttention after the CUDA device is selected."""
    global FLASH_ATTENTION_AVAILABLE
    global _FLASH_ATTENTION_IMPORT_ERROR
    global _flash_attn_varlen_func

    if _flash_attn_varlen_func is not None or not FLASH_ATTENTION_AVAILABLE:
        return
    try:
        from flash_attn import flash_attn_varlen_func

        _flash_attn_varlen_func = flash_attn_varlen_func
    except Exception as exc:
        _FLASH_ATTENTION_IMPORT_ERROR = exc
        FLASH_ATTENTION_AVAILABLE = False


_FLASHINFER_ATTENTION_IMPORT_ERROR: Exception | None = None
_FLASHINFER_MIXED_ATTENTION_IMPORT_ERROR: Exception | None = None
_BatchPrefillWithPagedKVCacheWrapper = None
_BatchDecodeWithPagedKVCacheWrapper = None
_BatchAttention = None
FLASHINFER_ATTENTION_AVAILABLE = _module_is_present("flashinfer")
FLASHINFER_IMPORT_ERROR: Exception | None = None
if not FLASHINFER_ATTENTION_AVAILABLE:
    FLASHINFER_IMPORT_ERROR = ModuleNotFoundError(
        "flashinfer-python==0.6.17 is not installed"
    )


def _load_flashinfer_attention() -> None:
    """Lazily import FlashInfer after the CUDA device is selected."""
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
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

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
            _FLASHINFER_MIXED_ATTENTION_IMPORT_ERROR = exc
    except Exception as exc:
        _FLASHINFER_ATTENTION_IMPORT_ERROR = exc
        FLASHINFER_IMPORT_ERROR = exc
        FLASHINFER_ATTENTION_AVAILABLE = False


class AttentionRoute(str, Enum):
    WARMUP = "warmup"
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED_UNIFIED = "mixed_unified"
    MIXED_SPLIT = "mixed_split"


@dataclass(frozen=True)
class AttentionPlan:
    """One explicit, backend-owned execution decision for a scheduled batch."""

    batch_type: BatchType
    route: AttentionRoute
    common: CommonAttentionMetadata
    metadata: Any

    @property
    def common_metadata(self) -> CommonAttentionMetadata:
        return self.common

    @property
    def backend_metadata(self) -> Any:
        return self.metadata


@dataclass(frozen=True)
class FlashAttentionMetadata:
    query_start_loc: torch.Tensor
    kv_start_loc: torch.Tensor | None
    block_tables: torch.Tensor | None
    max_q_len: int
    max_kv_len: int


@dataclass(frozen=True)
class FlashInferMetadata:
    query_start_loc: torch.Tensor
    kv_indptr: torch.Tensor | None
    page_indices: torch.Tensor | None
    last_page_len: torch.Tensor | None
    num_pages: int
    num_prefill_pages: int

    @property
    def is_cacheless(self) -> bool:
        return self.page_indices is None


@dataclass
class _FlashInferFullDecodeGraphState:
    batch_size: int
    page_indices_capacity: int
    wrapper: Any
    query_start_loc_buffer: torch.Tensor
    kv_indptr_buffer: torch.Tensor
    page_indices_buffer: torch.Tensor
    last_page_len_buffer: torch.Tensor
    output: torch.Tensor
    plan: AttentionPlan | None = None


class AttentionBackend:
    """Unified lifecycle implemented by every complete attention backend."""

    backend_name = "base"
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

    @classmethod
    def support_reason(
        cls,
        *,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        **_: Any,
    ) -> str | None:
        del block_size, dtype, device
        return None

    @classmethod
    def supports(cls, **capabilities: Any) -> bool:
        return cls.support_reason(**capabilities) is None

    def build_metadata(
        self,
        common: CommonAttentionMetadata,
    ) -> Any:
        raise NotImplementedError

    def build_plan(
        self,
        common: CommonAttentionMetadata,
        metadata: Any | None = None,
    ) -> AttentionPlan:
        raise NotImplementedError

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        plan: AttentionPlan,
    ) -> torch.Tensor:
        raise NotImplementedError

    def create_full_decode_graph_state(
        self,
        batch_size: int,
        page_indices_capacity: int,
        device: torch.device | str,
    ) -> Any:
        raise RuntimeError(
            f"{type(self).__name__} does not support full decode CUDA Graph"
        )

    def full_decode_graph_metadata_fits(
        self,
        state: Any,
        plan: AttentionPlan,
    ) -> bool:
        del state, plan
        return False

    def plan_full_decode_graph(
        self,
        state: Any,
        plan: AttentionPlan,
    ) -> None:
        raise RuntimeError(
            f"{type(self).__name__} does not support full decode CUDA Graph"
        )

    def activate_full_decode_graph(self, state: Any) -> None:
        raise RuntimeError(
            f"{type(self).__name__} does not support full decode CUDA Graph"
        )

    def deactivate_full_decode_graph(self) -> None:
        return None


class FlashAttentionBackend(AttentionBackend):
    """Dao FlashAttention varlen backend, including its paged-KV mode."""

    backend_name = "flashattention"

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
                "Dao FlashAttention paged KV cache requires block_size "
                "to be divisible by 256"
            )
        if dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "Dao FlashAttention supports only float16 and bfloat16"
            )
        self._planned_route: str | None = None
        self._route_counts = {
            route.value: 0 for route in AttentionRoute
        }

    @classmethod
    def support_reason(
        cls,
        *,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        attention_mode: str = "unified",
        **_: Any,
    ) -> str | None:
        if block_size % 256:
            return "block_size must be divisible by 256"
        if attention_mode != "unified":
            return (
                "attention_mode='split' requires the FlashInfer backend"
            )
        if dtype not in (torch.float16, torch.bfloat16):
            return "dtype must be float16 or bfloat16"
        selected_device = (
            torch.device(device)
            if device is not None
            else torch.device(
                "cuda",
                torch.cuda.current_device() if torch.cuda.is_available() else 0,
            )
        )
        if selected_device.type != "cuda" or not torch.cuda.is_available():
            return "a CUDA device is required"
        _load_flash_attention()
        if not FLASH_ATTENTION_AVAILABLE or _flash_attn_varlen_func is None:
            error = _FLASH_ATTENTION_IMPORT_ERROR
            return (
                "Dao FlashAttention is unavailable"
                + (f": {error}" if error is not None else "")
            )
        return None

    @property
    def planned_route(self) -> str | None:
        return self._planned_route

    @property
    def route_counts(self):
        return MappingProxyType(dict(self._route_counts))

    def build_metadata(
        self,
        common: CommonAttentionMetadata,
    ) -> FlashAttentionMetadata:
        _validate_common_metadata(common, require_cuda=True)
        if common.block_tables is None:
            return FlashAttentionMetadata(
                query_start_loc=common.query_start_loc,
                kv_start_loc=common.query_start_loc,
                block_tables=None,
                max_q_len=common.max_q_len,
                max_kv_len=common.max_kv_len,
            )

        kv_start_loc = torch.empty(
            common.num_seqs + 1,
            dtype=torch.int32,
            device=common.seq_lens.device,
        )
        kv_start_loc[0] = 0
        torch.cumsum(common.seq_lens, dim=0, out=kv_start_loc[1:])
        return FlashAttentionMetadata(
            query_start_loc=common.query_start_loc,
            kv_start_loc=kv_start_loc,
            block_tables=common.block_tables,
            max_q_len=common.max_q_len,
            max_kv_len=common.max_kv_len,
        )

    def build_plan(
        self,
        common: CommonAttentionMetadata,
        metadata: FlashAttentionMetadata | None = None,
    ) -> AttentionPlan:
        if metadata is None:
            metadata = self.build_metadata(common)
        if not isinstance(metadata, FlashAttentionMetadata):
            raise TypeError("FlashAttention requires FlashAttentionMetadata")

        batch_type = _classify_batch(common)
        if metadata.block_tables is None:
            route = AttentionRoute.WARMUP
        elif batch_type is BatchType.PURE_PREFILL:
            route = AttentionRoute.PREFILL
        elif batch_type is BatchType.PURE_DECODE:
            route = AttentionRoute.DECODE
        else:
            route = AttentionRoute.MIXED_UNIFIED

        plan = AttentionPlan(batch_type, route, common, metadata)
        self._planned_route = route.value
        self._route_counts[route.value] += 1
        return plan

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        plan: AttentionPlan,
    ) -> torch.Tensor:
        metadata = _require_plan_metadata(
            plan, FlashAttentionMetadata, self.backend_name
        )
        has_k_cache = _has_cache(k_cache)
        has_v_cache = _has_cache(v_cache)
        if has_k_cache != has_v_cache:
            raise ValueError(
                "key and value caches must both be present or absent"
            )

        route = plan.route
        if route is AttentionRoute.WARMUP:
            if has_k_cache:
                raise ValueError(
                    "FlashAttention warmup route requires an empty KV cache"
                )
            if metadata.block_tables is not None:
                raise ValueError(
                    "FlashAttention warmup route requires cacheless metadata"
                )
            return _cacheless_varlen_attention(q, k, v, plan.common)

        if route is AttentionRoute.PREFILL:
            expected_batch_type = BatchType.PURE_PREFILL
        elif route is AttentionRoute.DECODE:
            expected_batch_type = BatchType.PURE_DECODE
        elif route is AttentionRoute.MIXED_UNIFIED:
            expected_batch_type = BatchType.MIXED
        else:
            raise RuntimeError(
                f"invalid planned FlashAttention route: {route!r}"
            )
        if plan.batch_type is not expected_batch_type:
            raise ValueError(
                "FlashAttention route does not match the planned BatchType"
            )
        if not has_k_cache:
            raise ValueError(
                "paged FlashAttention route requires an allocated KV cache"
            )
        return self._forward_paged_attention(
            q,
            k_cache,
            v_cache,
            metadata,
        )

    def _forward_paged_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
        """Execute the common Dao paged-varlen implementation for one route."""
        _validate_cache_pair(k_cache, v_cache)
        if metadata.block_tables is None or metadata.kv_start_loc is None:
            raise ValueError(
                "paged FlashAttention requires backend cache metadata"
            )
        _load_flash_attention()
        if _flash_attn_varlen_func is None:
            error = _FLASH_ATTENTION_IMPORT_ERROR
            raise RuntimeError(
                "Dao FlashAttention is unavailable"
                + (f": {error}" if error is not None else "")
            )
        return _flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            max_seqlen_q=metadata.max_q_len,
            cu_seqlens_q=metadata.query_start_loc,
            max_seqlen_k=metadata.max_kv_len,
            cu_seqlens_k=metadata.kv_start_loc,
            softmax_scale=self.softmax_scale,
            causal=True,
            block_table=metadata.block_tables,
        )


class FlashInferBackend(AttentionBackend):
    """FlashInfer paged attention with phase-specialized and mixed routes."""

    backend_name = "flashinfer"

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
        if attention_mode not in ("unified", "split"):
            raise ValueError("attention_mode must be 'unified' or 'split'")
        if dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "FlashInfer attention supports only float16 and bfloat16"
            )
        _load_flashinfer_attention()
        if not FLASHINFER_ATTENTION_AVAILABLE:
            error = (
                _FLASHINFER_ATTENTION_IMPORT_ERROR
                or FLASHINFER_IMPORT_ERROR
            )
            raise RuntimeError(
                "FlashInfer attention is unavailable"
                + (f": {error}" if error is not None else "")
            )
        if not torch.cuda.is_available():
            raise RuntimeError("FlashInfer attention requires a CUDA device")

        self.attention_mode = attention_mode
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
        # FlashInfer requires this shared float workspace to be zero before its
        # first decode use.
        self.workspace.zero_()

        self.mixed_wrapper = None
        self._mixed_attention_initialization_error: Exception | None = None
        self._mixed_attention_unavailable_reason: str | None = None
        if self.supports_unified_mixed():
            try:
                self.mixed_wrapper = _BatchAttention(
                    kv_layout="NHD",
                    device=str(self.workspace.device),
                )
            except Exception as exc:
                self._mixed_attention_initialization_error = exc
        self.supports_unified_mixed()

        self._planned_route: str | None = None
        self._route_counts = {
            route.value: 0 for route in AttentionRoute
        }
        self._num_prefill_seqs = 0
        self._num_prefill_tokens = 0
        self._num_decode_seqs = 0
        self._num_decode_tokens = 0
        self._output_buffer: torch.Tensor | None = None
        self._lse_buffer: torch.Tensor | None = None
        self._active_graph_state: _FlashInferFullDecodeGraphState | None = None
        self.graph_workspace: torch.Tensor | None = None

    @classmethod
    def support_reason(
        cls,
        *,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        **_: Any,
    ) -> str | None:
        if block_size <= 0:
            return "block_size must be positive"
        if dtype not in (torch.float16, torch.bfloat16):
            return "dtype must be float16 or bfloat16"
        selected_device = (
            torch.device(device)
            if device is not None
            else torch.device(
                "cuda",
                torch.cuda.current_device() if torch.cuda.is_available() else 0,
            )
        )
        if selected_device.type != "cuda" or not torch.cuda.is_available():
            return "a CUDA device is required"
        _load_flashinfer_attention()
        if not FLASHINFER_ATTENTION_AVAILABLE:
            error = (
                _FLASHINFER_ATTENTION_IMPORT_ERROR
                or FLASHINFER_IMPORT_ERROR
            )
            return (
                "FlashInfer attention is unavailable"
                + (f": {error}" if error is not None else "")
            )
        return None

    @property
    def supports_full_decode_graph(self) -> bool:
        return self.attention_mode == "unified"

    @property
    def planned_route(self) -> str | None:
        return self._planned_route

    @property
    def route_counts(self):
        return MappingProxyType(dict(self._route_counts))

    @property
    def mixed_attention_available(self) -> bool:
        return self.mixed_wrapper is not None

    @property
    def mixed_attention_unavailable_reason(self) -> str | None:
        return self._mixed_attention_unavailable_reason

    def _unified_mixed_support_reason(self) -> str | None:
        if self.attention_mode != "unified":
            return "unified mixed attention is disabled in split mode"
        if _BatchAttention is None:
            error = _FLASHINFER_MIXED_ATTENTION_IMPORT_ERROR
            return (
                "FlashInfer BatchAttention is unavailable"
                + (f": {error}" if error is not None else "")
            )
        if self.head_dim > 256:
            return "FlashInfer BatchAttention requires head_dim <= 256"
        device_name = torch.cuda.get_device_name(self.workspace.device)
        # TODO: Replace this temporary product-name gate with a general
        # runtime capability/probe supplied by FlashInfer.
        if "RTX 5070" in device_name:
            return (
                "FlashInfer BatchAttention is temporarily disabled on "
                f"{device_name}"
            )
        if self._mixed_attention_initialization_error is not None:
            return (
                "FlashInfer BatchAttention initialization failed: "
                f"{self._mixed_attention_initialization_error}"
            )
        return None

    def supports_unified_mixed(self) -> bool:
        """Return whether the unified mixed kernel is usable on this device."""
        reason = self._unified_mixed_support_reason()
        self._mixed_attention_unavailable_reason = reason
        return reason is None

    def build_metadata(
        self,
        common: CommonAttentionMetadata,
    ) -> FlashInferMetadata:
        _validate_common_metadata(common, require_cuda=True)
        if common.block_tables is None:
            return FlashInferMetadata(
                query_start_loc=common.query_start_loc,
                kv_indptr=None,
                page_indices=None,
                last_page_len=None,
                num_pages=0,
                num_prefill_pages=0,
            )

        block_tables = common.block_tables
        max_pages_per_seq = block_tables.size(1)
        if common.trusted:
            block_counts = common.block_counts
            if not isinstance(block_counts, tuple):
                raise TypeError("trusted block_counts must be a tuple")
            if len(block_counts) != common.num_seqs:
                raise ValueError(
                    "trusted block_counts length must equal num_seqs"
                )
            if any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 < count <= max_pages_per_seq
                for count in block_counts
            ):
                raise ValueError(
                    "trusted block_counts contains an invalid row count"
                )
            offsets = [0]
            for count in block_counts:
                offsets.append(offsets[-1] + count)
            kv_indptr = torch.tensor(
                offsets,
                dtype=torch.int32,
                device=block_tables.device,
            )
            page_indices = torch.cat(
                tuple(
                    block_tables[row, :count]
                    for row, count in enumerate(block_counts)
                )
            ).contiguous()
            num_pages = common.num_kv_blocks
            num_prefill_pages = common.num_prefill_kv_blocks
            if num_pages != offsets[-1]:
                raise ValueError(
                    "trusted num_kv_blocks must match block_counts"
                )
            expected_prefill_pages = offsets[common.num_prefill_seqs]
            if num_prefill_pages != expected_prefill_pages:
                raise ValueError(
                    "trusted num_prefill_kv_blocks must match block_counts"
                )
        else:
            page_counts = torch.div(
                common.seq_lens + self.block_size - 1,
                self.block_size,
                rounding_mode="floor",
            ).to(dtype=torch.int32)
            kv_indptr = torch.empty(
                common.num_seqs + 1,
                dtype=torch.int32,
                device=common.seq_lens.device,
            )
            kv_indptr[0] = 0
            torch.cumsum(page_counts, dim=0, out=kv_indptr[1:])
            page_columns = torch.arange(
                max_pages_per_seq,
                dtype=torch.int32,
                device=block_tables.device,
            )
            valid_pages = (
                page_columns.unsqueeze(0) < page_counts.unsqueeze(1)
            )
            page_indices = block_tables[valid_pages].contiguous()
            num_pages = page_indices.numel()
            num_prefill_pages = int(
                page_counts[: common.num_prefill_seqs].sum().item()
            )
            if page_indices.numel() and not bool(
                torch.all(page_indices >= 0).item()
            ):
                raise ValueError("block_tables contains an invalid page index")

        last_page_len = (
            torch.remainder(common.seq_lens - 1, self.block_size) + 1
        ).to(dtype=torch.int32)

        return FlashInferMetadata(
            query_start_loc=common.query_start_loc,
            kv_indptr=kv_indptr,
            page_indices=page_indices,
            last_page_len=last_page_len,
            num_pages=num_pages,
            num_prefill_pages=num_prefill_pages,
        )

    def build_plan(
        self,
        common: CommonAttentionMetadata,
        metadata: FlashInferMetadata | None = None,
    ) -> AttentionPlan:
        if metadata is None:
            metadata = self.build_metadata(common)
        if not isinstance(metadata, FlashInferMetadata):
            raise TypeError("FlashInfer requires FlashInferMetadata")

        # This is the single phase classification point for the whole batch.
        batch_type = _classify_batch(common)
        if metadata.is_cacheless:
            route = AttentionRoute.WARMUP
            plan = AttentionPlan(batch_type, route, common, metadata)
            self._record_plan(plan)
            return plan

        q_indptr = metadata.query_start_loc
        kv_indptr = metadata.kv_indptr
        indices = metadata.page_indices
        last_page_len = metadata.last_page_len
        assert kv_indptr is not None
        assert indices is not None
        assert last_page_len is not None

        if common.trusted:
            if metadata.num_pages != indices.numel():
                raise ValueError(
                    "trusted FlashInfer page count does not match indices"
                )
        else:
            if int(q_indptr[0].item()) != 0:
                raise ValueError("query_start_loc must start at zero")
            if int(kv_indptr[0].item()) != 0:
                raise ValueError("FlashInfer KV indptr must start at zero")
            if int(kv_indptr[-1].item()) != indices.numel():
                raise ValueError(
                    "FlashInfer KV indptr does not match page indices"
                )

        if batch_type is BatchType.PURE_PREFILL:
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
            route = AttentionRoute.PREFILL
        elif batch_type is BatchType.PURE_DECODE:
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
            route = AttentionRoute.DECODE
        elif self.mixed_wrapper is not None:
            self.mixed_wrapper.plan(
                q_indptr,
                kv_indptr,
                indices,
                common.seq_lens,
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
            route = AttentionRoute.MIXED_UNIFIED
        else:
            prefill_page_end = metadata.num_prefill_pages
            self.prefill_wrapper.plan(
                q_indptr[: common.num_prefill_seqs + 1],
                kv_indptr[: common.num_prefill_seqs + 1],
                indices[:prefill_page_end],
                last_page_len[: common.num_prefill_seqs],
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                o_data_type=self.dtype,
            )
            decode_kv_indptr = (
                kv_indptr[common.num_prefill_seqs:] - prefill_page_end
            )
            self.decode_wrapper.plan(
                decode_kv_indptr,
                indices[prefill_page_end:],
                last_page_len[common.num_prefill_seqs:],
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                pos_encoding_mode="NONE",
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                o_data_type=self.dtype,
            )
            route = AttentionRoute.MIXED_SPLIT

        plan = AttentionPlan(batch_type, route, common, metadata)
        self._record_plan(plan)
        return plan

    def _record_plan(self, plan: AttentionPlan) -> None:
        common = plan.common
        self._num_prefill_seqs = common.num_prefill_seqs
        self._num_prefill_tokens = common.num_prefill_tokens
        self._num_decode_seqs = common.num_decode_seqs
        self._num_decode_tokens = common.num_decode_tokens
        self._planned_route = plan.route.value
        self._route_counts[plan.route.value] += 1

    def _get_reusable_attention_output(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        required_tokens = q.size(0)
        buffer = self._output_buffer
        if (
            buffer is None
            or buffer.size(0) < required_tokens
            or buffer.shape[1:] != q.shape[1:]
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
        plan: AttentionPlan,
    ) -> torch.Tensor:
        metadata = _require_plan_metadata(
            plan, FlashInferMetadata, self.backend_name
        )
        has_k_cache = _has_cache(k_cache)
        has_v_cache = _has_cache(v_cache)
        if has_k_cache != has_v_cache:
            raise ValueError(
                "key and value caches must both be present or absent"
            )

        graph_state = self._active_graph_state
        if graph_state is not None:
            if not has_k_cache:
                raise RuntimeError(
                    "full decode graph requires allocated KV cache"
                )
            _validate_flashinfer_inputs(self, q, k_cache, v_cache)
            output = graph_state.output
            if output.shape != q.shape:
                raise RuntimeError(
                    "decode graph attention output does not match query shape"
                )
            return graph_state.wrapper.run(
                q,
                (k_cache, v_cache),
                out=output,
            )

        if not has_k_cache:
            output = _cacheless_varlen_attention(q, k, v, plan.common)
            scratch = self._get_reusable_attention_output(q)
            scratch.copy_(output)
            return scratch

        _validate_flashinfer_inputs(self, q, k_cache, v_cache)
        if metadata.is_cacheless:
            raise ValueError("paged FlashInfer requires page metadata")
        if q.size(0) != plan.common.num_query_tokens:
            raise ValueError(
                "query token count does not match the attention plan"
            )

        cache = (k_cache, v_cache)
        output = self._get_reusable_attention_output(q)
        if plan.route is AttentionRoute.PREFILL:
            self.prefill_wrapper.run(q, cache, out=output)
            return output
        if plan.route is AttentionRoute.DECODE:
            self.decode_wrapper.run(q, cache, out=output)
            return output
        if plan.route is AttentionRoute.MIXED_UNIFIED:
            if self.mixed_wrapper is None:
                raise RuntimeError(
                    "planned unified mixed attention wrapper is unavailable"
                )
            result = self.mixed_wrapper.run(
                q,
                cache,
                out=output,
                lse=self._get_reusable_lse(q),
            )
            return result[0] if isinstance(result, tuple) else result
        if plan.route is not AttentionRoute.MIXED_SPLIT:
            raise RuntimeError("invalid planned FlashInfer attention route")

        split = plan.common.num_prefill_tokens
        if plan.common.num_prefill_seqs:
            self.prefill_wrapper.run(
                q[:split],
                cache,
                out=output[:split],
            )
        if plan.common.num_decode_seqs:
            self.decode_wrapper.run(
                q[split:],
                cache,
                out=output[split:],
            )
        return output

    def create_full_decode_graph_state(
        self,
        batch_size: int,
        page_indices_capacity: int,
        device: torch.device | str,
    ) -> _FlashInferFullDecodeGraphState:
        if not self.supports_full_decode_graph:
            raise RuntimeError(
                "full decode CUDA Graph requires unified FlashInfer attention"
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an int")
        if (
            isinstance(page_indices_capacity, bool)
            or not isinstance(page_indices_capacity, int)
        ):
            raise TypeError("page_indices_capacity must be an int")
        if batch_size <= 0 or page_indices_capacity < batch_size:
            raise ValueError(
                "graph state requires positive batch and page capacity"
            )
        graph_device = torch.device(device)
        if graph_device.type != "cuda":
            raise ValueError("full decode graph state must be on CUDA")

        if self.graph_workspace is None:
            self.graph_workspace = torch.zeros(
                FLASHINFER_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=graph_device,
            )
        query_start_loc_buffer = torch.empty(
            batch_size + 1, dtype=torch.int32, device=graph_device
        )
        kv_indptr_buffer = torch.empty_like(query_start_loc_buffer)
        page_indices_buffer = torch.empty(
            page_indices_capacity,
            dtype=torch.int32,
            device=graph_device,
        )
        last_page_len_buffer = torch.empty(
            batch_size, dtype=torch.int32, device=graph_device
        )
        wrapper = _BatchDecodeWithPagedKVCacheWrapper(
            self.graph_workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=kv_indptr_buffer,
            paged_kv_indices_buffer=page_indices_buffer,
            paged_kv_last_page_len_buffer=last_page_len_buffer,
            backend="auto",
        )
        output = torch.empty(
            batch_size,
            self.num_q_heads,
            self.head_dim,
            dtype=self.dtype,
            device=graph_device,
        )
        return _FlashInferFullDecodeGraphState(
            batch_size=batch_size,
            page_indices_capacity=page_indices_capacity,
            wrapper=wrapper,
            query_start_loc_buffer=query_start_loc_buffer,
            kv_indptr_buffer=kv_indptr_buffer,
            page_indices_buffer=page_indices_buffer,
            last_page_len_buffer=last_page_len_buffer,
            output=output,
        )

    def full_decode_graph_metadata_fits(
        self,
        state: Any,
        plan: AttentionPlan,
    ) -> bool:
        if not isinstance(state, _FlashInferFullDecodeGraphState):
            return False
        if not isinstance(plan, AttentionPlan):
            return False
        if (
            plan.batch_type is not BatchType.PURE_DECODE
            or plan.route is not AttentionRoute.DECODE
        ):
            return False
        if not isinstance(plan.metadata, FlashInferMetadata):
            return False
        common = plan.common
        metadata = plan.metadata
        tensors = (
            metadata.query_start_loc,
            metadata.kv_indptr,
            metadata.page_indices,
            metadata.last_page_len,
        )
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            return False
        q_indptr, kv_indptr, indices, last_page_len = tensors
        assert isinstance(q_indptr, torch.Tensor)
        assert isinstance(kv_indptr, torch.Tensor)
        assert isinstance(indices, torch.Tensor)
        assert isinstance(last_page_len, torch.Tensor)
        if common.num_prefill_seqs != 0 or common.num_prefill_tokens != 0:
            return False
        if common.num_decode_tokens != state.batch_size:
            return False
        if common.num_decode_seqs != state.batch_size:
            return False
        if q_indptr.numel() != state.batch_size + 1:
            return False
        if kv_indptr.numel() != state.batch_size + 1:
            return False
        if last_page_len.numel() != state.batch_size:
            return False
        if indices.numel() > state.page_indices_capacity:
            return False
        if any(
            tensor.dtype != torch.int32
            or tensor.device.type != "cuda"
            or tensor.ndim != 1
            for tensor in tensors
        ):
            return False
        if common.trusted:
            if metadata.num_pages != indices.numel():
                return False
            if metadata.num_prefill_pages != 0:
                return False
        else:
            if int(q_indptr[0].item()) != 0:
                return False
            if int(q_indptr[-1].item()) != state.batch_size:
                return False
            if int(kv_indptr[0].item()) != 0:
                return False
            if int(kv_indptr[-1].item()) != indices.numel():
                return False
            if not bool(
                torch.all(q_indptr[1:] - q_indptr[:-1] == 1).item()
            ):
                return False
            valid_last_page = (last_page_len > 0) & (
                last_page_len <= self.block_size
            )
            if not bool(torch.all(valid_last_page).item()):
                return False
        slot_mapping = common.slot_mapping
        return (
            slot_mapping.dtype == torch.int32
            and slot_mapping.device.type == "cuda"
            and slot_mapping.ndim == 1
            and slot_mapping.numel() == state.batch_size
        )

    def plan_full_decode_graph(
        self,
        state: Any,
        plan: AttentionPlan,
    ) -> None:
        if not self.full_decode_graph_metadata_fits(state, plan):
            raise ValueError(
                "attention plan does not fit full decode graph state"
            )
        assert isinstance(state, _FlashInferFullDecodeGraphState)
        metadata = plan.metadata
        assert isinstance(metadata, FlashInferMetadata)
        assert metadata.kv_indptr is not None
        assert metadata.page_indices is not None
        assert metadata.last_page_len is not None

        state.query_start_loc_buffer.copy_(metadata.query_start_loc)
        state.wrapper.plan(
            metadata.kv_indptr,
            metadata.page_indices,
            metadata.last_page_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            o_data_type=self.dtype,
        )
        state.plan = plan

    def activate_full_decode_graph(self, state: Any) -> None:
        if not isinstance(state, _FlashInferFullDecodeGraphState):
            raise TypeError("invalid FlashInfer decode graph state")
        if state.plan is None:
            raise RuntimeError(
                "decode graph state must be planned before activation"
            )
        if self._active_graph_state is not None:
            raise RuntimeError("a decode graph state is already active")
        self._active_graph_state = state

    def deactivate_full_decode_graph(self) -> None:
        self._active_graph_state = None


class AttentionBackendRegistry:
    """Static registry containing complete attention backend implementations."""

    _BACKENDS = {
        "flashattention": FlashAttentionBackend,
        "flashinfer": FlashInferBackend,
    }
    _ALIASES = {
        "legacy": "flashattention",
        "flash_attn": "flashattention",
    }

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        return tuple(cls._BACKENDS)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return cls.registered_names()

    @classmethod
    def register(
        cls,
        name: str,
        backend_type: type[AttentionBackend],
    ) -> None:
        """Register one complete backend without changing serving hot paths."""
        if not isinstance(name, str):
            raise TypeError("attention backend name must be a string")
        normalized = name.strip().lower()
        if not normalized or normalized == "auto":
            raise ValueError(
                "attention backend name must be non-empty and not 'auto'"
            )
        if normalized in cls._BACKENDS or normalized in cls._ALIASES:
            raise ValueError(
                f"attention backend name {normalized!r} is already registered"
            )
        if (
            not isinstance(backend_type, type)
            or not issubclass(backend_type, AttentionBackend)
        ):
            raise TypeError(
                "backend_type must be an AttentionBackend subclass"
            )
        cls._BACKENDS[normalized] = backend_type

    @staticmethod
    def _support_status(
        backend_type: type[AttentionBackend],
        capabilities: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Ask the public capability contract; fetch a reason only on failure."""
        supported = backend_type.supports(**capabilities)
        if type(supported) is not bool:
            raise TypeError(
                f"{backend_type.__name__}.supports() must return a bool"
            )
        if supported:
            return True, None

        reason = backend_type.support_reason(**capabilities)
        if reason is not None and not isinstance(reason, str):
            raise TypeError(
                f"{backend_type.__name__}.support_reason() must return "
                "a string or None"
            )
        return False, reason or "supports() returned False"

    @classmethod
    def create(
        cls,
        name: str = "auto",
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
        attention_mode: str = "unified",
        device: torch.device | str | None = None,
    ) -> AttentionBackend:
        if not isinstance(name, str):
            raise TypeError("attention backend name must be a string")
        requested = name.strip().lower()
        canonical = cls._ALIASES.get(requested, requested)
        capabilities = {
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "dtype": dtype,
            "device": device,
            "attention_mode": attention_mode,
        }
        init_kwargs = {
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "dtype": dtype,
        }

        if canonical == "auto":
            if block_size % 256 == 0:
                flashattention_supported, flashattention_reason = (
                    cls._support_status(
                        FlashAttentionBackend,
                        capabilities,
                    )
                )
                if flashattention_supported:
                    return FlashAttentionBackend(**init_kwargs)
            else:
                flashattention_reason = (
                    FlashAttentionBackend.support_reason(**capabilities)
                    or "block_size policy did not select FlashAttention"
                )

            flashinfer_supported, flashinfer_reason = cls._support_status(
                FlashInferBackend,
                capabilities,
            )
            if flashinfer_supported:
                return FlashInferBackend(
                    **init_kwargs,
                    attention_mode=attention_mode,
                )
            raise RuntimeError(
                "No usable attention backend. "
                f"flashattention: {flashattention_reason}; "
                f"flashinfer: {flashinfer_reason}"
            )

        backend_type = cls._BACKENDS.get(canonical)
        if backend_type is None:
            supported = ", ".join(("auto", *cls.registered_names()))
            raise ValueError(
                f"unknown attention backend {name!r}; choose from {supported}"
            )
        backend_supported, reason = cls._support_status(
            backend_type,
            capabilities,
        )
        if not backend_supported:
            raise RuntimeError(
                f"{canonical} attention backend is unusable: {reason}"
            )
        if backend_type is FlashInferBackend:
            return backend_type(
                **init_kwargs,
                attention_mode=attention_mode,
            )
        return backend_type(**init_kwargs)


def _validate_common_metadata(
    common: CommonAttentionMetadata,
    *,
    require_cuda: bool,
) -> None:
    if not isinstance(common, CommonAttentionMetadata):
        raise TypeError("common must be CommonAttentionMetadata")
    int_fields = (
        "num_prefill_seqs",
        "num_decode_seqs",
        "num_prefill_tokens",
        "num_decode_tokens",
        "max_q_len",
        "max_kv_len",
        "num_kv_blocks",
        "num_prefill_kv_blocks",
    )
    for name in int_fields:
        value = getattr(common, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"common.{name} must be an int")
        if value < 0:
            raise ValueError(f"common.{name} must be non-negative")
    if common.num_seqs <= 0:
        raise ValueError("attention metadata must contain at least one sequence")
    if common.num_decode_tokens != common.num_decode_seqs:
        raise ValueError(
            "decode suffix must contain one query token per sequence"
        )
    if not isinstance(common.trusted, bool):
        raise TypeError("common.trusted must be a bool")

    tensor_fields = (
        "query_start_loc",
        "seq_lens",
        "slot_mapping",
    )
    for name in tensor_fields:
        tensor = getattr(common, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"common.{name} must be a torch.Tensor")
        if tensor.dtype != torch.int32:
            raise TypeError(f"common.{name} must have dtype torch.int32")
        if tensor.ndim != 1:
            raise ValueError(f"common.{name} must be one-dimensional")
        if require_cuda and tensor.device.type != "cuda":
            raise ValueError(f"common.{name} must be on CUDA")

    if common.query_start_loc.numel() != common.num_seqs + 1:
        raise ValueError(
            "query_start_loc length must equal number of sequences + 1"
        )
    if common.seq_lens.numel() != common.num_seqs:
        raise ValueError("seq_lens length must equal number of sequences")
    if common.slot_mapping.numel() != common.num_query_tokens:
        raise ValueError(
            "slot_mapping length must equal number of query tokens"
        )
    if not common.trusted:
        if int(common.query_start_loc[0].item()) != 0:
            raise ValueError("query_start_loc must start at zero")
        if (
            int(common.query_start_loc[-1].item())
            != common.num_query_tokens
        ):
            raise ValueError(
                "query_start_loc must end at num_query_tokens"
            )
        query_lengths = (
            common.query_start_loc[1:] - common.query_start_loc[:-1]
        )
        if common.num_decode_seqs and not bool(
            torch.all(
                query_lengths[common.num_prefill_seqs:] == 1
            ).item()
        ):
            raise ValueError(
                "every decode sequence must have one query token"
            )
        if not bool(torch.all(common.seq_lens > 0).item()):
            raise ValueError("all KV sequence lengths must be positive")

    block_tables = common.block_tables
    if block_tables is not None:
        if not isinstance(block_tables, torch.Tensor):
            raise TypeError("common.block_tables must be a torch.Tensor")
        if block_tables.dtype != torch.int32:
            raise TypeError(
                "common.block_tables must have dtype torch.int32"
            )
        if block_tables.ndim != 2:
            raise ValueError("common.block_tables must be two-dimensional")
        if block_tables.size(0) != common.num_seqs:
            raise ValueError(
                "block_tables row count must equal number of sequences"
            )
        if require_cuda and block_tables.device.type != "cuda":
            raise ValueError("common.block_tables must be on CUDA")


def _classify_batch(common: CommonAttentionMetadata) -> BatchType:
    if common.num_prefill_seqs == 0:
        if common.num_decode_seqs <= 0:
            raise ValueError("pure decode batch must contain decode requests")
        return BatchType.PURE_DECODE
    if common.num_decode_seqs == 0:
        return BatchType.PURE_PREFILL
    return BatchType.MIXED


def _require_plan_metadata(
    plan: AttentionPlan,
    metadata_type: type,
    backend_name: str,
):
    if not isinstance(plan, AttentionPlan):
        raise TypeError("attention forward requires AttentionPlan")
    if not isinstance(plan.metadata, metadata_type):
        raise TypeError(
            f"{backend_name} plan carries incompatible backend metadata"
        )
    return plan.metadata


def _has_cache(cache: torch.Tensor | None) -> bool:
    return isinstance(cache, torch.Tensor) and cache.numel() > 0


def _validate_cache_pair(
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
) -> None:
    if not _has_cache(k_cache) or not _has_cache(v_cache):
        raise ValueError(
            "paged attention requires both key and value caches"
        )


def _validate_flashinfer_inputs(
    backend: FlashInferBackend,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    if q.device.type != "cuda" or q.dtype != backend.dtype:
        raise ValueError(
            "query must be a CUDA tensor with the backend dtype"
        )
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


def _cacheless_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    common: CommonAttentionMetadata,
) -> torch.Tensor:
    """Prefer Dao varlen attention, with a dependency-free warmup fallback."""
    _load_flash_attention()
    if _flash_attn_varlen_func is not None:
        return _flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=common.max_q_len,
            cu_seqlens_q=common.query_start_loc,
            max_seqlen_k=common.max_kv_len,
            cu_seqlens_k=common.query_start_loc,
            softmax_scale=1.0 / sqrt(q.size(-1)),
            causal=True,
            block_table=None,
        )
    return _torch_sdpa_varlen(q, k, v, common)


def _torch_sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    common: CommonAttentionMetadata,
) -> torch.Tensor:
    """Cacheless warmup reference that has no optional backend dependency."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("cacheless attention expects rank-3 Q/K/V")
    if k.shape != v.shape:
        raise ValueError("cacheless key and value shapes must match")
    if q.size(0) != k.size(0):
        raise ValueError(
            "cacheless warmup requires matching query and KV token counts"
        )
    if q.size(1) % k.size(1):
        raise ValueError(
            "query head count must be divisible by KV head count"
        )

    outputs = []
    starts = common.query_start_loc
    if common.trusted:
        # Warmup metadata is constructed on the host; this one-time path may
        # read its compact query offsets without affecting serving hot paths.
        boundaries = starts.cpu().tolist()
    else:
        boundaries = starts.tolist()
    kv_repeat = q.size(1) // k.size(1)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
        k_seq = k[start:end].transpose(0, 1).unsqueeze(0)
        v_seq = v[start:end].transpose(0, 1).unsqueeze(0)
        if kv_repeat != 1:
            k_seq = k_seq.repeat_interleave(kv_repeat, dim=1)
            v_seq = v_seq.repeat_interleave(kv_repeat, dim=1)
        output = F.scaled_dot_product_attention(
            q_seq,
            k_seq,
            v_seq,
            is_causal=True,
            scale=1.0 / sqrt(q.size(-1)),
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)


# Compatibility aliases for existing configuration and external imports.
LegacyFlashAttentionBackend = FlashAttentionBackend
FlashInferAttentionBackend = FlashInferBackend


__all__ = [
    "AttentionBackend",
    "AttentionBackendRegistry",
    "AttentionPlan",
    "AttentionRoute",
    "FlashAttentionBackend",
    "FlashAttentionMetadata",
    "FlashInferBackend",
    "FlashInferMetadata",
    "LegacyFlashAttentionBackend",
    "FlashInferAttentionBackend",
    "FLASH_ATTENTION_AVAILABLE",
    "FLASHINFER_ATTENTION_AVAILABLE",
    "FLASHINFER_IMPORT_ERROR",
    "FLASHINFER_WORKSPACE_BYTES",
]
