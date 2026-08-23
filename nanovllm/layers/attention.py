import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.layers.attention_backend import AttentionBackend
from nanovllm.layers.custom_op import CustomOp, CustomOpConfig
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class KVCacheStore(CustomOp):
    """Write K/V tensors into the NHD paged cache."""

    op_name = "kv_cache_store"

    def __init__(
        self,
        custom_op_config: CustomOpConfig | None = None,
    ) -> None:
        super().__init__(custom_op_config)
        implementation = self.requested_implementation
        supported = {"auto", "native", "native_torch", "native_triton"}
        if implementation not in supported:
            available = ", ".join(sorted(supported))
            raise ValueError(
                "unsupported 'kv_cache_store' implementation "
                f"{implementation!r}; available: {available}"
            )
        if self.platform == "cuda" and implementation != "native_torch":
            self.kv_store_provider_name = "native_triton"
            self.forward_impl = store_kvcache
        else:
            self.kv_store_provider_name = "native_torch"
            self.forward_impl = self.forward_native

    def forward_native(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if slot_mapping.numel() != key.shape[0]:
            raise ValueError("slot_mapping must contain one slot per token")
        valid = slot_mapping >= 0
        num_heads, head_dim = key.shape[-2:]
        flat_k_cache = k_cache.view(-1, num_heads, head_dim)
        flat_v_cache = v_cache.view(-1, num_heads, head_dim)
        slots = slot_mapping[valid].to(dtype=torch.long)
        flat_k_cache.index_copy_(0, slots, key[valid])
        flat_v_cache.index_copy_(0, slots, value[valid])

    def forward_cuda(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self.forward_impl(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
        )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        attention_backend: AttentionBackend | None = None,
        custom_op_config: CustomOpConfig | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        if attention_backend is None:
            raise ValueError(
                "Attention requires a selected AttentionBackend instance"
            )
        self.backend = attention_backend
        self.kv_store = KVCacheStore(
            custom_op_config=custom_op_config,
        )
        self.kv_store_provider_name = self.kv_store.kv_store_provider_name

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        metadata = context.attention_metadata
        plan = context.attention_plan
        if metadata is None or plan is None:
            raise RuntimeError(
                "attention metadata and plan must be prepared before forward"
            )
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.kv_store(k, v, k_cache, v_cache, metadata.slot_mapping)

        return self.backend.forward(q, k, v, k_cache, v_cache, plan)
