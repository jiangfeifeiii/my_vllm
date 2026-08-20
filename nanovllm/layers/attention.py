import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.layers.attention_backend import (
    AttentionBackend,
    LegacyFlashAttentionBackend,
)
from nanovllm.layers.operators import OperatorResolver, register_operator
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


@register_operator("kv_cache_store", "native_triton", priority=300)
def _bind_native_kv_cache_store(_):
    return store_kvcache


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        attention_backend: AttentionBackend | None = None,
        operator_resolver: OperatorResolver | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.backend = attention_backend or LegacyFlashAttentionBackend(
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=256,
            dtype=torch.get_default_dtype(),
        )
        resolver = operator_resolver or OperatorResolver()
        self.kv_store_provider_name, self.kv_store_impl = resolver.bind(
            "kv_cache_store", self, layout="NHD", head_dim=head_dim
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.kv_store_impl(k, v, k_cache, v_cache, context.slot_mapping)

        return self.backend.forward(q, k, v, k_cache, v_cache, context)
