from functools import lru_cache
import torch
from torch import nn

from nanovllm.layers.operators import OperatorResolver, register_operator


@register_operator("rotary_embedding", "native_torch", priority=100)
def _bind_native_rotary_embedding(layer):
    return layer.native_forward


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        operator_resolver: OperatorResolver | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        resolver = operator_resolver or OperatorResolver()
        self.provider_name, self.forward_impl = resolver.bind(
            "rotary_embedding", self, head_size=head_size
        )

    @torch.compile
    def native_forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_impl(positions, query, key)


@lru_cache(1)
def _get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    operator_resolver: OperatorResolver,
):
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        operator_resolver=operator_resolver,
    )
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float | None,
    rope_scaling: dict | None = None,
    operator_resolver: OperatorResolver | None = None,
):
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type != "default":
            raise NotImplementedError(f"RoPE type {rope_type!r} is not supported")
        base = rope_scaling.get("rope_theta", base)
    if base is None:
        raise ValueError("RoPE theta is missing from the model configuration")
    resolver = operator_resolver or OperatorResolver()
    return _get_rope(head_size, rotary_dim, max_position, float(base), resolver)
