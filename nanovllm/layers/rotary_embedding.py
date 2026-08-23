from functools import lru_cache
import torch

from nanovllm.layers.custom_op import CustomOp, CustomOpConfig


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(CustomOp):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        custom_op_config: CustomOpConfig | None = None,
    ) -> None:
        super().__init__(custom_op_config)
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.provider_name, self.forward_impl = self._select_cuda_impl()

    @torch.compile
    def forward_native(
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

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_impl(positions, query, key)

    def _select_cuda_impl(self):
        implementation = self.custom_op_config.implementation_for(
            "rotary_embedding"
        )
        supported = {"auto", "native", "native_torch", "flashinfer"}
        if implementation not in supported:
            available = ", ".join(sorted(supported))
            raise ValueError(
                "unsupported 'rotary_embedding' implementation "
                f"{implementation!r}; available: {available}"
            )
        if (
            self.platform != "cuda"
            and implementation not in ("auto", "native", "native_torch")
        ):
            raise RuntimeError(
                f"{implementation!r} for 'rotary_embedding' requires CUDA"
            )
        if implementation in ("native", "native_torch"):
            return "native_torch", self.forward_native
        if self.platform != "cuda":
            return "native_torch", self.forward_native

        from nanovllm.layers import flashinfer_ops

        dtype = self.custom_op_config.dtype
        flashinfer_available = (
            flashinfer_ops.FLASHINFER_AVAILABLE
            and dtype in (torch.float16, torch.bfloat16, None)
        )
        if implementation == "flashinfer" and not flashinfer_available:
            if dtype not in (torch.float16, torch.bfloat16, None):
                raise RuntimeError(
                    "'flashinfer' for 'rotary_embedding' does not "
                    f"support dtype {dtype}"
                )
        if implementation == "auto" and not flashinfer_available:
            return "native_torch", self.forward_native

        operation = flashinfer_ops.get_flashinfer_rotary_embedding()

        def forward(positions, query, key):
            return operation(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache.squeeze(1),
                is_neox=True,
            )

        return "flashinfer", forward


@lru_cache(1)
def _get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    custom_op_config: CustomOpConfig,
):
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        custom_op_config=custom_op_config,
    )
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float | None,
    rope_scaling: dict | None = None,
    custom_op_config: CustomOpConfig | None = None,
):
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type != "default":
            raise NotImplementedError(f"RoPE type {rope_type!r} is not supported")
        base = rope_scaling.get("rope_theta", base)
    if base is None:
        raise ValueError("RoPE theta is missing from the model configuration")
    dispatch = custom_op_config or CustomOpConfig()
    return _get_rope(
        head_size, rotary_dim, max_position, float(base), dispatch
    )
