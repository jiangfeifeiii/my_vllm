import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.operators import OperatorResolver, register_operator


@register_operator("silu_and_mul", "native_torch", priority=100)
def _bind_native_silu_and_mul(layer):
    return layer.native_forward


class SiluAndMul(nn.Module):

    def __init__(self, operator_resolver: OperatorResolver | None = None):
        super().__init__()
        resolver = operator_resolver or OperatorResolver()
        self.provider_name, self.forward_impl = resolver.bind(
            "silu_and_mul", self
        )

    @torch.compile
    def native_forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(x)
