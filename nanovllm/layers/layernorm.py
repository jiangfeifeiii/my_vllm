import torch
from torch import nn

from nanovllm.layers.operators import OperatorResolver, register_operator


@register_operator("rms_norm", "native_torch", priority=100)
def _bind_native_rms_norm(layer):
    return layer.rms_forward


@register_operator("fused_add_rms_norm", "native_torch", priority=100)
def _bind_native_fused_add_rms_norm(layer):
    return layer.add_rms_forward


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        operator_resolver: OperatorResolver | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        resolver = operator_resolver or OperatorResolver()
        self.rms_provider_name, self.rms_impl = resolver.bind(
            "rms_norm", self, hidden_size=hidden_size
        )
        self.add_rms_provider_name, self.add_rms_impl = resolver.bind(
            "fused_add_rms_norm", self, hidden_size=hidden_size
        )
        self.add_rms_inplace_impl = getattr(
            self.add_rms_impl,
            "_nanovllm_inplace",
            self.add_rms_impl,
        )

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_impl(x)
        else:
            return self.add_rms_impl(x, residual)

    def forward_inplace(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the provider fast path when both input values are dead."""
        return self.add_rms_inplace_impl(x, residual)
