import torch
from torch import nn

from nanovllm.layers.custom_op import CustomOp, CustomOpConfig


class RMSNorm(CustomOp):
    """RMSNorm and fused residual-add RMSNorm ordinary operator."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        custom_op_config: CustomOpConfig | None = None,
    ) -> None:
        super().__init__(custom_op_config)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.rms_provider_name, self.rms_impl, _ = self._select_impl(
            "rms_norm"
        )
        (
            self.add_rms_provider_name,
            self.add_rms_impl,
            self.add_rms_inplace_impl,
        ) = self._select_impl(
            "fused_add_rms_norm"
        )

    @torch.compile
    def _native_rms(
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
    def _native_add_rms(
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

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._native_rms(x)
        return self._native_add_rms(x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_impl(x)
        return self.add_rms_impl(x, residual)

    def forward_inplace(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the selected fast path when both incoming tensor values are dead."""
        return self.add_rms_inplace_impl(x, residual)

    def _select_impl(self, operator: str):
        implementation = self.custom_op_config.implementation_for(operator)
        supported = {
            "auto",
            "native",
            "native_torch",
            "flashinfer",
        }
        if implementation not in supported:
            available = ", ".join(sorted(supported))
            raise ValueError(
                f"unsupported {operator!r} implementation "
                f"{implementation!r}; available: {available}"
            )
        if (
            self.platform != "cuda"
            and implementation not in ("auto", "native", "native_torch")
        ):
            raise RuntimeError(f"{implementation!r} for {operator!r} requires CUDA")

        native = (
            self._native_rms
            if operator == "rms_norm"
            else self._native_add_rms
        )
        if implementation in ("native", "native_torch"):
            return "native_torch", native, native
        if self.platform != "cuda":
            return "native_torch", native, native

        from nanovllm.layers import flashinfer_ops

        dtype = self.custom_op_config.dtype
        flashinfer_available = (
            flashinfer_ops.FLASHINFER_AVAILABLE
            and dtype in (torch.float16, torch.bfloat16, None)
        )
        if implementation == "flashinfer" and not flashinfer_available:
            if dtype not in (torch.float16, torch.bfloat16, None):
                raise RuntimeError(
                    f"'flashinfer' for {operator!r} does not support "
                    f"dtype {dtype}"
                )
        if implementation == "auto" and not flashinfer_available:
            return "native_torch", native, native

        if operator == "rms_norm":
            operation = flashinfer_ops.get_flashinfer_rms_norm()

            def forward(x: torch.Tensor) -> torch.Tensor:
                return operation(x, self.weight, eps=self.eps)

            return "flashinfer", forward, forward

        operation = flashinfer_ops.get_flashinfer_fused_add_rms_norm()

        def forward(
            x: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            normalized = x.clone()
            new_residual = residual.clone()
            operation(normalized, new_residual, self.weight, eps=self.eps)
            return normalized, new_residual

        def forward_inplace(
            x: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            operation(x, residual, self.weight, eps=self.eps)
            return x, residual

        return "flashinfer", forward, forward_inplace
