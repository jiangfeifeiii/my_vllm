import torch
import torch.nn.functional as F

from nanovllm.layers.custom_op import CustomOp, CustomOpConfig


class SiluAndMul(CustomOp):
    """SwiGLU activation with construction-time platform dispatch."""

    op_name = "silu_and_mul"
    _CUDA_IMPLEMENTATIONS = frozenset(
        {
            "auto",
            "native",
            "native_torch",
            "flashinfer",
            "custom_cuda",
            "adaptive_cuda",
        }
    )

    def __init__(
        self,
        custom_op_config: CustomOpConfig | None = None,
    ) -> None:
        super().__init__(custom_op_config)
        implementation = self.require_cuda_implementation(
            self._CUDA_IMPLEMENTATIONS
        )
        self.provider_name, self.forward_impl = self._select_cuda_impl(
            implementation
        )

    @torch.compile
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(x)

    def _select_cuda_impl(self, implementation: str):
        if implementation in ("native", "native_torch"):
            return "native_torch", self.forward_native
        if self.platform != "cuda":
            return "native_torch", self.forward_native

        from nanovllm.layers import cuda_ops, flashinfer_ops

        dtype = self.custom_op_config.dtype
        dtype_supported = dtype in (
            torch.float16,
            torch.bfloat16,
            None,
        )
        if implementation != "auto" and not dtype_supported:
            raise RuntimeError(
                f"{implementation!r} for 'silu_and_mul' does not support "
                f"dtype {dtype}"
            )
        custom_available = cuda_ops.CUSTOM_CUDA_AVAILABLE and dtype_supported
        flashinfer_available = (
            flashinfer_ops.FLASHINFER_AVAILABLE and dtype_supported
        )

        if implementation == "adaptive_cuda":
            return "adaptive_cuda", cuda_ops.get_adaptive_silu_and_mul()
        if implementation == "custom_cuda":
            return "custom_cuda", cuda_ops.get_custom_silu_and_mul()
        if implementation == "flashinfer":
            return (
                "flashinfer",
                flashinfer_ops.get_flashinfer_silu_and_mul(),
            )
        if custom_available and flashinfer_available:
            return "adaptive_cuda", cuda_ops.get_adaptive_silu_and_mul()
        if custom_available:
            return "custom_cuda", cuda_ops.get_custom_silu_and_mul()
        if flashinfer_available:
            return (
                "flashinfer",
                flashinfer_ops.get_flashinfer_silu_and_mul(),
            )
        return "native_torch", self.forward_native
