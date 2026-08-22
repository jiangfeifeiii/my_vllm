import torch

from nanovllm.layers.flashinfer_ops import (
    FLASHINFER_AVAILABLE,
    get_flashinfer_silu_and_mul,
)
from nanovllm.layers.operators import register_operator


# Targeted RTX 5070 / BF16 / Qwen3-width-6144 calibration found a material
# custom-kernel advantage for decode-sized batches and a FlashInfer advantage
# as token parallelism grows. The boundary is intentionally device/shape
# scoped; do not extrapolate it to unmeasured GPUs, dtypes, or feature widths.
ADAPTIVE_SILU_FLASHINFER_MIN_TOKENS = 128
ADAPTIVE_SILU_BENCHMARK_WIDTH = 6144
ADAPTIVE_SILU_BENCHMARK_DEVICE = "NVIDIA GeForce RTX 5070"

try:
    from nanovllm import _C
except ImportError as error:
    raise ImportError(
        "nanovllm._C is not built; install with "
        "`pip install -e . --no-build-isolation`"
    ) from error


def _supports_cuda(*, device_type=None, dtype=None, **_):
    return device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)


def _supports_adaptive_cuda(**capabilities):
    return FLASHINFER_AVAILABLE and _supports_cuda(**capabilities)


@register_operator(
    "silu_and_mul", "custom_cuda", supports=_supports_cuda, priority=400
)
def _bind_custom_silu_and_mul(_layer):
    return _C.forward


@register_operator(
    "silu_and_mul",
    "adaptive_cuda",
    supports=_supports_adaptive_cuda,
    priority=500,
)
def _bind_adaptive_silu_and_mul(_layer):
    custom_operation = _C.forward
    flashinfer_operation = get_flashinfer_silu_and_mul()
    use_benchmarked_crossover = (
        torch.cuda.get_device_name() == ADAPTIVE_SILU_BENCHMARK_DEVICE
    )

    def forward(x: torch.Tensor) -> torch.Tensor:
        if x.ndim:
            width = x.shape[-1]
            flashinfer_compatible = (
                width > 0
                and width % 2 == 0
                and width * x.element_size() % 16 == 0
            )
            if flashinfer_compatible:
                rows = x.numel() // width
                use_measured_custom_path = (
                    use_benchmarked_crossover
                    and x.dtype == torch.bfloat16
                    and width == ADAPTIVE_SILU_BENCHMARK_WIDTH
                    and rows < ADAPTIVE_SILU_FLASHINFER_MIN_TOKENS
                )
                if not use_measured_custom_path:
                    return flashinfer_operation(x)
        return custom_operation(x)

    return forward
