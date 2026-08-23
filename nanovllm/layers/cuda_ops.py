import torch

from nanovllm.layers.flashinfer_ops import (
    FLASHINFER_AVAILABLE,
    get_flashinfer_silu_and_mul,
)


# Targeted RTX 5070 / BF16 / Qwen3-width-6144 calibration found a material
# custom-kernel advantage for decode-sized batches and a FlashInfer advantage
# as token parallelism grows. The boundary is intentionally device/shape
# scoped; do not extrapolate it to unmeasured GPUs, dtypes, or feature widths.
ADAPTIVE_SILU_FLASHINFER_MIN_TOKENS = 128
ADAPTIVE_SILU_BENCHMARK_WIDTH = 6144
ADAPTIVE_SILU_BENCHMARK_DEVICE = "NVIDIA GeForce RTX 5070"

try:
    from nanovllm import _C
except ImportError as error:  # The custom extension is optional.
    _C = None
    CUSTOM_CUDA_IMPORT_ERROR: Exception | None = error
else:
    CUSTOM_CUDA_IMPORT_ERROR = None


CUSTOM_CUDA_AVAILABLE = CUSTOM_CUDA_IMPORT_ERROR is None


def get_custom_silu_and_mul():
    if _C is None:
        raise RuntimeError(
            "nanovllm._C is unavailable; build the editable package first"
        ) from CUSTOM_CUDA_IMPORT_ERROR
    return _C.forward


def get_adaptive_silu_and_mul():
    if not FLASHINFER_AVAILABLE:
        raise RuntimeError("adaptive_cuda requires FlashInfer")
    custom_operation = get_custom_silu_and_mul()
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
