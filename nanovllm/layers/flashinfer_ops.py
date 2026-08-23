from collections.abc import Callable
from importlib.util import find_spec
import os

import torch


def _prepare_flashinfer_environment() -> Exception | None:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return None

    cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    capability = torch.cuda.get_device_capability()
    if capability[0] < 12 or cuda_version >= (12, 9):
        return None
    if find_spec("flashinfer_jit_cache") is None:
        return RuntimeError(
            "FlashInfer on SM 12.x with CUDA < 12.9 requires "
            "flashinfer-jit-cache==0.6.17+cu129"
        )

    os.environ.setdefault(
        "FLASHINFER_CUDA_ARCH_LIST",
        f"{capability[0]}.{capability[1]}f",
    )
    os.environ.setdefault("FLASHINFER_DISABLE_JIT", "1")
    return None


FLASHINFER_IMPORT_ERROR: Exception | None = _prepare_flashinfer_environment()

if FLASHINFER_IMPORT_ERROR is None:
    try:
        from flashinfer.activation import silu_and_mul as _flashinfer_silu_and_mul
        from flashinfer.norm import fused_add_rmsnorm as _flashinfer_fused_add_rmsnorm
        from flashinfer.norm import rmsnorm as _flashinfer_rmsnorm
        from flashinfer.rope import (
            apply_rope_with_cos_sin_cache as _flashinfer_apply_rope,
        )
    except Exception as exc:  # FlashInfer is an optional CUDA dependency.
        FLASHINFER_IMPORT_ERROR = exc
        _flashinfer_silu_and_mul = None
        _flashinfer_fused_add_rmsnorm = None
        _flashinfer_rmsnorm = None
        _flashinfer_apply_rope = None
else:
    _flashinfer_silu_and_mul = None
    _flashinfer_fused_add_rmsnorm = None
    _flashinfer_rmsnorm = None
    _flashinfer_apply_rope = None


FLASHINFER_AVAILABLE = FLASHINFER_IMPORT_ERROR is None


def _require_flashinfer(function: Callable | None, operator: str) -> Callable:
    if function is None:
        detail = (
            f": {FLASHINFER_IMPORT_ERROR}"
            if FLASHINFER_IMPORT_ERROR is not None
            else ""
        )
        raise RuntimeError(f"FlashInfer is unavailable for {operator}{detail}")
    return function


def get_flashinfer_silu_and_mul() -> Callable:
    """Return the validated FlashInfer SiLU-and-multiply callable."""
    return _require_flashinfer(_flashinfer_silu_and_mul, "silu_and_mul")


def get_flashinfer_rms_norm() -> Callable:
    """Return the validated FlashInfer RMSNorm callable."""
    return _require_flashinfer(_flashinfer_rmsnorm, "rms_norm")


def get_flashinfer_fused_add_rms_norm() -> Callable:
    """Return the validated FlashInfer fused add-RMSNorm callable."""
    return _require_flashinfer(
        _flashinfer_fused_add_rmsnorm, "fused_add_rms_norm"
    )


def get_flashinfer_rotary_embedding() -> Callable:
    """Return the validated FlashInfer rotary-embedding callable."""
    return _require_flashinfer(
        _flashinfer_apply_rope, "rotary_embedding"
    )
