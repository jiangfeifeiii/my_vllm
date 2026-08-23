from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any, Callable

import torch
from torch import nn


_NATIVE_IMPLEMENTATIONS = frozenset({"native", "native_torch"})


@dataclass(frozen=True)
class CustomOpConfig:
    """Construction-time platform and implementation choices for ordinary ops."""

    platform: str = "auto"
    dtype: torch.dtype | None = None
    overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        platform = self.platform.lower()
        if platform not in ("auto", "cuda", "cpu", "xpu"):
            raise ValueError("platform must be 'auto', 'cuda', 'cpu', or 'xpu'")
        normalized_overrides = {
            str(operator): str(implementation).lower()
            for operator, implementation in self.overrides.items()
        }
        object.__setattr__(self, "platform", platform)
        object.__setattr__(
            self,
            "overrides",
            MappingProxyType(normalized_overrides),
        )

    def implementation_for(self, operator: str) -> str:
        return self.overrides.get(operator, "auto")

    def __hash__(self) -> int:
        return hash(
            (
                self.platform,
                self.dtype,
                tuple(sorted(self.overrides.items())),
            )
        )


def _detect_platform() -> str:
    get_default_device = getattr(torch, "get_default_device", None)
    if get_default_device is not None:
        device = torch.device(get_default_device())
    else:  # pragma: no cover - compatibility with older supported Torch builds.
        device = torch.empty(0).device
    return device.type if device.type in ("cuda", "cpu", "xpu") else "cpu"


class CustomOp(nn.Module):
    """Platform-dispatched base class for ordinary operators."""

    op_name = "custom_op"

    def __init__(self, custom_op_config: CustomOpConfig | None = None) -> None:
        super().__init__()
        self.custom_op_config = custom_op_config or CustomOpConfig()
        configured_platform = self.custom_op_config.platform
        self.platform = (
            _detect_platform()
            if configured_platform == "auto"
            else configured_platform
        )
        self._forward_method = self.dispatch_forward()

    @property
    def requested_implementation(self) -> str:
        return self.custom_op_config.implementation_for(self.op_name)

    def dispatch_forward(self) -> Callable[..., Any]:
        method = getattr(self, f"forward_{self.platform}", None)
        if method is None:
            return self.forward_native
        return method

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_cuda(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_native(*args, **kwargs)

    def forward_xpu(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_native(*args, **kwargs)

    def require_cuda_implementation(
        self,
        supported: set[str] | frozenset[str],
    ) -> str:
        implementation = self.requested_implementation
        if implementation not in supported:
            available = ", ".join(sorted(supported))
            raise ValueError(
                f"unsupported {self.op_name!r} implementation "
                f"{implementation!r}; available: {available}"
            )
        if self.platform != "cuda" and implementation not in (
            "auto",
            *_NATIVE_IMPLEMENTATIONS,
        ):
            raise RuntimeError(
                f"{implementation!r} for {self.op_name!r} requires CUDA"
            )
        return implementation


def load_optional_implementations() -> dict[str, str]:
    """Probe optional ordinary-op CUDA modules without registering providers."""
    errors: dict[str, str] = {}
    for module_name, error_name in (
        ("nanovllm.layers.flashinfer_ops", "FLASHINFER_IMPORT_ERROR"),
        ("nanovllm.layers.cuda_ops", "CUSTOM_CUDA_IMPORT_ERROR"),
    ):
        try:
            module = import_module(module_name)
        except (ImportError, OSError, RuntimeError) as error:
            errors[module_name] = str(error)
            continue
        error = getattr(module, error_name, None)
        if error is not None:
            errors[module_name] = str(error)
    return errors
