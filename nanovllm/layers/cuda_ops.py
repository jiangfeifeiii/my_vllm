import torch

from nanovllm.layers.operators import register_operator

try:
    from nanovllm import _C
except ImportError as error:
    raise ImportError(
        "nanovllm._C is not built; install with "
        "`pip install -e . --no-build-isolation`"
    ) from error


def _supports_cuda(*, device_type=None, dtype=None, **_):
    return device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)


@register_operator(
    "silu_and_mul", "custom_cuda", supports=_supports_cuda, priority=400
)
def _bind_custom_silu_and_mul(_layer):
    return _C.forward
