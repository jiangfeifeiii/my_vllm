from importlib import import_module

import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.operators import OperatorResolver, REGISTRY


@pytest.fixture(scope="module")
def custom_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the custom SiLUAndMul operator")
    try:
        extension = import_module("nanovllm._C")
    except ImportError as error:
        pytest.skip(f"nanovllm._C is not built: {error}")
    assert hasattr(extension, "forward")
    return extension


@pytest.fixture(scope="module")
def registered_custom_provider(custom_ops):
    # Importing the provider modules is the production registration path.
    import_module("nanovllm.layers.flashinfer_ops")
    return import_module("nanovllm.layers.cuda_ops")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(18,), (31, 64), (2, 3, 14), (0, 16)])
def test_custom_silu_and_mul_matches_torch(custom_ops, dtype, shape):
    torch.manual_seed(23)
    input_tensor = torch.randn(shape, device="cuda", dtype=dtype)
    original = input_tensor.clone()

    output = custom_ops.forward(input_tensor)
    gate, up = original.chunk(2, dim=-1)
    expected = F.silu(gate) * up

    assert output.shape == expected.shape
    assert output.dtype == dtype
    assert output.device == input_tensor.device
    assert output.is_contiguous()
    assert torch.equal(input_tensor, original)
    if output.numel():
        assert output.data_ptr() != input_tensor.data_ptr()
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("input_factory", "error_pattern"),
    [
        (lambda: torch.randn(4, 7, device="cuda", dtype=torch.float16), "even"),
        (lambda: torch.randn(4, 8, device="cuda", dtype=torch.float32), "supports"),
        (
            lambda: torch.randn(4, 8, device="cuda", dtype=torch.float16).t(),
            "contiguous",
        ),
        (lambda: torch.randn(4, 8, dtype=torch.float16), "CUDA"),
    ],
)
def test_custom_silu_and_mul_rejects_unsupported_inputs(
    custom_ops, input_factory, error_pattern
):
    with pytest.raises(RuntimeError, match=error_pattern):
        custom_ops.forward(input_factory())


def test_custom_provider_priority_and_auto_selection(registered_custom_provider):
    providers = {
        provider.name: provider
        for provider in REGISTRY.providers("silu_and_mul")
    }

    assert {"native_torch", "flashinfer", "custom_cuda"} <= providers.keys()
    assert providers["custom_cuda"].priority == 400
    for dtype in (torch.float16, torch.bfloat16):
        selected = REGISTRY.resolve(
            "silu_and_mul",
            device_type="cuda",
            dtype=dtype,
        )
        assert selected.name == "custom_cuda"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_custom_provider_registered_layer_matches_torch(
    registered_custom_provider, dtype
):
    resolver = OperatorResolver(
        overrides={"silu_and_mul": "custom_cuda"},
        device_type="cuda",
        dtype=dtype,
    )
    layer = SiluAndMul(operator_resolver=resolver)
    torch.manual_seed(29)
    input_tensor = torch.randn(5, 192, device="cuda", dtype=dtype)
    original = input_tensor.clone()

    output = layer(input_tensor)
    gate, up = original.chunk(2, dim=-1)
    expected = F.silu(gate) * up

    assert layer.provider_name == "custom_cuda"
    assert output.shape == (5, 96)
    assert output.dtype == dtype
    assert torch.equal(input_tensor, original)
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
