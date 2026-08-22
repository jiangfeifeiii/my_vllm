from importlib import import_module

import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.flashinfer_ops import FLASHINFER_AVAILABLE
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

    assert {
        "native_torch",
        "flashinfer",
        "custom_cuda",
        "adaptive_cuda",
    } <= providers.keys()
    assert providers["custom_cuda"].priority == 400
    assert providers["adaptive_cuda"].priority == 500
    for dtype in (torch.float16, torch.bfloat16):
        selected = REGISTRY.resolve(
            "silu_and_mul",
            device_type="cuda",
            dtype=dtype,
        )
        expected = "adaptive_cuda" if FLASHINFER_AVAILABLE else "custom_cuda"
        assert selected.name == expected


def test_adaptive_provider_dispatches_on_token_threshold(
    registered_custom_provider,
    monkeypatch,
):
    if not FLASHINFER_AVAILABLE:
        pytest.skip("FlashInfer is required for adaptive dispatch")
    cuda_ops = registered_custom_provider
    calls = []

    def custom_operation(x):
        calls.append(("custom_cuda", x.shape[0]))
        return x[..., : x.shape[-1] // 2]

    def flashinfer_operation(x):
        calls.append(("flashinfer", x.shape[0]))
        return x[..., : x.shape[-1] // 2]

    monkeypatch.setattr(cuda_ops._C, "forward", custom_operation)
    monkeypatch.setattr(
        cuda_ops,
        "get_flashinfer_silu_and_mul",
        lambda: flashinfer_operation,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda *_: cuda_ops.ADAPTIVE_SILU_BENCHMARK_DEVICE,
    )
    provider = next(
        item
        for item in REGISTRY.providers("silu_and_mul")
        if item.name == "adaptive_cuda"
    )
    forward = provider.callable(None)
    threshold = cuda_ops.ADAPTIVE_SILU_FLASHINFER_MIN_TOKENS
    width = cuda_ops.ADAPTIVE_SILU_BENCHMARK_WIDTH

    forward(torch.empty(threshold - 1, width, dtype=torch.bfloat16))
    forward(torch.empty(threshold, width, dtype=torch.bfloat16))

    assert calls == [
        ("custom_cuda", threshold - 1),
        ("flashinfer", threshold),
    ]


def test_adaptive_provider_does_not_extrapolate_measured_crossover(
    registered_custom_provider,
    monkeypatch,
):
    if not FLASHINFER_AVAILABLE:
        pytest.skip("FlashInfer is required for adaptive dispatch")
    cuda_ops = registered_custom_provider
    calls = []

    def custom_operation(x):
        calls.append("custom_cuda")
        return x[..., : x.shape[-1] // 2]

    def flashinfer_operation(x):
        calls.append("flashinfer")
        return x[..., : x.shape[-1] // 2]

    monkeypatch.setattr(cuda_ops._C, "forward", custom_operation)
    monkeypatch.setattr(
        cuda_ops,
        "get_flashinfer_silu_and_mul",
        lambda: flashinfer_operation,
    )
    provider = next(
        item
        for item in REGISTRY.providers("silu_and_mul")
        if item.name == "adaptive_cuda"
    )
    width = cuda_ops.ADAPTIVE_SILU_BENCHMARK_WIDTH

    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda *_: cuda_ops.ADAPTIVE_SILU_BENCHMARK_DEVICE,
    )
    measured_device_forward = provider.callable(None)
    measured_device_forward(torch.empty(1, width, dtype=torch.float16))
    measured_device_forward(torch.empty(1, width + 16, dtype=torch.bfloat16))
    measured_device_forward(torch.empty(1, 10, dtype=torch.bfloat16))

    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda *_: "unmeasured CUDA device",
    )
    unmeasured_device_forward = provider.callable(None)
    unmeasured_device_forward(torch.empty(1, width, dtype=torch.bfloat16))

    assert calls == [
        "flashinfer",
        "flashinfer",
        "custom_cuda",
        "flashinfer",
    ]


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


@pytest.mark.parametrize("rows", [8, 128])
def test_adaptive_provider_registered_layer_matches_torch(
    registered_custom_provider,
    rows,
):
    if not FLASHINFER_AVAILABLE:
        pytest.skip("FlashInfer is required for adaptive dispatch")
    resolver = OperatorResolver(
        overrides={"silu_and_mul": "adaptive_cuda"},
        device_type="cuda",
        dtype=torch.bfloat16,
    )
    layer = SiluAndMul(operator_resolver=resolver)
    torch.manual_seed(31 + rows)
    input_tensor = torch.randn(
        rows,
        registered_custom_provider.ADAPTIVE_SILU_BENCHMARK_WIDTH,
        device="cuda",
        dtype=torch.bfloat16,
    )
    original = input_tensor.clone()

    output = layer(input_tensor)
    gate, up = original.chunk(2, dim=-1)
    expected = F.silu(gate) * up

    assert layer.provider_name == "adaptive_cuda"
    assert torch.equal(input_tensor, original)
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
