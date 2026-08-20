from unittest import TestCase, main, skipUnless

import torch
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.flashinfer_ops import (
    FLASHINFER_AVAILABLE,
    FLASHINFER_IMPORT_ERROR,
)
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.operators import OperatorResolver, REGISTRY
from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb


def _resolver(*operators: str, dtype: torch.dtype) -> OperatorResolver:
    return OperatorResolver(
        overrides={operator: "flashinfer" for operator in operators},
        device_type="cuda",
        dtype=dtype,
    )


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 3e-3, 3e-3


class FlashInferRegistrationTest(TestCase):

    def test_providers_have_expected_priority_and_capability_filter(self):
        for operator in (
            "silu_and_mul",
            "rms_norm",
            "fused_add_rms_norm",
            "rotary_embedding",
        ):
            with self.subTest(operator=operator):
                provider = next(
                    item
                    for item in REGISTRY.providers(operator)
                    if item.name == "flashinfer"
                )
                self.assertEqual(provider.priority, 200)
                self.assertFalse(
                    provider.supports(device_type="cpu", dtype=torch.float16)
                )
                self.assertFalse(
                    provider.supports(device_type="cuda", dtype=torch.float32)
                )
                self.assertEqual(
                    provider.supports(
                        device_type=torch.device("cuda"),
                        dtype=torch.bfloat16,
                    ),
                    FLASHINFER_AVAILABLE,
                )


_SKIP_REASON = (
    "FlashInfer functional tests require CUDA and an importable FlashInfer"
    + (
        f" ({FLASHINFER_IMPORT_ERROR})"
        if FLASHINFER_IMPORT_ERROR is not None
        else ""
    )
)


@skipUnless(torch.cuda.is_available() and FLASHINFER_AVAILABLE, _SKIP_REASON)
class FlashInferOperatorTest(TestCase):
    dtypes = (torch.float16, torch.bfloat16)

    def test_silu_and_mul_matches_reference_and_is_out_of_place(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                torch.manual_seed(101)
                layer = SiluAndMul(
                    operator_resolver=_resolver("silu_and_mul", dtype=dtype)
                ).cuda()
                x = torch.randn(7, 256, device="cuda", dtype=dtype)
                original = x.clone()

                output = layer(x)
                gate, value = original.chunk(2, dim=-1)
                expected = F.silu(gate) * value
                atol, rtol = _tolerances(dtype)

                self.assertEqual(layer.provider_name, "flashinfer")
                self.assertEqual(output.shape, (7, 128))
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(torch.equal(x, original))
                self.assertNotEqual(output.data_ptr(), x.data_ptr())
                torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)

    def test_rms_norm_matches_fp32_reference_and_is_out_of_place(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                torch.manual_seed(103)
                layer = RMSNorm(
                    128,
                    eps=1e-6,
                    operator_resolver=_resolver(
                        "rms_norm", "fused_add_rms_norm", dtype=dtype
                    ),
                ).cuda().to(dtype=dtype)
                layer.weight.data.copy_(
                    torch.linspace(
                        0.75, 1.25, 128, device="cuda", dtype=dtype
                    )
                )
                x = torch.randn(9, 128, device="cuda", dtype=dtype)
                original = x.clone()

                output = layer(x)
                x_fp32 = original.float()
                expected = x_fp32 * torch.rsqrt(
                    x_fp32.square().mean(dim=-1, keepdim=True) + layer.eps
                )
                expected = expected.to(dtype).mul(layer.weight)
                atol, rtol = _tolerances(dtype)

                self.assertEqual(layer.rms_provider_name, "flashinfer")
                self.assertEqual(output.shape, x.shape)
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(torch.equal(x, original))
                self.assertNotEqual(output.data_ptr(), x.data_ptr())
                torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)

    def test_fused_add_rms_norm_preserves_inputs_and_tuple_contract(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                torch.manual_seed(107)
                layer = RMSNorm(
                    128,
                    eps=1e-6,
                    operator_resolver=_resolver(
                        "rms_norm", "fused_add_rms_norm", dtype=dtype
                    ),
                ).cuda().to(dtype=dtype)
                layer.weight.data.copy_(
                    torch.linspace(
                        0.8, 1.2, 128, device="cuda", dtype=dtype
                    )
                )
                x = torch.randn(11, 128, device="cuda", dtype=dtype)
                residual = torch.randn(11, 128, device="cuda", dtype=dtype)
                original_x = x.clone()
                original_residual = residual.clone()

                result = layer(x, residual)
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                output, new_residual = result

                expected_residual = (original_x.float() + original_residual.float()).to(
                    dtype
                )
                summed_fp32 = expected_residual.float()
                expected = summed_fp32 * torch.rsqrt(
                    summed_fp32.square().mean(dim=-1, keepdim=True) + layer.eps
                )
                expected = expected.to(dtype).mul(layer.weight)
                atol, rtol = _tolerances(dtype)

                self.assertEqual(layer.add_rms_provider_name, "flashinfer")
                self.assertEqual(output.shape, x.shape)
                self.assertEqual(new_residual.shape, residual.shape)
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(new_residual.dtype, dtype)
                self.assertTrue(torch.equal(x, original_x))
                self.assertTrue(torch.equal(residual, original_residual))
                self.assertNotEqual(output.data_ptr(), x.data_ptr())
                self.assertNotEqual(new_residual.data_ptr(), residual.data_ptr())
                torch.testing.assert_close(
                    new_residual, expected_residual, atol=atol, rtol=rtol
                )
                torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)

    def test_rope_matches_neox_reference_and_is_out_of_place(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                torch.manual_seed(109)
                layer = RotaryEmbedding(
                    head_size=64,
                    rotary_dim=64,
                    max_position_embeddings=128,
                    base=10000.0,
                    operator_resolver=_resolver("rotary_embedding", dtype=dtype),
                ).cuda()
                positions = torch.tensor(
                    [0, 3, 17, 63], device="cuda", dtype=torch.int32
                )
                query = torch.randn(4, 8, 64, device="cuda", dtype=dtype)
                key = torch.randn(4, 2, 64, device="cuda", dtype=dtype)
                original_query = query.clone()
                original_key = key.clone()

                output_query, output_key = layer(positions, query, key)
                cos_sin = layer.cos_sin_cache[positions.long()]
                cos, sin = cos_sin.chunk(2, dim=-1)
                expected_query = apply_rotary_emb(
                    original_query, cos, sin
                )
                expected_key = apply_rotary_emb(original_key, cos, sin)
                atol, rtol = _tolerances(dtype)

                self.assertEqual(layer.provider_name, "flashinfer")
                self.assertEqual(output_query.shape, query.shape)
                self.assertEqual(output_key.shape, key.shape)
                self.assertEqual(output_query.dtype, dtype)
                self.assertEqual(output_key.dtype, dtype)
                self.assertTrue(torch.equal(query, original_query))
                self.assertTrue(torch.equal(key, original_key))
                self.assertNotEqual(output_query.data_ptr(), query.data_ptr())
                self.assertNotEqual(output_key.data_ptr(), key.data_ptr())
                torch.testing.assert_close(
                    output_query, expected_query, atol=atol, rtol=rtol
                )
                torch.testing.assert_close(
                    output_key, expected_key, atol=atol, rtol=rtol
                )


if __name__ == "__main__":
    main()
