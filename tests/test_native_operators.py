from contextlib import ExitStack, contextmanager
from unittest import TestCase, main
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding


def _eager_callable(compiled_callable):
    """Return the function wrapped by torch.compile when one is present."""
    return getattr(compiled_callable, "_torchdynamo_orig_callable", compiled_callable)


@contextmanager
def _eager_native_methods(*methods):
    """Exercise public module calls without compiling kernels in CPU tests."""
    with ExitStack() as stack:
        for owner, name in methods:
            stack.enter_context(
                patch.object(owner, name, _eager_callable(getattr(owner, name)))
            )
        yield


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 3e-3, 3e-3
    return 1e-6, 1e-6


class NativeOperatorTest(TestCase):
    dtypes = (torch.float16, torch.bfloat16)

    def test_fused_add_rmsnorm_internal_fast_path_safely_falls_back(self):
        with _eager_native_methods((RMSNorm, "add_rms_forward")):
            norm = RMSNorm(16, eps=1e-6).to(dtype=torch.bfloat16)
            x = torch.randn(6, 16, dtype=torch.bfloat16)
            residual = torch.randn(6, 16, dtype=torch.bfloat16)
            original_x = x.clone()
            original_residual = residual.clone()

            output, new_residual = norm.forward_inplace(x, residual)

            self.assertTrue(torch.equal(x, original_x))
            self.assertTrue(torch.equal(residual, original_residual))
            self.assertNotEqual(output.data_ptr(), x.data_ptr())
            self.assertNotEqual(
                new_residual.data_ptr(), residual.data_ptr()
            )

    def test_silu_and_mul_value_shape_dtype_and_input_aliasing(self):
        with _eager_native_methods((SiluAndMul, "native_forward")):
            for dtype in self.dtypes:
                with self.subTest(dtype=dtype):
                    torch.manual_seed(7)
                    x = torch.randn(5, 24, dtype=dtype)
                    original = x.clone()

                    output = SiluAndMul()(x)
                    gate, up = original.chunk(2, dim=-1)
                    expected = F.silu(gate) * up
                    atol, rtol = _tolerances(dtype)

                    self.assertEqual(output.shape, (5, 12))
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(torch.equal(x, original))
                    self.assertNotEqual(output.data_ptr(), x.data_ptr())
                    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)

    def test_rmsnorm_plain_value_and_public_interface(self):
        with _eager_native_methods((RMSNorm, "rms_forward")):
            for dtype in self.dtypes:
                with self.subTest(dtype=dtype):
                    torch.manual_seed(11)
                    norm = RMSNorm(16, eps=1e-6).to(dtype=dtype)
                    norm.weight.data.copy_(
                        torch.linspace(0.75, 1.25, 16, dtype=dtype)
                    )
                    x = torch.randn(3, 4, 16, dtype=dtype)
                    original = x.clone()

                    output = norm(x)
                    x_fp32 = original.float()
                    expected = x_fp32 * torch.rsqrt(
                        x_fp32.pow(2).mean(dim=-1, keepdim=True) + norm.eps
                    )
                    expected = expected.to(dtype).mul(norm.weight)
                    atol, rtol = _tolerances(dtype)

                    self.assertIsInstance(output, torch.Tensor)
                    self.assertEqual(output.shape, x.shape)
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(torch.equal(x, original))
                    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)

    def test_fused_add_rmsnorm_value_tuple_and_residual_contract(self):
        with _eager_native_methods((RMSNorm, "add_rms_forward")):
            for dtype in self.dtypes:
                with self.subTest(dtype=dtype):
                    torch.manual_seed(13)
                    norm = RMSNorm(16, eps=1e-6).to(dtype=dtype)
                    norm.weight.data.copy_(
                        torch.linspace(0.8, 1.2, 16, dtype=dtype)
                    )
                    x = torch.randn(6, 16, dtype=dtype)
                    residual = torch.randn(6, 16, dtype=dtype)
                    original_x = x.clone()
                    original_residual = residual.clone()

                    result = norm(x, residual)
                    self.assertIsInstance(result, tuple)
                    self.assertEqual(len(result), 2)
                    output, new_residual = result

                    summed_fp32 = original_x.float() + original_residual.float()
                    expected_residual = summed_fp32.to(dtype)
                    expected_output = summed_fp32 * torch.rsqrt(
                        summed_fp32.pow(2).mean(dim=-1, keepdim=True) + norm.eps
                    )
                    expected_output = expected_output.to(dtype).mul(norm.weight)
                    atol, rtol = _tolerances(dtype)

                    self.assertEqual(output.shape, x.shape)
                    self.assertEqual(new_residual.shape, residual.shape)
                    self.assertEqual(output.dtype, dtype)
                    self.assertEqual(new_residual.dtype, dtype)
                    self.assertTrue(torch.equal(x, original_x))
                    self.assertTrue(torch.equal(residual, original_residual))
                    torch.testing.assert_close(
                        new_residual, expected_residual, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        output, expected_output, atol=atol, rtol=rtol
                    )

    def test_rope_value_tuple_shape_dtype_and_out_of_place_contract(self):
        with _eager_native_methods((RotaryEmbedding, "native_forward")):
            for dtype in self.dtypes:
                with self.subTest(dtype=dtype):
                    torch.manual_seed(17)
                    rope = RotaryEmbedding(
                        head_size=8,
                        rotary_dim=8,
                        max_position_embeddings=32,
                        base=10000.0,
                    )
                    positions = torch.tensor([0, 3, 11], dtype=torch.int64)
                    query = torch.randn(3, 4, 8, dtype=dtype)
                    key = torch.randn(3, 2, 8, dtype=dtype)
                    original_query = query.clone()
                    original_key = key.clone()

                    result = rope(positions, query, key)
                    self.assertIsInstance(result, tuple)
                    self.assertEqual(len(result), 2)
                    output_query, output_key = result

                    cos_sin = rope.cos_sin_cache[positions]
                    cos, sin = cos_sin.chunk(2, dim=-1)

                    def reference(x):
                        first, second = x.float().chunk(2, dim=-1)
                        return torch.cat(
                            (first * cos - second * sin,
                             second * cos + first * sin),
                            dim=-1,
                        ).to(dtype)

                    atol, rtol = _tolerances(dtype)
                    self.assertEqual(output_query.shape, query.shape)
                    self.assertEqual(output_key.shape, key.shape)
                    self.assertEqual(output_query.dtype, dtype)
                    self.assertEqual(output_key.dtype, dtype)
                    self.assertTrue(torch.equal(query, original_query))
                    self.assertTrue(torch.equal(key, original_key))
                    self.assertNotEqual(output_query.data_ptr(), query.data_ptr())
                    self.assertNotEqual(output_key.data_ptr(), key.data_ptr())
                    torch.testing.assert_close(
                        output_query, reference(original_query), atol=atol, rtol=rtol
                    )
                    torch.testing.assert_close(
                        output_key, reference(original_key), atol=atol, rtol=rtol
                    )


if __name__ == "__main__":
    main()
