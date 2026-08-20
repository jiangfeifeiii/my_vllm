#include <torch/extension.h>

torch::Tensor silu_and_mul_cuda(const torch::Tensor& input);

namespace {

torch::Tensor silu_and_mul(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "silu_and_mul expects a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "silu_and_mul expects a contiguous tensor");
  TORCH_CHECK(input.dim() >= 1, "silu_and_mul expects at least one dimension");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat16 ||
          input.scalar_type() == torch::kBFloat16,
      "silu_and_mul only supports float16 and bfloat16");
  TORCH_CHECK(
      input.size(-1) > 0 && input.size(-1) % 2 == 0,
      "silu_and_mul expects the last dimension to be a positive even number");

  return silu_and_mul_cuda(input);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward",
      &silu_and_mul,
      pybind11::arg("input"),
      "Fused SiLU and multiply (CUDA)");
}
