#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

template <typename scalar_t>
__global__ void silu_and_mul_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t output_elements,
    int64_t hidden_size) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t output_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += stride) {
    const int64_t row = output_index / hidden_size;
    const int64_t column = output_index - row * hidden_size;
    const int64_t gate_index = row * (hidden_size * 2) + column;

    const float gate = static_cast<float>(input[gate_index]);
    const float up = static_cast<float>(input[gate_index + hidden_size]);
    const float silu = gate / (1.0f + expf(-gate));
    output[output_index] = static_cast<scalar_t>(silu * up);
  }
}

template <typename scalar_t>
void launch_silu_and_mul(
    const torch::Tensor& input,
    torch::Tensor& output,
    int64_t hidden_size) {
  constexpr int threads = 256;
  const int64_t output_elements = output.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      (output_elements + threads - 1) / threads, 65535));
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  silu_and_mul_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
      input.const_data_ptr<scalar_t>(),
      output.mutable_data_ptr<scalar_t>(),
      output_elements,
      hidden_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor silu_and_mul_cuda(const torch::Tensor& input) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output_shape = input.sizes().vec();
  const int64_t hidden_size = output_shape.back() / 2;
  output_shape.back() = hidden_size;
  auto output = torch::empty(output_shape, input.options());

  if (output.numel() == 0) {
    return output;
  }

  switch (input.scalar_type()) {
    case torch::kFloat16:
      launch_silu_and_mul<at::Half>(input, output, hidden_size);
      break;
    case torch::kBFloat16:
      launch_silu_and_mul<at::BFloat16>(input, output, hidden_size);
      break;
    default:
      TORCH_CHECK(false, "silu_and_mul only supports float16 and bfloat16");
  }

  return output;
}
