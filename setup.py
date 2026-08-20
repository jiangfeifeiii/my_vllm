from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    ext_modules=[
        CUDAExtension(
            "nanovllm._C",
            [
                "csrc/silu_and_mul.cpp",
                "csrc/silu_and_mul_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True),
    },
)
