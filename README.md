# Nano-vLLM-v1

Nano-vLLM-v1 is a compact Qwen3 inference engine for studying and testing a
vLLM-v1-style execution path. It includes paged KV cache, prefix caching,
cache-aware scheduling, chunked prefill, tensor parallelism, pluggable
operators, and phase-specialized FlashInfer attention.

The production inference path currently targets NVIDIA CUDA GPUs and model
weights in FP16 or BF16. Linear and tensor-parallel projection layers
intentionally remain on `torch.nn.functional.linear`; GEMM is not part of the
operator registry.

See [Architecture](docs/architecture.md) for the provider rules, attention
backends, cache layout, and scheduler invariants.

## Installation

Python 3.10-3.12, a CUDA-capable PyTorch build, a matching CUDA toolkit, and a
local Qwen3 checkpoint are required. The editable install builds the custom
CUDA SiLU-and-multiply extension, so install PyTorch before building the
project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools<82" wheel ninja packaging pytest

# Install the PyTorch build matching the host CUDA driver/toolkit first.
# Then build nano-vLLM and install its runtime dependencies.
python -m pip install -e . --no-build-isolation
```

### Reproducible SM120 / CUDA 12.8 setup

The following pinned setup is the validated route for an SM120 GPU with a
CUDA 12.8 toolkit. FlashInfer's cu129 AOT cache is required because FlashInfer
0.6.17 cannot JIT-compile SM120 kernels with a CUDA toolkit older than 12.9.
The cache is distributed from FlashInfer's official package indexes.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools<82" wheel ninja packaging pytest
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install "flashinfer-python==0.6.17"
python -m pip install "flashinfer-cubin==0.6.17" \
  --index-url https://flashinfer.ai/whl
python -m pip install "flashinfer-jit-cache==0.6.17+cu129" \
  --index-url https://flashinfer.ai/whl/cu129

export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export FLASHINFER_DISABLE_JIT=1

python -m pip install -e . --no-build-isolation
```

Keep the two `FLASHINFER_*` variables set for tests and inference. With
`FLASHINFER_DISABLE_JIT=1`, missing AOT kernels fail immediately instead of
silently starting a local JIT build.

## Quick start

The default attention path uses FlashInfer with 16-token KV-cache pages. Set
`chunked_prefill=True` when prompts may exceed the per-batch token budget.

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/absolute/path/to/Qwen3-model",
    enforce_eager=True,
    tensor_parallel_size=1,
    attention_backend="flashinfer",
    kvcache_block_size=16,
    chunked_prefill=True,
)
params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], params)
print(outputs[0]["text"])
```

Operator selection defaults to `auto`: the highest-priority supported
provider is selected for each registered operator. A provider can be pinned
without changing layer code:

```python
llm = LLM(
    "/absolute/path/to/Qwen3-model",
    operator_overrides={
        "silu_and_mul": "custom_cuda",
        "rms_norm": "flashinfer",
        "fused_add_rms_norm": "flashinfer",
        "rotary_embedding": "flashinfer",
        "kv_cache_store": "native",
    },
)
```

`native` is an alias for the highest-priority supported provider whose name
starts with `native_`; exact names such as `native_torch`, `native_triton`,
`flashinfer`, and `custom_cuda` are also accepted.

### Legacy rollback

The attention backend is independent of the operator registry. To restore the
original FlashAttention paged path and native registered operators, select it
explicitly and use a block size divisible by 256:

```python
llm = LLM(
    "/absolute/path/to/Qwen3-model",
    attention_backend="legacy",
    kvcache_block_size=256,
    operator_overrides={
        "silu_and_mul": "native",
        "rms_norm": "native",
        "fused_add_rms_norm": "native",
        "rotary_embedding": "native",
        "kv_cache_store": "native",
    },
)
```

The legacy path still requires `flash-attn`; it is a rollback path for
attention and provider selection, not a CPU inference mode.

## Tests

Run the full suite after installation. CUDA/model-dependent tests skip when
their prerequisites are absent:

```bash
python -m pytest -q tests
```

Focused operator, scheduler, and attention coverage:

```bash
python -m pytest -q \
  tests/test_operator_registry.py \
  tests/test_native_operators.py \
  tests/test_block_manager_baseline.py \
  tests/test_block_manager_chunked.py \
  tests/test_block_manager_planning.py \
  tests/test_scheduler_cache_aware.py

FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
python -m pytest -q \
  tests/test_flashinfer_operators.py \
  tests/test_custom_cuda_silu.py \
  tests/test_attention_backend.py \
  tests/test_model_runner_metadata.py
```

For end-to-end generation, point the tests at a local Qwen3 checkpoint:

```bash
export NANOVLLM_TEST_MODEL=/absolute/path/to/Qwen3-model
export NANOVLLM_E2E_GPU_MEMORY_UTILIZATION=0.35
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export FLASHINFER_DISABLE_JIT=1
python -m pytest -q \
  tests/test_e2e_baseline.py \
  tests/test_e2e_flashinfer_attention.py
```

## Benchmarks

`serving_bench.py` drives requests with exponentially distributed arrival
times and reports throughput, TTFT, TPOT, and average latency:

```bash
python serving_bench.py \
  --model /absolute/path/to/Qwen3-model \
  --request-rate 10 \
  --num-requests 1024 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1024 \
  --random-input-len 128 \
  --random-output-len 100 \
  --chunked-prefill \
  --enforce-eager
```

`bench.py` is the offline throughput harness; set its local model path and
workload constants, then run `python bench.py`.

`bench_attention.py` is a model-free microbenchmark for the exact mixed
attention decision in this refactor: one unified paged-prefill wrapper versus
the phase-specialized prefill/decode wrappers, including the split path's
output concatenation. Planning occurs once outside the timed region, matching
one batch plan reused across transformer layers.

```bash
FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
python bench_attention.py \
  --dtype bf16 \
  --block-size 16 \
  --num-prefill 4 \
  --prefill-q-len 128 \
  --num-decode 64 \
  --decode-kv-len 2048 \
  --warmup 20 \
  --iters 200 \
  --flashinfer-cuda-arch-list 12.0f
```

The following measurements were reproduced on commit `844f634`. They are
kernel-path microbenchmarks, not end-to-end throughput claims. On this consumer
SM120 GPU the split path was about 3-4% slower for both sampled workloads; the
phase split is retained for explicit execution semantics and future
backend-specific tuning, not presented as a speedup.

| Commit | GPU / software | Dtype / layout | Workload | Unified | Split | Unified / split | Max abs diff |
|---|---|---|---|---:|---:|---:|---:|
| `844f634` | RTX 5070; Torch 2.11.0+cu128; CUDA 12.8; FlashInfer 0.6.17 + cu129 AOT cache | BF16; B16; Q16/KV4/HD128 | P=4x128, D=64x1, decode KV=2048; warmup=20, iters=200 | 0.4726 ms | 0.4926 ms | 0.960x | 0.003906 |
| `844f634` | same | BF16; B16; Q16/KV4/HD128 | P=2x64, D=128x1, decode KV=4096; warmup=20, iters=100 | 1.7651 ms | 1.8252 ms | 0.967x | 0.007812 |

Always record the commit, hardware, software versions, dtype, backend, block
size, workload, warmup, and exact command when comparing runs.
