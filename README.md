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
from nanovllm import CUDAGraphPolicy, LLM, SamplingParams

llm = LLM(
    "/absolute/path/to/Qwen3-model",
    cudagraph_mode=CUDAGraphPolicy.NONE,
    tensor_parallel_size=1,
    attention_backend="flashinfer",
    attention_mode="unified",
    kvcache_block_size=16,
    chunked_prefill=True,
)
params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], params)
print(outputs[0]["text"])
```

`attention_mode="unified"` is the default and sends the complete packed
`[Prefill | Decode]` query through one paged prefill call. Select `"split"`
to use phase-specialized prefill/decode wrappers writing directly into one
reusable output buffer; the split path does not concatenate temporary outputs.

### CUDA Graph policy

`cudagraph_mode` accepts `CUDAGraphPolicy.NONE` (`"none"`) and
`CUDAGraphPolicy.FULL_DECODE_ONLY` (`"full_decode_only"`). The engine default
is `FULL_DECODE_ONLY`; the compatibility setting `enforce_eager=True` maps to
`NONE`.

The policy and the execution mode selected for one scheduled batch are
separate:

| Policy and batch conditions | Runtime mode |
|---|---|
| `NONE`, any batch | `EAGER` |
| `FULL_DECODE_ONLY`, pure decode, unified FlashInfer, tensor parallel size 1, one query token per request, exact captured batch-size bucket, valid page metadata | `FULL_GRAPH` |
| `FULL_DECODE_ONLY`, any unmet condition above | `EAGER` fallback |

The default configured buckets are `1, 2, 4, 8, 16, 32, 64`, limited by
`max_num_seqs` and `max_num_batched_tokens` during initialization. Replay
requires an exact batch-size match: a batch of 12 is never padded to a bucket
of 16. Pure prefill, mixed batches, split attention, legacy attention, tensor
parallel execution, missing buckets, and metadata-capacity failures remain
Eager.

For an eligible replay, FlashInfer planning and metadata/input updates happen
outside the graph. Replay captures the embedding, transformer layers (including
unified attention and MLP), and final RMSNorm. Logit computation, sampling,
scheduler/postprocessing, and prefix-cache management remain outside. Runtime
coverage can be inspected without a profiler:

```python
stats = llm.model_runner.get_cudagraph_stats()
print(stats["full_graph_replay_steps"])
print(stats["eager_fallback_steps"])
print(stats["graph_bucket_hits"], stats["graph_bucket_misses"])
print(stats["captured_batch_sizes"])
```

On PyTorch 2.11 with FlashInfer 0.6.17, a process captures at most one
full-decode graph-engine session. `ModelRunner.exit()` explicitly resets every
graph exec; if another `FULL_DECODE_ONLY` engine is then created in the same
process, it emits a `RuntimeWarning` and safely runs Eager. Start the new engine
in a fresh process when it also needs Graph capture. This guard avoids a known
multi-wrapper CUDA teardown/reinitialization failure.

These statistics validate runtime routing; they are not scheduler performance
results. Cross-framework scheduler comparisons must disable CUDA Graph in both
engines.

For a runnable example that explicitly enables FlashInfer attention, 16-token
KV pages, chunked prefill, cache-aware LPM, the custom CUDA SiLU kernel, and
the FlashInfer normalization/RoPE providers, run:

```bash
# Eager (also the example's default)
python example_optimized.py \
  --model /absolute/path/to/Qwen3-model \
  --cudagraph-mode none

# Full transformer-body graph for eligible unified pure-decode buckets;
# all other batches automatically use Eager.
python example_optimized.py \
  --model /absolute/path/to/Qwen3-model \
  --attention-mode unified \
  --cudagraph-mode full_decode_only
```

The example prints the captured buckets, capture time, additional graph memory,
replay/fallback step counts, and exact-bucket hit/miss counts. Add `--debug` to
pause in `pdb` immediately before the first generation call.

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
  tests/test_scheduler_cache_aware.py \
  tests/test_cudagraph_runtime.py

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

The real-GPU full-decode Graph suite is opt-in and additionally needs the
local model path:

```bash
NANOVLLM_TEST_MODEL=/absolute/path/to/Qwen3-model \
NANOVLLM_CUDAGRAPH_GPU_MEMORY_UTILIZATION=0.35 \
NANOVLLM_RUN_CUDAGRAPH_TESTS=1 \
FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
python -m pytest -q \
  tests/test_cudagraph_flashinfer.py
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

### Targeted scheduler and attention experiments

`bench_scheduler.py` runs one real Qwen3-0.6B scheduler variant per process.
It keeps the physical GPU cache intact while replacing only the idle
scheduler's `BlockManager` with the recorded logical block limit.
`bench_attention.py` is model-free and measures the exact Qwen3-8B attention
shape. Both scripts fail rather than silently changing the requested workload.

These are the exact commands used for the checked-in RTX 5070 results:

```bash
PYTHONHASHSEED=0 FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
/tmp/nanovllm-flashinfer-env/bin/python bench_scheduler.py lpm \
  --mode fcfs --model /workspace/aiinfra/models/Qwen3-0.6B \
  --logical-kv-blocks 896 --gpu-memory-utilization 0.5 \
  --attention-mode unified --enforce-eager --seed 2026 \
  --output benchmark_results/rtx5070/lpm_fcfs.json

PYTHONHASHSEED=0 FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
/tmp/nanovllm-flashinfer-env/bin/python bench_scheduler.py lpm \
  --mode lpm --model /workspace/aiinfra/models/Qwen3-0.6B \
  --logical-kv-blocks 896 --gpu-memory-utilization 0.5 \
  --attention-mode unified --enforce-eager --seed 2026 \
  --output benchmark_results/rtx5070/lpm.json

PYTHONHASHSEED=0 FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
/tmp/nanovllm-flashinfer-env/bin/python bench_scheduler.py in-batch \
  --mode off --model /workspace/aiinfra/models/Qwen3-0.6B \
  --logical-kv-blocks 640 --gpu-memory-utilization 0.5 \
  --attention-mode unified --enforce-eager --seed 2026 \
  --output benchmark_results/rtx5070/in_batch_off.json

PYTHONHASHSEED=0 FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
/tmp/nanovllm-flashinfer-env/bin/python bench_scheduler.py in-batch \
  --mode on --model /workspace/aiinfra/models/Qwen3-0.6B \
  --logical-kv-blocks 640 --gpu-memory-utilization 0.5 \
  --attention-mode unified --enforce-eager --seed 2026 \
  --output benchmark_results/rtx5070/in_batch_on.json

PYTHONHASHSEED=0 FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
/tmp/nanovllm-flashinfer-env/bin/python bench_attention.py \
  --output benchmark_results/rtx5070/attention.json \
  --case case1 --case case2 --warmup 50 --iters 500 --repeats 5 \
  --workspace-mib 64 --seed 2026 --device 0 --backend auto
```

The runs used RTX 5070 SM120, driver 596.49, Torch 2.11.0+cu128,
CUDA 12.8, FlashInfer/cubin 0.6.17, and the
`flashinfer-jit-cache 0.6.17+cu129` AOT cache with JIT disabled. They were
captured from the Stage-3 worktree based on `bf16747`; the raw JSON records
the dirty state and exact script hashes
(`bench_scheduler.py=1b52cff92b4b9f339646c713b40b82c0072cf51ddc2dac7836179c9d659ae21e`,
`bench_attention.py=8c41c805e5169fc0894472db589d79eaa358c939d83c0a222cb2ed8acb5b7b34`).

#### Scheduler LPM

The measured phase contains 12 cold requests followed by 12 followers of three
resident 4096-token prefixes. The first FCFS step scheduled `Cold1..Cold4`
and directly recorded 103 evictions; the first LPM step scheduled
`A1,B1,C1,A2`, claimed 16,384 cached tokens, and recorded no eviction.

| Policy | Persistent hit tokens | Computed prompt tokens | Cached-block evictions | Preemptions | P95 TTFT (ms) | Throughput (req/s) | Completion (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 40,528 | 19,983 | 1,242 | 0 | 6,787.689 | 3.192 | 7.519 |
| LPM | 49,152 | 11,359 | 704 | 0 | 5,433.597 | 3.609 | 6.651 |
| LPM vs FCFS | +21.3% | -43.2% | -43.3% | unchanged | -19.9% | +13.1% | -11.6% |

Raw request/step traces:
[`lpm_fcfs.json`](benchmark_results/rtx5070/lpm_fcfs.json) and
[`lpm.json`](benchmark_results/rtx5070/lpm.json). The total eviction counter
includes all measured phase-2 steps; only the per-step trace supports direct
request attribution.

#### In-batch Prefix Burst

All 16 requests arrive together in four groups with 2048 shared tokens.
Temporary OFF admits `A1..A4` first. Temporary ON marks 12 followers and
admits `A1,B1,C1,D1`; followers later receive ordinary persistent hits.

| Policy | Temporary followers | Later persistent hit tokens | Computed prompt tokens | Duplicate prefill tokens | P95 TTFT (ms) | Throughput (req/s) | Completion (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LPM | 0 | 0 | 34,816 | 24,576 | 3,601.997 | 3.695 | 4.330 |
| LPM + Temporary Deprioritization | 12 | 24,576 | 10,240 | 0 | 3,184.553 | 3.894 | 4.108 |
| Temporary ON vs OFF | +12 | +24,576 | -70.6% | -100.0% | -11.6% | +5.4% | -5.1% |

Raw request/step traces:
[`in_batch_off.json`](benchmark_results/rtx5070/in_batch_off.json) and
[`in_batch_on.json`](benchmark_results/rtx5070/in_batch_on.json).

#### Attention

Planning and caller-owned outputs are outside timed regions for Unified and
Zero-copy Split. Old Split exactly retains phase-output allocation plus
`torch.cat`; CUDA Events capture device work and the cat copy, not host-only
allocator latency. Each raw value below is the per-call mean of 500 iterations.

| Case | Method | Raw repeats (ms) | Median (ms) | Min / max (ms) | Latency vs Unified | Max abs diff vs Unified |
|---|---|---|---:|---:|---:|---:|
| P16/KV4096 + D128/KV8192 | Unified | 7.204317, 7.205540, 7.208503, 7.206069, 7.204852 | 7.205540 | 7.204317 / 7.208503 | baseline | 0 |
| same | Old Split + Cat | 7.269072, 7.267829, 7.266360, 7.265813, 7.271242 | 7.267829 | 7.265813 / 7.271242 | +0.86% | 0.00001526 |
| same | Zero-copy Split | 7.263309, 7.262338, 7.262393, 7.261421, 7.259636 | 7.262338 | 7.259636 / 7.263309 | +0.79% | 0.00001526 |
| P32/KV4096 + D64/KV16384 | Unified | 7.226809, 7.220091, 7.230055, 7.228425, 7.225559 | 7.226809 | 7.220091 / 7.230055 | baseline | 0 |
| same | Old Split + Cat | 7.494944, 7.503711, 7.498145, 7.499335, 7.500299 | 7.499335 | 7.494944 / 7.503711 | +3.77% | 0.00001526 |
| same | Zero-copy Split | 7.501136, 7.500984, 7.488825, 7.498392, 7.503362 | 7.500984 | 7.488825 / 7.503362 | +3.79% | 0.00001526 |

The pure-decode diagnostic also favored the paged-prefill wrapper on this
stack: 7.167986 vs 7.213052 ms in Case 1 and 7.196127 vs 7.426044 ms in Case 2.
Unified therefore remains the default; Split is retained as an explicit
phase-specialized path, not presented as a speedup. Full metadata, correctness,
execution order, and diagnostic repeats are in
[`attention.json`](benchmark_results/rtx5070/attention.json).
