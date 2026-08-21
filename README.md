# Nano-vLLM-v1

Nano-vLLM-v1 is a compact Qwen3 inference engine for studying and testing a
vLLM-v1-style execution path. It includes paged KV cache, prefix caching,
stable cache-aware LPM, same-step full-block prefix reuse, chunked prefill,
tensor parallelism, pluggable operators, and phase-specialized attention.

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
    enable_lpm=True,
    enable_same_step_prefix_reuse=True,
)
params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], params)
print(outputs[0]["text"])
```

`attention_mode="unified"` is the default and sends the complete packed
`[Prefill | Decode]` query through one paged prefill call. Select `"split"`
to use phase-specialized prefill/decode wrappers writing directly into one
reusable output buffer; the split path does not concatenate temporary outputs.

### Prefix-cache admission

`enable_lpm=True` and `enable_same_step_prefix_reuse=True` are independent,
default-on scheduler controls. At the start of a step, the scheduler snapshots
the Waiting queue, matches the persistent cache once, and performs a stable
longest-prefix-match sort. Equal initial hits retain FCFS order. That ranked
order is then fixed for the rest of the step.

With Same-step Prefix Reuse enabled, each ranked request is committed in
`latest lookup -> capacity check -> claim -> allocate/publish` order. A later
request can therefore reuse full prefix blocks published by an earlier request
in the same step. Its latest hit changes only the remaining prompt work and KV
capacity required; it never changes the fixed LPM order. Capacity includes both
cached-free blocks that must be claimed and new blocks for the uncached suffix.
Decode and resumed-chunk reservations remain protected.

This same-step lookup/publish behavior follows the vLLM-style request commit
model; it is combined here with stable persistent-hit LPM ordering. LPM decides
which request consumes surviving cache first, while Same-step Prefix Reuse
removes duplicate work after that order has been fixed.

Set `enable_same_step_prefix_reuse=False` for the frozen-plan ablation. In that
mode, all Waiting admissions use only the initial persistent match snapshot;
prefixes published by an earlier request are not looked up again until a later
step. This does not disable persistent prefix caching or LPM. Likewise,
`enable_lpm=False` restores FCFS ranking without disabling Same-step Prefix
Reuse.

Only complete, block-aligned pages are published. Partial pages keep hash `-1`
and every request computes its own unaligned tail. Scheduler accounting keeps
initial persistent hits separate from newly acquired same-step hits; token
totals are block counts multiplied by `kvcache_block_size`. Computed prompt
tokens count only newly scheduled prompt work, while same-step reused requests
and blocks count requests with a positive same-step gain and the corresponding
newly reused full blocks.

Chained hashes remain candidate keys rather than correctness proofs. Each
published block also records an immutable canonical lineage identity and its
parent identity. Equivalent re-publications inherit that identity, preserving
valid descendants across different physical pages; a real hash collision or
destructive page reuse gets a new identity and safely stops matching.

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
  tests/test_block_manager_config.py \
  tests/test_block_manager_planning.py \
  tests/test_scheduler_cache_aware.py \
  tests/test_bench_scheduler_metrics.py \
  tests/test_bench_eager_compare.py \
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
  --logical-kv-blocks 2240 --gpu-memory-utilization 0.6 \
  --attention-mode unified --enforce-eager --seed 2026 \
  --output benchmark_results/rtx5070/in_batch_off.json

PYTHONHASHSEED=0 FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
/tmp/nanovllm-flashinfer-env/bin/python bench_scheduler.py in-batch \
  --mode on --model /workspace/aiinfra/models/Qwen3-0.6B \
  --logical-kv-blocks 2240 --gpu-memory-utilization 0.6 \
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
`flashinfer-jit-cache 0.6.17+cu129` AOT cache with JIT disabled. Each schema-v2
JSON records the source commit/dirty state, command, runtime configuration, and
exact benchmark-script SHA-256.

#### Scheduler LPM

The measured phase contains 12 cold requests followed by 12 followers of three
resident 4096-token prefixes. The first FCFS step schedules `Cold1..Cold4`;
the first LPM step schedules `A1,B1,C1,A2` and consumes the surviving
persistent prefixes before the cold requests can evict them.

| Policy | Initial persistent hits | Same-step hits | Computed prompt | Cached-block evictions | Preemptions | P95 TTFT (ms) | Throughput (req/s) | Completion (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 32,768 | 4,096 | 23,647 | 1,471 | 0 | 5,980.160 | 3.408 | 7.043 |
| LPM | 49,152 | 0 | 11,359 | 704 | 0 | 4,401.418 | 4.694 | 5.113 |

Raw request/step traces:
[`lpm_fcfs.json`](benchmark_results/rtx5070/lpm_fcfs.json) and
[`lpm.json`](benchmark_results/rtx5070/lpm.json). The total eviction counter
includes all measured phase-2 steps; only the per-step trace supports direct
request attribution.

#### In-batch Prefix Burst

All 16 requests arrive together as `A1..A4, B1..B4, C1..C4, D1..D4`; each
group shares 2048 tokens and has a 128-token unique suffix. Both variants use
the same 2,240-block cap, which is the cold worst case, so neither result is
helped by preemption. OFF freezes the initial misses. ON keeps exactly the same
stable order, but each follower observes its group leader's newly published
full blocks.

| Policy | Initial hits | Same-step hits | Computed prompt | Duplicate prefill | First-step admissions | Reused requests / blocks | Preemptions | P95 TTFT (ms) | Throughput (req/s) | Completion (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Same-step OFF | 0 | 0 | 34,816 | 24,576 | 4 | 0 / 0 | 0 | 959.989 | 8.238 | 1.942 |
| Same-step ON | 0 | 24,576 | 10,240 | 0 | 16 | 12 / 1,536 | 0 | 382.066 | 12.331 | 1.298 |

The requested `same_step_reused_requests` count is serialized as
`same_step_reused_request_count`; `same_step_reused_blocks` is the exact
full-block total. Prompt-token conservation is asserted per request and in
aggregate in both files.

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

### Runtime feature validation: full-decode CUDA Graph

This experiment is intentionally separate from the scheduler results above and
the cross-framework comparison below. `bench_cudagraph.py` compares only
nano-vLLM `NONE` with nano-vLLM `FULL_DECODE_ONLY` on exact-size unified pure
decode batches. Each case has one excluded follower-prefill step, an independent
8-step warm-up batch, and five measured repeats of 64 decode steps. The first
measured KV length is shown in the case name and increases by one per step; no
request or graph-bucket padding is used.

The final RTX 5070 runs used the same Qwen3-0.6B BF16 model, seed 2026,
FlashInfer unified attention, 16-token pages, and script SHA-256
`c9947824d05cf9c9a0feb2a3545dd0393ff1a53c877201a5b0b024c982c9fc31`.
The hardened comparison validator rederived every repeat and case summary from
the raw 64-step timings, checked KV-length and counter progression, and confirmed
the complete 15-case matrix, identical configuration except graph policy,
identical model/source provenance, and the same GPU and software stack.

| Case | NONE step median (ms) | FULL step median (ms) | Step delta | NONE TPOT (ms) | FULL TPOT (ms) | TPOT delta | NONE tok/s | FULL tok/s | Replay NONE / FULL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 / KV512 | 9.436 | 4.465 | -52.69% | 12.922 | 4.562 | -64.69% | 77.4 | 219.2 | 0% / 100% |
| B1 / KV2048 | 9.292 | 5.404 | -41.84% | 11.038 | 5.772 | -47.71% | 90.6 | 173.3 | 0% / 100% |
| B1 / KV4096 | 9.463 | 6.561 | -30.67% | 12.572 | 6.934 | -44.84% | 79.5 | 144.2 | 0% / 100% |
| B4 / KV512 | 9.384 | 4.759 | -49.29% | 10.742 | 4.961 | -53.82% | 372.4 | 806.3 | 0% / 100% |
| B4 / KV2048 | 9.455 | 5.747 | -39.22% | 10.829 | 6.124 | -43.44% | 369.4 | 653.1 | 0% / 100% |
| B4 / KV4096 | 9.172 | 6.930 | -24.45% | 10.303 | 7.294 | -29.20% | 388.2 | 548.4 | 0% / 100% |
| B8 / KV512 | 9.075 | 4.851 | -46.54% | 9.844 | 5.233 | -46.84% | 812.7 | 1,528.7 | 0% / 100% |
| B8 / KV2048 | 9.209 | 5.794 | -37.08% | 10.145 | 6.001 | -40.85% | 788.6 | 1,333.1 | 0% / 100% |
| B8 / KV4096 | 9.409 | 6.777 | -27.97% | 10.388 | 7.067 | -31.97% | 770.1 | 1,132.1 | 0% / 100% |
| B16 / KV512 | 9.524 | 5.130 | -46.13% | 11.147 | 5.382 | -51.72% | 1,435.3 | 2,973.1 | 0% / 100% |
| B16 / KV2048 | 9.433 | 6.890 | -26.96% | 10.359 | 7.102 | -31.45% | 1,544.5 | 2,253.0 | 0% / 100% |
| B16 / KV4096 | 10.270 | 9.341 | -9.05% | 12.034 | 9.602 | -20.20% | 1,329.6 | 1,666.3 | 0% / 100% |
| B32 / KV512 | 9.410 | 6.274 | -33.32% | 10.667 | 6.609 | -38.05% | 2,999.9 | 4,842.1 | 0% / 100% |
| B32 / KV2048 | 9.948 | 9.095 | -8.58% | 11.196 | 9.302 | -16.92% | 2,858.1 | 3,440.3 | 0% / 100% |
| B32 / KV4096 | 13.402 | 12.567 | -6.22% | 14.284 | 12.776 | -10.55% | 2,240.3 | 2,504.7 | 0% / 100% |

`FULL_DECODE_ONLY` captured buckets `1,4,8,16,32` in 298.588 ms and used
118,684,160 additional bytes (113.186 MiB); `NONE` captured nothing and used
no graph memory. All 4,800 measured FULL decode steps replayed and all 4,800
exact-bucket decisions hit. The benchmark also compares every autoregressive
completion hash: 62/75 case-repeat hashes matched. This hash is diagnostic,
not a correctness gate, because tolerance-level BF16 logit differences can
cross a sampling boundary and accumulate; hidden-state and logit tolerance
tests provide the graph correctness check.

Reproduce the two policies in separate processes, then validate and render the
comparison:

```bash
PYTHONHASHSEED=0 FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
/tmp/nanovllm-flashinfer-env/bin/python bench_cudagraph.py \
  --model /workspace/aiinfra/models/Qwen3-0.6B --mode none \
  --output benchmark_results/cudagraph/cudagraph_none.json

PYTHONHASHSEED=0 FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
/tmp/nanovllm-flashinfer-env/bin/python bench_cudagraph.py \
  --model /workspace/aiinfra/models/Qwen3-0.6B --mode full_decode_only \
  --output benchmark_results/cudagraph/cudagraph_full_decode_only.json

/tmp/nanovllm-flashinfer-env/bin/python bench_cudagraph.py compare \
  --none benchmark_results/cudagraph/cudagraph_none.json \
  --full benchmark_results/cudagraph/cudagraph_full_decode_only.json \
  --output benchmark_results/cudagraph/cudagraph_comparison.json \
  --markdown-output benchmark_results/cudagraph/cudagraph_comparison.md
```

Raw per-step timings and runtime counters are in
[`cudagraph_none.json`](benchmark_results/cudagraph/cudagraph_none.json) and
[`cudagraph_full_decode_only.json`](benchmark_results/cudagraph/cudagraph_full_decode_only.json).
The validated machine-readable and rendered tables are
[`cudagraph_comparison.json`](benchmark_results/cudagraph/cudagraph_comparison.json)
and [`cudagraph_comparison.md`](benchmark_results/cudagraph/cudagraph_comparison.md).
These figures validate the runtime feature only and are not evidence about LPM,
Same-step Prefix Reuse, or vLLM scheduler performance.

### Eager-only nano-vLLM versus vLLM

`bench_eager_compare.py` runs nano-vLLM and vLLM in separate processes over
the same immutable token-ID trace. This is an end-to-end comparison of the
nano-vLLM cache-aware scheduler and vLLM's default Eager scheduler on two
targeted shared-prefix workloads; it is not an isolated scheduler
microbenchmark.

The comparison validator confirmed identical input token IDs, arrival order
and time, output lengths, request manifests, model path and file hashes,
tokenizer files, seed, ignore-EOS setting, and benchmark script
(`ea3ad3c150a64df358c92b0dcdf3d1c53bc807154ab4e72476e8a451b6edef6f`).
Both runs used the same RTX 5070
(`GPU-d579820b-3886-c645-9f70-5649b0bdf393`) with driver 596.49, Torch
2.11.0+cu128, BF16, and seed 2026 reset before
the measured phase, temperature 1.0, tensor parallel size 1, prefix caching,
chunked prefill, and 16-token blocks. LPM uses `max_num_seqs=4`; In-batch
uses `max_num_seqs=16`. Both result files
report `cudagraph_mode=none` and `enforce_eager=true`: nano-vLLM explicitly
uses `CUDAGraphPolicy.NONE`, while vLLM uses `enforce_eager=True`.

| Workload | Measured requests (+ priming) | Logical KV blocks | Max model / batched tokens | Trace SHA-256 | Manifest SHA-256 |
|---|---:|---:|---:|---|---|
| Long-prefix KV pressure (LPM) | 24 (+3) | 896 | 4,352 / 16,384 | `a3f29bf406cd8cbeff837712e92ba9e66fd1522f2bf4d525d44c2a8fd42e1a53` | `bd4ae18070ec2fd8961642aa50b539941068a8f2ab7d8abc8f7345100dd1906b` |
| In-batch prefix burst | 16 (+0) | 2,240 | 2,304 / 10,240 | `4b6aca655ccd03cd4da344f53cfcac27b9fb654a2d98f8a07f990d56c17c159a` | `1bc44d69e6ea521d48d05178ac6193109aae4558cd06ac8e0bc40fb6346c928c` |

Both traces identify the same Qwen3-0.6B model and tokenizer files:

| File | SHA-256 |
|---|---|
| `config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| `generation_config.json` | `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` |
| `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |

The checked-in measurements compare only P95 TTFT, request throughput, and
total batch completion time. nano-vLLM scheduler counters and vLLM's reported
cached-token counters have different definitions and are retained in the raw
files only; they are not compared across backends.

| Workload | Backend | P95 TTFT (ms) | Throughput (req/s) | Completion (s) |
|---|---|---:|---:|---:|
| Long-prefix KV pressure | nano-vLLM | 4,345.851 | 4.719 | 5.086 |
| Long-prefix KV pressure | vLLM | **3,650.615** | **5.674** | **4.230** |
| In-batch prefix burst | nano-vLLM | 657.895 | 10.646 | 1.503 |
| In-batch prefix burst | vLLM | **283.458** | **16.191** | **0.988** |

vLLM was faster on all three comparable metrics in both RTX 5070 runs. This
result is limited to these traces and software stacks; it must not be
generalized to other workloads or attributed to CUDA Graph, because both
backends were forced to Eager execution.

Cache counters remain backend-native rather than cross-framework comparable.
On the In-batch trace, nano-vLLM records 24,576 same-step hit tokens and 10,240
computed prompt tokens; vLLM reports 24,576 cached prompt tokens and a derived
10,240 prompt-minus-cache count. On the LPM trace, nano-vLLM records 49,152
initial persistent-hit tokens, zero same-step tokens, and 11,359 computed
prompt tokens; vLLM reports 36,864 cached tokens and a derived 23,647. These
numbers are useful diagnostics, but only P95 TTFT, request throughput, and
completion time are validated as definition-equivalent.

#### Reproduction

Generate both deterministic traces with the nano-vLLM environment:

```bash
MODEL_DIR=/workspace/aiinfra/models/Qwen3-0.6B
RESULT_DIR=benchmark_results/eager_compare
NANO_PY=/tmp/nanovllm-flashinfer-env/bin/python

for WORKLOAD_NAME in lpm in-batch; do
  "$NANO_PY" bench_eager_compare.py generate-trace "$WORKLOAD_NAME" \
    --model "$MODEL_DIR" --seed 2026 \
    --output "$RESULT_DIR/$WORKLOAD_NAME.trace.json"
done
```

Run nano-vLLM with its validated FlashInfer 0.6.17 cu129 AOT cache and JIT
disabled. Engine initialization and any prefix-priming phase are outside the
measured interval.

```bash
export CUDA_HOME=/usr/local/cuda
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export FLASHINFER_DISABLE_JIT=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

for WORKLOAD_NAME in lpm in-batch; do
  "$NANO_PY" bench_eager_compare.py run-nano \
    --trace "$RESULT_DIR/$WORKLOAD_NAME.trace.json" \
    --output "$RESULT_DIR/$WORKLOAD_NAME.nano.json"
done
```

The measured vLLM environment used its packaged CUDA 13 toolkit through this
shim. Its FlashInfer 0.6.14 installation has no `flashinfer-jit-cache` AOT
package, so `FLASHINFER_DISABLE_JIT` is intentionally unset: missing kernels
may compile during engine startup, before the measured interval. Do not reuse
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` with untrusted code; this benchmark uses
it only to send the seed-reset callback to its trusted local worker through
`collective_rpc`.

```bash
VLLM_ENV_DIR=/workspace/aiinfra/.venvs/nanovllm-awq
VLLM_PY="$VLLM_ENV_DIR/bin/python"
VLLM_CUDA_DIR="$VLLM_ENV_DIR/lib/python3.10/site-packages/nvidia/cu13"
VLLM_SHIM_DIR=/tmp/vllm-cuda13-shim

mkdir -p "$VLLM_SHIM_DIR/lib64"
ln -sfn "$VLLM_CUDA_DIR/bin" "$VLLM_SHIM_DIR/bin"
ln -sfn "$VLLM_CUDA_DIR/include" "$VLLM_SHIM_DIR/include"
ln -sfn "$VLLM_CUDA_DIR/nvvm" "$VLLM_SHIM_DIR/nvvm"
ln -sfn "$VLLM_CUDA_DIR/lib/libcudart.so.13" \
  "$VLLM_SHIM_DIR/lib64/libcudart.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so \
  "$VLLM_SHIM_DIR/lib64/libcuda.so"

export PATH="$VLLM_SHIM_DIR/bin:$PATH"
export CUDA_HOME="$VLLM_SHIM_DIR"
export FLASHINFER_CUDA_ARCH_LIST=12.0f
unset FLASHINFER_DISABLE_JIT
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export LD_LIBRARY_PATH="$VLLM_CUDA_DIR/lib:$VLLM_ENV_DIR/lib/python3.10/site-packages/torch/lib:$VLLM_SHIM_DIR/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for WORKLOAD_NAME in lpm in-batch; do
  "$VLLM_PY" bench_eager_compare.py run-vllm \
    --trace "$RESULT_DIR/$WORKLOAD_NAME.trace.json" \
    --output "$RESULT_DIR/$WORKLOAD_NAME.vllm.json"
done
```

Finally, validate the fairness contract and produce the comparison files:

```bash
for WORKLOAD_NAME in lpm in-batch; do
  "$NANO_PY" bench_eager_compare.py compare \
    --trace "$RESULT_DIR/$WORKLOAD_NAME.trace.json" \
    --nano-result "$RESULT_DIR/$WORKLOAD_NAME.nano.json" \
    --vllm-result "$RESULT_DIR/$WORKLOAD_NAME.vllm.json" \
    --output "$RESULT_DIR/$WORKLOAD_NAME.comparison.json"
done
```

Raw traces, per-request results, runtime provenance, and validated comparisons:

| Workload | Trace | nano-vLLM | vLLM | Comparison |
|---|---|---|---|---|
| Long-prefix KV pressure | [`lpm.trace.json`](benchmark_results/eager_compare/lpm.trace.json) | [`lpm.nano.json`](benchmark_results/eager_compare/lpm.nano.json) | [`lpm.vllm.json`](benchmark_results/eager_compare/lpm.vllm.json) | [`lpm.comparison.json`](benchmark_results/eager_compare/lpm.comparison.json) |
| In-batch prefix burst | [`in-batch.trace.json`](benchmark_results/eager_compare/in-batch.trace.json) | [`in-batch.nano.json`](benchmark_results/eager_compare/in-batch.nano.json) | [`in-batch.vllm.json`](benchmark_results/eager_compare/in-batch.vllm.json) | [`in-batch.comparison.json`](benchmark_results/eager_compare/in-batch.comparison.json) |
