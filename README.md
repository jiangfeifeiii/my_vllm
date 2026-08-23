# Nano-vLLM-v1

Nano-vLLM-v1 is a compact Qwen3 inference engine for studying and testing a
vLLM-v1-style execution path. It includes paged KV cache, prefix caching,
stable cache-aware LPM, same-step full-block prefix reuse, chunked prefill,
tensor parallelism, pluggable operators, and phase-specialized attention.

The production inference path currently targets NVIDIA CUDA GPUs and model
weights in FP16 or BF16. Linear and tensor-parallel projection layers
intentionally remain on `torch.nn.functional.linear`; GEMM is not part of the
ordinary-operator `CustomOp` layer.

See [Architecture](docs/architecture.md) for the ordinary-operator dispatch rules, attention
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

`attention_mode="unified"` is the default. `PURE_PREFILL` uses FlashInfer's
paged-prefill wrapper, `PURE_DECODE` uses its paged-decode wrapper, and `MIXED`
uses holistic `BatchAttention` where FlashInfer/device capability allows it.
Unsupported mixed configurations (including FlashInfer 0.6.17 on SM120) fall
back to phase-specialized prefill/decode calls writing directly into one
reusable output buffer. Select `"split"` to force that zero-copy mixed path.

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
specialized decode attention and MLP), and final RMSNorm. Logit computation,
sampling, scheduler/postprocessing, and prefix-cache management remain outside.
Runtime coverage can be inspected without a profiler:

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
KV pages, chunked prefill, cache-aware LPM, adaptive SiLU dispatch, and the
FlashInfer normalization/RoPE implementations, run:

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

The example prints holistic-mixed availability/fallback, eager attention-route
counts, captured buckets, capture time, additional graph memory, and replay/
fallback hit counts. Its default workload includes a resumed long prefill next
to active decode requests, so the `MIXED` route is exercised. Add `--debug` to
pause in `pdb` immediately before the first generation call.

Ordinary operators inherit `CustomOp`. During model construction, each module
binds its platform entry point (`forward_cuda`, `forward_cpu`, `forward_xpu`,
or `forward_native`) and selects its concrete implementation. There is no
central implementation registry or name lookup in the Forward hot path.
`auto` is the default; an implementation can still be pinned through the
public `operator_overrides` option without changing layer code:

```python
llm = LLM(
    "/absolute/path/to/Qwen3-model",
    operator_overrides={
        "silu_and_mul": "adaptive_cuda",
        "rms_norm": "flashinfer",
        "fused_add_rms_norm": "flashinfer",
        "rotary_embedding": "flashinfer",
        "kv_cache_store": "native",
    },
)
```

`native` selects the operator's platform-native implementation; exact names
such as `native_torch`, `native_triton`,
`flashinfer`, `custom_cuda`, and `adaptive_cuda` are also accepted. On the
measured RTX 5070 BF16/Qwen3 width-6144 path, adaptive dispatch uses the custom
kernel below 128 rows and FlashInfer from 128 rows; other compatible devices,
dtypes, and widths default to FlashInfer instead of extrapolating that targeted
serving heuristic. Existing fields such as `provider_name`,
`rms_provider_name`, `add_rms_provider_name`, and `kv_store_provider_name`
remain available as diagnostic names for the bound implementations.

### Legacy rollback

The attention backend is independent of ordinary-operator `CustomOp`
dispatch. To restore the original FlashAttention paged path and native
ordinary-operator implementations, select it explicitly and use a block size
divisible by 256:

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
attention and ordinary-operator implementation selection, not a CPU inference
mode.

## Tests

Run the full suite after installation. CUDA/model-dependent tests skip when
their prerequisites are absent:

```bash
python -m pytest -q tests
```

Focused operator, scheduler, and attention coverage:

```bash
python -m pytest -q \
  tests/test_custom_op_dispatch.py \
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
`bench_attention_dispatch.py` measures the current production backend on the
Qwen3-0.6B `P128 + D3` mixed shape and records the asserted route, capability
fallback, exact page metadata, raw repeats, numerical difference, and source
provenance. The older `bench_attention.py` is a model-free raw-wrapper study
for a Qwen3-8B shape. These scripts fail rather than silently changing the
requested workload.

Run the current production Qwen3-0.6B mixed-dispatch protocol with:

```bash
# On the documented SM120 stack, make the safety fallback an explicit assert.
PYTHONHASHSEED=0 FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_DISABLE_JIT=1 \
/tmp/nanovllm-flashinfer-env/bin/python bench_attention_dispatch.py \
  --output /tmp/nanovllm-attention-dispatch.json \
  --expected-route mixed_split
```

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
exact benchmark-script SHA-256. The post-audit runs use source commit
`1544a51` and
`bench_scheduler.py=5f39e7910c2234324460e8342a8a5a7e169a94b7c1ee8222c0625df34d46f23c`.

#### Scheduler LPM

The measured phase contains 12 cold requests followed by 12 followers of three
resident 4096-token prefixes. The first FCFS step schedules `Cold1..Cold4`;
the first LPM step schedules `A1,B1,C1,A2` and consumes the surviving
persistent prefixes before the cold requests can evict them.

| Policy | Initial persistent hits | Same-step hits | Computed prompt | Cached-block evictions | Preemptions | P95 TTFT (ms) | Throughput (req/s) | Completion (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 32,768 | 4,096 | 23,647 | 1,471 | 0 | 5,384.363 | 3.915 | 6.130 |
| LPM | 49,152 | 0 | 11,359 | 704 | 0 | 4,179.009 | 4.919 | 4.879 |

In this targeted run, LPM reduced computed prompt work by 51.96%, cache
evictions by 52.14%, P95 TTFT by 22.39%, and completion time by 20.40%;
request throughput increased by 25.63%. The causal gate—not timing alone—also
requires the first LPM batch to be `A1,B1,C1,A2`.

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
| Same-step OFF | 0 | 0 | 34,816 | 24,576 | 4 | 0 / 0 | 0 | 973.228 | 8.465 | 1.890 |
| Same-step ON | 0 | 24,576 | 10,240 | 0 | 16 | 12 / 1,536 | 0 | 383.414 | 12.666 | 1.263 |

The requested `same_step_reused_requests` count is serialized as
`same_step_reused_request_count`; `same_step_reused_blocks` is the exact
full-block total. Prompt-token conservation is asserted per request and in
aggregate in both files. ON reduced computed prompt work by 70.59%, P95 TTFT
by 60.60%, and completion time by 33.17%; throughput increased by 49.63%.

Raw request/step traces:
[`in_batch_off.json`](benchmark_results/rtx5070/in_batch_off.json) and
[`in_batch_on.json`](benchmark_results/rtx5070/in_batch_on.json).

#### Attention

> Historical scope: the table below predates commit `7c68108` and benchmarks
> raw FlashInfer wrappers for a Qwen3-8B shape. Its “Unified” method means the
> retired all-batch paged-prefill route; it is not a measurement of the current
> BatchType-aware production backend and must not be used as a current dispatch
> performance claim.

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

The historical pure-decode diagnostic also favored the paged-prefill wrapper
on that old Qwen3-8B workload: 7.167986 vs 7.213052 ms in Case 1 and 7.196127
vs 7.426044 ms in Case 2. Current pure-decode dispatch instead uses the
dedicated decode wrapper. Full metadata, correctness, execution order, and
diagnostic repeats for the historical experiment are in
[`attention.json`](benchmark_results/rtx5070/attention.json).

### Runtime feature validation: full-decode CUDA Graph

> Historical scope: the checked-in performance table below predates the
> BatchType-aware decode-wrapper graph change in `7c68108`. Current full-model
> eager/graph correctness and metadata freshness are covered by
> `tests/test_cudagraph_flashinfer.py`; rerun `bench_cudagraph.py` before making
> current performance claims.

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

### Eager-only five-metric LPM comparison

`bench_eager_compare.py` compares the complete nano-vLLM cache-aware LPM
configuration with vLLM's default asynchronous FCFS policy. Both engines run
in separate processes over the same immutable, simultaneous-arrival token-ID
trace. Same-step Prefix Reuse is intentionally excluded from the
cross-framework headline comparison: it remains a functional/internal
ablation rather than an innovation claim.

The LPM stress trace primes three independent 4,096-token prefixes, then
submits 12 cold requests followed by 12 prefix-sharing followers. It uses
`max_num_seqs=4`, `max_num_batched_tokens=16,384`, 16-token blocks, and exactly
896 usable KV blocks, so persistent prefixes are repeatedly exposed to
allocation pressure and LPM has frequent opportunities to change admission
order. The schema-v3 trace SHA-256 is
`96000c13be2e12c4576742c0e9416e143feb1a0a6ac2959a9a846d65b0e60052`;
its request-manifest SHA-256 remains
`bd4ae18070ec2fd8961642aa50b539941068a8f2ab7d8abc8f7345100dd1906b`,
identical to the earlier LPM case.

#### Five comparable metrics

| Metric | nano-vLLM LPM | vLLM FCFS | nano relative to vLLM |
|---|---:|---:|---:|
| Computed prompt tokens | **11,359** | 23,647 | **-51.96%** |
| Cached-block evictions | **704** | 1,465 | **-51.95%** |
| P95 TTFT | 4,767.668 ms | **4,279.198 ms** | +11.42% |
| Request throughput | 4.350 req/s | **4.844 req/s** | -10.22% |
| Total batch completion | 5.518 s | **4.954 s** | +11.38% |

The mechanism improvement exists in the full-module comparison: nano-vLLM
computed 12,288 fewer prompt tokens and evicted 761 fewer cached physical
blocks. It did not translate into a cross-framework latency win in this single
run: nano-vLLM had 11.42% higher P95 TTFT, 10.22% lower request throughput, and
11.38% longer completion time. Computed work and eviction are mechanism
metrics, not a promise that end-to-end time changes proportionally.

Preemption is a control metric rather than a sixth headline metric; both runs
recorded zero preemptions. Throughput and completion are also mathematical
reciprocals for this fixed 24-request batch, so they are two presentations of
one completion-rate outcome rather than independent evidence.

#### Counter definitions and validation

`computed_prompt_tokens` sums the scheduled interval that intersects each
request's prompt before vLLM advances `num_computed_tokens`. It includes prompt
recomputation after any preemption and excludes decode tokens.
`cached_block_eviction_count` increments once when a physical block allocation
actually removes at least one live prefix-cache mapping; cache claims, touches,
and explicit non-allocation invalidations are excluded. These definitions
match nano-vLLM's `SchedulerMetrics` hooks.

vLLM is observed by
[`bench_vllm_metrics.py`](bench_vllm_metrics.py), a benchmark-only subclass of
its existing `AsyncScheduler`. It does not change FCFS order, admission, block
selection, or execution. The timed path adds only Python integer accounting;
measurement snapshots use EngineCore's utility IPC before and after the timed
phase. The observer source SHA-256 is
`c3916058bf7c9611c4f622b69f6df13f2261537c4a6e128cdf014b39c580f461`.

The checked-in result passed all of these conservation checks:

- vLLM direct prompt work, native `local_compute` counter delta, and
  `sum(prompt_len - num_cached_tokens)` all equal 23,647.
- vLLM observed 1,465 destructive cached-block evictions within 1,579 measured
  physical allocations.
- nano-vLLM's initial hits + same-step hits + computed prompt tokens equal all
  60,511 measured prompt tokens.
- Both preemption counters are zero, all 24 requests generated exactly 64
  output tokens, and every prompt hash/order matches the trace.

vLLM internally reserves block 0 as a null block. The runner therefore
configures 897 physical blocks and validates `897 - 1 = 896` usable blocks;
nano-vLLM's measured `BlockManager` exposes 896 directly. This corrects the
older comparison's off-by-one capacity mismatch.

#### Fairness and interpretation boundary

Both official runs used the same RTX 5070
(`GPU-d579820b-3886-c645-9f70-5649b0bdf393`), driver 596.49, Torch
2.11.0+cu128, BF16, seed 2026 reset immediately before measurement,
temperature 1.0, ignore-EOS, tensor/data/pipeline parallel size 1, prefix
caching, chunked prefill, `gpu_memory_utilization=0.7`, and Eager-only
execution. `gpu_memory_utilization` controls the backing allocation, while the
measured scheduler capacity remains fixed at 896 usable blocks. The benchmark
script SHA-256 in both raw results is
`119ac5aeaba88f69378c8b28251a92cf00f30d504d797d3a14fc380817d54787`.

This is a complete-engine comparison, not an isolated proof that LPM alone
caused the timing delta. nano-vLLM used FlashInfer 0.6.17 with the 0.6.17+cu129
AOT cache and JIT disabled; the local vLLM 0.26.1.dev0 build used FlashInfer
0.6.14 without `flashinfer-jit-cache`. The exact mechanism counters do not
depend on that package difference, but timing does. Use the earlier
nano-vLLM LPM-versus-nano-vLLM FCFS ablation for causal LPM claims. This is one
run on one model, workload, and GPU; it must not be generalized or attributed
to CUDA Graph.

#### Reproduction

Generate the deterministic LPM trace with the nano-vLLM environment:

```bash
MODEL_DIR=/workspace/aiinfra/models/Qwen3-0.6B
RESULT_DIR=benchmark_results/eager_compare
NANO_PY=/tmp/nanovllm-flashinfer-env/bin/python

"$NANO_PY" bench_eager_compare.py generate-trace lpm \
  --model "$MODEL_DIR" --seed 2026 \
  --output "$RESULT_DIR/lpm.trace.json"
```

Run nano-vLLM. Initialization and prefix priming are outside the measured
interval:

```bash
export CUDA_HOME=/usr/local/cuda
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export FLASHINFER_DISABLE_JIT=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

"$NANO_PY" bench_eager_compare.py run-nano \
  --trace "$RESULT_DIR/lpm.trace.json" \
  --gpu-memory-utilization 0.7 \
  --output "$RESULT_DIR/lpm.nano.json"
```

The measured vLLM environment uses its packaged CUDA 13 toolkit through a
shim. `FLASHINFER_DISABLE_JIT` is intentionally unset because that environment
has no matching AOT JIT-cache package. Do not enable
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` for untrusted code; here it transports
only the local seed-reset callback.

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

"$VLLM_PY" bench_eager_compare.py run-vllm \
  --trace "$RESULT_DIR/lpm.trace.json" \
  --gpu-memory-utilization 0.7 \
  --output "$RESULT_DIR/lpm.vllm.json"
```

Finally, validate all fairness, provenance, exact-counter, capacity, and
prompt-conservation gates:

```bash
"$NANO_PY" bench_eager_compare.py compare \
  --trace "$RESULT_DIR/lpm.trace.json" \
  --nano-result "$RESULT_DIR/lpm.nano.json" \
  --vllm-result "$RESULT_DIR/lpm.vllm.json" \
  --output "$RESULT_DIR/lpm.comparison.json"
```

Checked-in artifacts:

| Trace | nano-vLLM | vLLM | Validated comparison |
|---|---|---|---|
| [`lpm.trace.json`](benchmark_results/eager_compare/lpm.trace.json) | [`lpm.nano.json`](benchmark_results/eager_compare/lpm.nano.json) | [`lpm.vllm.json`](benchmark_results/eager_compare/lpm.vllm.json) | [`lpm.comparison.json`](benchmark_results/eager_compare/lpm.comparison.json) |

The older `in-batch.*` files in this directory are historical schema-v2
artifacts. They are not accepted by the schema-v3 five-metric comparator and
are not part of the current cross-framework claim.
