# Architecture

Nano-vLLM-v1 keeps the control plane small while separating operator choice,
attention execution, and cache-aware scheduling. The serving path is designed
for NVIDIA CUDA with FP16 or BF16 model weights. CPU tests cover portable
control-plane and reference behavior, but CPU inference and non-NVIDIA GPU
backends are outside the current scope.

## Operator registry

Layers request an operation by logical name through `OperatorResolver`.
Providers register a factory, capability predicate, and integer priority. At
layer construction time the resolver binds one callable; there is no provider
lookup in the token loop.

Selection has three modes:

1. `auto` (the default) chooses the highest-priority provider whose capability
   predicate accepts the device and dtype.
2. `native` considers only names beginning with `native_`, then applies the
   same supported/highest-priority rule.
3. An exact override, such as `flashinfer` or `custom_cuda`, selects only that
   provider and raises an actionable error if it is missing or unsupported.

Optional provider modules are imported before model construction. Import or
binary-loading failures are recorded, native providers remain available, and
`auto` can fall back. An explicitly requested unavailable provider fails
instead of changing the requested implementation.

The current registrations are intentionally minimal:

| Logical operator | Provider | Priority | Implementation and support |
|---|---:|---:|---|
| `silu_and_mul` | `native_torch` | 100 | `torch.nn.functional.silu` + multiply |
| `silu_and_mul` | `flashinfer` | 200 | FlashInfer, CUDA FP16/BF16 |
| `silu_and_mul` | `custom_cuda` | 400 | In-tree `nanovllm._C`, CUDA FP16/BF16 |
| `rms_norm` | `native_torch` | 100 | PyTorch reference path |
| `rms_norm` | `flashinfer` | 200 | FlashInfer, CUDA FP16/BF16 |
| `fused_add_rms_norm` | `native_torch` | 100 | PyTorch reference path |
| `fused_add_rms_norm` | `flashinfer` | 200 | FlashInfer, CUDA FP16/BF16 |
| `rotary_embedding` | `native_torch` | 100 | PyTorch NeoX-style RoPE |
| `rotary_embedding` | `flashinfer` | 200 | FlashInfer, CUDA FP16/BF16 |
| `kv_cache_store` | `native_triton` | 300 | In-tree Triton NHD cache store |

Thus, on a supported CUDA FP16/BF16 installation, `auto` uses the custom CUDA
kernel for SiLU-and-multiply, FlashInfer for normalization and RoPE, and the
native Triton KV-cache store. If the custom extension is not loadable,
SiLU-and-multiply falls through to FlashInfer and then `native_torch`.

Linear projections are deliberately excluded. `ReplicatedLinear`, column-
parallel, QKV, and row-parallel layers continue to call
`torch.nn.functional.linear`; row-parallel output is reduced with the existing
tensor-parallel collective. Provider work must not silently change GEMM.

## Attention backends and B16 cache

`AttentionBackend` has two lifecycle methods:

- `plan(context)` runs once per scheduled batch and prepares shared metadata.
- `forward(q, k, v, k_cache, v_cache, context)` runs once per attention layer.

Each layer first stores its newly produced K/V through the selected
`kv_cache_store` provider. The backend then reads the paged cache, so a mixed
batch can attend to K/V written earlier in the same layer invocation.

### FlashInfer backend

The default `FlashInferAttentionBackend` uses NHD KV pages shaped
`[num_pages, 16, num_kv_heads, head_dim]`. The 16-token page size is the engine
default and is carried consistently by `Config`, `Sequence`, `BlockManager`,
model-runner metadata, cache allocation, and FlashInfer planning.

One backend instance is shared by all transformer layers. For Eager execution
it owns one 64 MiB CUDA `uint8` workspace and two wrappers:

- `BatchPrefillWithPagedKVCacheWrapper` for prefill query lengths of one or
  more tokens;
- `BatchDecodeWithPagedKVCacheWrapper` for the decode suffix, where every
  sequence contributes exactly one query token.

The scheduler naturally emits a packed batch in this order:

```text
[prefill sequences, including a resumed chunk] [running decode sequences]
|<----------- prefill query tokens ----------->|<- one token per sequence ->|
```

`ModelRunner.prepare_model_input` turns that scheduler-provided phase boundary
into one explicit `Context.batch_type`: `PURE_PREFILL`, `PURE_DECODE`, or
`MIXED`. The type describes execution semantics rather than tensor shape. A
single-token chunk continuation remains `PURE_PREFILL`, while a mixed batch in
which every request happens to have `q_len=1` remains `MIXED`. Page tables and
Q/K length relationships continue to describe cache layout and metadata; they
are not used to infer the batch type.

The model runner records `num_prefill_seqs`, `num_prefill_tokens`, and
`num_decode_tokens` while building one set of page CSR metadata. In the default
`attention_mode="unified"`, `plan` passes the full metadata to the paged
prefill wrapper and every layer runs the complete packed `[P | D]` query in
one call. In `attention_mode="split"`, `plan` slices the metadata at the phase
boundary and prepares both phase-specialized wrappers.

The split path owns a lazily grown contiguous output buffer. Prefill and decode
write directly into its `[P]` and `[D]` slices with FlashInfer's `out=` API;
there are no per-phase output tensors, `torch.cat`, or full-output copy. The
decode slice is zeroed before use for compatibility with FlashInfer backends
that require zero-initialized caller output. This scratch reuse relies on the
current inference-only, sequential layer execution on one CUDA stream. Pure
prefill, pure decode, and mixed batches use the same backend object. GQA is
represented by independent query-head and KV-head counts, with `num_q_heads`
divisible by `num_kv_heads`.

The cacheless model-memory warmup occurs before KV allocation and uses the
existing ragged `flash_attn_varlen_func` path. Split mode also allocates its
reusable output scratch during warmup so KV capacity profiling includes that
memory. Warmup is not a serving fallback: a serving batch with page metadata
must be planned before any layer runs.

### Legacy rollback

`LegacyFlashAttentionBackend` preserves the original
`flash_attn_varlen_func` behavior, including its paged `block_table` path. Its
cache block size must be divisible by 256. Select it with
`attention_backend="legacy"` and `kvcache_block_size=256`; attention backend
selection is separate from per-operator overrides. Legacy attention accepts
only `attention_mode="unified"`. Both paths store K/V in NHD layout before
attention.

## Batch execution and CUDA Graph dispatch

The scheduler decides which requests run and how many tokens they receive. It
does not select an attention topology or CUDA Graph route. From the scheduler's
prefill/decode boundary, `ModelRunner.prepare_model_input` records
`Context.batch_type`. The attention mode chooses unified or phase-specialized
attention, while the CUDA Graph policy determines whether the transformer body
is eligible for replay. These are independent decisions.

The user-facing `CUDAGraphPolicy` has two values: `NONE` and
`FULL_DECODE_ONLY`. `enforce_eager=True` is retained as a compatibility mapping
to `NONE`. The dispatcher records its per-batch decision separately as
`RuntimeExecutionMode.EAGER` or `RuntimeExecutionMode.FULL_GRAPH`.

| Policy and runtime conditions | Selected mode |
|---|---|
| `NONE` | `EAGER` |
| `FULL_DECODE_ONLY` + `PURE_DECODE` + unified FlashInfer capability + tensor parallel size 1 + one query token per request + exact captured bucket + valid fixed-buffer metadata | `FULL_GRAPH` |
| `FULL_DECODE_ONLY` with any condition above unmet | `EAGER` fallback |

The dispatcher reads the backend's `supports_full_decode_graph` capability;
it does not infer support from a backend-name string. Unified FlashInfer is the
only backend that currently advertises the capability. Split FlashInfer and
legacy FlashAttention always execute Eager. Pure prefill and mixed batches also
execute Eager even when every packed query happens to have length one.

Graph buckets default to exact decode batch sizes `1, 2, 4, 8, 16, 32, 64` and
are configurable through `cudagraph_batch_sizes`. Initialization omits buckets
larger than either `max_num_seqs` or `max_num_batched_tokens`. Runtime selection
never pads a smaller batch to a larger graph; a missing exact bucket is an
Eager fallback. Invalid tensor dtype/device/shape, page-index capacity overflow,
invalid page bounds, and malformed slot metadata likewise fall back instead of
rejecting the scheduled batch.

### Full-decode graph state lifecycle

For each retained bucket, `ModelRunner` owns one `DecodeGraphState` containing:

- one graph-aware FlashInfer unified wrapper with a fixed batch size;
- fixed-address input IDs, positions, slot mapping, and page CSR buffers;
- fixed-address attention and final hidden-state outputs;
- page-index capacity and the captured `torch.cuda.CUDAGraph`.

The graph-aware wrappers are separate from the Eager wrappers. They share a
dedicated, stable 64 MiB graph workspace because graph buckets execute
serially; each bucket still has independent fixed metadata and output buffers.
Their lifetime is the `ModelRunner` lifetime and they are released by
`ModelRunner.exit`, which explicitly calls `CUDAGraph.reset()` before releasing
the wrappers and fixed buffers.

With the currently pinned PyTorch 2.11 and FlashInfer 0.6.17 stack, only one
full-decode capture session is initialized per CUDA process. Reinitializing a
multi-bucket graph-wrapper set after teardown can hang or access stale CUDA
state. A later `FULL_DECODE_ONLY` engine in the same process therefore emits a
warning and falls back to Eager; a fresh process is required to capture again.

Initialization follows this order:

```text
load extensions/providers and run an Eager model warmup
    -> allocate per-bucket graph wrappers and fixed buffers
    -> allocate the KV-cache pool
    -> warm and capture every retained bucket
```

Capture uses exact-size synthetic decode inputs and a non-writing slot mapping.
Those startup inputs are not dummy runtime requests and are not graph-bucket
padding. Imports, provider resolution, kernel preparation, workspace creation,
and dynamic tensor allocation all happen before capture.

At serving time the selected paths are:

```text
Scheduler output
    -> prepare packed tensors, page metadata, and BatchType
    -> dispatch
       EAGER: FlashInfer plan -> model body -> LM head
       FULL_GRAPH:
           graph-aware FlashInfer plan and fixed-metadata update (outside graph)
           -> input ID / position / slot copies (outside graph)
           -> CUDA Graph replay
           -> LM head (outside graph)
```

FlashInfer `plan()` remains outside capture because KV length, page indices,
and last-page length can change on every decode step. Planning updates the
wrapper's fixed-address buffers and kernel plan before replay; only `run()`
inside each layer consumes that stable state.

The captured transformer-body boundary is:

```text
embedding -> all transformer layers (unified attention and MLP) -> final RMSNorm
```

Logit computation/LM head, sampling, scheduler postprocessing, request
completion, and prefix-cache management remain outside. The implementation does
not support prefill or mixed full graphs, split-attention graphs, bucket
padding, tensor-parallel graphs, piecewise graphs, speculative-decoding graphs,
or whole-serving-loop capture.

`ModelRunner.get_cudagraph_stats()` exposes the captured bucket sizes, capture
time, additional allocated memory, `full_graph_replay_steps`,
`eager_fallback_steps`, `graph_bucket_hits`, and `graph_bucket_misses`. These
counters describe runtime coverage only. CUDA Graph measurements are an
internal `NONE` versus `FULL_DECODE_ONLY` runtime validation and must not be
mixed into Eager-only cross-framework scheduler conclusions.

## Cache-aware scheduler

The scheduler owns three request states:

- `waiting`: new or preempted requests eligible for prefix matching;
- `chunked_req`: at most one partially completed prompt, resumed before new
  waiting work after decode reservations;
- `running`: prompts that completed prefill and now contribute one decode
  token per step.

Each scheduling step preserves decode capacity first, then follows this
planning/commit pipeline:

```text
match -> rank -> admit -> protect -> allocate
```

1. **Match.** `BlockManager.match_prefix` walks full prompt blocks through a
   chained xxHash64 index. It validates both hash and token content and can
   match blocks that are currently used or cached-free.
2. **Rank.** Temporary-prefix detection first scans the original FCFS waiting
   order. With the SGLang-compatible defaults, requests with at most 32 real
   cached tokens are checked, and a 32-token temporary match marks later
   requests as followers. Ordinary requests are then sorted by longest real
   persistent match. All followers share one lower priority, so their internal
   order remains FCFS. `enable_lpm=False` disables both LPM and temporary
   deprioritization; `enable_in_batch_prefix_deprioritization=False` keeps LPM
   while disabling only the temporary policy.
3. **Admit.** The scheduler jointly checks sequence slots, remaining token
   budget, decode reservations, and required KV blocks. Only persistent cache
   matches reduce token or KV budgets. Without chunked prefill, the uncached
   suffix after persistent matching must fit the configured batch budget or
   scheduling raises an actionable error; with chunking enabled, one prompt
   may consume a partial chunk and becomes `chunked_req`.
4. **Protect.** All persistent matched cached-free blocks for admitted
   requests are claimed before any allocation. This prevents a later
   allocation from resetting a block selected during the read-only planning
   pass.
5. **Allocate.** Every admitted request allocates its own uncached suffix.
   A follower admitted in the leader's step remains a cold prefill: temporary
   matches never affect cached-token counts, block tables, claims, or attention
   metadata. With limited budget it normally waits, then reuses the leader's
   committed full blocks through the ordinary persistent index on a later
   step. Partial blocks remain unpublished until a later append completes
   them.

Decode requests are preempted from the running tail only when the one-token
decode reservation or its boundary KV block cannot fit. Decode work has
priority over a paused chunk; a paused chunk in turn is considered before new
waiting requests. A step returns `prefill_seqs + decode_seqs`, which is the
`[P | D]` order consumed directly by the split FlashInfer wrappers.

After sampling, completed or stopped requests release their block references.
A fully prefetched prompt moves to `running`; the sole incomplete prefill is
retained as `chunked_req`. Full cached blocks remain hash-addressable after
their reference count reaches zero and can be protected and reclaimed by a
future matching request.

## Correctness and measurement rules

- Model execution is NVIDIA CUDA FP16/BF16 only. Unsupported dtypes or an
  unavailable explicitly selected provider fail early.
- FlashInfer page metadata is CUDA `int32`, one-dimensional CSR metadata; the
  decode suffix must have query length one for every sequence.
- Prefix hashes are candidate keys, not proof of equality: token contents and
  block lifecycle state are checked before reuse.
- Performance comparisons must use the same model, dtype, prompt/output
  distribution, batch limits, warmup, backend, and block size. No performance
  number belongs in project documentation until its command, environment, and
  commit are recorded in the README result table.
