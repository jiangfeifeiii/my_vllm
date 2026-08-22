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
| `silu_and_mul` | `adaptive_cuda` | 500 | Shape/device dispatch between FlashInfer and the in-tree CUDA kernel |
| `rms_norm` | `native_torch` | 100 | PyTorch reference path |
| `rms_norm` | `flashinfer` | 200 | FlashInfer, CUDA FP16/BF16 |
| `fused_add_rms_norm` | `native_torch` | 100 | PyTorch reference path |
| `fused_add_rms_norm` | `flashinfer` | 200 | FlashInfer, CUDA FP16/BF16 |
| `rotary_embedding` | `native_torch` | 100 | PyTorch NeoX-style RoPE |
| `rotary_embedding` | `flashinfer` | 200 | FlashInfer, CUDA FP16/BF16 |
| `kv_cache_store` | `native_triton` | 300 | In-tree Triton NHD cache store |

Thus, on a supported CUDA FP16/BF16 installation, `auto` binds one adaptive
SiLU callable at layer construction. On the measured RTX 5070, BF16 Qwen3
width 6144 uses the in-tree CUDA kernel below 128 rows and FlashInfer from 128
rows onward. This is a targeted serving heuristic: measurements show a
material custom-kernel advantage for decode-sized batches and a FlashInfer
advantage as token parallelism grows, while the exact boundary is sensitive to
clock/workload conditions. The rule is therefore not extrapolated: other
devices, dtypes, and FlashInfer-compatible widths use FlashInfer, while
incompatible layouts fall back to the in-tree CUDA kernel. This adds only a
host-side shape branch in the token loop, not a registry lookup.

Normalization and RoPE use FlashInfer. The public fused-add RMSNorm provider
keeps its out-of-place contract, while Qwen3 calls an internal in-place form
only where both incoming tensor values are dead, eliminating two clones per
eligible normalization. KV-cache store remains the existing Triton kernel:
FlashInfer append requires different batch/position metadata and would disturb
the `slot_mapping=-1` CUDA Graph contract. Linear projections remain
`torch.nn.functional.linear` (cuBLAS); changing GEMM or sampling would require
broader model/RNG semantics work and is outside this low-risk optimization.

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
it owns one zero-initialized 64 MiB CUDA `uint8` workspace and two always-
available wrappers:

- `BatchPrefillWithPagedKVCacheWrapper` for prefill query lengths of one or
  more tokens;
- `BatchDecodeWithPagedKVCacheWrapper` for the decode suffix, where every
  sequence contributes exactly one query token.

In unified mode, supported devices additionally create FlashInfer 0.6.17's
holistic `BatchAttention` wrapper for mixed batches. That upstream wrapper owns
384 MiB of CUDA float workspace, 8 MiB of CUDA integer workspace, and 8 MiB of
pinned host workspace. Split mode never creates it. FlashInfer 0.6.17 hard-
codes two cooperative CTAs per SM for head dimension 128; this launch is not
valid on the tested SM120 RTX 5070, so SM120 is explicitly capability-gated to
the zero-copy split fallback instead of failing during serving. Head dimensions
above FlashInfer's 256-element limit use the same fallback.

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
`num_decode_tokens` while building one set of page CSR metadata. Eager dispatch
uses the explicit batch type: `PURE_PREFILL` plans only the paged-prefill
wrapper and `PURE_DECODE` plans only the paged-decode wrapper. For `MIXED`, the
default `attention_mode="unified"` uses holistic `BatchAttention` when the
runtime capability gate passes and otherwise uses the zero-copy specialized
composition. `attention_mode="split"` always uses that composition.

All Eager routes write through FlashInfer's `out=` API into one lazily grown
contiguous output buffer. The mixed split route writes its `[P]` and `[D]`
slices directly; there are no per-phase output tensors, `torch.cat`, or full-
output copy. Holistic mixed attention also reuses one FP32 LSE buffer. The
standard AOT decode backend overwrites caller output, so it does not zero that
slice on every layer; only the shared float workspace is zero-initialized
before first decode use. Scratch reuse relies on inference-only, sequential
layer execution on one CUDA stream. GQA is represented by independent query-
head and KV-head counts, with `num_q_heads` divisible by `num_kv_heads`.

For cached FlashInfer serving, `ModelRunner` transfers only page CSR metadata,
slot mappings, query offsets, and logits indices. Legacy-only KV offsets,
context lengths, and dense block tables are not constructed. Cacheless startup
warmup retains the ragged FlashAttention metadata, while the legacy backend
builds only its legacy representation. ModelRunner-produced page metadata also
carries trusted host page counts, avoiding device `.item()`/`torch.all()`
round trips in per-step planning without weakening validation for externally
constructed contexts.

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
`Context.batch_type`. The attention mode controls whether mixed batches may use
holistic attention or are forced to the phase-specialized composition, while
the CUDA Graph policy determines whether the transformer body is eligible for
replay. These are independent decisions.

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

- one graph-aware FlashInfer decode wrapper with a fixed batch size;
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
embedding -> all transformer layers (decode attention and MLP) -> final RMSNorm
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
ranking/commit pipeline:

```text
Waiting snapshot
    -> initial persistent match
    -> stable LPM rank (order fixed)
    -> per-request latest match -> capacity check -> claim -> allocate/publish
```

1. **Initial match.** `BlockManager.match_prefix` walks full prompt blocks
   through the persistent chained-xxHash64 index. It validates the candidate
   hash, exact token contents, block state, and reference count, and can match
   blocks that are currently used or cached-free. This read-only pass records
   the cache state visible at the beginning of the step.
2. **Stable LPM rank.** With `enable_lpm=True` (the default), requests are
   sorted by descending initial persistent-match length. Python's stable sort
   preserves FCFS order for equal matches. The resulting order is fixed before
   any request is committed. A cache hit acquired later in the same step cannot
   re-sort the request or move it ahead of an earlier-ranked request.
3. **Sequential commit.** With `enable_same_step_prefix_reuse=True` (the
   default), each ranked request performs a fresh lookup against the latest
   persistent index, checks token/sequence/KV capacity without mutation, claims
   the complete matched prefix, and allocates its uncached suffix. Every newly
   formed full block is published immediately, so the next ranked request can
   match it in the same scheduler step. No destructive BlockManager operation
   occurs between one request's latest lookup and claim.
4. **Capacity and strict priority.** Admission capacity is the number of
   matched cached-free blocks that claim will remove from the free queue plus
   the number of new physical blocks required by the uncached suffix. Blocks
   reserved for running Decode and the selected `chunked_req` remain excluded.
   If the current request cannot fit, the scheduler stops rather than bypassing
   it for a lower-ranked request. Without chunked prefill, an uncached prompt
   larger than the batch token limit raises an actionable error; with chunking,
   at most one partial prompt remains `chunked_req`.
5. **Frozen OFF baseline.** `enable_same_step_prefix_reuse=False` retains the
   former frozen-plan admission behavior for ablation: only initial persistent
   matches determine cached tokens and resource needs, all admitted cached-free
   prefixes are protected before allocation, and blocks published in this step
   become visible to Waiting requests only on a later step. Persistent prefix
   caching and LPM remain enabled unless independently disabled.

`enable_lpm=False` restores stable FCFS ranking but does not disable Same-step
Prefix Reuse. Conversely, disabling Same-step Prefix Reuse does not disable
LPM. The removed Temporary Deprioritization index and follower soft-priority
policy are not part of the default admission path.

Each `Sequence` caches one chained hash per complete token block. Prompt hashes
are built at construction, and `append_token` adds a hash only when it completes
a new full block. Lookup, claim validation, and publication reuse these values,
but a hash remains only a candidate key: BlockManager still compares exact
token IDs before reuse. A partial block has no cached hash entry, retains
physical hash `-1`, and is not inserted into the persistent index until a later
append fills it. Same-step reuse therefore covers only block-aligned prefixes;
unaligned tails are computed independently.

Exact token equality on one block is not sufficient to prove a chained prefix
under an adversarial hash collision. Every published block therefore carries an
immutable canonical lineage identity plus its parent identity. A duplicate
publication inherits the existing identity only when hash, exact tokens, and
parent lineage all agree; this keeps valid descendants reachable when the same
content occupies a different physical page. A collision or destructive page
reuse receives a new physical-generation identity, so an old child cannot be
spliced onto the new parent. Claiming a cached-free page preserves its identity;
only destructive reset increments the physical generation. With the current
single-owner hash index, collisions may cause a safe miss, never a cross-lineage
hit.

The scheduler exposes initial persistent-hit blocks separately from same-step
hit blocks. For an admitted request, initial hits are blocks that were already
present in the initial snapshot and were still claimed; same-step hits are the
additional blocks in its latest match. Requests left Waiting are not counted.
Token metrics multiply those block counts by the configured block size.
`computed_prompt_tokens` counts prompt tokens actually scheduled as new work;
`same_step_reused_request_count` (the requested
`same_step_reused_requests` count) counts requests with a positive same-step hit;
and `same_step_reused_blocks` sums only the blocks gained after the initial
snapshot. Keeping these categories separate prevents LPM cache survival from
being attributed to same-step publication. For the targeted grouped-prefix
workload, `duplicate_prefill_tokens` is the computed shared-prefix work beyond
one required copy per prefix group;
`first_step_prefill_admission_count` records how many Prefill requests entered
the first Forward.

Decode requests are preempted from the running tail only when the one-token
decode reservation or its boundary KV block cannot fit. Decode work has
priority over a paused chunk; a paused chunk in turn is considered before new
waiting requests. A step returns `prefill_seqs + decode_seqs`, which is the
`[P | D]` order consumed directly by both attention modes. Same-step admission
does not change Decode policy, the single-`chunked_req` rule, ModelRunner token
packing, mixed-batch layout, AttentionBackend selection, or CUDA Graph routing.

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
- Prefix hashes are candidate keys, not proof of equality: token contents,
  canonical parent lineage, and block lifecycle state are checked before reuse.
- Admission fit checks happen before mutation and stop the lower-ranked Waiting
  tail when the current request cannot fit. A stale plan, collision validation
  failure, or impossible internal invariant raises immediately; the scheduler
  does not attempt transactional rollback or silently retry a different order.
- Performance comparisons must use the same model, dtype, prompt/output
  distribution, batch limits, warmup, backend, and block size. No performance
  number belongs in project documentation until its command, environment, and
  commit are recorded in the README result table.
