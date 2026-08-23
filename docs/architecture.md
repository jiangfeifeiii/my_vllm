# Architecture

Nano-vLLM-v1 keeps the control plane small while separating operator choice,
attention execution, and cache-aware scheduling. The serving path is designed
for NVIDIA CUDA with FP16 or BF16 model weights. CPU tests cover portable
control-plane and reference behavior, but CPU inference and non-NVIDIA GPU
backends are outside the current scope.

## Ordinary operators and `CustomOp`

RMSNorm, RoPE, SiLU-and-multiply, and KV-cache store inherit `CustomOp`.
`ModelRunner` creates one `CustomOpConfig` containing the platform, model
dtype, and public `operator_overrides`; it passes that immutable configuration
through model construction. Each operator then binds its platform entry point
(`forward_cuda`, `forward_cpu`, `forward_xpu`, or `forward_native`) once in its
constructor. A CUDA entry point owns its implementation choice among in-tree
CUDA/Triton, FlashInfer, and native Torch callables. `forward()` invokes the
already-bound method, so there is no central registry, capability-table scan,
or dynamic name lookup in the Forward hot path.

Selection has three forms:

1. `auto` (the default) lets that operator choose its supported CUDA fast path
   and fall back to its native reference implementation when optional code is
   unavailable.
2. `native` selects the operator's platform-native path. Exact native names
   such as `native_torch` and `native_triton` remain available where supported.
3. An exact override such as `flashinfer`, `custom_cuda`, or `adaptive_cuda`
   requests that implementation and raises an actionable error when it is
   unavailable or incompatible.

Optional CUDA modules are probed before model construction. Import or binary-
loading failures are recorded so `auto` can fall back, while an explicitly
requested unavailable implementation fails instead of silently changing the
request. The current implementation choices are intentionally small:

| Logical operator | Implementation | Implementation and support |
|---|---|---|
| `silu_and_mul` | `native_torch` | `torch.nn.functional.silu` + multiply |
| `silu_and_mul` | `flashinfer` | FlashInfer, CUDA FP16/BF16 |
| `silu_and_mul` | `custom_cuda` | In-tree `nanovllm._C`, CUDA FP16/BF16 |
| `silu_and_mul` | `adaptive_cuda` | Shape/device dispatch between FlashInfer and the in-tree CUDA kernel |
| `rms_norm` | `native_torch` | PyTorch reference path |
| `rms_norm` | `flashinfer` | FlashInfer, CUDA FP16/BF16 |
| `fused_add_rms_norm` | `native_torch` | PyTorch reference path |
| `fused_add_rms_norm` | `flashinfer` | FlashInfer, CUDA FP16/BF16 |
| `rotary_embedding` | `native_torch` | PyTorch NeoX-style RoPE |
| `rotary_embedding` | `flashinfer` | FlashInfer, CUDA FP16/BF16 |
| `kv_cache_store` | `native_triton` | In-tree Triton NHD cache store |

Thus, on a supported CUDA FP16/BF16 installation, `auto` binds one adaptive
SiLU callable at construction. On the measured RTX 5070, BF16 Qwen3 width 6144
uses the in-tree CUDA kernel below 128 rows and FlashInfer from 128 rows onward.
This is a targeted serving heuristic: measurements show a material custom-
kernel advantage for decode-sized batches and a FlashInfer advantage as token
parallelism grows, while the exact boundary is sensitive to clock/workload
conditions. The rule is therefore not extrapolated: other devices, dtypes, and
FlashInfer-compatible widths use FlashInfer, while incompatible layouts fall
back to the in-tree CUDA kernel. This adds only an implementation-local host-
side shape branch in the token loop.

Normalization and RoPE use FlashInfer. The public fused-add RMSNorm operation
keeps its out-of-place contract, while Qwen3 calls an internal in-place form
only where both incoming tensor values are dead, eliminating two clones per
eligible normalization. KV-cache store remains the existing Triton kernel:
FlashInfer append requires different batch/position metadata and would disturb
the `slot_mapping=-1` CUDA Graph contract. Linear projections remain
`torch.nn.functional.linear` (cuBLAS); changing GEMM or sampling would require
broader model/RNG semantics work and is outside this low-risk optimization.

## Attention backend registry and neutral metadata

Attention is a complete, stateful subsystem rather than an ordinary
`CustomOp`. The static `AttentionBackendRegistry` contains two implementations:
`FlashAttentionBackend` (Dao FlashAttention) and `FlashInferBackend`. Every
backend implements the same lifecycle:

- `supports(...)` checks whether the complete backend can serve the requested
  device, dtype, block size, and mode;
- `build_metadata(common)` converts backend-neutral input into private kernel
  metadata;
- `build_plan(common, metadata)` classifies the batch and prepares exactly one
  explicit `AttentionPlan` for the scheduled step;
- `forward(q, k, v, k_cache, v_cache, plan)` executes that plan once per layer.

`attention_backend="auto"` is the default. When `block_size % 256 == 0` and
Dao FlashAttention is usable, the registry selects `FlashAttentionBackend`;
otherwise it selects `FlashInferBackend`. If neither complete backend is
usable, initialization fails with the support reason for both candidates.
Explicit requests never silently switch to another backend. New complete
backends can be registered without adding scheduler or transformer-layer
branches.

`ModelRunner` knows only `CommonAttentionMetadata`: prefill/decode sequence and
token counts, `query_start_loc`, `seq_lens`, `block_tables`, `slot_mapping`,
maximum query/KV lengths, and trusted host block counts. It does not build
FlashInfer page CSR or Dao-specific offsets. The selected backend converts this
common object and stores the resulting plan in the execution context; attention
layers consume the opaque plan. Each layer first stores its new K/V through the
bound `kv_cache_store` `CustomOp`, then the backend reads the paged cache, so a
mixed batch can attend to K/V written earlier in the same layer invocation.

### Explicit batch plan and FlashInfer routes

The scheduler emits one packed batch in this order:

```text
[prefill sequences, including a resumed chunk] [running decode sequences]
|<----------- prefill query tokens ----------->|<- one token per sequence ->|
```

The scheduler and model runner preserve that boundary as counts; they do not
choose a kernel. The selected backend's `build_plan` is the only classification
point. Zero prefill sequences means `PURE_DECODE`, zero decode sequences means
`PURE_PREFILL`, and nonzero counts on both sides mean `MIXED`. Therefore a
single-token chunk continuation is still prefill, and a mixed batch remains
mixed even when every query length happens to be one. `AttentionPlan` records
that `BatchType`, the backend-owned metadata, and one explicit route.

`FlashInferBackend` converts dense common block tables to CUDA `int32` page CSR
inside `build_metadata`. With the default 16-token NHD KV pages it owns one
paged-prefill wrapper and one paged-decode wrapper. Its Eager plans are:

| Batch type | Route | Execution |
|---|---|---|
| `PURE_PREFILL` | `prefill` | paged-prefill wrapper |
| `PURE_DECODE` | `decode` | paged-decode wrapper |
| `MIXED` | `mixed_unified` | FlashInfer `BatchAttention`, when usable |
| `MIXED` | `mixed_split` | specialized prefill and decode wrappers |

`attention_mode="unified"` permits `mixed_unified` and automatically falls back
to `mixed_split` when the unified wrapper is unavailable;
`attention_mode="split"` forces the split plan. The current FlashInfer version
has a temporary, centralized gate for device names containing `RTX 5070`.
It does not infer that every SM120 device is unsupported. A TODO marks this
product-name check for replacement with a general runtime probe supplied by
FlashInfer or CUDA capability testing.

All FlashInfer Eager routes use one lazily grown contiguous output buffer.
`mixed_split` writes prefill and decode results directly into its `[P]` and
`[D]` slices, so it creates no per-phase result tensors and performs no
`torch.cat` or whole-output copy. `mixed_unified` also reuses a backend-owned
FP32 LSE buffer. Scratch reuse relies on inference-only, sequential layer
execution on one CUDA stream. GQA is represented by independent query-head and
KV-head counts, with `num_q_heads` divisible by `num_kv_heads`.

Cacheless model-memory warmup uses the same common metadata contract and a
backend-independent varlen implementation. A cached serving batch must carry a
backend-built plan before any layer runs.

### Dao FlashAttention backend and compatibility alias

`FlashAttentionBackend` hides `flash_attn_varlen_func`, its cumulative KV
offsets, and paged `block_table` call behind the same metadata/plan/forward
interface. Dao paged attention requires a cache block size divisible by 256 and
supports only `attention_mode="unified"` in this engine.
`attention_backend="legacy"` remains a configuration compatibility alias for
`"flashattention"`; it does not register or instantiate a third backend.
Attention selection remains independent of ordinary-operator overrides. Both
complete backends store K/V in NHD layout before attention.

## Batch execution and CUDA Graph dispatch

The scheduler decides which requests run and how many tokens they receive. It
does not select an attention topology or CUDA Graph route.
`ModelRunner.prepare_model_input` preserves the scheduler's prefill/decode
boundary in `CommonAttentionMetadata`; the selected backend records `BatchType`
and route in `AttentionPlan`. The attention mode controls whether FlashInfer
mixed batches may use `mixed_unified` or are forced to `mixed_split`, while
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
Dao FlashAttention always execute Eager. Pure prefill and mixed batches also
execute Eager even when every packed query happens to have length one.

Graph buckets default to exact decode batch sizes `1, 2, 4, 8, 16, 32, 64` and
are configurable through `cudagraph_batch_sizes`. Initialization omits buckets
larger than either `max_num_seqs` or `max_num_batched_tokens`. Runtime selection
never pads a smaller batch to a larger graph; a missing exact bucket is an
Eager fallback. Invalid tensor dtype/device/shape, page-index capacity overflow,
invalid page bounds, and malformed slot metadata likewise fall back instead of
rejecting the scheduled batch.

### Full-decode graph state lifecycle

For each retained bucket, `ModelRunner` owns one outer `DecodeGraphState`
containing fixed-address model inputs, the final hidden-state output, the graph
exec, and an opaque attention state created by the selected backend. For
FlashInfer, that backend-owned state contains the fixed-batch decode wrapper,
private query/page CSR buffers, attention output, and page-index capacity.

The model runner asks the backend whether a common plan fits, then asks it to
update, activate, and deactivate that state; it never reaches into FlashInfer
wrapper or page-buffer fields. Graph-aware wrappers are separate from Eager
wrappers and share a stable backend workspace because graph buckets execute
serially. Their lifetime is the engine lifetime. `ModelRunner.exit` explicitly
resets each `CUDAGraph` before releasing the opaque backend state and model
buffers.

With the currently pinned PyTorch 2.11 and FlashInfer 0.6.17 stack, only one
full-decode capture session is initialized per CUDA process. Reinitializing a
multi-bucket graph-wrapper set after teardown can hang or access stale CUDA
state. A later `FULL_DECODE_ONLY` engine in the same process therefore emits a
warning and falls back to Eager; a fresh process is required to capture again.

Initialization follows this order:

```text
load optional ordinary-operator implementations and run an Eager model warmup
    -> allocate per-bucket graph wrappers and fixed buffers
    -> allocate the KV-cache pool
    -> warm and capture every retained bucket
```

Capture uses exact-size synthetic decode inputs and a non-writing slot mapping.
Those startup inputs are not dummy runtime requests and are not graph-bucket
padding. Imports, CustomOp construction-time binding, kernel preparation, workspace creation,
and dynamic tensor allocation all happen before capture.

At serving time the selected paths are:

```text
Scheduler output
    -> prepare packed tensors and CommonAttentionMetadata
    -> selected backend builds backend metadata and AttentionPlan
    -> dispatch
       EAGER: model body -> LM head
       FULL_GRAPH:
           backend-owned graph plan and fixed-metadata update (outside graph)
           -> input ID / position / slot copies (outside graph)
           -> CUDA Graph replay
           -> LM head (outside graph)
```

Backend `build_plan()` and full-decode graph-state update remain outside
capture because KV length, page indices, and last-page length can change on
every decode step. FlashInfer updates its private fixed-address buffers and
kernel plan before replay; only wrapper `run()` inside each layer consumes that
stable state.

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
  unavailable explicitly selected implementation fail early.
- Common attention tensors are CUDA `int32`; the decode suffix has one query
  token per sequence. FlashInfer alone converts dense block tables to its
  one-dimensional page CSR representation.
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
