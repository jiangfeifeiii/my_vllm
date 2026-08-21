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

One backend instance is shared by all transformer layers. It owns one 64 MiB
CUDA `uint8` workspace and two available wrappers:

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

Neither attention backend currently advertises CUDA Graph support. The
`enforce_eager` option therefore does not turn phase-specialized attention
into a captured graph implicitly.

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
