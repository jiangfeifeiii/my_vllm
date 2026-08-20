# Operator baseline audit

This document freezes the operator-level baseline before introducing the
multi-provider registry. It describes code that exists in this repository; it
does not imply that a FlashInfer provider is currently available.

## Baseline environment

The audit was performed at repository HEAD `318567e` (`Fix Transformers 5
compatibility`) with the following local runtime:

- Python compatibility declared by the project: `>=3.10,<3.13`.
- PyTorch: `2.11.0+cu128`; CUDA runtime: 12.8.
- CUDA compiler: 12.8 (`nvcc` 12.8.93).
- GPU: one NVIDIA GeForce RTX 5070, compute capability 12.0.
- FlashAttention: `2.8.3.post1`.
- FlashInfer: not installed and not declared in `pyproject.toml`.

Consequently, native CPU operator tests and the installed FlashAttention API
can be checked now. FlashInfer call signatures must be version-pinned and
validated after installation before any FlashInfer provider is enabled.

## Current implementations

| Semantic operator | Current implementation and public call | Provider decision | FlashInfer equivalent visible in local code |
| --- | --- | --- | --- |
| SiLU-and-multiply | `nanovllm/layers/activation.py:6-14`: `SiluAndMul.forward(x)`. It splits the last dimension in half and evaluates `F.silu(gate) * up` under `torch.compile`. | Stateless. Register the existing module factory or eager callable directly as `native_torch`; do not create another module class. | No locally installed API can be inspected. The local vLLM source does not use a general FlashInfer SiLU-and-multiply function. A possible FlashInfer activation provider must remain unavailable until its pinned package exposes and passes an interface test. |
| RMSNorm | `nanovllm/layers/layernorm.py:16-26`: `RMSNorm.rms_forward(x)`. Reduction is performed in FP32, then converted to the input dtype and multiplied by the learned weight. | Use a thin adapter for the semantic signature `(x, weight, eps) -> y`, because the existing method closes over module state. Preserve the parameter on `RMSNorm` so checkpoint keys do not change. | Local vLLM code only demonstrates `flashinfer.norm.gemma_rmsnorm`. Gemma RMSNorm has a different weight convention and is not a Qwen RMSNorm replacement. A general `rmsnorm` candidate must be verified after installation. |
| Fused add + RMSNorm | `nanovllm/layers/layernorm.py:28-40`: `RMSNorm.add_rms_forward(x, residual) -> (normalized, new_residual)`. `RMSNorm.forward` selects this path when `residual` is not `None` (`:42-50`). | Register separately from plain RMSNorm, using a thin adapter `(x, residual, weight, eps) -> (y, residual_sum)`. This is necessary for independent provider overrides. | Local vLLM only demonstrates `gemma_fused_add_rmsnorm`, which mutates tensors in place. It is not automatically equivalent to the current return and alias contract. |
| RoPE | `nanovllm/layers/rotary_embedding.py:6-48`: NeoX split-half rotation, full `head_size` rotary dimension, cached cosine/sine values, and FP32 arithmetic followed by conversion to the input dtype. Public call is `RotaryEmbedding.forward(positions, query, key) -> (query, key)`. | `RotaryEmbedding`/`get_rope` can be registered directly as the native factory. A functional registry contract needs only a thin state-binding adapter for the cache and head size. | The local vLLM source calls `flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(positions, query, key, head_size, cos_sin_cache, is_neox)`. A wrapper is required because it is in-place while the native implementation returns new tensors. |
| KV-cache store | `nanovllm/layers/attention.py:10-40`: custom Triton `store_kvcache(key, value, k_cache, v_cache, slot_mapping)`. A slot of `-1` is skipped. The cache is treated as contiguous NHD pages. | The top-level callable already matches a useful operator interface and can be registered directly as `native_triton`. Its `supports` check must include layout/contiguity and an even, valid head layout. | No equivalent FlashInfer cache-write call is demonstrated by local code. Local vLLM's FlashInfer backend still writes through its own `reshape_and_cache_flash` custom op, so the current Triton implementation remains authoritative. |
| Attention | `nanovllm/layers/attention.py:43-71`: `Attention.forward(q, k, v)`. It reads global execution metadata, stores new K/V, and calls installed `flash_attn_varlen_func` with optional paged `block_table`. `flash_attn_with_kvcache` is imported but unused. | Do not put attention in the ordinary operator registry. Preserve this class as the legacy FlashAttention baseline behind a thin stateful attention-backend boundary. | Local vLLM source demonstrates `BatchPrefillWithPagedKVCacheWrapper` and `BatchDecodeWithPagedKVCacheWrapper`, each with persistent workspace plus a metadata `plan()` followed by `run()`. This is a backend, not a stateless callable. |
| Dense/TP linear | `nanovllm/layers/linear.py:37-153`: replicated, column, merged-column, QKV, and row-parallel layers all call `F.linear`; row parallel performs an all-reduce. Their weight-loader signatures encode different shard rules. | Keep unchanged and outside the registry for FP16/BF16 inference. | Local FlashInfer references concern quantized GEMM paths, not a required replacement for dense FP16/BF16 `F.linear`. |
| Embedding and LM head | `nanovllm/layers/embed_head.py:9-67`: `F.embedding` with TP masking/all-reduce and `F.linear` with context-sensitive last-token selection and TP gather. | Keep unchanged and outside the registry. | No local equivalent is required or demonstrated. |
| Sampling | `nanovllm/layers/sampler.py:5-15`: temperature scaling, softmax, exponential-noise sampling, and argmax. | It is directly callable but should remain out of the first registry change. | Local vLLM demonstrates FlashInfer top-k/top-p sampling only. Those paths are statistically equivalent, not necessarily token-for-token RNG equivalent to the current sampler. |

## Model and runner coupling

The Qwen model uses the operators in these distinct semantic locations:

- Q/K RMSNorm and RoPE: `nanovllm/models/qwen3.py:68-85`.
- MLP SiLU-and-multiply: `nanovllm/models/qwen3.py:99-115`.
- Plain and fused residual RMSNorm paths: `nanovllm/models/qwen3.py:142-157`.
- Final fused residual RMSNorm: `nanovllm/models/qwen3.py:170-181`.

`ModelRunner` owns important attention state:

- KV cache allocation and per-layer cache injection are in
  `nanovllm/engine/model_runner.py:116-134`. Each layer receives separate K and
  V tensors with logical shape `[num_blocks, block_size, num_kv_heads,
  head_dim]`.
- Query/KV sequence metadata, block tables, slot mappings, and context lengths
  are built in `model_runner.py:143-190`.
- CUDA graph buffers and capture are fixed in `model_runner.py:230-265`.

A FlashInfer attention backend therefore has to translate the dense block table
to paged `indptr`, page indices, and last-page lengths, plan once per batch, and
reuse planned wrappers for every layer. It must not allocate a large workspace
or repeat planning once per decoder layer.

## Minimal registry/config integration point

The smallest compatible design is one registry module containing a provider
record and `register`/`resolve` functions. A provider needs only `name`, a
callable or factory, `supports`, and `priority`. Native providers stay next to
their existing implementations.

`Config` should add an `operator_overrides` mapping with a `default_factory`.
`ModelRunner` should configure the resolver immediately before model
construction at `model_runner.py:34`. Layers resolve and save their callable in
`__init__`; forward methods must not consult the registry.

The override value `native` should mean the best registered `native_*`
provider, while explicit values such as `native_torch` and `native_triton`
remain useful for tests. Forced unavailable providers must raise a clear error
during initialization instead of silently falling back.

Never register one global `nn.Module` instance: that would share parameters or
cache state across decoder layers and could alter `state_dict` names. Register a
factory or a stateless function, leaving learned parameters on their existing
semantic layer.

## Baseline invariants and risks

1. With every override set to `native`, numerical results, output structure,
   dtypes, and checkpoint keys must remain unchanged.
2. The installed FlashAttention paged-KV API requires its page block size to be
   a multiple of 256, matching `Config.__post_init__` today. A future default of
   16 cannot run the legacy backend. Legacy rollback must explicitly use both
   `attention_backend=legacy` and a compatible block size such as 256.
3. FlashInfer RoPE and fused normalization candidates can be in-place. Thin
   wrappers must preserve the native alias/return contract or document and test
   an intentional contract change.
4. For FlashInfer paged attention, a sequence ending exactly at a page boundary
   has `last_page_len == block_size`, not zero.
5. FlashInfer planning and buffers must be CUDA-graph safe and have stable
   addresses before replay.
6. The local machine can validate only tensor parallel size one. TP greater than
   one needs a multi-GPU test environment.
7. The repository's root `test.py` is a tensor-indexing scratch script, not a
   correctness baseline. Full model/logit and generation comparisons also need
   a concrete local Qwen model path.

The first native tests intentionally avoid FlashInfer and exercise the public
module calls in eager mode by unwrapping only the `torch.compile` decorator.
This keeps CPU CI deterministic while testing the exact native calculation and
interface used by the compiled methods.
