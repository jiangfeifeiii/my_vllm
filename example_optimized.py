#!/usr/bin/env python3
"""Run nano-vLLM with automatic attention selection and optimized ops."""

import argparse
import atexit
import os
from pathlib import Path


OPERATOR_OVERRIDES = {
    "silu_and_mul": "adaptive_cuda",
    "rms_norm": "flashinfer",
    "fused_add_rms_norm": "flashinfer",
    "rotary_embedding": "flashinfer",
    "kv_cache_store": "native_triton",
}


def parse_args() -> argparse.Namespace:
    default_model = os.environ.get("NANOVLLM_MODEL") or os.environ.get(
        "NANOVLLM_TEST_MODEL"
    )
    parser = argparse.ArgumentParser(
        description="Run and debug the fully optimized nano-vLLM path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Local Qwen3 model directory (or set NANOVLLM_MODEL).",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "flashattention", "flashinfer", "legacy"),
        default="auto",
        help=(
            "attention backend registry selection; legacy is a "
            "FlashAttention compatibility alias"
        ),
    )
    parser.add_argument(
        "--attention-mode",
        choices=("unified", "split"),
        default="unified",
        help="mixed-batch execution mode; split requires FlashInfer",
    )
    parser.add_argument(
        "--kvcache-block-size",
        type=int,
        default=16,
        help=(
            "KV-cache page size; Dao FlashAttention requires a multiple "
            "of 256"
        ),
    )
    parser.add_argument(
        "--cudagraph-mode",
        choices=("none", "full_decode_only"),
        default="none",
        help=("CUDA Graph policy; full_decode_only captures only eligible "
              "unified pure-decode batches."),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pause in pdb after engine initialization and before generation.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable generation progress bars.",
    )
    return parser.parse_args()


def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def collect_operator_bindings(llm) -> dict[str, set[str]]:
    """Read compatibility diagnostics for construction-bound CustomOps."""
    bindings: dict[str, set[str]] = {
        operator: set() for operator in OPERATOR_OVERRIDES
    }
    for module in llm.model_runner.model.modules():
        if hasattr(module, "kv_store_provider_name"):
            bindings["kv_cache_store"].add(module.kv_store_provider_name)
        if hasattr(module, "rms_provider_name"):
            bindings["rms_norm"].add(module.rms_provider_name)
        if hasattr(module, "add_rms_provider_name"):
            bindings["fused_add_rms_norm"].add(module.add_rms_provider_name)
        if type(module).__name__ == "RotaryEmbedding":
            bindings["rotary_embedding"].add(module.provider_name)
        if type(module).__name__ == "SiluAndMul":
            bindings["silu_and_mul"].add(module.provider_name)
    return bindings


def print_runtime_configuration(llm) -> None:
    config = llm.config
    bindings = collect_operator_bindings(llm)
    attention_backend = llm.model_runner.attention_backend
    print("\nEnabled runtime configuration")
    print(f"  backend requested : {config.attention_backend}")
    print(f"  backend selected  : {attention_backend.backend_name}")
    print(f"  attention mode    : {config.attention_mode}")
    if hasattr(attention_backend, "mixed_attention_available"):
        print(
            "  unified mixed     : "
            f"{attention_backend.mixed_attention_available}"
        )
        reason = attention_backend.mixed_attention_unavailable_reason
        if reason:
            print(f"  mixed fallback    : {reason}")
    print(f"  KV-cache page size: {config.kvcache_block_size}")
    print(f"  chunked prefill   : {config.chunked_prefill}")
    print(f"  batch token budget: {config.max_num_batched_tokens}")
    print("  prefix cache/LPM  : scheduler built-in")
    graph_stats = llm.model_runner.get_cudagraph_stats()
    captured = graph_stats["captured_batch_sizes"] or "none"
    print(f"  CUDA Graph policy : {graph_stats['policy']}")
    print(f"  captured batches  : {captured}")
    print(f"  graph capture time: {graph_stats['capture_time_ms']:.2f} ms")
    print(
        f"  graph extra memory: "
        f"{graph_stats['extra_memory_bytes'] / 2**20:.2f} MiB"
    )
    for operator, expected_implementation in OPERATOR_OVERRIDES.items():
        actual = bindings[operator]
        print(f"  {operator:<19}: {', '.join(sorted(actual))}")
        if actual != {expected_implementation}:
            raise RuntimeError(
                f"{operator} expected implementation "
                f"{expected_implementation!r}, got {actual!r}"
            )


def print_cache_stats(llm, label: str) -> None:
    manager = llm.scheduler.block_manager
    cached_free = sum(
        block.ref_count == 0 and block.hash != -1 for block in manager.blocks
    )
    print(
        f"{label}: used={len(manager.used_block_ids)}, "
        f"free={len(manager.free_block_ids)}, cached_free={cached_free}"
    )


def print_cudagraph_stats(llm) -> None:
    stats = llm.model_runner.get_cudagraph_stats()
    print("\nCUDA Graph runtime statistics")
    print(f"  full graph replay steps: {stats['full_graph_replay_steps']}")
    print(f"  eager fallback steps   : {stats['eager_fallback_steps']}")
    print(f"  graph bucket hits      : {stats['graph_bucket_hits']}")
    print(f"  graph bucket misses    : {stats['graph_bucket_misses']}")
    attention_backend = llm.model_runner.attention_backend
    if hasattr(attention_backend, "route_counts"):
        print("  eager attention routes :")
        for route, count in attention_backend.route_counts.items():
            print(f"    {route:<16}: {count}")


def print_outputs(label: str, prompts: list[str], outputs: list[dict]) -> None:
    print(f"\n{label}")
    for index, (prompt, output) in enumerate(zip(prompts, outputs), start=1):
        print(f"[{index}] prompt chars: {len(prompt)}")
        print(f"[{index}] completion: {output['text']!r}")
        print(f"[{index}] token ids : {output['token_ids']}")


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("pass --model /path/to/Qwen3 or set NANOVLLM_MODEL")
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"model directory does not exist: {model_path}")
    if (
        args.batch_tokens <= 0
        or args.max_model_len <= 1
        or args.kvcache_block_size <= 0
    ):
        raise SystemExit("batch/model/page limits must be positive")
    if (
        args.attention_backend in ("flashattention", "legacy")
        and args.kvcache_block_size % 256
    ):
        raise SystemExit(
            "Dao FlashAttention requires --kvcache-block-size divisible by 256"
        )
    if args.batch_tokens >= args.max_model_len:
        raise SystemExit(
            "--batch-tokens must be smaller than --max-model-len "
            "to demonstrate chunked prefill"
        )
    if args.max_tokens <= 0 or args.temperature <= 1e-10:
        raise SystemExit("--max-tokens and --temperature must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")

    import torch
    from transformers import AutoTokenizer

    from nanovllm import CUDAGraphPolicy, LLM, SamplingParams

    if not torch.cuda.is_available():
        raise SystemExit("nano-vLLM requires an NVIDIA CUDA GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    shared_sentence = (
        "Nano-vLLM is a lightweight inference engine using paged KV cache, "
        "registry-selected attention, chunked prefill, longest-prefix "
        "matching, and statically bound GPU operators."
    )
    shared_context = shared_sentence
    prime_prompt = build_chat_prompt(
        tokenizer,
        shared_context,
        "Summarize the execution path in two short sentences.",
    )
    while len(tokenizer.encode(prime_prompt)) <= args.batch_tokens:
        shared_context = f"{shared_context} {shared_sentence}"
        prime_prompt = build_chat_prompt(
            tokenizer,
            shared_context,
            "Summarize the execution path in two short sentences.",
        )

    cached_prompt = build_chat_prompt(
        tokenizer,
        shared_context,
        "Which configured features reduce repeated prompt work?",
    )
    cold_prompt = build_chat_prompt(
        tokenizer,
        "Answer accurately and concisely.",
        "What is a paged KV cache?",
    )
    long_cold_context = (
        "This request has an unrelated uncached context and is intentionally "
        "long enough to require chunked prefill."
    )
    long_cold_prompt = build_chat_prompt(
        tokenizer,
        long_cold_context,
        "Explain why bounded batches help an inference engine.",
    )
    while len(tokenizer.encode(long_cold_prompt)) <= args.batch_tokens:
        long_cold_context = f"{long_cold_context} {long_cold_context}"
        long_cold_prompt = build_chat_prompt(
            tokenizer,
            long_cold_context,
            "Explain why bounded batches help an inference engine.",
        )
    all_prompts = [
        prime_prompt,
        cached_prompt,
        cold_prompt,
        long_cold_prompt,
    ]
    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in all_prompts]
    longest_request = max(prompt_lengths) + args.max_tokens
    if longest_request > args.max_model_len:
        raise SystemExit(
            f"prompt + completion needs {longest_request} tokens, but "
            f"--max-model-len is {args.max_model_len}; increase it"
        )

    llm = None
    try:
        llm = LLM(
            str(model_path),
            cudagraph_mode=CUDAGraphPolicy(args.cudagraph_mode),
            tensor_parallel_size=1,
            attention_backend=args.attention_backend,
            attention_mode=args.attention_mode,
            kvcache_block_size=args.kvcache_block_size,
            chunked_prefill=True,
            operator_overrides=OPERATOR_OVERRIDES,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.batch_tokens,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        # LLMEngine registers this automatically; this script owns cleanup.
        atexit.unregister(llm.exit)
        print_runtime_configuration(llm)
        print(f"  prompt token lengths: {prompt_lengths}")

        if args.debug:
            print("\nDebugger paused before llm.generate().")
            print("Set forward breakpoints, then use `continue`.")
            breakpoint()

        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            ignore_eos=True,
        )

        # This prompt is deliberately larger than the batch budget, so it must
        # use chunked prefill and leaves reusable cached-free prefix pages.
        prime_outputs = llm.generate(
            [prime_prompt],
            sampling_params,
            use_tqdm=not args.no_tqdm,
        )
        print_outputs(
            "Chunked-prefill cache priming",
            [prime_prompt],
            prime_outputs,
        )
        print_cache_stats(llm, "After priming")

        # The short cold request is submitted first on purpose. Cache-aware
        # LPM can rank the second request first because it reuses the primed
        # prefix. The third request exceeds the batch budget, so after the two
        # short prefills finish its resumed chunk runs beside their decode
        # tokens and exercises the explicit MIXED attention route.
        ranked_prompts = [cold_prompt, cached_prompt, long_cold_prompt]
        ranked_outputs = llm.generate(
            ranked_prompts,
            sampling_params,
            use_tqdm=not args.no_tqdm,
        )
        print_outputs("Cache-aware LPM batch", ranked_prompts, ranked_outputs)
        print_cache_stats(llm, "After LPM batch")
        print_cudagraph_stats(llm)
    finally:
        try:
            if llm is not None:
                llm.exit()
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
