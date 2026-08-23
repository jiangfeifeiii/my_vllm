from math import sqrt
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch

import nanovllm.layers.attention_backend as attention_backend_module
from nanovllm.layers.attention_backend import (
    FLASHINFER_ATTENTION_AVAILABLE,
    FLASHINFER_WORKSPACE_BYTES,
    AttentionPlan,
    AttentionRoute,
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashInferBackend,
    FlashInferMetadata,
)
from nanovllm.utils.context import BatchType, CommonAttentionMetadata


class AttentionBackendValidationTest(TestCase):

    def test_flashattention_paged_cache_rejects_block_size_16(self):
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            FlashAttentionBackend(
                num_q_heads=16,
                num_kv_heads=4,
                head_dim=128,
                block_size=16,
                dtype=torch.bfloat16,
            )

    def test_flashattention_accepts_native_block_size(self):
        backend = FlashAttentionBackend(
            num_q_heads=16,
            num_kv_heads=4,
            head_dim=128,
            block_size=256,
            dtype=torch.bfloat16,
        )
        self.assertEqual(backend.block_size, 256)

    def test_flashattention_dispatches_all_paged_routes_explicitly(self):
        backend = FlashAttentionBackend(16, 4, 128, 256, torch.bfloat16)
        cache = torch.ones(1)
        cases = (
            (BatchType.PURE_PREFILL, AttentionRoute.PREFILL),
            (BatchType.PURE_DECODE, AttentionRoute.DECODE),
            (BatchType.MIXED, AttentionRoute.MIXED_UNIFIED),
        )

        for batch_type, route in cases:
            with self.subTest(route=route):
                plan = _flashattention_plan(batch_type, route)
                q = torch.empty(
                    plan.common.num_query_tokens,
                    16,
                    128,
                )
                expected = torch.full_like(q, 7)
                with patch.object(
                    backend,
                    "_forward_paged_attention",
                    return_value=expected,
                ) as paged_forward:
                    output = backend.forward(
                        q, q, q, cache, cache, plan
                    )

                self.assertIs(output, expected)
                paged_forward.assert_called_once_with(
                    q,
                    cache,
                    cache,
                    plan.metadata,
                )

    def test_flashattention_dispatches_warmup_without_paged_kernel(self):
        backend = FlashAttentionBackend(16, 4, 128, 256, torch.bfloat16)
        plan = _flashattention_plan(
            BatchType.PURE_PREFILL,
            AttentionRoute.WARMUP,
            cached=False,
        )
        q = torch.empty(plan.common.num_query_tokens, 16, 128)
        empty_cache = torch.empty(0)
        expected = torch.full_like(q, 3)

        with (
            patch.object(
                attention_backend_module,
                "_cacheless_varlen_attention",
                return_value=expected,
            ) as warmup_forward,
            patch.object(backend, "_forward_paged_attention") as paged_forward,
        ):
            output = backend.forward(
                q, q, q, empty_cache, empty_cache, plan
            )

        self.assertIs(output, expected)
        warmup_forward.assert_called_once_with(q, q, q, plan.common)
        paged_forward.assert_not_called()

    def test_flashattention_rejects_incompatible_or_unknown_route(self):
        backend = FlashAttentionBackend(16, 4, 128, 256, torch.bfloat16)
        cache = torch.ones(1)
        mixed = _flashattention_plan(
            BatchType.MIXED,
            AttentionRoute.MIXED_UNIFIED,
        )
        mismatched = AttentionPlan(
            BatchType.PURE_DECODE,
            AttentionRoute.MIXED_UNIFIED,
            mixed.common,
            mixed.metadata,
        )
        unsupported = AttentionPlan(
            BatchType.MIXED,
            AttentionRoute.MIXED_SPLIT,
            mixed.common,
            mixed.metadata,
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            backend.forward(
                torch.empty(3, 16, 128),
                torch.empty(3, 16, 128),
                torch.empty(3, 16, 128),
                cache,
                cache,
                mismatched,
            )
        with self.assertRaisesRegex(RuntimeError, "invalid planned"):
            backend.forward(
                torch.empty(3, 16, 128),
                torch.empty(3, 16, 128),
                torch.empty(3, 16, 128),
                cache,
                cache,
                unsupported,
            )

    def test_flashattention_warmup_rejects_allocated_cache(self):
        backend = FlashAttentionBackend(16, 4, 128, 256, torch.bfloat16)
        plan = _flashattention_plan(
            BatchType.PURE_PREFILL,
            AttentionRoute.WARMUP,
            cached=False,
        )
        q = torch.empty(2, 16, 128)
        cache = torch.ones(1)

        with self.assertRaisesRegex(ValueError, "requires an empty KV cache"):
            backend.forward(q, q, q, cache, cache, plan)

    def test_flashinfer_dependency_error_is_actionable(self):
        missing = RuntimeError("missing flashinfer-jit-cache")
        with (
            patch.object(
                attention_backend_module,
                "FLASHINFER_ATTENTION_AVAILABLE",
                False,
            ),
            patch.object(
                attention_backend_module,
                "FLASHINFER_IMPORT_ERROR",
                missing,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "FlashInfer attention is unavailable.*flashinfer-jit-cache",
            ):
                FlashInferBackend(16, 4, 128, 16, torch.bfloat16)

    def test_flashinfer_rejects_unknown_attention_mode(self):
        with self.assertRaisesRegex(ValueError, "attention_mode"):
            FlashInferBackend(
                16, 4, 128, 16, torch.bfloat16, attention_mode="auto"
            )


def _flashattention_plan(
    batch_type: BatchType,
    route: AttentionRoute,
    *,
    cached: bool = True,
) -> AttentionPlan:
    if batch_type is BatchType.PURE_PREFILL:
        query_lengths = (2,)
        num_prefill_seqs = 1
    elif batch_type is BatchType.PURE_DECODE:
        query_lengths = (1,)
        num_prefill_seqs = 0
    else:
        query_lengths = (2, 1)
        num_prefill_seqs = 1

    query_start_loc = [0]
    for query_length in query_lengths:
        query_start_loc.append(query_start_loc[-1] + query_length)
    num_seqs = len(query_lengths)
    num_decode_seqs = num_seqs - num_prefill_seqs
    num_prefill_tokens = sum(query_lengths[:num_prefill_seqs])
    num_decode_tokens = sum(query_lengths[num_prefill_seqs:])
    query_start_tensor = torch.tensor(
        query_start_loc, dtype=torch.int32
    )
    block_tables = (
        torch.arange(num_seqs, dtype=torch.int32).view(num_seqs, 1)
        if cached
        else None
    )
    common = CommonAttentionMetadata(
        num_prefill_seqs=num_prefill_seqs,
        num_decode_seqs=num_decode_seqs,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        query_start_loc=query_start_tensor,
        seq_lens=torch.full((num_seqs,), 256, dtype=torch.int32),
        slot_mapping=torch.arange(
            query_start_loc[-1], dtype=torch.int32
        ),
        block_tables=block_tables,
        max_q_len=max(query_lengths),
        max_kv_len=256,
        block_counts=(1,) * num_seqs if cached else (),
        num_kv_blocks=num_seqs if cached else 0,
        num_prefill_kv_blocks=num_prefill_seqs if cached else 0,
        trusted=True,
    )
    metadata = FlashAttentionMetadata(
        query_start_loc=query_start_tensor,
        kv_start_loc=(
            torch.arange(
                0,
                (num_seqs + 1) * 256,
                256,
                dtype=torch.int32,
            )
            if cached
            else query_start_tensor
        ),
        block_tables=block_tables,
        max_q_len=common.max_q_len,
        max_kv_len=common.max_kv_len,
    )
    return AttentionPlan(batch_type, route, common, metadata)


_SKIP_REASON = "FlashInfer attention tests require CUDA and FlashInfer AOT kernels"


@skipUnless(
    torch.cuda.is_available() and FLASHINFER_ATTENTION_AVAILABLE,
    _SKIP_REASON,
)
class FlashInferAttentionBackendTest(TestCase):
    num_q_heads = 16
    num_kv_heads = 4
    head_dim = 128
    block_size = 16
    dtype = torch.bfloat16

    def _backend(
        self,
        dtype: torch.dtype | None = None,
        attention_mode: str = "unified",
    ) -> FlashInferBackend:
        return FlashInferBackend(
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            dtype or self.dtype,
            attention_mode=attention_mode,
        )

    def test_common_metadata_is_converted_to_flashinfer_page_csr(self):
        common, _, _, _ = _phase_case(
            self.dtype,
            (5, 1),
            (21, 13),
            1,
        )
        backend = self._backend()

        metadata = backend.build_metadata(common)

        self.assertIsInstance(metadata, FlashInferMetadata)
        self.assertIs(metadata.query_start_loc, common.query_start_loc)
        self.assertEqual(metadata.kv_indptr.tolist(), [0, 2, 3])
        self.assertEqual(metadata.page_indices.tolist(), [0, 1, 2])
        self.assertEqual(metadata.last_page_len.tolist(), [5, 13])
        self.assertEqual(metadata.num_pages, 3)
        self.assertEqual(metadata.num_prefill_pages, 2)
        self.assertFalse(hasattr(common, "page_kv_indptr"))
        self.assertFalse(hasattr(common, "page_indices"))

    def test_explicit_plans_cover_pure_and_mixed_workloads(self):
        cases = (
            (
                "pure_prefill",
                (5, 3),
                (21, 13),
                2,
                BatchType.PURE_PREFILL,
                AttentionRoute.PREFILL,
            ),
            (
                "pure_decode",
                (1, 1),
                (21, 13),
                0,
                BatchType.PURE_DECODE,
                AttentionRoute.DECODE,
            ),
            (
                "mixed",
                (5, 1),
                (21, 13),
                1,
                BatchType.MIXED,
                None,
            ),
        )
        for attention_mode in ("unified", "split"):
            backend = self._backend(attention_mode=attention_mode)
            for (
                phase,
                query_lengths,
                kv_lengths,
                num_prefill,
                batch_type,
                fixed_route,
            ) in cases:
                with self.subTest(mode=attention_mode, phase=phase):
                    common, _, _, _ = _phase_case(
                        self.dtype,
                        query_lengths,
                        kv_lengths,
                        num_prefill,
                    )

                    plan = backend.build_plan(common)

                    self.assertIsInstance(plan, AttentionPlan)
                    self.assertIs(plan.batch_type, batch_type)
                    self.assertIs(plan.common_metadata, common)
                    self.assertIsInstance(
                        plan.backend_metadata, FlashInferMetadata
                    )
                    if fixed_route is not None:
                        self.assertIs(plan.route, fixed_route)
                    elif attention_mode == "split":
                        self.assertIs(plan.route, AttentionRoute.MIXED_SPLIT)
                    else:
                        self.assertIn(
                            plan.route,
                            (
                                AttentionRoute.MIXED_UNIFIED,
                                AttentionRoute.MIXED_SPLIT,
                            ),
                        )

    def test_paged_pure_and_mixed_results_match_reference(self):
        cases = (
            ((5, 3), (21, 13), 2),
            ((1, 1), (21, 13), 0),
            ((5, 1), (21, 13), 1),
        )
        for attention_mode in ("unified", "split"):
            backend = self._backend(attention_mode=attention_mode)
            self.assertEqual(backend.workspace.dtype, torch.uint8)
            self.assertEqual(
                backend.workspace.numel(), FLASHINFER_WORKSPACE_BYTES
            )
            for case_index, (
                query_lengths,
                kv_lengths,
                num_prefill,
            ) in enumerate(cases):
                with self.subTest(
                    attention_mode=attention_mode,
                    query_lengths=query_lengths,
                ):
                    torch.manual_seed(307 + case_index)
                    common, q, k_cache, v_cache = _phase_case(
                        self.dtype,
                        query_lengths,
                        kv_lengths,
                        num_prefill,
                    )
                    plan = backend.build_plan(common)
                    unused = q.new_empty(
                        (0, self.num_kv_heads, self.head_dim)
                    )

                    output = backend.forward(
                        q, unused, unused, k_cache, v_cache, plan
                    )

                    metadata = plan.backend_metadata
                    expected = _paged_attention_reference(
                        q,
                        k_cache,
                        v_cache,
                        metadata.kv_indptr,
                        metadata.page_indices,
                        query_lengths,
                        kv_lengths,
                    )
                    self.assertEqual(output.shape, q.shape)
                    self.assertEqual(output.dtype, self.dtype)
                    torch.testing.assert_close(
                        output, expected, atol=2e-2, rtol=2e-2
                    )

    def test_pure_routes_call_only_the_phase_specialized_wrapper(self):
        cases = (
            ((5, 3), (21, 13), 2, AttentionRoute.PREFILL),
            ((1, 1), (21, 13), 0, AttentionRoute.DECODE),
        )
        for query_lengths, kv_lengths, num_prefill, route in cases:
            with self.subTest(route=route):
                backend = self._backend()
                common, q, k_cache, v_cache = _phase_case(
                    self.dtype,
                    query_lengths,
                    kv_lengths,
                    num_prefill,
                )
                with (
                    patch.object(backend.prefill_wrapper, "plan") as prefill_plan,
                    patch.object(backend.decode_wrapper, "plan") as decode_plan,
                ):
                    plan = backend.build_plan(common)

                def fake_run(_, __, *, out):
                    out.fill_(6)
                    return out

                with (
                    patch.object(
                        backend.prefill_wrapper,
                        "run",
                        side_effect=fake_run,
                    ) as prefill_run,
                    patch.object(
                        backend.decode_wrapper,
                        "run",
                        side_effect=fake_run,
                    ) as decode_run,
                ):
                    unused = q.new_empty(
                        (0, self.num_kv_heads, self.head_dim)
                    )
                    output = backend.forward(
                        q, unused, unused, k_cache, v_cache, plan
                    )

                self.assertIs(plan.route, route)
                self.assertTrue(bool(torch.all(output == 6).item()))
                if route is AttentionRoute.PREFILL:
                    prefill_plan.assert_called_once()
                    decode_plan.assert_not_called()
                    prefill_run.assert_called_once()
                    decode_run.assert_not_called()
                else:
                    prefill_plan.assert_not_called()
                    decode_plan.assert_called_once()
                    prefill_run.assert_not_called()
                    decode_run.assert_called_once()

    def test_mixed_split_writes_output_views_without_torch_cat(self):
        backend = self._backend(attention_mode="split")
        common, q, k_cache, v_cache = _phase_case(
            self.dtype,
            (5, 1, 1),
            (21, 13, 18),
            1,
        )
        with (
            patch.object(backend.prefill_wrapper, "plan"),
            patch.object(backend.decode_wrapper, "plan"),
        ):
            plan = backend.build_plan(common)
        self.assertIs(plan.route, AttentionRoute.MIXED_SPLIT)

        output_views = []

        def prefill_run(_, __, *, out):
            output_views.append(out)
            out.fill_(2)
            return out

        def decode_run(_, __, *, out):
            output_views.append(out)
            out.fill_(3)
            return out

        unused = q.new_empty((0, self.num_kv_heads, self.head_dim))
        with (
            patch.object(
                backend.prefill_wrapper, "run", side_effect=prefill_run
            ),
            patch.object(
                backend.decode_wrapper, "run", side_effect=decode_run
            ),
            patch(
                "torch.cat",
                side_effect=AssertionError("mixed split must not concatenate"),
            ),
        ):
            output = backend.forward(
                q, unused, unused, k_cache, v_cache, plan
            )

        split = common.num_prefill_tokens
        self.assertEqual(len(output_views), 2)
        self.assertEqual(output_views[0].data_ptr(), output.data_ptr())
        self.assertEqual(
            output_views[1].data_ptr(), output[split:].data_ptr()
        )
        self.assertTrue(bool(torch.all(output[:split] == 2).item()))
        self.assertTrue(bool(torch.all(output[split:] == 3).item()))

    def test_trusted_plan_avoids_gpu_value_reads(self):
        backend = self._backend(attention_mode="split")
        common, _, _, _ = _phase_case(
            self.dtype,
            (5, 1),
            (21, 13),
            1,
        )

        with (
            patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("device item read"),
            ),
            patch(
                "torch.all",
                side_effect=AssertionError("device all read"),
            ),
            patch.object(backend.prefill_wrapper, "plan"),
            patch.object(backend.decode_wrapper, "plan"),
        ):
            plan = backend.build_plan(common)

        self.assertIs(plan.route, AttentionRoute.MIXED_SPLIT)

    def test_cacheless_warmup_uses_backend_owned_fallback(self):
        torch.manual_seed(223)
        common, q, _, _ = _phase_case(
            self.dtype,
            (3, 1),
            (3, 1),
            2,
            cache=False,
        )
        k = torch.randn(
            4,
            self.num_kv_heads,
            self.head_dim,
            device="cuda",
            dtype=self.dtype,
        )
        v = torch.randn_like(k)
        empty_cache = torch.empty(0, device="cuda")
        backend = self._backend()

        plan = backend.build_plan(common)
        output = backend.forward(
            q, k, v, empty_cache, empty_cache, plan
        )

        self.assertIs(plan.route, AttentionRoute.WARMUP)
        expected = _ragged_attention_reference(q, k, v, (3, 1))
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)

    def test_paged_forward_requires_attention_plan(self):
        backend = self._backend()
        q = torch.empty(
            1, 16, 128, device="cuda", dtype=torch.bfloat16
        )
        cache = torch.empty(
            1, 16, 4, 128, device="cuda", dtype=torch.bfloat16
        )
        with self.assertRaisesRegex(TypeError, "requires AttentionPlan"):
            backend.forward(q, q[:, :4], q[:, :4], cache, cache, object())


def _phase_case(
    dtype: torch.dtype,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    num_prefill_seqs: int,
    *,
    cache: bool = True,
):
    query_start_loc = [0]
    for query_length in query_lengths:
        query_start_loc.append(query_start_loc[-1] + query_length)

    block_counts = tuple(
        (kv_length + 15) // 16 for kv_length in kv_lengths
    )
    block_tables = None
    if cache:
        width = max(block_counts)
        rows = []
        page_index = 0
        for page_count in block_counts:
            row = list(range(page_index, page_index + page_count))
            row.extend([-1] * (width - page_count))
            rows.append(row)
            page_index += page_count
        block_tables = torch.tensor(
            rows, device="cuda", dtype=torch.int32
        )

    common = CommonAttentionMetadata(
        num_prefill_seqs=num_prefill_seqs,
        num_decode_seqs=len(query_lengths) - num_prefill_seqs,
        num_prefill_tokens=sum(query_lengths[:num_prefill_seqs]),
        num_decode_tokens=sum(query_lengths[num_prefill_seqs:]),
        query_start_loc=torch.tensor(
            query_start_loc, device="cuda", dtype=torch.int32
        ),
        seq_lens=torch.tensor(
            kv_lengths, device="cuda", dtype=torch.int32
        ),
        slot_mapping=torch.arange(
            query_start_loc[-1], device="cuda", dtype=torch.int32
        ),
        block_tables=block_tables,
        max_q_len=max(query_lengths),
        max_kv_len=max(kv_lengths),
        block_counts=block_counts if cache else (),
        num_kv_blocks=sum(block_counts) if cache else 0,
        num_prefill_kv_blocks=(
            sum(block_counts[:num_prefill_seqs]) if cache else 0
        ),
        trusted=True,
    )
    q = torch.randn(
        query_start_loc[-1], 16, 128, device="cuda", dtype=dtype
    )
    num_pages = sum(block_counts) if cache else 0
    k_cache = torch.randn(
        num_pages, 16, 4, 128, device="cuda", dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    return common, q, k_cache, v_cache


def _paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
) -> torch.Tensor:
    outputs = []
    query_start = 0
    group_size = q.shape[1] // k_cache.shape[2]
    for request, (query_length, kv_length) in enumerate(
        zip(query_lengths, kv_lengths)
    ):
        indices = page_indices[
            kv_indptr[request] : kv_indptr[request + 1]
        ].long()
        keys = k_cache[indices].flatten(0, 1)[:kv_length].float()
        values = v_cache[indices].flatten(0, 1)[:kv_length].float()
        keys = keys.repeat_interleave(group_size, dim=1)
        values = values.repeat_interleave(group_size, dim=1)
        query = q[query_start : query_start + query_length].float()
        scores = torch.einsum("qhd,khd->hqk", query, keys) / sqrt(
            q.shape[-1]
        )
        query_positions = torch.arange(
            kv_length - query_length,
            kv_length,
            device=q.device,
        )
        key_positions = torch.arange(kv_length, device=q.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        scores.masked_fill_(~causal_mask[None, :, :], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum("hqk,khd->qhd", probabilities, values)
        )
        query_start += query_length
    return torch.cat(outputs).to(q.dtype)


def _ragged_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sequence_lengths: tuple[int, ...],
) -> torch.Tensor:
    outputs = []
    start = 0
    group_size = q.shape[1] // k.shape[1]
    for length in sequence_lengths:
        query = q[start : start + length].float()
        keys = k[start : start + length].float().repeat_interleave(
            group_size, dim=1
        )
        values = v[start : start + length].float().repeat_interleave(
            group_size, dim=1
        )
        scores = torch.einsum("qhd,khd->hqk", query, keys) / sqrt(
            q.shape[-1]
        )
        causal_mask = torch.arange(length, device=q.device)[None, :] <= (
            torch.arange(length, device=q.device)[:, None]
        )
        scores.masked_fill_(~causal_mask[None, :, :], -torch.inf)
        outputs.append(
            torch.einsum(
                "hqk,khd->qhd",
                torch.softmax(scores, dim=-1),
                values,
            )
        )
        start += length
    return torch.cat(outputs).to(q.dtype)
