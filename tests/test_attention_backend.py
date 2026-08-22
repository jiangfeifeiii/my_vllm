from math import sqrt
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch

import nanovllm.layers.attention_backend as attention_backend_module
from nanovllm.layers.attention_backend import (
    FLASHINFER_ATTENTION_AVAILABLE,
    FLASHINFER_WORKSPACE_BYTES,
    FlashInferAttentionBackend,
    LegacyFlashAttentionBackend,
)
from nanovllm.utils.context import BatchType


class AttentionBackendValidationTest(TestCase):

    def test_legacy_paged_cache_rejects_block_size_16(self):
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            LegacyFlashAttentionBackend(
                num_q_heads=16,
                num_kv_heads=4,
                head_dim=128,
                block_size=16,
                dtype=torch.bfloat16,
            )

    def test_legacy_accepts_native_block_size(self):
        backend = LegacyFlashAttentionBackend(
            num_q_heads=16,
            num_kv_heads=4,
            head_dim=128,
            block_size=256,
            dtype=torch.bfloat16,
        )
        self.assertEqual(backend.block_size, 256)

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
                FlashInferAttentionBackend(16, 4, 128, 16, torch.bfloat16)

    def test_flashinfer_rejects_unknown_attention_mode(self):
        with self.assertRaisesRegex(ValueError, "attention_mode"):
            FlashInferAttentionBackend(
                16, 4, 128, 16, torch.bfloat16, attention_mode="auto"
            )

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
    ) -> FlashInferAttentionBackend:
        return FlashInferAttentionBackend(
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            dtype or self.dtype,
            attention_mode=attention_mode,
        )

    def test_unified_bf16_gqa_page_size_16_matches_reference(self):
        torch.manual_seed(211)
        query_lengths = (5, 1)
        kv_lengths = (21, 13)
        page_q_indptr = torch.tensor(
            [0, 5, 6], device="cuda", dtype=torch.int32
        )
        page_kv_indptr = torch.tensor(
            [0, 2, 3], device="cuda", dtype=torch.int32
        )
        page_indices = torch.tensor(
            [2, 0, 3], device="cuda", dtype=torch.int32
        )
        page_last_page_len = torch.tensor(
            [5, 13], device="cuda", dtype=torch.int32
        )
        context = SimpleNamespace(
            page_q_indptr=page_q_indptr,
            page_kv_indptr=page_kv_indptr,
            page_indices=page_indices,
            page_last_page_len=page_last_page_len,
        )

        q = torch.randn(
            sum(query_lengths),
            self.num_q_heads,
            self.head_dim,
            device="cuda",
            dtype=self.dtype,
        )
        k_cache = torch.randn(
            4,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            device="cuda",
            dtype=self.dtype,
        )
        v_cache = torch.randn_like(k_cache)
        unused_k = torch.empty(
            0,
            self.num_kv_heads,
            self.head_dim,
            device="cuda",
            dtype=self.dtype,
        )
        unused_v = torch.empty_like(unused_k)

        backend = self._backend()
        self.assertEqual(backend.workspace.dtype, torch.uint8)
        self.assertEqual(backend.workspace.numel(), FLASHINFER_WORKSPACE_BYTES)
        backend.plan(context)
        output = backend.forward(
            q, unused_k, unused_v, k_cache, v_cache, context
        )

        expected = _paged_attention_reference(
            q,
            k_cache,
            v_cache,
            page_kv_indptr,
            page_indices,
            query_lengths,
            kv_lengths,
        )
        self.assertEqual(
            output.shape,
            (sum(query_lengths), self.num_q_heads, self.head_dim),
        )
        self.assertEqual(output.dtype, self.dtype)
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)

    def test_unified_pure_prefill_decode_and_mixed(self):
        cases = (
            ("pure_prefill", (5, 3), (21, 13), 2),
            ("pure_decode", (1, 1), (21, 13), 0),
            ("mixed", (5, 1, 1), (21, 13, 18), 1),
        )
        for dtype in (torch.float16, torch.bfloat16):
            backend = self._backend(dtype)
            self.assertEqual(
                backend.prefill_wrapper._float_workspace_buffer.data_ptr(),
                backend.workspace.data_ptr(),
            )
            self.assertEqual(
                backend.decode_wrapper._float_workspace_buffer.data_ptr(),
                backend.workspace.data_ptr(),
            )
            for case_index, (
                phase,
                query_lengths,
                kv_lengths,
                num_prefill_seqs,
            ) in enumerate(cases):
                with self.subTest(dtype=dtype, phase=phase):
                    torch.manual_seed(307 + case_index)
                    context, q, k_cache, v_cache = _phase_case(
                        dtype,
                        query_lengths,
                        kv_lengths,
                        num_prefill_seqs,
                    )
                    unused_k = q.new_empty(
                        (0, self.num_kv_heads, self.head_dim)
                    )
                    unused_v = torch.empty_like(unused_k)

                    backend.plan(context)
                    output = backend.forward(
                        q,
                        unused_k,
                        unused_v,
                        k_cache,
                        v_cache,
                        context,
                    )
                    self.assertIsNotNone(backend._output_buffer)
                    self.assertEqual(
                        output.data_ptr(),
                        backend._output_buffer.data_ptr(),
                    )
                    expected = _paged_attention_reference(
                        q,
                        k_cache,
                        v_cache,
                        context.page_kv_indptr,
                        context.page_indices,
                        query_lengths,
                        kv_lengths,
                    )
                    tolerance = 5e-3 if dtype == torch.float16 else 2e-2

                    self.assertEqual(
                        backend._num_prefill_seqs, num_prefill_seqs
                    )
                    self.assertEqual(
                        backend._num_prefill_tokens,
                        sum(query_lengths[:num_prefill_seqs]),
                    )
                    self.assertEqual(
                        backend._num_decode_seqs,
                        len(query_lengths) - num_prefill_seqs,
                    )
                    self.assertEqual(
                        backend._num_decode_tokens,
                        sum(query_lengths[num_prefill_seqs:]),
                    )
                    torch.testing.assert_close(
                        output,
                        expected,
                        atol=tolerance,
                        rtol=tolerance,
                    )

                    if phase == "mixed":
                        second_q = torch.randn_like(q)
                        second_k_cache = torch.randn_like(k_cache)
                        second_v_cache = torch.randn_like(v_cache)
                        second_output = backend.forward(
                            second_q,
                            unused_k,
                            unused_v,
                            second_k_cache,
                            second_v_cache,
                            context,
                        )
                        self.assertEqual(
                            second_output.data_ptr(), output.data_ptr()
                        )
                        second_expected = _paged_attention_reference(
                            second_q,
                            second_k_cache,
                            second_v_cache,
                            context.page_kv_indptr,
                            context.page_indices,
                            query_lengths,
                            kv_lengths,
                        )
                        torch.testing.assert_close(
                            second_output,
                            second_expected,
                            atol=tolerance,
                            rtol=tolerance,
                        )

    def test_zero_copy_split_pure_prefill_decode_and_mixed(self):
        cases = (
            ("pure_decode", (1, 1), (21, 13), 0),
            ("pure_prefill", (5, 3), (21, 13), 2),
            ("mixed", (5, 1, 1), (21, 13, 18), 1),
        )
        for dtype in (torch.float16, torch.bfloat16):
            backend = self._backend(dtype, "split")
            self.assertFalse(backend.mixed_attention_available)
            self.assertEqual(
                backend.prefill_wrapper._float_workspace_buffer.data_ptr(),
                backend.workspace.data_ptr(),
            )
            self.assertEqual(
                backend.decode_wrapper._float_workspace_buffer.data_ptr(),
                backend.workspace.data_ptr(),
            )
            previous_ptr = None
            capacity = 0
            for case_index, (
                phase,
                query_lengths,
                kv_lengths,
                num_prefill_seqs,
            ) in enumerate(cases):
                with self.subTest(dtype=dtype, phase=phase):
                    torch.manual_seed(401 + case_index)
                    context, q, k_cache, v_cache = _phase_case(
                        dtype,
                        query_lengths,
                        kv_lengths,
                        num_prefill_seqs,
                    )
                    unused_k = q.new_empty(
                        (0, self.num_kv_heads, self.head_dim)
                    )
                    unused_v = torch.empty_like(unused_k)

                    backend.plan(context)
                    output = backend.forward(
                        q,
                        unused_k,
                        unused_v,
                        k_cache,
                        v_cache,
                        context,
                    )
                    expected = _paged_attention_reference(
                        q,
                        k_cache,
                        v_cache,
                        context.page_kv_indptr,
                        context.page_indices,
                        query_lengths,
                        kv_lengths,
                    )
                    tolerance = 5e-3 if dtype == torch.float16 else 2e-2

                    self.assertIsNotNone(backend._output_buffer)
                    self.assertEqual(
                        output.data_ptr(),
                        backend._output_buffer.data_ptr(),
                    )
                    if previous_ptr is not None:
                        if q.size(0) <= capacity:
                            self.assertEqual(output.data_ptr(), previous_ptr)
                        else:
                            self.assertNotEqual(output.data_ptr(), previous_ptr)
                    previous_ptr = output.data_ptr()
                    capacity = max(capacity, q.size(0))
                    torch.testing.assert_close(
                        output,
                        expected,
                        atol=tolerance,
                        rtol=tolerance,
                    )

                    if phase == "mixed":
                        second_q = torch.randn_like(q)
                        second_output = backend.forward(
                            second_q,
                            unused_k,
                            unused_v,
                            k_cache,
                            v_cache,
                            context,
                        )
                        second_expected = _paged_attention_reference(
                            second_q,
                            k_cache,
                            v_cache,
                            context.page_kv_indptr,
                            context.page_indices,
                            query_lengths,
                            kv_lengths,
                        )
                        self.assertEqual(second_output.data_ptr(), previous_ptr)
                        torch.testing.assert_close(
                            second_output,
                            second_expected,
                            atol=tolerance,
                            rtol=tolerance,
                        )

    def test_plan_and_run_route_pure_batches_to_specialized_wrappers(self):
        cases = (
            ((5, 3), 2, "prefill"),
            ((1, 1), 0, "decode"),
        )
        for query_lengths, num_prefill, expected_route in cases:
            context, q, k_cache, v_cache = _phase_case(
                torch.bfloat16,
                query_lengths,
                (21, 13),
                num_prefill,
            )
            unused = q.new_empty((0, self.num_kv_heads, self.head_dim))
            for attention_mode in ("unified", "split"):
                with self.subTest(
                    expected_route=expected_route,
                    attention_mode=attention_mode,
                ):
                    backend = self._backend(attention_mode=attention_mode)
                    with (
                        patch.object(
                            backend.prefill_wrapper, "plan"
                        ) as prefill_plan,
                        patch.object(
                            backend.decode_wrapper, "plan"
                        ) as decode_plan,
                    ):
                        backend.plan(context)

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
                        output = backend.forward(
                            q, unused, unused, k_cache, v_cache, context
                        )

                    self.assertEqual(backend.planned_route, expected_route)
                    self.assertEqual(
                        backend.route_counts[expected_route], 1
                    )
                    self.assertEqual(
                        output.data_ptr(), backend._output_buffer.data_ptr()
                    )
                    self.assertTrue(bool(torch.all(output == 6).item()))
                    if expected_route == "prefill":
                        prefill_plan.assert_called_once()
                        decode_plan.assert_not_called()
                        prefill_run.assert_called_once()
                        decode_run.assert_not_called()
                    else:
                        prefill_plan.assert_not_called()
                        decode_plan.assert_called_once()
                        prefill_run.assert_not_called()
                        decode_run.assert_called_once()
                    with self.assertRaises(TypeError):
                        backend.route_counts[expected_route] = 9

    def test_mixed_fallback_slices_prefill_and_decode_metadata(self):
        context, _, _, _ = _phase_case(
            torch.bfloat16,
            (5, 1),
            (21, 13),
            1,
        )
        for attention_mode in ("unified", "split"):
            with self.subTest(attention_mode=attention_mode):
                backend = self._backend(
                    torch.bfloat16,
                    attention_mode,
                )
                backend.mixed_wrapper = None
                with (
                    patch.object(backend.prefill_wrapper, "plan") as prefill,
                    patch.object(backend.decode_wrapper, "plan") as decode,
                ):
                    backend.plan(context)

                self.assertEqual(backend.planned_route, "mixed_split")
                prefill.assert_called_once()
                decode.assert_called_once()
                prefill_args = prefill.call_args.args
                decode_args = decode.call_args.args
                self.assertEqual(prefill_args[0].tolist(), [0, 5])
                self.assertEqual(prefill_args[1].tolist(), [0, 2])
                self.assertEqual(prefill_args[2].tolist(), [0, 1])
                self.assertEqual(prefill_args[3].tolist(), [5])
                self.assertEqual(decode_args[0].tolist(), [0, 1])
                self.assertEqual(decode_args[1].tolist(), [2])
                self.assertEqual(decode_args[2].tolist(), [13])

    def test_trusted_mixed_plan_uses_host_page_counts_without_value_reads(self):
        context, _, _, _ = _phase_case(
            torch.bfloat16,
            (5, 1),
            (21, 13),
            1,
        )
        context.page_metadata_trusted = True
        context.num_pages = context.page_indices.numel()
        context.num_prefill_pages = 2
        backend = self._backend(torch.bfloat16, "unified")
        backend.mixed_wrapper = None

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
            patch.object(backend.prefill_wrapper, "plan") as prefill,
            patch.object(backend.decode_wrapper, "plan") as decode,
        ):
            backend.plan(context)

        self.assertEqual(backend.planned_route, "mixed_split")
        self.assertEqual(prefill.call_args.args[2].numel(), 2)
        self.assertEqual(decode.call_args.args[1].numel(), 1)

    def test_trusted_plan_rejects_host_page_count_drift(self):
        context, _, _, _ = _phase_case(
            torch.bfloat16,
            (5, 1),
            (21, 13),
            1,
        )
        context.page_metadata_trusted = True
        context.num_pages = context.page_indices.numel() + 1
        context.num_prefill_pages = 2

        with self.assertRaisesRegex(ValueError, "num_pages"):
            self._backend().plan(context)

    def test_mixed_holistic_restores_kv_lengths_and_reuses_buffers(self):
        context, q, k_cache, v_cache = _phase_case(
            torch.bfloat16,
            (1, 1, 1),
            (1, 16, 17),
            1,
        )
        backend = self._backend(torch.bfloat16, "unified")
        holistic = Mock()
        backend.mixed_wrapper = holistic
        with (
            patch.object(backend.prefill_wrapper, "plan") as prefill,
            patch.object(backend.decode_wrapper, "plan") as decode,
        ):
            backend.plan(context)

        self.assertEqual(backend.planned_route, "mixed_holistic")
        holistic.plan.assert_called_once()
        prefill.assert_not_called()
        decode.assert_not_called()
        self.assertEqual(
            holistic.plan.call_args.args[3].tolist(),
            [1, 16, 17],
        )

        lse_pointers = []

        def fake_run(_, __, *, out, lse):
            lse_pointers.append(lse.data_ptr())
            self.assertEqual(lse.dtype, torch.float32)
            out.fill_(4)
            lse.fill_(1)
            return out, lse

        holistic.run.side_effect = fake_run
        unused = q.new_empty((0, self.num_kv_heads, self.head_dim))
        first = backend.forward(
            q, unused, unused, k_cache, v_cache, context
        )
        second = backend.forward(
            q, unused, unused, k_cache, v_cache, context
        )
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(lse_pointers[0], lse_pointers[1])
        self.assertEqual(holistic.run.call_count, 2)
        self.assertTrue(bool(torch.all(second == 4).item()))

    def test_decode_workspace_starts_zero(self):
        backend = self._backend()
        self.assertEqual(torch.count_nonzero(backend.workspace).item(), 0)

    def test_sm120_disables_holistic_mixed_attention(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120-specific FlashInfer 0.6.17 safety gate")
        backend = self._backend(attention_mode="unified")
        self.assertFalse(backend.mixed_attention_available)
        self.assertIn("SM120", backend.mixed_attention_unavailable_reason)

    def test_large_head_dim_disables_holistic_mixed_attention(self):
        backend = FlashInferAttentionBackend(
            self.num_q_heads,
            self.num_kv_heads,
            384,
            self.block_size,
            self.dtype,
            attention_mode="unified",
        )
        self.assertFalse(backend.mixed_attention_available)
        self.assertIn(
            "head_dim <= 256",
            backend.mixed_attention_unavailable_reason,
        )

    def test_decode_output_is_not_zeroed_per_layer(self):
        context, q, k_cache, v_cache = _phase_case(
            torch.bfloat16,
            (1, 1),
            (19, 17),
            0,
        )
        backend = self._backend(torch.bfloat16, "unified")
        backend.plan(context)
        backend._get_reusable_attention_output(q).fill_(9)
        unused_k = q.new_empty((0, self.num_kv_heads, self.head_dim))
        unused_v = torch.empty_like(unused_k)

        def fake_decode(_, __, *, out):
            self.assertTrue(bool(torch.all(out == 9).item()))
            out.fill_(3)
            return out

        with patch.object(
            backend.decode_wrapper,
            "run",
            side_effect=fake_decode,
        ):
            output = backend.forward(
                q,
                unused_k,
                unused_v,
                k_cache,
                v_cache,
                context,
            )

        self.assertTrue(bool(torch.all(output == 3).item()))

    def test_active_full_decode_graph_overrides_eager_decode_route(self):
        context, q, k_cache, v_cache = _phase_case(
            torch.bfloat16,
            (1, 1),
            (19, 17),
            0,
        )
        backend = self._backend(torch.bfloat16, "unified")
        backend.plan(context)
        self.assertEqual(backend.planned_route, "decode")
        graph_wrapper = Mock()
        graph_output = torch.empty_like(q)

        def fake_graph_run(_, __, *, out):
            out.fill_(8)
            return out

        graph_wrapper.run.side_effect = fake_graph_run
        backend.activate_full_decode_graph(graph_wrapper, graph_output)
        unused = q.new_empty((0, self.num_kv_heads, self.head_dim))
        with (
            patch.object(backend.prefill_wrapper, "run") as prefill,
            patch.object(backend.decode_wrapper, "run") as decode,
        ):
            output = backend.forward(
                q, unused, unused, k_cache, v_cache, context
            )
        backend.deactivate_full_decode_graph()

        graph_wrapper.run.assert_called_once()
        prefill.assert_not_called()
        decode.assert_not_called()
        self.assertEqual(output.data_ptr(), graph_output.data_ptr())
        self.assertTrue(bool(torch.all(output == 8).item()))

    def test_trusted_full_decode_graph_plan_avoids_device_value_reads(self):
        context, _, _, _ = _phase_case(
            torch.bfloat16,
            (1, 1),
            (19, 17),
            0,
        )
        context.page_metadata_trusted = True
        context.num_pages = context.page_indices.numel()
        context.num_prefill_pages = 0
        backend = self._backend(torch.bfloat16, "unified")
        wrapper = Mock()
        wrapper._nanovllm_qo_indptr_buffer = torch.empty_like(
            context.page_q_indptr
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
        ):
            backend.plan_full_decode_graph(wrapper, context)

        wrapper.plan.assert_called_once()
        self.assertTrue(
            torch.equal(
                wrapper._nanovllm_qo_indptr_buffer,
                context.page_q_indptr,
            )
        )

    def test_plan_rejects_batch_type_boundary_mismatch(self):
        context, _, _, _ = _phase_case(
            torch.bfloat16,
            (3, 1),
            (19, 17),
            1,
        )
        context.batch_type = BatchType.PURE_DECODE

        with self.assertRaisesRegex(
            ValueError,
            "batch_type does not match",
        ):
            self._backend().plan(context)

    def test_cacheless_warmup_uses_ragged_fallback(self):
        torch.manual_seed(223)
        sequence_lengths = (3, 1)
        cu_seqlens = torch.tensor(
            [0, 3, 4], device="cuda", dtype=torch.int32
        )
        context = SimpleNamespace(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            page_q_indptr=cu_seqlens,
            page_kv_indptr=torch.zeros(3, device="cuda", dtype=torch.int32),
            page_indices=torch.empty(0, device="cuda", dtype=torch.int32),
            page_last_page_len=torch.zeros(2, device="cuda", dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
        )
        q = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(4, 4, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        empty_cache = torch.tensor([], device="cuda")
        expected = _ragged_attention_reference(q, k, v, sequence_lengths)

        for attention_mode in ("unified", "split"):
            with self.subTest(attention_mode=attention_mode):
                backend = self._backend(attention_mode=attention_mode)
                backend.plan(context)
                output = backend.forward(
                    q, k, v, empty_cache, empty_cache, context
                )

                self.assertFalse(backend._planned)
                if attention_mode == "split":
                    self.assertIsNotNone(backend._output_buffer)
                    self.assertEqual(
                        output.data_ptr(),
                        backend._output_buffer.data_ptr(),
                    )
                else:
                    self.assertIsNone(backend._output_buffer)
                torch.testing.assert_close(
                    output, expected, atol=2e-2, rtol=2e-2
                )

    def test_paged_forward_requires_plan(self):
        backend = self._backend()
        q = torch.empty(
            1, 16, 128, device="cuda", dtype=torch.bfloat16
        )
        cache = torch.empty(
            1, 16, 4, 128, device="cuda", dtype=torch.bfloat16
        )
        with self.assertRaisesRegex(RuntimeError, r"plan\(context\)"):
            backend.forward(q, q[:, :4], q[:, :4], cache, cache, object())


def _phase_case(
    dtype: torch.dtype,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    num_prefill_seqs: int,
):
    q_indptr = [0]
    kv_indptr = [0]
    last_page_len = []
    for query_length, kv_length in zip(query_lengths, kv_lengths):
        q_indptr.append(q_indptr[-1] + query_length)
        kv_indptr.append(kv_indptr[-1] + (kv_length + 15) // 16)
        last_page_len.append((kv_length - 1) % 16 + 1)

    page_q_indptr = torch.tensor(
        q_indptr, device="cuda", dtype=torch.int32
    )
    page_kv_indptr = torch.tensor(
        kv_indptr, device="cuda", dtype=torch.int32
    )
    page_indices = torch.arange(
        kv_indptr[-1], device="cuda", dtype=torch.int32
    )
    if num_prefill_seqs == 0:
        batch_type = BatchType.PURE_DECODE
    elif num_prefill_seqs == len(query_lengths):
        batch_type = BatchType.PURE_PREFILL
    else:
        batch_type = BatchType.MIXED
    context = SimpleNamespace(
        page_q_indptr=page_q_indptr,
        page_kv_indptr=page_kv_indptr,
        page_indices=page_indices,
        page_last_page_len=torch.tensor(
            last_page_len, device="cuda", dtype=torch.int32
        ),
        num_prefill_seqs=num_prefill_seqs,
        num_prefill_tokens=sum(query_lengths[:num_prefill_seqs]),
        num_decode_tokens=sum(query_lengths[num_prefill_seqs:]),
        batch_type=batch_type,
    )
    q = torch.randn(
        q_indptr[-1], 16, 128, device="cuda", dtype=dtype
    )
    k_cache = torch.randn(
        kv_indptr[-1], 16, 4, 128, device="cuda", dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    return context, q, k_cache, v_cache


def _paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_kv_indptr: torch.Tensor,
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
            page_kv_indptr[request] : page_kv_indptr[request + 1]
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
                "hqk,khd->qhd", torch.softmax(scores, dim=-1), values
            )
        )
        start += length
    return torch.cat(outputs).to(q.dtype)


if __name__ == "__main__":
    main()
