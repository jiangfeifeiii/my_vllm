"""CPU-safe contract tests for attention backend selection and planning."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

import nanovllm.layers.attention_backend as attention_backend_module
from nanovllm.layers.attention_backend import (
    AttentionBackend,
    AttentionBackendRegistry,
    AttentionPlan,
    AttentionRoute,
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashInferAttentionBackend,
    FlashInferBackend,
    FlashInferMetadata,
    LegacyFlashAttentionBackend,
)
from nanovllm.utils.context import BatchType, CommonAttentionMetadata


_BACKEND_KWARGS = {
    "num_q_heads": 16,
    "num_kv_heads": 4,
    "head_dim": 128,
    "dtype": torch.bfloat16,
}


class CommonAttentionMetadataTest(TestCase):

    def test_metadata_is_backend_neutral_and_exposes_batch_totals(self):
        common = _common_metadata((5, 1, 1), (21, 13, 18), 1)

        self.assertEqual(common.num_prefill_seqs, 1)
        self.assertEqual(common.num_decode_seqs, 2)
        self.assertEqual(common.num_prefill_tokens, 5)
        self.assertEqual(common.num_decode_tokens, 2)
        self.assertEqual(common.num_seqs, 3)
        self.assertEqual(common.num_query_tokens, 7)
        self.assertEqual(common.query_start_loc.tolist(), [0, 5, 6, 7])
        self.assertEqual(common.seq_lens.tolist(), [21, 13, 18])
        self.assertEqual(common.block_counts, (2, 1, 2))
        self.assertEqual(common.max_q_len, 5)
        self.assertEqual(common.max_kv_len, 21)

        # Backend-owned page CSR/wrappers must not leak into common metadata.
        self.assertFalse(hasattr(common, "page_kv_indptr"))
        self.assertFalse(hasattr(common, "page_indices"))
        self.assertFalse(hasattr(common, "wrapper"))

    def test_metadata_is_immutable_for_one_runtime_step(self):
        common = _common_metadata((1,), (17,), 0)
        with self.assertRaises(FrozenInstanceError):
            common.num_decode_tokens = 2


class AttentionBackendRegistryTest(TestCase):

    def _create(self, name: str, block_size: int):
        return AttentionBackendRegistry.create(
            name,
            block_size=block_size,
            attention_mode="unified",
            device="cuda",
            **_BACKEND_KWARGS,
        )

    def test_registry_contains_only_complete_attention_backends(self):
        self.assertEqual(
            AttentionBackendRegistry.registered_names(),
            ("flashattention", "flashinfer"),
        )
        self.assertEqual(
            AttentionBackendRegistry.names(),
            ("flashattention", "flashinfer"),
        )

    def test_controlled_registration_keeps_the_policy_extensible(self):
        class ExperimentalBackend(AttentionBackend):
            pass

        class LocalRegistry(AttentionBackendRegistry):
            _BACKENDS = dict(AttentionBackendRegistry._BACKENDS)
            _ALIASES = dict(AttentionBackendRegistry._ALIASES)

        LocalRegistry.register("experimental", ExperimentalBackend)

        self.assertEqual(
            LocalRegistry.registered_names(),
            ("flashattention", "flashinfer", "experimental"),
        )
        self.assertEqual(
            AttentionBackendRegistry.registered_names(),
            ("flashattention", "flashinfer"),
        )

    def test_registered_backend_may_override_only_supports(self):
        class SupportsOnlyBackend(AttentionBackend):
            enabled = False

            @classmethod
            def supports(cls, **_):
                return cls.enabled

        class LocalRegistry(AttentionBackendRegistry):
            _BACKENDS = dict(AttentionBackendRegistry._BACKENDS)
            _ALIASES = dict(AttentionBackendRegistry._ALIASES)

        LocalRegistry.register("supports_only", SupportsOnlyBackend)
        create_kwargs = {
            **_BACKEND_KWARGS,
            "block_size": 16,
            "attention_mode": "unified",
            "device": "cuda",
        }

        with self.assertRaisesRegex(
            RuntimeError, r"supports\(\) returned False"
        ):
            LocalRegistry.create("supports_only", **create_kwargs)

        SupportsOnlyBackend.enabled = True
        backend = LocalRegistry.create("supports_only", **create_kwargs)
        self.assertIsInstance(backend, SupportsOnlyBackend)

    def test_auto_selection_treats_supports_as_authoritative(self):
        with (
            patch.object(
                FlashAttentionBackend, "supports", return_value=False
            ) as flashattention_supports,
            patch.object(
                FlashAttentionBackend, "support_reason", return_value=None
            ) as flashattention_reason,
            patch.object(
                FlashInferBackend, "supports", return_value=True
            ) as flashinfer_supports,
            patch.object(
                FlashInferBackend,
                "support_reason",
                side_effect=AssertionError(
                    "diagnostics must not run for a supported backend"
                ),
            ),
            patch.object(FlashInferBackend, "__init__", return_value=None),
        ):
            backend = self._create("auto", 256)

        self.assertIsInstance(backend, FlashInferBackend)
        flashattention_supports.assert_called_once()
        flashattention_reason.assert_called_once()
        flashinfer_supports.assert_called_once()

    def test_auto_prefers_flashattention_for_native_page_size(self):
        with (
            patch.object(
                FlashAttentionBackend, "support_reason", return_value=None
            ) as flashattention_support,
            patch.object(
                FlashInferBackend, "support_reason", return_value=None
            ) as flashinfer_support,
            patch.object(FlashAttentionBackend, "__init__", return_value=None),
        ):
            backend = self._create("auto", 256)

        self.assertIsInstance(backend, FlashAttentionBackend)
        flashattention_support.assert_called_once()
        flashinfer_support.assert_not_called()

    def test_auto_uses_flashinfer_for_small_pages(self):
        with (
            patch.object(
                FlashAttentionBackend, "support_reason", return_value=None
            ),
            patch.object(
                FlashInferBackend, "support_reason", return_value=None
            ) as flashinfer_support,
            patch.object(FlashInferBackend, "__init__", return_value=None),
        ):
            backend = self._create("auto", 16)

        self.assertIsInstance(backend, FlashInferBackend)
        flashinfer_support.assert_called_once()

    def test_auto_falls_back_when_flashattention_import_failed(self):
        with (
            patch.object(
                FlashAttentionBackend,
                "support_reason",
                return_value="Dao FlashAttention is unavailable: import failed",
            ),
            patch.object(
                FlashInferBackend, "support_reason", return_value=None
            ),
            patch.object(FlashInferBackend, "__init__", return_value=None),
        ):
            backend = self._create("auto", 256)

        self.assertIsInstance(backend, FlashInferBackend)

    def test_auto_reports_both_reasons_when_no_backend_is_usable(self):
        with (
            patch.object(
                FlashAttentionBackend,
                "support_reason",
                return_value="flash-attn import failed",
            ),
            patch.object(
                FlashInferBackend,
                "support_reason",
                return_value="flashinfer import failed",
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "No usable attention backend.*flash-attn import failed.*"
                "flashinfer import failed",
            ):
                self._create("auto", 256)

    def test_explicit_backend_and_legacy_alias_remain_supported(self):
        with (
            patch.object(
                FlashAttentionBackend, "support_reason", return_value=None
            ),
            patch.object(FlashAttentionBackend, "__init__", return_value=None),
        ):
            backend = self._create("legacy", 256)

        self.assertIsInstance(backend, FlashAttentionBackend)
        self.assertIs(LegacyFlashAttentionBackend, FlashAttentionBackend)
        self.assertIs(FlashInferAttentionBackend, FlashInferBackend)

    def test_explicit_flashinfer_bypasses_auto_preference(self):
        with (
            patch.object(
                FlashInferBackend, "support_reason", return_value=None
            ) as flashinfer_support,
            patch.object(FlashInferBackend, "__init__", return_value=None),
        ):
            backend = self._create("flashinfer", 256)

        self.assertIsInstance(backend, FlashInferBackend)
        flashinfer_support.assert_called_once()

    def test_unknown_backend_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "unknown attention backend"):
            self._create("prefill", 16)


class AttentionPlanContractTest(TestCase):

    def test_flashinfer_classifies_pure_batches_once(self):
        cases = (
            (
                (5, 3),
                (21, 13),
                2,
                BatchType.PURE_PREFILL,
                AttentionRoute.PREFILL,
            ),
            (
                (1, 1),
                (21, 13),
                0,
                BatchType.PURE_DECODE,
                AttentionRoute.DECODE,
            ),
        )
        for query_lengths, seq_lens, num_prefill, batch_type, route in cases:
            with self.subTest(route=route):
                backend = _bare_flashinfer()
                common = _common_metadata(
                    query_lengths, seq_lens, num_prefill
                )
                metadata = _flashinfer_metadata(common)

                plan = backend.build_plan(common, metadata)

                self.assertIsInstance(plan, AttentionPlan)
                self.assertIs(plan.batch_type, batch_type)
                self.assertIs(plan.route, route)
                self.assertIs(plan.common_metadata, common)
                self.assertIs(plan.backend_metadata, metadata)
                self.assertEqual(backend.planned_route, route.value)
                self.assertEqual(backend.route_counts[route.value], 1)
                if route is AttentionRoute.PREFILL:
                    backend.prefill_wrapper.plan.assert_called_once()
                    backend.decode_wrapper.plan.assert_not_called()
                else:
                    backend.prefill_wrapper.plan.assert_not_called()
                    backend.decode_wrapper.plan.assert_called_once()

    def test_flashinfer_mixed_unified_is_one_explicit_route(self):
        backend = _bare_flashinfer(mixed=True)
        common = _common_metadata((5, 1), (21, 13), 1)
        metadata = _flashinfer_metadata(common)

        plan = backend.build_plan(common, metadata)

        self.assertIs(plan.batch_type, BatchType.MIXED)
        self.assertIs(plan.route, AttentionRoute.MIXED_UNIFIED)
        backend.mixed_wrapper.plan.assert_called_once()
        backend.prefill_wrapper.plan.assert_not_called()
        backend.decode_wrapper.plan.assert_not_called()

    def test_flashinfer_mixed_split_slices_backend_metadata(self):
        backend = _bare_flashinfer(mixed=False)
        common = _common_metadata((5, 1), (21, 13), 1)
        metadata = _flashinfer_metadata(common)

        plan = backend.build_plan(common, metadata)

        self.assertIs(plan.batch_type, BatchType.MIXED)
        self.assertIs(plan.route, AttentionRoute.MIXED_SPLIT)
        prefill_args = backend.prefill_wrapper.plan.call_args.args
        decode_args = backend.decode_wrapper.plan.call_args.args
        self.assertEqual(prefill_args[0].tolist(), [0, 5])
        self.assertEqual(prefill_args[1].tolist(), [0, 2])
        self.assertEqual(prefill_args[2].tolist(), [0, 1])
        self.assertEqual(prefill_args[3].tolist(), [5])
        self.assertEqual(decode_args[0].tolist(), [0, 1])
        self.assertEqual(decode_args[1].tolist(), [2])
        self.assertEqual(decode_args[2].tolist(), [13])

    def test_flashattention_uses_the_same_plan_contract(self):
        backend = _bare_flashattention()
        cases = (
            ((4, 2), 2, BatchType.PURE_PREFILL, AttentionRoute.PREFILL),
            ((1, 1), 0, BatchType.PURE_DECODE, AttentionRoute.DECODE),
            ((4, 1), 1, BatchType.MIXED, AttentionRoute.MIXED_UNIFIED),
        )
        for query_lengths, num_prefill, batch_type, route in cases:
            with self.subTest(route=route):
                common = _common_metadata(
                    query_lengths, (256, 256), num_prefill, block_size=256
                )
                metadata = FlashAttentionMetadata(
                    common.query_start_loc,
                    torch.tensor([0, 256, 512], dtype=torch.int32),
                    common.block_tables,
                    common.max_q_len,
                    common.max_kv_len,
                )

                plan = backend.build_plan(common, metadata)

                self.assertIs(plan.batch_type, batch_type)
                self.assertIs(plan.route, route)


class FlashInferUnifiedMixedCapabilityTest(TestCase):

    def test_full_decode_graph_capability_follows_attention_mode(self):
        backend = _bare_flashinfer()

        backend.attention_mode = "unified"
        self.assertTrue(backend.supports_full_decode_graph)

        backend.attention_mode = "split"
        self.assertFalse(backend.supports_full_decode_graph)

    def test_rtx_5070_name_disables_unified_mixed(self):
        backend = _bare_flashinfer(mixed=True)
        with (
            patch.object(attention_backend_module, "_BatchAttention", object()),
            patch.object(
                torch.cuda,
                "get_device_name",
                return_value="NVIDIA GeForce RTX 5070",
            ),
        ):
            self.assertFalse(backend.supports_unified_mixed())

        self.assertIn("RTX 5070", backend.mixed_attention_unavailable_reason)

    def test_other_sm120_device_is_not_rejected_by_architecture(self):
        backend = _bare_flashinfer(mixed=True)
        with (
            patch.object(attention_backend_module, "_BatchAttention", object()),
            patch.object(
                torch.cuda,
                "get_device_name",
                return_value="NVIDIA GeForce RTX 5090",
            ),
            patch.object(
                torch.cuda, "get_device_capability", return_value=(12, 0)
            ) as get_capability,
        ):
            self.assertTrue(backend.supports_unified_mixed())

        get_capability.assert_not_called()
        self.assertIsNone(backend.mixed_attention_unavailable_reason)


def _common_metadata(
    query_lengths: tuple[int, ...],
    seq_lens: tuple[int, ...],
    num_prefill_seqs: int,
    *,
    block_size: int = 16,
    cache: bool = True,
) -> CommonAttentionMetadata:
    if len(query_lengths) != len(seq_lens):
        raise ValueError("query_lengths and seq_lens must have equal length")
    query_start_loc = [0]
    for length in query_lengths:
        query_start_loc.append(query_start_loc[-1] + length)

    block_counts = tuple(
        (length + block_size - 1) // block_size for length in seq_lens
    )
    block_tables = None
    if cache:
        width = max(block_counts)
        rows = []
        page = 0
        for count in block_counts:
            row = list(range(page, page + count))
            row.extend([-1] * (width - count))
            rows.append(row)
            page += count
        block_tables = torch.tensor(rows, dtype=torch.int32)

    num_prefill_tokens = sum(query_lengths[:num_prefill_seqs])
    num_decode_seqs = len(query_lengths) - num_prefill_seqs
    num_decode_tokens = sum(query_lengths[num_prefill_seqs:])
    return CommonAttentionMetadata(
        num_prefill_seqs=num_prefill_seqs,
        num_decode_seqs=num_decode_seqs,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        slot_mapping=torch.arange(query_start_loc[-1], dtype=torch.int32),
        block_tables=block_tables,
        max_q_len=max(query_lengths),
        max_kv_len=max(seq_lens),
        block_counts=block_counts if cache else (),
        num_kv_blocks=sum(block_counts) if cache else 0,
        num_prefill_kv_blocks=(
            sum(block_counts[:num_prefill_seqs]) if cache else 0
        ),
        trusted=True,
    )


def _flashinfer_metadata(
    common: CommonAttentionMetadata,
) -> FlashInferMetadata:
    kv_indptr = [0]
    for count in common.block_counts:
        kv_indptr.append(kv_indptr[-1] + count)
    last_page_len = [
        (length - 1) % 16 + 1 for length in common.seq_lens.tolist()
    ]
    return FlashInferMetadata(
        query_start_loc=common.query_start_loc,
        kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
        page_indices=torch.arange(common.num_kv_blocks, dtype=torch.int32),
        last_page_len=torch.tensor(last_page_len, dtype=torch.int32),
        num_pages=common.num_kv_blocks,
        num_prefill_pages=common.num_prefill_kv_blocks,
    )


def _bare_flashinfer(*, mixed: bool = False) -> FlashInferBackend:
    backend = object.__new__(FlashInferBackend)
    AttentionBackend.__init__(backend, block_size=16, **_BACKEND_KWARGS)
    backend.attention_mode = "unified"
    backend.workspace = SimpleNamespace(device=torch.device("cuda", 0))
    backend.prefill_wrapper = Mock()
    backend.decode_wrapper = Mock()
    backend.mixed_wrapper = Mock() if mixed else None
    backend._mixed_attention_initialization_error = None
    backend._mixed_attention_unavailable_reason = None
    backend._planned_route = None
    backend._route_counts = {route.value: 0 for route in AttentionRoute}
    backend._num_prefill_seqs = 0
    backend._num_prefill_tokens = 0
    backend._num_decode_seqs = 0
    backend._num_decode_tokens = 0
    backend._output_buffer = None
    backend._lse_buffer = None
    backend._active_graph_state = None
    backend.graph_workspace = None
    return backend


def _bare_flashattention() -> FlashAttentionBackend:
    backend = object.__new__(FlashAttentionBackend)
    AttentionBackend.__init__(
        backend, block_size=256, **_BACKEND_KWARGS
    )
    backend._planned_route = None
    backend._route_counts = {route.value: 0 for route in AttentionRoute}
    return backend
