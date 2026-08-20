from math import sqrt
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

import nanovllm.layers.attention_backend as attention_backend_module
from nanovllm.layers.attention_backend import (
    FLASHINFER_ATTENTION_AVAILABLE,
    FLASHINFER_WORKSPACE_BYTES,
    FlashInferAttentionBackend,
    LegacyFlashAttentionBackend,
)


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

    def _backend(self) -> FlashInferAttentionBackend:
        return FlashInferAttentionBackend(
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            self.dtype,
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
        q = torch.randn(
            4, 16, 128, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(4, 4, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        empty_cache = torch.tensor([], device="cuda")

        backend = self._backend()
        backend.plan(context)
        output = backend.forward(
            q, k, v, empty_cache, empty_cache, context
        )
        expected = _ragged_attention_reference(q, k, v, sequence_lengths)

        self.assertFalse(backend._planned)
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


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
