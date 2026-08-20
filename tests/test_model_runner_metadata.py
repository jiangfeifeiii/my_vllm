from unittest import TestCase, main, skipUnless

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


def _tolist(tensor: torch.Tensor) -> list[int]:
    return tensor.cpu().tolist()


class _BackendRecorder:
    def plan(self, context):
        self.context = context


@skipUnless(torch.cuda.is_available(), "ModelRunner metadata preparation uses CUDA")
class ModelRunnerMetadataTest(TestCase):

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.runner = object.__new__(ModelRunner)
        self.runner.block_size = 16

        self.runner.attention_backend = _BackendRecorder()

    def test_mixed_prefix_extension_and_fresh_prefill_page_metadata(self):
        prefix_extend = Sequence(list(range(100, 119)), block_size=16)
        prefix_extend.num_cached_tokens = 16
        prefix_extend.num_new_tokens = 3
        prefix_extend.block_table = [7, 4]

        fresh_prefill = Sequence(list(range(200, 205)), block_size=16)
        fresh_prefill.num_cached_tokens = 0
        fresh_prefill.num_new_tokens = 5
        fresh_prefill.block_table = [9]

        input_ids, positions = self.runner.prepare_model_input(
            [prefix_extend, fresh_prefill]
        )
        context = get_context()

        self.assertEqual(prefix_extend.num_context_tokens, 19)
        self.assertEqual(fresh_prefill.num_context_tokens, 5)
        self.assertEqual(_tolist(input_ids), [116, 117, 118, 200, 201, 202, 203, 204])
        self.assertEqual(_tolist(positions), [16, 17, 18, 0, 1, 2, 3, 4])
        self.assertEqual(_tolist(context.page_q_indptr), [0, 3, 8])
        self.assertEqual(_tolist(context.page_kv_indptr), [0, 2, 3])
        self.assertEqual(_tolist(context.page_indices), [7, 4, 9])
        self.assertEqual(_tolist(context.page_last_page_len), [3, 5])
        self.assertEqual(
            _tolist(context.slot_mapping),
            [64, 65, 66, 144, 145, 146, 147, 148],
        )
        self.assertEqual(_tolist(context.context_lens), [19, 5])
        self.assertEqual(_tolist(context.seq_need_compute_logits), [0, 1])
        self.assertEqual(
            context.block_tables.cpu().tolist(),
            [[7, 4], [9, -1]],
        )
        self.assertEqual(context.max_seqlen_q, 5)
        self.assertEqual(context.max_seqlen_k, 19)
        self.assertEqual(context.num_prefill_seqs, 2)
        self.assertEqual(context.num_prefill_tokens, 8)
        self.assertEqual(context.num_decode_tokens, 0)

    def test_warmup_without_block_table_has_empty_page_metadata(self):
        warmup = Sequence([10, 11, 12, 13], block_size=16)
        warmup.num_cached_tokens = 0
        warmup.num_new_tokens = 4

        input_ids, positions = self.runner.prepare_model_input([warmup])
        context = get_context()

        self.assertEqual(_tolist(input_ids), [10, 11, 12, 13])
        self.assertEqual(_tolist(positions), [0, 1, 2, 3])
        self.assertEqual(_tolist(context.page_q_indptr), [0, 4])
        self.assertEqual(_tolist(context.page_kv_indptr), [0, 0])
        self.assertEqual(_tolist(context.page_indices), [])
        self.assertEqual(_tolist(context.page_last_page_len), [0])
        self.assertEqual(_tolist(context.slot_mapping), [])
        self.assertEqual(_tolist(context.context_lens), [4])
        self.assertIsNone(context.seq_need_compute_logits)
        self.assertIsNone(context.block_tables)
        self.assertEqual(context.num_prefill_seqs, 1)
        self.assertEqual(context.num_prefill_tokens, 4)
        self.assertEqual(context.num_decode_tokens, 0)

    def test_contiguous_prefill_decode_phase_boundary(self):
        prefill = Sequence([1, 2, 3], block_size=16)
        prefill.num_new_tokens = 3
        prefill.block_table = [2]

        decode = Sequence(list(range(100, 118)), block_size=16)
        decode.num_cached_tokens = 17
        decode.num_new_tokens = 1
        decode.block_table = [4, 5]

        input_ids, positions = self.runner.prepare_model_input(
            [prefill, decode],
            num_prefill_seqs=1,
        )
        context = get_context()

        self.assertEqual(_tolist(input_ids), [1, 2, 3, 117])
        self.assertEqual(_tolist(positions), [0, 1, 2, 17])
        self.assertEqual(_tolist(context.page_q_indptr), [0, 3, 4])
        self.assertEqual(_tolist(context.page_kv_indptr), [0, 1, 3])
        self.assertEqual(_tolist(context.page_indices), [2, 4, 5])
        self.assertEqual(_tolist(context.page_last_page_len), [3, 2])
        self.assertEqual(_tolist(context.slot_mapping), [32, 33, 34, 81])
        self.assertEqual(context.num_prefill_seqs, 1)
        self.assertEqual(context.num_prefill_tokens, 3)
        self.assertEqual(context.num_decode_tokens, 1)


if __name__ == "__main__":
    main()
