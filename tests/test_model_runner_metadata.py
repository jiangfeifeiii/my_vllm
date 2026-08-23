from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch

from nanovllm.config import CUDAGraphPolicy
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import (
    BatchType,
    CommonAttentionMetadata,
    get_context,
    reset_context,
    set_context,
)


def _tolist(tensor: torch.Tensor) -> list[int]:
    return tensor.cpu().tolist()


class _BackendRecorder:
    def build_plan(self, metadata):
        self.metadata = metadata
        if metadata.num_prefill_seqs == 0:
            batch_type = BatchType.PURE_DECODE
        elif metadata.num_decode_seqs == 0:
            batch_type = BatchType.PURE_PREFILL
        else:
            batch_type = BatchType.MIXED
        return SimpleNamespace(batch_type=batch_type)


class _ModelRecorder:
    def __call__(self, input_ids, positions):
        return torch.stack((input_ids, positions), dim=-1)

    def compute_logits(self, hidden_states):
        return hidden_states


class ModelRunnerPlanDelegationTest(TestCase):

    def tearDown(self):
        reset_context()

    def test_backend_plan_is_the_only_batch_classification_source(self):
        cases = (
            (2, 0, 5, 0, BatchType.PURE_PREFILL),
            (0, 2, 0, 2, BatchType.PURE_DECODE),
            (1, 1, 3, 1, BatchType.MIXED),
        )
        for (
            prefill_seqs,
            decode_seqs,
            prefill_tokens,
            decode_tokens,
            expected,
        ) in cases:
            with self.subTest(expected=expected):
                metadata = CommonAttentionMetadata(
                    num_prefill_seqs=prefill_seqs,
                    num_decode_seqs=decode_seqs,
                    num_prefill_tokens=prefill_tokens,
                    num_decode_tokens=decode_tokens,
                    query_start_loc=torch.tensor(
                        [0, 1, 2], dtype=torch.int32
                    ),
                    seq_lens=torch.tensor([1, 1], dtype=torch.int32),
                    slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
                )
                runner = object.__new__(ModelRunner)
                runner.cudagraph_policy = CUDAGraphPolicy.NONE
                runner.attention_backend = _BackendRecorder()
                runner.model = _ModelRecorder()
                set_context(attention_metadata=metadata)

                runner.run_model(
                    torch.tensor([7, 8]),
                    torch.tensor([0, 0]),
                )

                context = get_context()
                self.assertIs(runner.attention_backend.metadata, metadata)
                self.assertIs(context.batch_type, expected)
                reset_context()


class ModelRunnerWarmupTest(TestCase):

    def test_warmup_covers_maximum_packed_token_capacity(self):
        cases = (
            (16384, 4352, 4, [4352, 4352, 4352, 3328]),
            (16384, 4352, 2, [4352, 4352]),
            (8, 96, 4, [8]),
        )
        for batch_tokens, model_len, max_seqs, expected in cases:
            with self.subTest(
                batch_tokens=batch_tokens,
                model_len=model_len,
                max_seqs=max_seqs,
            ):
                runner = object.__new__(ModelRunner)
                runner.config = SimpleNamespace(
                    max_num_batched_tokens=batch_tokens,
                    max_model_len=model_len,
                    max_num_seqs=max_seqs,
                )
                runner.block_size = 16
                runner.run = Mock()
                with (
                    patch("torch.cuda.empty_cache"),
                    patch("torch.cuda.reset_peak_memory_stats"),
                ):
                    runner.warmup_model()

                sequences = runner.run.call_args.args[0]
                self.assertEqual([len(seq) for seq in sequences], expected)
                self.assertEqual(
                    [seq.num_new_tokens for seq in sequences],
                    expected,
                )


@skipUnless(torch.cuda.is_available(), "ModelRunner metadata preparation uses CUDA")
class ModelRunnerMetadataTest(TestCase):

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.runner = object.__new__(ModelRunner)
        self.runner.block_size = 16
        self.runner.config = SimpleNamespace(attention_backend="flashinfer")

        self.runner.attention_backend = _BackendRecorder()

    def test_mixed_prefix_extension_and_fresh_prefill_common_metadata(self):
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
        metadata = context.attention_metadata

        self.assertIsNotNone(metadata)
        self.assertIsNone(context.attention_plan)
        self.assertEqual(prefix_extend.num_context_tokens, 19)
        self.assertEqual(fresh_prefill.num_context_tokens, 5)
        self.assertEqual(
            _tolist(input_ids),
            [116, 117, 118, 200, 201, 202, 203, 204],
        )
        self.assertEqual(
            _tolist(positions), [16, 17, 18, 0, 1, 2, 3, 4]
        )
        self.assertEqual(_tolist(metadata.query_start_loc), [0, 3, 8])
        self.assertEqual(_tolist(metadata.seq_lens), [19, 5])
        self.assertEqual(
            _tolist(metadata.slot_mapping),
            [64, 65, 66, 144, 145, 146, 147, 148],
        )
        self.assertEqual(_tolist(context.seq_need_compute_logits), [0, 1])
        self.assertEqual(
            metadata.block_tables.cpu().tolist(),
            [[7, 4], [9, -1]],
        )
        self.assertTrue(metadata.trusted)
        self.assertEqual(metadata.block_counts, (2, 1))
        self.assertEqual(metadata.num_kv_blocks, 3)
        self.assertEqual(metadata.num_prefill_kv_blocks, 3)
        self.assertEqual(metadata.max_q_len, 5)
        self.assertEqual(metadata.max_kv_len, 19)
        self.assertEqual(metadata.num_prefill_seqs, 2)
        self.assertEqual(metadata.num_decode_seqs, 0)
        self.assertEqual(metadata.num_prefill_tokens, 8)
        self.assertEqual(metadata.num_decode_tokens, 0)
        self.assertIsNone(context.batch_type)

    def test_warmup_without_block_table_has_cacheless_common_metadata(self):
        warmup = Sequence([10, 11, 12, 13], block_size=16)
        warmup.num_cached_tokens = 0
        warmup.num_new_tokens = 4

        input_ids, positions = self.runner.prepare_model_input([warmup])
        context = get_context()
        metadata = context.attention_metadata

        self.assertIsNotNone(metadata)
        self.assertEqual(_tolist(input_ids), [10, 11, 12, 13])
        self.assertEqual(_tolist(positions), [0, 1, 2, 3])
        self.assertEqual(_tolist(metadata.query_start_loc), [0, 4])
        self.assertEqual(_tolist(metadata.seq_lens), [4])
        self.assertEqual(_tolist(metadata.slot_mapping), [-1, -1, -1, -1])
        self.assertIsNone(context.seq_need_compute_logits)
        self.assertIsNone(metadata.block_tables)
        self.assertEqual(metadata.block_counts, ())
        self.assertTrue(metadata.trusted)
        self.assertEqual(metadata.num_kv_blocks, 0)
        self.assertEqual(metadata.num_prefill_kv_blocks, 0)
        self.assertEqual(metadata.max_q_len, 4)
        self.assertEqual(metadata.max_kv_len, 4)
        self.assertEqual(metadata.num_prefill_seqs, 1)
        self.assertEqual(metadata.num_decode_seqs, 0)
        self.assertEqual(metadata.num_prefill_tokens, 4)
        self.assertEqual(metadata.num_decode_tokens, 0)
        self.assertIsNone(context.attention_plan)
        self.assertIsNone(context.batch_type)

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
        metadata = context.attention_metadata

        self.assertIsNotNone(metadata)
        self.assertEqual(_tolist(input_ids), [1, 2, 3, 117])
        self.assertEqual(_tolist(positions), [0, 1, 2, 17])
        self.assertEqual(_tolist(metadata.query_start_loc), [0, 3, 4])
        self.assertEqual(_tolist(metadata.seq_lens), [3, 18])
        self.assertEqual(_tolist(metadata.slot_mapping), [32, 33, 34, 81])
        self.assertEqual(
            metadata.block_tables.cpu().tolist(),
            [[2, -1], [4, 5]],
        )
        self.assertTrue(metadata.trusted)
        self.assertEqual(metadata.block_counts, (1, 2))
        self.assertEqual(metadata.num_kv_blocks, 3)
        self.assertEqual(metadata.num_prefill_kv_blocks, 1)
        self.assertEqual(metadata.num_prefill_seqs, 1)
        self.assertEqual(metadata.num_decode_seqs, 1)
        self.assertEqual(metadata.num_prefill_tokens, 3)
        self.assertEqual(metadata.num_decode_tokens, 1)
        self.assertIsNone(context.attention_plan)
        self.assertIsNone(context.batch_type)

    def test_multi_token_chunk_continuation_is_pure_prefill(self):
        chunked = Sequence(list(range(20)), block_size=16)
        chunked.num_cached_tokens = 16
        chunked.num_new_tokens = 4
        chunked.block_table = [3, 4]

        self.runner.prepare_model_input([chunked], num_prefill_seqs=1)
        context = get_context()
        metadata = context.attention_metadata

        self.assertEqual(metadata.max_q_len, 4)
        self.assertEqual(metadata.num_prefill_seqs, 1)
        self.assertEqual(metadata.num_decode_seqs, 0)
        self.assertIsNone(context.attention_plan)
        self.assertIsNone(context.batch_type)

    def test_single_token_chunk_continuation_is_pure_prefill(self):
        chunked = Sequence(list(range(17)), block_size=16)
        chunked.num_cached_tokens = 16
        chunked.num_new_tokens = 1
        chunked.block_table = [3, 4]

        self.runner.prepare_model_input([chunked], num_prefill_seqs=1)
        context = get_context()
        metadata = context.attention_metadata

        self.assertEqual(metadata.max_q_len, 1)
        self.assertEqual(metadata.num_prefill_seqs, 1)
        self.assertEqual(metadata.num_decode_seqs, 0)
        self.assertIsNone(context.attention_plan)
        self.assertIsNone(context.batch_type)

    def test_normal_decode_is_pure_decode(self):
        decode = Sequence(list(range(17)), block_size=16)
        decode.num_cached_tokens = 16
        decode.num_new_tokens = 1
        decode.block_table = [3, 4]

        self.runner.prepare_model_input([decode], num_prefill_seqs=0)
        context = get_context()
        metadata = context.attention_metadata

        self.assertEqual(metadata.max_q_len, 1)
        self.assertEqual(metadata.num_prefill_seqs, 0)
        self.assertEqual(metadata.num_decode_seqs, 1)
        self.assertEqual(metadata.num_prefill_tokens, 0)
        self.assertEqual(metadata.num_decode_tokens, 1)
        self.assertTrue(metadata.trusted)
        self.assertEqual(metadata.block_counts, (2,))
        self.assertEqual(metadata.num_kv_blocks, 2)
        self.assertEqual(metadata.num_prefill_kv_blocks, 0)
        self.assertIsNone(context.attention_plan)
        self.assertIsNone(context.batch_type)

    def test_backend_selection_does_not_change_common_metadata(self):
        snapshots = []
        for backend_name in ("flashinfer", "legacy"):
            self.runner.config.attention_backend = backend_name
            decode = Sequence(list(range(17)), block_size=16)
            decode.num_cached_tokens = 16
            decode.num_new_tokens = 1
            decode.block_table = [3, 4]

            self.runner.prepare_model_input([decode], num_prefill_seqs=0)
            metadata = get_context().attention_metadata
            snapshots.append(
                {
                    "query_start_loc": _tolist(metadata.query_start_loc),
                    "seq_lens": _tolist(metadata.seq_lens),
                    "slot_mapping": _tolist(metadata.slot_mapping),
                    "block_tables": metadata.block_tables.cpu().tolist(),
                    "block_counts": metadata.block_counts,
                    "num_kv_blocks": metadata.num_kv_blocks,
                    "num_prefill_kv_blocks": (
                        metadata.num_prefill_kv_blocks
                    ),
                }
            )
            reset_context()

        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(
            snapshots[0],
            {
                "query_start_loc": [0, 1],
                "seq_lens": [17],
                "slot_mapping": [64],
                "block_tables": [[3, 4]],
                "block_counts": (2,),
                "num_kv_blocks": 2,
                "num_prefill_kv_blocks": 0,
            },
        )

    def test_all_single_token_mixed_batch_is_still_mixed(self):
        prefill = Sequence([7], block_size=16)
        prefill.num_new_tokens = 1
        prefill.block_table = [2]

        decode = Sequence(list(range(100, 117)), block_size=16)
        decode.num_cached_tokens = 16
        decode.num_new_tokens = 1
        decode.block_table = [4, 5]

        self.runner.prepare_model_input(
            [prefill, decode],
            num_prefill_seqs=1,
        )
        context = get_context()
        metadata = context.attention_metadata

        self.assertEqual(metadata.max_q_len, 1)
        self.assertEqual(metadata.num_prefill_seqs, 1)
        self.assertEqual(metadata.num_decode_seqs, 1)
        self.assertEqual(metadata.num_prefill_tokens, 1)
        self.assertEqual(metadata.num_decode_tokens, 1)
        self.assertIsNone(context.attention_plan)
        self.assertIsNone(context.batch_type)

    def test_empty_execution_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty execution batch"):
            self.runner.prepare_model_input([])


if __name__ == "__main__":
    main()
