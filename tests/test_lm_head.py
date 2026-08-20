from unittest import TestCase, main
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.utils.context import reset_context, set_context


class ParallelLMHeadTest(TestCase):

    def setUp(self):
        self.rank_patch = patch("torch.distributed.get_rank", return_value=0)
        self.world_size_patch = patch(
            "torch.distributed.get_world_size", return_value=1
        )
        self.rank_patch.start()
        self.world_size_patch.start()

        self.vocab_size = 7
        self.hidden_size = 3
        self.head = ParallelLMHead(self.vocab_size, self.hidden_size)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.arange(
                    self.vocab_size * self.hidden_size,
                    dtype=torch.float32,
                ).reshape(self.vocab_size, self.hidden_size)
                / 10
            )
        self.hidden_states = torch.arange(
            5 * self.hidden_size, dtype=torch.float32
        ).reshape(5, self.hidden_size)
        self.cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)

    def tearDown(self):
        reset_context()
        self.world_size_patch.stop()
        self.rank_patch.stop()

    def test_empty_logits_selector_preserves_empty_batch_dimension(self):
        set_context(
            cu_seqlens_q=self.cu_seqlens_q,
            seq_need_compute_logits=torch.tensor([], dtype=torch.int32),
        )

        logits = self.head(self.hidden_states)

        self.assertEqual(logits.shape, torch.Size([0, self.vocab_size]))
        self.assertEqual(logits.dtype, self.hidden_states.dtype)
        self.assertEqual(logits.numel(), 0)

    def test_nonempty_logits_selector_selects_requested_sequences(self):
        selector = torch.tensor([1, 0], dtype=torch.int32)
        set_context(
            cu_seqlens_q=self.cu_seqlens_q,
            seq_need_compute_logits=selector,
        )

        logits = self.head(self.hidden_states)

        last_token_indices = torch.tensor([4, 1], dtype=torch.int64)
        expected = F.linear(
            self.hidden_states[last_token_indices], self.head.weight
        )
        self.assertEqual(logits.shape, torch.Size([2, self.vocab_size]))
        torch.testing.assert_close(logits, expected, atol=0, rtol=0)


if __name__ == "__main__":
    main()
