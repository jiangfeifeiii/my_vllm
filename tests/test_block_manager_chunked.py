import pickle

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


@pytest.mark.parametrize("block_size", [16, 256])
def test_partial_chunk_is_unpublished_until_the_block_is_full(block_size: int):
    manager = BlockManager(num_blocks=4, block_size=block_size)
    token_ids = list(range(block_size * 2 + 1))
    first_chunk_size = block_size // 2
    seq = Sequence(token_ids, block_size=block_size)

    seq.num_new_tokens = first_chunk_size
    assert manager.can_allocate(first_chunk_size)
    manager.allocate(seq)

    prefix_block_id = seq.block_table[0]
    prefix_block = manager.blocks[prefix_block_id]
    assert prefix_block.hash == -1
    assert prefix_block.token_ids == []
    assert -1 not in manager.hash_to_block_id

    # Mirror scheduler postprocessing, then complete the same physical block.
    seq.num_cached_tokens = first_chunk_size
    seq.num_new_tokens = block_size - first_chunk_size
    assert manager.can_append(seq, seq.num_new_tokens)
    manager.may_append(seq)

    expected_tokens = token_ids[:block_size]
    expected_hash = manager.compute_hash(expected_tokens)
    assert prefix_block.hash == expected_hash
    assert prefix_block.token_ids == expected_tokens
    assert manager.hash_to_block_id[expected_hash] == prefix_block_id

    # A full block becomes visible immediately to a later request in this batch.
    follower = Sequence(expected_tokens + [99_999], block_size=block_size)
    assert manager.get_token_layout(follower) == (block_size, 0, 1)
    follower.num_new_tokens = 1
    manager.allocate(follower)
    assert follower.block_table[0] == prefix_block_id
    assert prefix_block.ref_count == 2

    manager.deallocate(follower)
    manager.deallocate(seq)
    assert prefix_block.ref_count == 0
    assert prefix_block_id in manager.free_block_ids


@pytest.mark.parametrize(
    "operation",
    ["get_token_layout", "allocate", "deallocate", "can_append", "may_append"],
)
def test_all_sequence_entrypoints_reject_mismatched_block_size(operation: str):
    manager = BlockManager(num_blocks=2, block_size=16)
    seq = Sequence([1, 2], block_size=32)
    seq.num_new_tokens = 1

    with pytest.raises(AssertionError, match="does not match"):
        if operation == "can_append":
            manager.can_append(seq, 1)
        else:
            getattr(manager, operation)(seq)


def test_sequence_pickle_preserves_runtime_semantics_and_block_size():
    params = SamplingParams(temperature=0.7, max_tokens=11, ignore_eos=True)
    seq = Sequence(list(range(20)), params, block_size=16)
    seq.append_token(999)
    seq.status = SequenceStatus.RUNNING
    seq.num_cached_tokens = 16
    seq.num_new_tokens = 5
    seq.block_table = [3, 5]

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.__dict__ == seq.__dict__
    assert restored.block_size == 16
    assert restored.status is SequenceStatus.RUNNING
    assert restored.max_tokens == 11
    assert restored.ignore_eos is True
    assert restored.token_ids is not seq.token_ids
    assert restored.block_table is not seq.block_table
