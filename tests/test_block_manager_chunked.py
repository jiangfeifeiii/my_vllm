import pickle

import pytest

import nanovllm.engine.block_manager as block_manager_module
import nanovllm.engine.sequence as sequence_module
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
    assert prefix_block.parent_block_id == -1
    assert prefix_block.parent_generation == -1
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
    assert prefix_block.parent_block_id == -1
    assert prefix_block.parent_generation == -1
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


def test_sequence_caches_only_chained_full_block_hashes():
    tokens = list(range(2 * 16 + 3))
    seq = Sequence(tokens, block_size=16)
    first_hash = BlockManager.compute_hash(tokens[:16])
    second_hash = BlockManager.compute_hash(tokens[16:32], first_hash)

    assert seq.block_hashes == [first_hash, second_hash]

    for token_id in range(12):
        seq.append_token(100_000 + token_id)
        assert seq.block_hashes == [first_hash, second_hash]

    seq.append_token(100_012)
    expected_third = BlockManager.compute_hash(seq.block(2), second_hash)
    assert seq.block_hashes == [first_hash, second_hash, expected_third]

    with pytest.raises(AttributeError):
        seq.block_size = 32


def test_sequence_pickle_preserves_hash_cache_and_continues_incrementally():
    seq = Sequence(list(range(31)), block_size=16)
    restored = pickle.loads(pickle.dumps(seq))

    assert restored.block_hashes == seq.block_hashes
    restored.append_token(99_001)

    expected_second = BlockManager.compute_hash(
        restored.block(1), restored.block_hashes[0]
    )
    assert restored.block_hashes == [seq.block_hashes[0], expected_second]


def test_sequence_legacy_pickle_state_rebuilds_hashes_before_boundary_append():
    original = Sequence(list(range(31)), block_size=16)
    legacy_state = original.__getstate__()
    legacy_state["block_size"] = legacy_state.pop("_block_size")
    legacy_state.pop("block_hashes")
    restored = Sequence.__new__(Sequence)

    restored.__setstate__(legacy_state)
    restored.append_token(99_002)

    assert restored.block_size == 16
    assert restored.block_hashes == Sequence(
        restored.token_ids,
        block_size=16,
    ).block_hashes


def test_sequence_boundary_hash_failure_is_atomic(monkeypatch):
    seq = Sequence(list(range(15)), block_size=16)
    state_before = pickle.dumps(seq.__dict__)

    def fail_hash(*_args, **_kwargs):
        raise RuntimeError("injected block hash failure")

    monkeypatch.setattr(sequence_module, "compute_block_hash", fail_hash)
    with pytest.raises(RuntimeError, match="injected block hash failure"):
        seq.append_token(99_003)

    assert pickle.dumps(seq.__dict__) == state_before


def test_block_manager_hot_paths_do_not_rehash_cached_sequence_blocks(
    monkeypatch,
):
    manager = BlockManager(num_blocks=4, block_size=16)
    prefix = list(range(32))
    leader = Sequence(prefix + [91_001], block_size=16)
    follower = Sequence(prefix + [91_002], block_size=16)

    def fail_rehash(*_args, **_kwargs):
        raise AssertionError("full block hash was recomputed")

    monkeypatch.setattr(sequence_module, "compute_block_hash", fail_rehash)
    monkeypatch.setattr(block_manager_module, "compute_block_hash", fail_rehash)

    leader.num_new_tokens = len(leader)
    manager.allocate_new(leader)
    latest = manager.match_prefix(follower)
    manager.claim_prefix(follower, latest)
    follower.num_new_tokens = 1
    manager.allocate_new(follower)

    assert latest == leader.block_table[:2]
    assert follower.block_table[:2] == leader.block_table[:2]

    manager.deallocate(leader)
    manager.deallocate(follower)


def test_hash_collision_candidate_never_reuses_different_tokens():
    manager = BlockManager(num_blocks=3, block_size=16)
    source = Sequence(list(range(16)) + [92_001], block_size=16)
    source.num_new_tokens = len(source)
    manager.allocate_new(source)
    source_block_id = source.block_table[0]
    manager.deallocate(source)

    target = Sequence(list(range(100, 116)) + [92_002], block_size=16)
    target.block_hashes[0] = source.block_hashes[0]

    assert manager.hash_to_block_id[target.block_hashes[0]] == source_block_id
    assert manager.match_prefix(target) == []
    assert target.block_table == []
    assert target.num_cached_tokens == 0


@pytest.mark.parametrize(
    "operation",
    [
        "match_prefix",
        "claim_prefix",
        "allocate_new",
        "get_token_layout",
        "allocate",
        "deallocate",
        "num_blocks_to_append",
        "can_append",
        "may_append",
    ],
)
def test_all_sequence_entrypoints_reject_mismatched_block_size(operation: str):
    manager = BlockManager(num_blocks=2, block_size=16)
    seq = Sequence([1, 2], block_size=32)
    seq.num_new_tokens = 1

    with pytest.raises(AssertionError, match="does not match"):
        if operation in ("can_append", "num_blocks_to_append"):
            getattr(manager, operation)(seq, 1)
        elif operation == "claim_prefix":
            manager.claim_prefix(seq, [])
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
