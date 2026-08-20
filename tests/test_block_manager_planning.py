from copy import deepcopy

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


BLOCK_SIZE = 16


def _allocate_new(manager: BlockManager, seq: Sequence, num_tokens: int) -> None:
    seq.num_new_tokens = num_tokens
    manager.allocate_new(seq)


def _seed_free_prefix(manager: BlockManager, prefix: list[int]) -> list[int]:
    seed = Sequence(prefix + [90_001], block_size=manager.block_size)
    _allocate_new(manager, seed, len(seed))
    prefix_ids = seed.block_table[:-1]
    manager.deallocate(seed)
    return prefix_ids


def _manager_state(manager: BlockManager):
    return (
        tuple(manager.free_block_ids),
        frozenset(manager.used_block_ids),
        dict(manager.hash_to_block_id),
        tuple(
            (block.ref_count, block.hash, tuple(block.token_ids))
            for block in manager.blocks
        ),
    )


def _assert_invariants(manager: BlockManager) -> None:
    free_ids = tuple(manager.free_block_ids)
    free = set(free_ids)
    used = manager.used_block_ids
    all_ids = set(range(len(manager.blocks)))

    assert len(free_ids) == len(free)
    assert free.isdisjoint(used)
    assert free | used == all_ids
    for block_id, block in enumerate(manager.blocks):
        if block_id in free:
            assert block.ref_count == 0
        else:
            assert block.ref_count > 0
        if block.hash != -1:
            assert len(block.token_ids) == manager.block_size
            assert manager.hash_to_block_id[block.hash] == block_id
    for block_hash, block_id in manager.hash_to_block_id.items():
        block = manager.blocks[block_id]
        assert block.hash == block_hash
        assert len(block.token_ids) == manager.block_size


def test_match_prefix_is_pure_read_and_honors_max_blocks():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    prefix = list(range(BLOCK_SIZE * 3))
    expected_ids = _seed_free_prefix(manager, prefix)
    seq = Sequence(prefix + [90_002], block_size=BLOCK_SIZE)
    manager_before = _manager_state(manager)
    seq_before = deepcopy(seq.__dict__)

    assert manager.match_prefix(seq, max_blocks=0) == []
    assert manager.match_prefix(seq, max_blocks=2) == expected_ids[:2]
    assert manager.match_prefix(seq) == expected_ids

    assert _manager_state(manager) == manager_before
    assert seq.__dict__ == seq_before
    _assert_invariants(manager)


def test_match_prefix_never_reuses_the_requests_last_block():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    tokens = list(range(BLOCK_SIZE * 3))
    expected_ids = _seed_free_prefix(manager, tokens)
    exact_request = Sequence(tokens, block_size=BLOCK_SIZE)

    assert len(expected_ids) == 3
    assert manager.match_prefix(exact_request) == expected_ids[:2]


def test_two_plans_share_one_cached_free_claim_without_double_consumption():
    manager = BlockManager(num_blocks=5, block_size=BLOCK_SIZE)
    prefix = list(range(BLOCK_SIZE))
    prefix_id = _seed_free_prefix(manager, prefix)[0]
    first = Sequence(prefix + [90_003], block_size=BLOCK_SIZE)
    second = Sequence(prefix + [90_004], block_size=BLOCK_SIZE)

    first_plan = manager.match_prefix(first)
    second_plan = manager.match_prefix(second)
    assert first_plan == second_plan == [prefix_id]
    free_before = len(manager.free_block_ids)

    manager.claim_prefix(first, first_plan)
    assert len(manager.free_block_ids) == free_before - 1
    assert manager.blocks[prefix_id].ref_count == 1
    manager.claim_prefix(second, second_plan)

    assert len(manager.free_block_ids) == free_before - 1
    assert manager.blocks[prefix_id].ref_count == 2
    assert first.block_table == second.block_table == [prefix_id]
    assert first.num_cached_tokens == second.num_cached_tokens == BLOCK_SIZE
    _assert_invariants(manager)

    manager.deallocate(first)
    manager.deallocate(second)
    _assert_invariants(manager)


def test_claimed_prefix_is_not_recycled_by_other_allocations():
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    prefix = list(range(BLOCK_SIZE))
    prefix_id = _seed_free_prefix(manager, prefix)[0]
    protected = Sequence(prefix + [90_005], block_size=BLOCK_SIZE)
    manager.claim_prefix(protected, manager.match_prefix(protected))
    protected_hash = manager.blocks[prefix_id].hash
    protected_tokens = manager.blocks[prefix_id].token_ids.copy()

    other = Sequence(list(range(1_000, 1_000 + BLOCK_SIZE * 3)), block_size=BLOCK_SIZE)
    _allocate_new(manager, other, len(other))

    assert prefix_id in manager.used_block_ids
    assert prefix_id not in manager.free_block_ids
    assert manager.blocks[prefix_id].ref_count == 1
    assert manager.blocks[prefix_id].hash == protected_hash
    assert manager.blocks[prefix_id].token_ids == protected_tokens
    _assert_invariants(manager)

    manager.deallocate(other)
    manager.deallocate(protected)
    _assert_invariants(manager)


def test_follower_continues_matching_after_leader_publishes_a_real_block():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    first_tokens = list(range(BLOCK_SIZE))
    second_tokens = list(range(100, 100 + BLOCK_SIZE))
    first_id = _seed_free_prefix(manager, first_tokens)[0]
    request_tokens = first_tokens + second_tokens + [90_006]
    leader = Sequence(request_tokens, block_size=BLOCK_SIZE)
    follower = Sequence(request_tokens, block_size=BLOCK_SIZE)

    manager.claim_prefix(leader, manager.match_prefix(leader, max_blocks=1))
    manager.claim_prefix(follower, manager.match_prefix(follower, max_blocks=1))
    assert leader.block_table == follower.block_table == [first_id]

    _allocate_new(manager, leader, BLOCK_SIZE)
    second_id = leader.block_table[1]
    continuation = manager.match_prefix(follower)
    assert continuation == [second_id]
    manager.claim_prefix(follower, continuation)

    assert follower.num_cached_tokens == BLOCK_SIZE * 2
    assert follower.block_table == [first_id, second_id]
    assert manager.blocks[first_id].ref_count == 2
    assert manager.blocks[second_id].ref_count == 2
    _assert_invariants(manager)

    manager.deallocate(leader)
    manager.deallocate(follower)
    _assert_invariants(manager)


def test_claim_prefix_rejects_a_stale_plan_without_mutating_sequence():
    manager = BlockManager(num_blocks=3, block_size=BLOCK_SIZE)
    prefix = list(range(BLOCK_SIZE))
    target = Sequence(prefix + [90_007], block_size=BLOCK_SIZE)
    _seed_free_prefix(manager, prefix)
    plan = manager.match_prefix(target)
    target_before = deepcopy(target.__dict__)

    reclaimer = Sequence(
        list(range(2_000, 2_000 + BLOCK_SIZE * 3)),
        block_size=BLOCK_SIZE,
    )
    _allocate_new(manager, reclaimer, len(reclaimer))

    with pytest.raises(AssertionError, match="stale"):
        manager.claim_prefix(target, plan)

    assert target.__dict__ == target_before
    _assert_invariants(manager)


def test_num_blocks_to_append_counts_partial_capacity_exactly():
    manager = BlockManager(num_blocks=3, block_size=BLOCK_SIZE)
    seq = Sequence(list(range(BLOCK_SIZE * 2)), block_size=BLOCK_SIZE)
    _allocate_new(manager, seq, BLOCK_SIZE // 2)
    seq.num_cached_tokens = BLOCK_SIZE // 2
    seq.num_new_tokens = 0

    assert manager.num_blocks_to_append(seq, 0) == 0
    assert manager.num_blocks_to_append(seq, BLOCK_SIZE // 2) == 0
    assert manager.num_blocks_to_append(seq, BLOCK_SIZE // 2 + 1) == 1
    assert manager.can_append(seq, BLOCK_SIZE // 2 + 1)
