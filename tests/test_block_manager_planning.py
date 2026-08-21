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
            (
                block.ref_count,
                block.generation,
                block.hash,
                block.lineage_block_id,
                block.lineage_generation,
                block.parent_block_id,
                block.parent_generation,
                tuple(block.token_ids),
            )
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
            assert block.lineage_block_id != -1
            assert block.lineage_generation != -1
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


def test_claiming_cached_free_chain_preserves_physical_lineage():
    manager = BlockManager(num_blocks=6, block_size=BLOCK_SIZE)
    prefix = list(range(BLOCK_SIZE * 2))
    prefix_ids = _seed_free_prefix(manager, prefix)
    generations = [
        manager.blocks[block_id].generation for block_id in prefix_ids
    ]
    target = Sequence(prefix + [90_008], block_size=BLOCK_SIZE)

    plan = manager.match_prefix(target)
    assert plan == prefix_ids
    manager.claim_prefix(target, plan)

    assert [
        manager.blocks[block_id].generation for block_id in prefix_ids
    ] == generations
    child = manager.blocks[prefix_ids[1]]
    assert child.parent_block_id == prefix_ids[0]
    assert child.parent_generation == generations[0]
    _assert_invariants(manager)


def test_hash_collision_cannot_splice_child_from_different_parent():
    manager = BlockManager(num_blocks=10, block_size=BLOCK_SIZE)
    first_a = list(range(BLOCK_SIZE))
    first_b = list(range(1_000, 1_000 + BLOCK_SIZE))
    second = list(range(2_000, 2_000 + BLOCK_SIZE))

    chain_a = Sequence(first_a + second + [90_009], block_size=BLOCK_SIZE)
    _allocate_new(manager, chain_a, len(chain_a))
    parent_a_id, child_a_id = chain_a.block_table[:2]
    manager.deallocate(chain_a)

    seed_b = Sequence(first_b + [90_010], block_size=BLOCK_SIZE)
    _allocate_new(manager, seed_b, len(seed_b))
    parent_b_id = seed_b.block_table[0]
    manager.deallocate(seed_b)

    target = Sequence(first_b + second + [90_011], block_size=BLOCK_SIZE)
    target.block_hashes[1] = chain_a.block_hashes[1]
    assert manager.hash_to_block_id[target.block_hashes[0]] == parent_b_id
    assert manager.hash_to_block_id[target.block_hashes[1]] == child_a_id
    assert parent_a_id != parent_b_id
    assert manager.match_prefix(target) == [parent_b_id]

    manager_before = _manager_state(manager)
    target_before = deepcopy(target.__dict__)
    with pytest.raises(AssertionError, match="lineage"):
        manager.claim_prefix(target, [parent_b_id, child_a_id])

    assert _manager_state(manager) == manager_before
    assert target.__dict__ == target_before
    _assert_invariants(manager)


def test_parent_generation_prevents_aba_lineage_splice():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    first_a = list(range(BLOCK_SIZE))
    first_b = list(range(3_000, 3_000 + BLOCK_SIZE))
    second = list(range(4_000, 4_000 + BLOCK_SIZE))

    chain_a = Sequence(first_a + second + [90_012], block_size=BLOCK_SIZE)
    _allocate_new(manager, chain_a, len(chain_a))
    parent_id, child_id = chain_a.block_table[:2]
    old_generation = manager.blocks[parent_id].generation
    manager.deallocate(chain_a)

    # Force the old physical parent page to be destructively reused. The
    # synthetic hash collision keeps both map lookups viable, so only the
    # generation in the child's lineage can reject the stale chain.
    manager.free_block_ids.remove(parent_id)
    manager.free_block_ids.appendleft(parent_id)
    replacement = Sequence(first_b + [90_013], block_size=BLOCK_SIZE)
    replacement.block_hashes[0] = chain_a.block_hashes[0]
    _allocate_new(manager, replacement, len(replacement))

    assert replacement.block_table[0] == parent_id
    assert manager.blocks[parent_id].generation == old_generation + 1
    assert manager.hash_to_block_id[chain_a.block_hashes[1]] == child_id

    target = Sequence(first_b + second + [90_014], block_size=BLOCK_SIZE)
    target.block_hashes[:] = chain_a.block_hashes
    assert manager.match_prefix(target) == [parent_id]
    _assert_invariants(manager)


def test_duplicate_root_publication_preserves_longer_cached_chain():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    first = list(range(BLOCK_SIZE))
    second = list(range(5_000, 5_000 + BLOCK_SIZE))

    long_seq = Sequence(first + second + [90_015], block_size=BLOCK_SIZE)
    _allocate_new(manager, long_seq, len(long_seq))
    long_parent_id, child_id = long_seq.block_table[:2]
    canonical_lineage = manager._lineage(long_parent_id)
    manager.deallocate(long_seq)

    short_seq = Sequence(first + [90_016], block_size=BLOCK_SIZE)
    _allocate_new(manager, short_seq, len(short_seq))
    duplicate_parent_id = short_seq.block_table[0]
    manager.deallocate(short_seq)

    assert duplicate_parent_id != long_parent_id
    assert manager._lineage(duplicate_parent_id) == canonical_lineage
    target = Sequence(first + second + [90_017], block_size=BLOCK_SIZE)
    assert manager.match_prefix(target) == [duplicate_parent_id, child_id]
    _assert_invariants(manager)


def test_duplicate_middle_publication_preserves_cached_descendant():
    manager = BlockManager(num_blocks=10, block_size=BLOCK_SIZE)
    first = list(range(BLOCK_SIZE))
    second = list(range(6_000, 6_000 + BLOCK_SIZE))
    third = list(range(7_000, 7_000 + BLOCK_SIZE))

    long_seq = Sequence(
        first + second + third + [90_018],
        block_size=BLOCK_SIZE,
    )
    _allocate_new(manager, long_seq, len(long_seq))
    root_id, middle_id, child_id = long_seq.block_table[:3]
    canonical_middle_lineage = manager._lineage(middle_id)
    manager.deallocate(long_seq)

    shorter = Sequence(first + second, block_size=BLOCK_SIZE)
    manager.claim_prefix(shorter, manager.match_prefix(shorter))
    shorter.num_new_tokens = BLOCK_SIZE
    manager.allocate_new(shorter)
    duplicate_middle_id = shorter.block_table[1]
    manager.deallocate(shorter)

    assert manager.hash_to_block_id[long_seq.block_hashes[0]] == root_id
    assert duplicate_middle_id != middle_id
    assert manager._lineage(duplicate_middle_id) == canonical_middle_lineage
    target = Sequence(
        first + second + third + [90_019],
        block_size=BLOCK_SIZE,
    )
    assert manager.match_prefix(target) == [
        root_id,
        duplicate_middle_id,
        child_id,
    ]
    _assert_invariants(manager)


def test_recycling_collision_non_owner_preserves_current_hash_owner():
    manager = BlockManager(num_blocks=5, block_size=BLOCK_SIZE)
    first = Sequence(list(range(BLOCK_SIZE)) + [90_020], block_size=BLOCK_SIZE)
    _allocate_new(manager, first, len(first))
    non_owner_id = first.block_table[0]
    collision_hash = first.block_hashes[0]
    manager.deallocate(first)

    second = Sequence(
        list(range(8_000, 8_000 + BLOCK_SIZE)) + [90_021],
        block_size=BLOCK_SIZE,
    )
    second.block_hashes[0] = collision_hash
    _allocate_new(manager, second, len(second))
    owner_id = second.block_table[0]
    manager.deallocate(second)

    assert owner_id != non_owner_id
    assert manager.hash_to_block_id[collision_hash] == owner_id
    manager._allocate_block(non_owner_id)
    assert manager.hash_to_block_id[collision_hash] == owner_id

    manager.blocks[non_owner_id].ref_count = 0
    manager._deallocate_block(non_owner_id)
    manager._allocate_block(owner_id)
    assert collision_hash not in manager.hash_to_block_id


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
