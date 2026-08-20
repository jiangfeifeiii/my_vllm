from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


BLOCK_SIZE = 256


def _allocate(manager: BlockManager, seq: Sequence, num_new_tokens: int) -> None:
    seq.num_new_tokens = num_new_tokens
    assert manager.can_allocate(num_new_tokens)
    manager.allocate(seq)


def test_cached_free_hit_and_shared_refcount_lifecycle() -> None:
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    shared_prefix = list(range(BLOCK_SIZE))

    cold = Sequence(shared_prefix + [10_001])
    assert manager.get_token_layout(cold) == (0, 0, BLOCK_SIZE + 1)
    _allocate(manager, cold, len(cold))

    prefix_block_id = cold.block_table[0]
    prefix_hash = manager.blocks[prefix_block_id].hash
    assert prefix_hash != -1
    assert manager.blocks[prefix_block_id].ref_count == 1
    assert prefix_block_id in manager.used_block_ids

    manager.deallocate(cold)

    assert manager.blocks[prefix_block_id].ref_count == 0
    assert prefix_block_id in manager.free_block_ids
    assert prefix_block_id not in manager.used_block_ids
    assert manager.hash_to_block_id[prefix_hash] == prefix_block_id
    assert manager.blocks[prefix_block_id].token_ids == shared_prefix

    first_hit = Sequence(shared_prefix + [10_002])
    assert manager.get_token_layout(first_hit) == (0, BLOCK_SIZE, 1)
    _allocate(manager, first_hit, 1)

    assert first_hit.num_cached_tokens == BLOCK_SIZE
    assert first_hit.block_table[0] == prefix_block_id
    assert manager.blocks[prefix_block_id].ref_count == 1
    assert prefix_block_id in manager.used_block_ids
    assert prefix_block_id not in manager.free_block_ids

    shared_hit = Sequence(shared_prefix + [10_003])
    assert manager.get_token_layout(shared_hit) == (BLOCK_SIZE, 0, 1)
    _allocate(manager, shared_hit, 1)

    assert shared_hit.num_cached_tokens == BLOCK_SIZE
    assert shared_hit.block_table[0] == prefix_block_id
    assert manager.blocks[prefix_block_id].ref_count == 2

    manager.deallocate(first_hit)
    assert manager.blocks[prefix_block_id].ref_count == 1
    assert prefix_block_id in manager.used_block_ids
    assert prefix_block_id not in manager.free_block_ids

    manager.deallocate(shared_hit)
    assert manager.blocks[prefix_block_id].ref_count == 0
    assert prefix_block_id in manager.free_block_ids
    assert prefix_block_id not in manager.used_block_ids
    assert manager.hash_to_block_id[prefix_hash] == prefix_block_id


def test_last_full_block_is_not_prefix_cache_hit() -> None:
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    token_ids = list(range(BLOCK_SIZE * 2))

    seed = Sequence(token_ids)
    _allocate(manager, seed, len(seed))
    first_block_id, last_block_id = seed.block_table
    assert manager.blocks[first_block_id].hash != -1
    assert manager.blocks[last_block_id].hash != -1

    manager.deallocate(seed)

    repeated = Sequence(token_ids)
    assert manager.get_token_layout(repeated) == (0, BLOCK_SIZE, BLOCK_SIZE)
    _allocate(manager, repeated, BLOCK_SIZE)

    assert repeated.num_cached_tokens == BLOCK_SIZE
    assert repeated.block_table[0] == first_block_id
    assert repeated.block_table[1] != last_block_id

