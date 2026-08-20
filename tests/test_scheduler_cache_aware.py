from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


BLOCK_SIZE = 16


def _scheduler(
    *,
    num_blocks: int = 12,
    token_budget: int = 64,
    max_num_seqs: int = 8,
    chunked: bool = True,
) -> Scheduler:
    config = SimpleNamespace(
        chunked_prefill=chunked,
        max_model_len=256,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=token_budget,
        eos=-1,
        num_kvcache_blocks=num_blocks,
        kvcache_block_size=BLOCK_SIZE,
    )
    return Scheduler(config)


def _allocate_new(scheduler: Scheduler, seq: Sequence, num_tokens: int) -> None:
    seq.num_new_tokens = num_tokens
    scheduler.block_manager.allocate_new(seq)


def _publish_free_prefix(scheduler: Scheduler, prefix: list[int]) -> list[int]:
    seed = Sequence(prefix + [900_001], block_size=BLOCK_SIZE)
    _allocate_new(scheduler, seed, len(seed))
    prefix_ids = seed.block_table[:-1]
    scheduler.block_manager.deallocate(seed)
    return prefix_ids


def _add_decode(scheduler: Scheduler, token_base: int) -> Sequence:
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    seq = Sequence(
        list(range(token_base, token_base + BLOCK_SIZE)),
        params,
        block_size=BLOCK_SIZE,
    )
    _allocate_new(scheduler, seq, BLOCK_SIZE)
    seq.num_cached_tokens = BLOCK_SIZE
    seq.num_new_tokens = 0
    seq.status = SequenceStatus.RUNNING
    seq.append_token(token_base + 10_000)
    scheduler.running.append(seq)
    return seq


def _assert_block_invariants(scheduler: Scheduler) -> None:
    manager = scheduler.block_manager
    free_ids = tuple(manager.free_block_ids)
    free = set(free_ids)
    used = manager.used_block_ids

    assert len(free_ids) == len(free)
    assert free.isdisjoint(used)
    assert free | used == set(range(len(manager.blocks)))
    for block_id, block in enumerate(manager.blocks):
        assert block.ref_count == 0 if block_id in free else block.ref_count > 0
    for block_hash, block_id in manager.hash_to_block_id.items():
        block = manager.blocks[block_id]
        assert block.hash == block_hash
        assert len(block.token_ids) == BLOCK_SIZE


def test_waiting_to_single_chunked_to_decode_to_finished():
    scheduler = _scheduler(token_budget=8, max_num_seqs=1)
    params = SamplingParams(max_tokens=2, ignore_eos=True)
    seq = Sequence(list(range(20)), params, block_size=BLOCK_SIZE)
    scheduler.add(seq)

    first = scheduler.schedule()
    assert first == [seq]
    assert seq.num_new_tokens == 8
    scheduler.postprocess(first, [], [])
    assert scheduler.chunked_req is seq
    assert not scheduler.running
    assert seq.num_cached_tokens == 8

    second = scheduler.schedule()
    assert second == [seq]
    scheduler.postprocess(second, [], [])
    assert scheduler.chunked_req is seq
    assert seq.num_cached_tokens == 16

    final_prefill = scheduler.schedule()
    assert final_prefill == [seq]
    assert seq.num_new_tokens == 4
    scheduler.postprocess(final_prefill, [70_001], [0])
    assert scheduler.chunked_req is None
    assert list(scheduler.running) == [seq]
    assert seq.num_cached_tokens == 20

    decode = scheduler.schedule()
    assert decode == [seq]
    assert scheduler.num_scheduled_prefill_seqs == 0
    scheduler.postprocess(decode, [70_002], [0])

    assert seq.status is SequenceStatus.FINISHED
    assert scheduler.is_finished()
    assert not seq.block_table
    _assert_block_invariants(scheduler)


def test_decode_budget_is_reserved_and_batch_order_is_prefill_then_decode():
    scheduler = _scheduler(
        num_blocks=3,
        token_budget=2,
        max_num_seqs=3,
    )
    decode = _add_decode(scheduler, 100)
    first_prefill = Sequence([1], block_size=BLOCK_SIZE)
    second_prefill = Sequence([2], block_size=BLOCK_SIZE)
    scheduler.add(first_prefill)
    scheduler.add(second_prefill)

    scheduled = scheduler.schedule()

    assert scheduled == [first_prefill, decode]
    assert scheduler.num_scheduled_prefill_seqs == 1
    assert sum(seq.num_new_tokens for seq in scheduled) == 2
    assert list(scheduler.waiting) == [second_prefill]
    assert len(scheduler.block_manager.free_block_ids) == 0

    scheduler.block_manager.deallocate(first_prefill)
    scheduler.block_manager.deallocate(decode)
    _assert_block_invariants(scheduler)


def test_longest_prefix_match_sort_is_stable_for_ties():
    scheduler = _scheduler()
    first_block = list(range(BLOCK_SIZE))
    second_block = list(range(100, 100 + BLOCK_SIZE))
    prefix = first_block + second_block
    _publish_free_prefix(scheduler, prefix)
    low = Sequence(
        first_block + list(range(500, 500 + BLOCK_SIZE)) + [1],
        block_size=BLOCK_SIZE,
    )
    high_first = Sequence(prefix + [2], block_size=BLOCK_SIZE)
    high_second = Sequence(prefix + [3], block_size=BLOCK_SIZE)
    scheduler.waiting.extend([low, high_first, high_second])

    ranked, plans = scheduler._match_waiting()

    assert ranked == [high_first, high_second, low]
    assert len(plans[high_first]) == len(plans[high_second]) == 2
    assert len(plans[low]) == 1


def test_real_prefixes_are_claimed_before_any_destructive_allocation(monkeypatch):
    scheduler = _scheduler(num_blocks=5, token_budget=17, max_num_seqs=2)
    prefix = list(range(BLOCK_SIZE))
    prefix_id = _publish_free_prefix(scheduler, prefix)[0]
    protected = Sequence(prefix + [800_001], block_size=BLOCK_SIZE)
    fresh = Sequence(
        list(range(1_000, 1_000 + BLOCK_SIZE)),
        block_size=BLOCK_SIZE,
    )
    scheduler.add(protected)
    scheduler.add(fresh)

    manager = scheduler.block_manager
    manager.free_block_ids.remove(prefix_id)
    manager.free_block_ids.appendleft(prefix_id)
    assert manager.free_block_ids[0] == prefix_id
    prefix_hash = manager.blocks[prefix_id].hash
    prefix_tokens = manager.blocks[prefix_id].token_ids.copy()
    events = []
    labels = {protected: "protected", fresh: "fresh"}
    original_claim = manager.claim_prefix
    original_allocate = manager.allocate_new

    def recording_claim(seq, block_ids):
        events.append(("claim", labels[seq], tuple(block_ids)))
        return original_claim(seq, block_ids)

    def recording_allocate(seq):
        events.append(("allocate", labels[seq], ()))
        return original_allocate(seq)

    monkeypatch.setattr(manager, "claim_prefix", recording_claim)
    monkeypatch.setattr(manager, "allocate_new", recording_allocate)

    scheduled = scheduler.schedule()

    assert scheduled == [protected, fresh]
    first_allocate = next(i for i, event in enumerate(events) if event[0] == "allocate")
    assert all(event[0] == "claim" for event in events[:first_allocate])
    assert events[0] == ("claim", "protected", (prefix_id,))
    assert manager.blocks[prefix_id].hash == prefix_hash
    assert manager.blocks[prefix_id].token_ids == prefix_tokens
    assert manager.hash_to_block_id[prefix_hash] == prefix_id
    assert prefix_id in manager.used_block_ids
    _assert_block_invariants(scheduler)

    manager.deallocate(protected)
    manager.deallocate(fresh)


def test_two_real_prefix_plans_share_cached_free_capacity_and_refcount():
    scheduler = _scheduler(num_blocks=4, token_budget=2, max_num_seqs=2)
    prefix = list(range(BLOCK_SIZE))
    prefix_id = _publish_free_prefix(scheduler, prefix)[0]
    first = Sequence(prefix + [810_001], block_size=BLOCK_SIZE)
    second = Sequence(prefix + [810_002], block_size=BLOCK_SIZE)
    scheduler.add(first)
    scheduler.add(second)
    free_before = len(scheduler.block_manager.free_block_ids)

    scheduled = scheduler.schedule()

    assert scheduled == [first, second]
    assert first.block_table[0] == second.block_table[0] == prefix_id
    assert scheduler.block_manager.blocks[prefix_id].ref_count == 2
    assert len(scheduler.block_manager.free_block_ids) == free_before - 3
    _assert_block_invariants(scheduler)

    scheduler.block_manager.deallocate(first)
    scheduler.block_manager.deallocate(second)
    assert scheduler.block_manager.blocks[prefix_id].ref_count == 0
    _assert_block_invariants(scheduler)


def test_in_batch_follower_reuses_leaders_full_block_and_runs_after_leader():
    scheduler = _scheduler(num_blocks=4, token_budget=34, max_num_seqs=2)
    shared = list(range(BLOCK_SIZE))
    leader = Sequence(shared + [820_001], block_size=BLOCK_SIZE)
    follower = Sequence(shared + [820_002], block_size=BLOCK_SIZE)
    scheduler.add(leader)
    scheduler.add(follower)

    scheduled = scheduler.schedule()

    assert scheduled == [leader, follower]
    assert follower.seq_id in scheduler.temporary_deprioritized
    shared_id = leader.block_table[0]
    assert follower.block_table[0] == shared_id
    assert scheduler.block_manager.blocks[shared_id].ref_count == 2
    assert scheduler.block_manager.blocks[shared_id].token_ids == shared
    _assert_block_invariants(scheduler)

    scheduler.block_manager.deallocate(leader)
    scheduler.block_manager.deallocate(follower)
    _assert_block_invariants(scheduler)


def test_decode_kv_shortage_preempts_running_tail_without_ref_leak():
    scheduler = _scheduler(num_blocks=3, token_budget=1, max_num_seqs=2)
    head = _add_decode(scheduler, 2_000)
    tail = _add_decode(scheduler, 3_000)
    tail_block_id = tail.block_table[0]
    assert list(scheduler.running) == [head, tail]
    assert len(scheduler.block_manager.free_block_ids) == 1

    scheduled = scheduler.schedule()

    assert scheduled == [head]
    assert list(scheduler.running) == [head]
    assert list(scheduler.waiting) == [tail]
    assert tail.status is SequenceStatus.WAITING
    assert tail.num_cached_tokens == tail.num_new_tokens == 0
    assert tail.block_table == []
    assert scheduler.block_manager.blocks[tail_block_id].ref_count == 0
    assert tail_block_id in scheduler.block_manager.free_block_ids
    assert sum(block.ref_count for block in scheduler.block_manager.blocks) == 2
    _assert_block_invariants(scheduler)

    scheduler.block_manager.deallocate(head)
    _assert_block_invariants(scheduler)
