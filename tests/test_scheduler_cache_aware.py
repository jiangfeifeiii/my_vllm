from types import SimpleNamespace

import pytest

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
    enable_lpm: bool = True,
    same_step: bool = True,
) -> Scheduler:
    config = SimpleNamespace(
        chunked_prefill=chunked,
        max_model_len=256,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=token_budget,
        eos=-1,
        num_kvcache_blocks=num_blocks,
        kvcache_block_size=BLOCK_SIZE,
        enable_lpm=enable_lpm,
        enable_same_step_prefix_reuse=same_step,
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


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, False])
def test_sampling_params_require_positive_integer_max_tokens(max_tokens):
    with pytest.raises(AssertionError, match="positive integer"):
        SamplingParams(max_tokens=max_tokens)


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


def test_unchunked_request_over_budget_fails_actionably():
    scheduler = _scheduler(token_budget=8, chunked=False)
    seq = Sequence(list(range(9)), block_size=BLOCK_SIZE)
    scheduler.add(seq)

    with pytest.raises(ValueError, match="enable chunked_prefill"):
        scheduler.schedule()

    assert list(scheduler.waiting) == [seq]
    assert seq.block_table == []
    _assert_block_invariants(scheduler)

@pytest.mark.parametrize("same_step", [False, True])
def test_oversized_lower_rank_is_deferred_after_prior_commit(same_step):
    scheduler = _scheduler(
        num_blocks=3,
        token_budget=4,
        max_num_seqs=2,
        chunked=False,
        same_step=same_step,
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    first = Sequence([73_001], params, block_size=BLOCK_SIZE)
    oversized = Sequence(list(range(5)), params, block_size=BLOCK_SIZE)
    scheduler.waiting.extend([first, oversized])

    scheduled = scheduler.schedule()

    assert scheduled == [first]
    assert first.status is SequenceStatus.RUNNING
    assert list(scheduler.waiting) == [oversized]
    assert oversized.status is SequenceStatus.WAITING
    assert oversized.block_table == []
    scheduler.postprocess(scheduled, [73_002], [0])
    assert first.status is SequenceStatus.FINISHED

    with pytest.raises(ValueError, match="enable chunked_prefill"):
        scheduler.schedule()

    assert list(scheduler.waiting) == [oversized]
    assert oversized.block_table == []
    assert not scheduler.block_manager.used_block_ids
    _assert_block_invariants(scheduler)


def test_oversized_waiting_request_does_not_block_decode_work():
    scheduler = _scheduler(
        num_blocks=3,
        token_budget=4,
        max_num_seqs=2,
        chunked=False,
    )
    decode = _add_decode(scheduler, 6_000)
    oversized = Sequence(list(range(5)), block_size=BLOCK_SIZE)
    scheduler.add(oversized)

    scheduled = scheduler.schedule()

    assert scheduled == [decode]
    assert decode.num_new_tokens == 1
    assert list(scheduler.waiting) == [oversized]
    assert oversized.block_table == []
    _assert_block_invariants(scheduler)

    scheduler.block_manager.deallocate(decode)
    scheduler.running.clear()
    _assert_block_invariants(scheduler)


def test_completed_chunk_waiting_prefill_and_decode_share_one_p_d_batch():
    scheduler = _scheduler(
        num_blocks=6,
        token_budget=10,
        max_num_seqs=3,
    )
    chunked = Sequence(
        list(range(20)),
        SamplingParams(max_tokens=2, ignore_eos=True),
        block_size=BLOCK_SIZE,
    )
    _allocate_new(scheduler, chunked, BLOCK_SIZE)
    chunked.num_cached_tokens = BLOCK_SIZE
    chunked.num_new_tokens = 0
    chunked.status = SequenceStatus.RUNNING
    scheduler.chunked_req = chunked

    decode = _add_decode(scheduler, 4000)
    waiting = Sequence(
        list(range(5000, 5005)),
        SamplingParams(max_tokens=2, ignore_eos=True),
        block_size=BLOCK_SIZE,
    )
    scheduler.add(waiting)

    scheduled = scheduler.schedule()

    assert scheduled == [chunked, waiting, decode]
    assert scheduler.num_scheduled_prefill_seqs == 2
    assert scheduler.num_scheduled_prefill_tokens == 9
    assert [seq.num_new_tokens for seq in scheduled] == [4, 5, 1]

    scheduler.postprocess(
        scheduled,
        [71_001, 71_002, 71_003],
        [0, 1, 2],
    )
    assert scheduler.chunked_req is None
    assert list(scheduler.running) == [decode, chunked, waiting]
    assert [seq.num_cached_tokens for seq in scheduled] == [20, 5, 17]
    _assert_block_invariants(scheduler)

    for seq in list(scheduler.running):
        scheduler.block_manager.deallocate(seq)
    scheduler.running.clear()
    _assert_block_invariants(scheduler)


def test_completed_chunk_publishes_before_waiting_latest_lookup():
    scheduler = _scheduler(
        num_blocks=3,
        token_budget=10,
        max_num_seqs=2,
    )
    shared = list(range(BLOCK_SIZE))
    chunked = Sequence(
        shared + [72_001],
        SamplingParams(max_tokens=1, ignore_eos=True),
        block_size=BLOCK_SIZE,
    )
    _allocate_new(scheduler, chunked, 8)
    chunked.num_cached_tokens = 8
    chunked.num_new_tokens = 0
    chunked.status = SequenceStatus.RUNNING
    scheduler.chunked_req = chunked

    follower = Sequence(
        shared + [72_002],
        SamplingParams(max_tokens=1, ignore_eos=True),
        block_size=BLOCK_SIZE,
    )
    scheduler.add(follower)

    scheduled = scheduler.schedule()

    assert scheduled == [chunked, follower]
    assert [seq.num_new_tokens for seq in scheduled] == [9, 1]
    assert follower.num_cached_tokens == BLOCK_SIZE
    assert follower.block_table[0] == chunked.block_table[0]
    assert scheduler.initial_persistent_hit_blocks_by_seq == {}
    assert scheduler.same_step_hit_blocks_by_seq == {follower.seq_id: 1}
    assert scheduler.block_manager.blocks[chunked.block_table[0]].ref_count == 2
    _assert_block_invariants(scheduler)

    scheduler.block_manager.deallocate(chunked)
    scheduler.block_manager.deallocate(follower)
    scheduler.chunked_req = None
    _assert_block_invariants(scheduler)


def test_same_step_off_keeps_completed_chunk_follower_as_frozen_miss():
    scheduler = _scheduler(
        num_blocks=4,
        token_budget=26,
        max_num_seqs=2,
        same_step=False,
    )
    shared = list(range(BLOCK_SIZE))
    chunked = Sequence(
        shared + [72_101],
        SamplingParams(max_tokens=1, ignore_eos=True),
        block_size=BLOCK_SIZE,
    )
    _allocate_new(scheduler, chunked, 8)
    chunked.num_cached_tokens = 8
    chunked.num_new_tokens = 0
    chunked.status = SequenceStatus.RUNNING
    scheduler.chunked_req = chunked
    follower = Sequence(
        shared + [72_102],
        SamplingParams(max_tokens=1, ignore_eos=True),
        block_size=BLOCK_SIZE,
    )
    scheduler.add(follower)

    scheduled = scheduler.schedule()

    assert scheduled == [chunked, follower]
    assert [seq.num_new_tokens for seq in scheduled] == [9, 17]
    assert follower.num_cached_tokens == 0
    assert follower.block_table[0] != chunked.block_table[0]
    assert scheduler.initial_persistent_hit_blocks_by_seq == {}
    assert scheduler.same_step_hit_blocks_by_seq == {}
    _assert_block_invariants(scheduler)

    scheduler.block_manager.deallocate(chunked)
    scheduler.block_manager.deallocate(follower)
    scheduler.chunked_req = None
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
    scheduler = _scheduler(same_step=False)
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

    waiting, plans = scheduler._match_waiting()
    ranked = scheduler._rank_waiting(waiting, plans)

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


def test_cached_free_claim_is_counted_before_any_request_mutation():
    scheduler = _scheduler(
        num_blocks=3,
        token_budget=17,
        max_num_seqs=1,
        chunked=False,
    )
    prefix = list(range(2 * BLOCK_SIZE))
    prefix_ids = _publish_free_prefix(scheduler, prefix)
    target = Sequence(
        prefix + list(range(100, 117)),
        block_size=BLOCK_SIZE,
    )
    scheduler.add(target)
    free_before = tuple(scheduler.block_manager.free_block_ids)
    refs_before = [
        block.ref_count for block in scheduler.block_manager.blocks
    ]

    with pytest.raises(RuntimeError, match="cannot make progress"):
        scheduler.schedule()

    assert tuple(scheduler.block_manager.free_block_ids) == free_before
    assert [
        block.ref_count for block in scheduler.block_manager.blocks
    ] == refs_before
    assert target.block_table == []
    assert target.num_cached_tokens == target.num_new_tokens == 0
    assert scheduler.block_manager.match_prefix(target) == prefix_ids
    _assert_block_invariants(scheduler)


@pytest.mark.parametrize("same_step", [False, True])
def test_kv_strict_break_does_not_bypass_higher_lpm_request(same_step):
    scheduler = _scheduler(
        num_blocks=3,
        token_budget=64,
        max_num_seqs=2,
        chunked=False,
        same_step=same_step,
    )
    prefix = list(range(2 * BLOCK_SIZE))
    _publish_free_prefix(scheduler, prefix)
    high = Sequence(
        prefix + list(range(9_000, 9_000 + BLOCK_SIZE + 1)),
        block_size=BLOCK_SIZE,
    )
    low = Sequence([91_001], block_size=BLOCK_SIZE)
    scheduler.waiting.extend([low, high])

    with pytest.raises(RuntimeError, match="cannot make progress"):
        scheduler.schedule()

    assert list(scheduler.waiting) == [low, high]
    assert high.block_table == low.block_table == []
    assert high.num_cached_tokens == low.num_cached_tokens == 0
    assert high.num_new_tokens == low.num_new_tokens == 0
    _assert_block_invariants(scheduler)


def test_same_step_reuse_keeps_fcfs_and_shares_leader_blocks():
    scheduler = _scheduler(
        num_blocks=6,
        token_budget=36,
        max_num_seqs=4,
        chunked=False,
    )
    shared = list(range(2 * BLOCK_SIZE))
    requests = [
        Sequence(shared + [820_000 + index], block_size=BLOCK_SIZE)
        for index in range(1, 5)
    ]
    scheduler.waiting.extend(requests)

    scheduled = scheduler.schedule()

    assert scheduled == requests
    assert [seq.num_new_tokens for seq in scheduled] == [33, 1, 1, 1]
    assert [seq.num_cached_tokens for seq in scheduled] == [0, 32, 32, 32]
    prefix_ids = requests[0].block_table[:2]
    assert all(seq.block_table[:2] == prefix_ids for seq in requests)
    assert all(
        scheduler.block_manager.blocks[block_id].ref_count == 4
        for block_id in prefix_ids
    )
    assert scheduler.initial_persistent_hit_blocks_by_seq == {}
    assert scheduler.same_step_hit_blocks_by_seq == {
        seq.seq_id: 2 for seq in requests[1:]
    }
    _assert_block_invariants(scheduler)

    for seq in requests:
        scheduler.block_manager.deallocate(seq)
    _assert_block_invariants(scheduler)


def test_latest_lookup_claim_allocate_publish_is_atomic_per_request(
    monkeypatch,
):
    scheduler = _scheduler(
        num_blocks=4,
        token_budget=34,
        max_num_seqs=2,
        chunked=False,
    )
    shared = list(range(2 * BLOCK_SIZE))
    first = Sequence(shared + [821_001], block_size=BLOCK_SIZE)
    second = Sequence(shared + [821_002], block_size=BLOCK_SIZE)
    scheduler.waiting.extend([first, second])
    labels = {first: "A1", second: "A2"}
    events = []
    manager = scheduler.block_manager
    original_match = manager.match_prefix
    original_claim = manager.claim_prefix
    original_allocate = manager.allocate_new

    def recording_match(seq, *args, **kwargs):
        events.append(("match", labels[seq]))
        return original_match(seq, *args, **kwargs)

    def recording_claim(seq, block_ids):
        events.append(("claim", labels[seq]))
        return original_claim(seq, block_ids)

    def recording_allocate(seq):
        events.append(("allocate", labels[seq]))
        return original_allocate(seq)

    monkeypatch.setattr(manager, "match_prefix", recording_match)
    monkeypatch.setattr(manager, "claim_prefix", recording_claim)
    monkeypatch.setattr(manager, "allocate_new", recording_allocate)

    assert scheduler.schedule() == [first, second]
    assert events == [
        ("match", "A1"),
        ("match", "A2"),
        ("match", "A1"),
        ("claim", "A1"),
        ("allocate", "A1"),
        ("match", "A2"),
        ("claim", "A2"),
        ("allocate", "A2"),
    ]

    manager.deallocate(first)
    manager.deallocate(second)
    _assert_block_invariants(scheduler)


def test_same_step_hit_does_not_change_initial_lpm_order():
    scheduler = _scheduler(
        num_blocks=14,
        token_budget=36,
        max_num_seqs=4,
        chunked=False,
    )
    p1_prefix = list(range(100, 100 + 4 * BLOCK_SIZE))
    p2_prefix = list(range(1_000, 1_000 + 2 * BLOCK_SIZE))
    _publish_free_prefix(scheduler, p1_prefix)
    _publish_free_prefix(scheduler, p2_prefix)
    shared = list(range(2_000, 2_000 + 2 * BLOCK_SIZE))
    p1 = Sequence(p1_prefix + [822_001], block_size=BLOCK_SIZE)
    p2 = Sequence(p2_prefix + [822_002], block_size=BLOCK_SIZE)
    a1 = Sequence(shared + [822_003], block_size=BLOCK_SIZE)
    a2 = Sequence(shared + [822_004], block_size=BLOCK_SIZE)
    scheduler.waiting.extend([a1, p2, a2, p1])

    scheduled = scheduler.schedule()

    assert scheduled == [p1, p2, a1, a2]
    assert [seq.num_new_tokens for seq in scheduled] == [1, 1, 33, 1]
    assert scheduler.initial_persistent_hit_blocks_by_seq == {
        p1.seq_id: 4,
        p2.seq_id: 2,
    }
    assert scheduler.same_step_hit_blocks_by_seq == {a2.seq_id: 2}

    for seq in scheduled:
        scheduler.block_manager.deallocate(seq)
    _assert_block_invariants(scheduler)


def test_same_step_reuse_expands_admission_with_same_budgets():
    shared = list(range(2 * BLOCK_SIZE))

    def run(enabled: bool):
        scheduler = _scheduler(
            num_blocks=6,
            token_budget=36,
            max_num_seqs=4,
            chunked=False,
            same_step=enabled,
        )
        requests = [
            Sequence(shared + [823_000 + index], block_size=BLOCK_SIZE)
            for index in range(1, 5)
        ]
        scheduler.waiting.extend(requests)
        return scheduler, requests, scheduler.schedule()

    off_scheduler, off_requests, off_scheduled = run(False)
    on_scheduler, on_requests, on_scheduled = run(True)

    assert off_scheduled == off_requests[:1]
    assert on_scheduled == on_requests
    assert sum(seq.num_new_tokens for seq in off_scheduled) == 33
    assert sum(seq.num_new_tokens for seq in on_scheduled) == 36
    assert len(on_scheduler.block_manager.used_block_ids) == 6

    for seq in off_scheduled:
        off_scheduler.block_manager.deallocate(seq)
    for seq in on_scheduled:
        on_scheduler.block_manager.deallocate(seq)
    _assert_block_invariants(off_scheduler)
    _assert_block_invariants(on_scheduler)


def test_same_step_reuses_only_full_block_aligned_prefix():
    scheduler = _scheduler(
        num_blocks=4,
        token_budget=44,
        max_num_seqs=2,
        chunked=False,
    )
    full_prefix = list(range(2 * BLOCK_SIZE))
    partial_prefix = list(range(500, 505))
    leader = Sequence(
        full_prefix + partial_prefix + [824_001],
        block_size=BLOCK_SIZE,
    )
    follower = Sequence(
        full_prefix + partial_prefix + [824_002],
        block_size=BLOCK_SIZE,
    )
    scheduler.waiting.extend([leader, follower])

    scheduled = scheduler.schedule()

    assert scheduled == [leader, follower]
    assert [seq.num_new_tokens for seq in scheduled] == [38, 6]
    assert follower.num_cached_tokens == 2 * BLOCK_SIZE
    assert follower.block_table[:2] == leader.block_table[:2]
    assert follower.block_table[2] != leader.block_table[2]
    assert scheduler.block_manager.blocks[leader.block_table[2]].hash == -1
    assert scheduler.block_manager.blocks[follower.block_table[2]].hash == -1
    assert scheduler.same_step_hit_blocks_by_seq == {follower.seq_id: 2}

    scheduler.block_manager.deallocate(leader)
    scheduler.block_manager.deallocate(follower)
    _assert_block_invariants(scheduler)


def test_disabling_lpm_preserves_fcfs_and_same_step_reuse():
    scheduler = _scheduler(
        token_budget=66,
        max_num_seqs=2,
        enable_lpm=False,
    )
    shared = list(range(2 * BLOCK_SIZE))
    first = Sequence(shared + [829_001], block_size=BLOCK_SIZE)
    follower = Sequence(shared + [829_002], block_size=BLOCK_SIZE)
    other = Sequence(list(range(500, 533)), block_size=BLOCK_SIZE)
    scheduler.waiting.extend([first, follower, other])

    scheduled = scheduler.schedule()

    assert scheduled == [first, follower]
    assert [seq.num_new_tokens for seq in scheduled] == [33, 1]
    assert follower.num_cached_tokens == 2 * BLOCK_SIZE
    assert scheduler.initial_persistent_hit_blocks_by_seq == {}
    assert scheduler.same_step_hit_blocks_by_seq == {follower.seq_id: 2}
    scheduler.block_manager.deallocate(first)
    scheduler.block_manager.deallocate(follower)
    _assert_block_invariants(scheduler)


def test_disabling_same_step_reuse_keeps_persistent_lpm_enabled():
    scheduler = _scheduler(
        num_blocks=8,
        token_budget=33,
        max_num_seqs=1,
        same_step=False,
    )
    prefix = list(range(2 * BLOCK_SIZE))
    prefix_ids = _publish_free_prefix(scheduler, prefix)
    cold = Sequence(list(range(500, 533)), block_size=BLOCK_SIZE)
    cached = Sequence(prefix + [829_101], block_size=BLOCK_SIZE)
    scheduler.waiting.extend([cold, cached])

    scheduled = scheduler.schedule()

    assert scheduled == [cached]
    assert list(scheduler.waiting) == [cold]
    assert cached.block_table[:2] == prefix_ids
    assert cached.num_cached_tokens == 2 * BLOCK_SIZE
    assert cached.num_new_tokens == 1
    assert scheduler.initial_persistent_hit_blocks_by_seq == {
        cached.seq_id: 2
    }
    assert scheduler.same_step_hit_blocks_by_seq == {}
    scheduler.block_manager.deallocate(cached)
    _assert_block_invariants(scheduler)


def test_max_num_seqs_one_reuses_prefix_only_on_next_step():
    scheduler = _scheduler(num_blocks=8, token_budget=33, max_num_seqs=1)
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    shared = list(range(2 * BLOCK_SIZE))
    cold = Sequence(
        list(range(900, 900 + 2 * BLOCK_SIZE + 1)),
        params,
        block_size=BLOCK_SIZE,
    )
    leader = Sequence(
        shared + [830_001],
        params,
        block_size=BLOCK_SIZE,
    )
    follower = Sequence(
        shared + [830_002],
        params,
        block_size=BLOCK_SIZE,
    )
    scheduler.waiting.extend([leader, cold, follower])

    first = scheduler.schedule()

    assert first == [leader]
    assert scheduler.same_step_hit_blocks_by_seq == {}
    assert list(scheduler.waiting) == [cold, follower]
    assert follower.block_table == []
    assert follower.num_cached_tokens == 0
    leader_prefix_ids = leader.block_table[:2]

    scheduler.postprocess(first, [83_100], [0])

    assert leader.status is SequenceStatus.FINISHED
    assert scheduler.block_manager.match_prefix(follower) == leader_prefix_ids
    assert all(
        block_id in scheduler.block_manager.free_block_ids
        for block_id in leader_prefix_ids
    )

    second = scheduler.schedule()

    assert second == [follower]
    assert follower.block_table[:2] == leader_prefix_ids
    assert list(scheduler.waiting) == [cold]
    assert follower.num_cached_tokens == 2 * BLOCK_SIZE
    assert follower.num_new_tokens == 1
    scheduler.postprocess(second, [83_101], [0])
    assert scheduler.initial_persistent_hit_blocks_by_seq == {
        follower.seq_id: 2
    }
    assert scheduler.same_step_hit_blocks_by_seq == {}
    assert follower.status is SequenceStatus.FINISHED
    assert list(scheduler.waiting) == [cold]

    third = scheduler.schedule()
    assert third == [cold]
    scheduler.postprocess(third, [83_102], [0])
    assert cold.status is SequenceStatus.FINISHED
    assert scheduler.is_finished()
    _assert_block_invariants(scheduler)


def test_same_step_off_keeps_frozen_misses_as_independent_cold_prefills():
    scheduler = _scheduler(
        num_blocks=8, token_budget=66, max_num_seqs=2, same_step=False
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    shared = list(range(2 * BLOCK_SIZE))
    leader = Sequence(
        shared + [840_001],
        params,
        block_size=BLOCK_SIZE,
    )
    follower = Sequence(
        shared + [840_002],
        params,
        block_size=BLOCK_SIZE,
    )
    scheduler.waiting.extend([leader, follower])

    scheduled = scheduler.schedule()

    assert scheduled == [leader, follower]
    assert leader.num_cached_tokens == follower.num_cached_tokens == 0
    assert leader.num_new_tokens == follower.num_new_tokens == 2 * BLOCK_SIZE + 1
    assert set(leader.block_table[:2]).isdisjoint(follower.block_table[:2])
    assert all(
        scheduler.block_manager.blocks[block_id].ref_count == 1
        for block_id in leader.block_table[:2] + follower.block_table[:2]
    )

    scheduler.postprocess(scheduled, [84_100, 84_101], [0, 1])
    assert scheduler.initial_persistent_hit_blocks_by_seq == {}
    assert scheduler.same_step_hit_blocks_by_seq == {}
    assert leader.status is follower.status is SequenceStatus.FINISHED
    _assert_block_invariants(scheduler)


def test_same_step_admission_preserves_decode_boundary_block_reservation():
    scheduler = _scheduler(
        num_blocks=5,
        token_budget=19,
        max_num_seqs=3,
        chunked=False,
    )
    decode = _add_decode(scheduler, 10_000)
    shared = list(range(20_000, 20_000 + BLOCK_SIZE))
    leader = Sequence(shared + [85_001], block_size=BLOCK_SIZE)
    follower = Sequence(shared + [85_002], block_size=BLOCK_SIZE)
    scheduler.waiting.extend([leader, follower])

    scheduled = scheduler.schedule()

    assert scheduled == [leader, follower, decode]
    assert scheduler.num_scheduled_prefill_seqs == 2
    assert [seq.num_new_tokens for seq in scheduled] == [17, 1, 1]
    assert follower.num_cached_tokens == BLOCK_SIZE
    assert follower.block_table[0] == leader.block_table[0]
    assert scheduler.same_step_hit_blocks_by_seq == {follower.seq_id: 1}
    assert len(scheduler.block_manager.free_block_ids) == 0
    assert len(decode.block_table) == 2
    _assert_block_invariants(scheduler)

    scheduler.block_manager.deallocate(leader)
    scheduler.block_manager.deallocate(follower)
    scheduler.block_manager.deallocate(decode)
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
