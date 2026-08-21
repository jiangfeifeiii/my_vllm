from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, config: Config):
        self.enable_chunked = config.chunked_prefill
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_lpm = getattr(config, "enable_lpm", True)
        self.enable_same_step_prefix_reuse = getattr(
            config,
            "enable_same_step_prefix_reuse",
            True,
        )
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        self.waiting: deque[Sequence] = deque()
        self.chunked_req: Sequence | None = None
        self.running: deque[Sequence] = deque()
        self.num_scheduled_prefill_seqs = 0
        self.num_scheduled_prefill_tokens = 0
        self.initial_persistent_hit_blocks_by_seq: dict[int, int] = {}
        self.same_step_hit_blocks_by_seq: dict[int, int] = {}

    def is_finished(self) -> bool:
        return not self.waiting and self.chunked_req is None and not self.running

    def add(self, seq: Sequence) -> None:
        assert len(seq) <= self.max_model_len - 1, (
            "Sequence length exceeds max_model_len"
        )
        self.waiting.append(seq)

    def _preempt_decode_tail(self) -> None:
        block_manager = self.block_manager
        while self.running:
            decode_blocks = sum(
                block_manager.num_blocks_to_append(seq, 1)
                for seq in self.running
            )
            if (
                len(self.running) <= self.max_num_batched_tokens
                and decode_blocks <= len(block_manager.free_block_ids)
            ):
                return
            self.preempt(self.running.pop())

    def _match_waiting(self) -> tuple[list[Sequence], dict[Sequence, list[int]]]:
        waiting = list(self.waiting)
        real_prefixes = {
            seq: self.block_manager.match_prefix(seq)
            for seq in waiting
        }
        return waiting, real_prefixes

    def _rank_waiting(
        self,
        waiting: list[Sequence],
        initial_prefixes: dict[Sequence, list[int]],
    ) -> list[Sequence]:
        # Python's sort is stable, so requests with the same priority preserve
        # their original FCFS order. Only the initial persistent snapshot can
        # affect LPM order; later same-step hits never trigger a re-sort.
        return sorted(
            waiting,
            key=lambda seq: (
                -len(initial_prefixes[seq]) if self.enable_lpm else 0
            ),
        )

    def _record_prefix_hits(
        self,
        seq: Sequence,
        initial_blocks: int,
        latest_blocks: int,
    ) -> None:
        claimed_initial = min(initial_blocks, latest_blocks)
        same_step_blocks = max(latest_blocks - initial_blocks, 0)
        if claimed_initial:
            self.initial_persistent_hit_blocks_by_seq[seq.seq_id] = (
                claimed_initial
            )
        if same_step_blocks:
            self.same_step_hit_blocks_by_seq[seq.seq_id] = same_step_blocks

    def _commit_waiting_same_step(
        self,
        ranked: list[Sequence],
        initial_prefixes: dict[Sequence, list[int]],
        token_budget: int,
        reserved_blocks: int,
        active: int,
    ) -> list[Sequence]:
        block_manager = self.block_manager
        committed: list[Sequence] = []

        for seq in ranked:
            if active + len(committed) >= self.max_num_seqs:
                break

            # No BlockManager mutation may occur between this lookup and claim.
            latest_blocks = block_manager.match_prefix(seq)
            cached_tokens = (
                seq.num_cached_tokens
                + len(latest_blocks) * block_manager.block_size
            )
            remaining = len(seq) - cached_tokens
            assert remaining > 0
            if not self.enable_chunked and remaining > token_budget:
                break
            num_new_tokens = min(remaining, token_budget)
            if num_new_tokens <= 0:
                break

            required_blocks = block_manager.num_blocks_required_for_admission(
                latest_blocks,
                num_new_tokens,
            )
            available_blocks = (
                len(block_manager.free_block_ids) - reserved_blocks
            )
            assert available_blocks >= 0
            if required_blocks > available_blocks:
                break

            block_manager.claim_prefix(seq, latest_blocks)
            seq.num_new_tokens = num_new_tokens
            block_manager.allocate_new(seq)
            self._record_prefix_hits(
                seq,
                len(initial_prefixes[seq]),
                len(latest_blocks),
            )
            seq.status = SequenceStatus.RUNNING
            self.waiting.remove(seq)
            committed.append(seq)
            token_budget -= num_new_tokens
            if num_new_tokens < remaining:
                break

        return committed

    def schedule(self) -> list[Sequence]:
        block_manager = self.block_manager
        self.num_scheduled_prefill_seqs = 0
        self.num_scheduled_prefill_tokens = 0
        self.initial_persistent_hit_blocks_by_seq = {}
        self.same_step_hit_blocks_by_seq = {}

        # Reserve one token and any boundary block for each decode request.
        self._preempt_decode_tail()
        decode_seqs = list(self.running)
        decode_blocks = sum(
            block_manager.num_blocks_to_append(seq, 1)
            for seq in decode_seqs
        )
        token_budget = self.max_num_batched_tokens - len(decode_seqs)
        free_budget = len(block_manager.free_block_ids) - decode_blocks
        assert token_budget >= 0 and free_budget >= 0

        scheduled_chunked = None
        chunked_blocks = 0
        chunked_tokens = 0
        allow_waiting = True
        if self.chunked_req is not None:
            chunked = self.chunked_req
            remaining = len(chunked) - chunked.num_cached_tokens
            assert remaining > 0
            chunked_tokens = min(remaining, token_budget)
            chunked_blocks = block_manager.num_blocks_to_append(
                chunked,
                chunked_tokens,
            )
            if chunked_tokens == 0 or chunked_blocks > free_budget:
                # Decode has priority. Do not let new requests bypass a paused
                # chunk and starve it while decode owns the remaining budget.
                allow_waiting = False
                chunked_tokens = 0
                chunked_blocks = 0
            else:
                scheduled_chunked = chunked
                token_budget -= chunked_tokens
                free_budget -= chunked_blocks
                if chunked_tokens < remaining:
                    allow_waiting = False

        ranked: list[Sequence] = []
        initial_prefixes: dict[Sequence, list[int]] = {}
        if allow_waiting and self.waiting and token_budget > 0:
            waiting, initial_prefixes = self._match_waiting()
            ranked = self._rank_waiting(waiting, initial_prefixes)

        active = len(decode_seqs) + (self.chunked_req is not None)
        waiting_prefills: list[Sequence]
        prefill_seqs: list[Sequence] = []
        admitted: list[Sequence] = []
        admission: dict[Sequence, tuple[list[int], int]] = {}
        if self.enable_same_step_prefix_reuse:
            # Ranking is already frozen from the initial snapshot. Commit the
            # higher-priority chunk first so later Waiting lookups may reuse a
            # full block it publishes without changing their LPM order.
            if scheduled_chunked is not None:
                scheduled_chunked.num_new_tokens = chunked_tokens
                block_manager.may_append(scheduled_chunked)
                prefill_seqs.append(scheduled_chunked)
            waiting_prefills = self._commit_waiting_same_step(
                ranked,
                initial_prefixes,
                token_budget,
                decode_blocks,
                active,
            )
        else:
            # The OFF ablation retains the old frozen-plan behavior. Claim all
            # admitted prefixes before allocation so cached-free plans cannot
            # become stale while a preceding cold request allocates blocks.
            protected_free_ids: set[int] = set()
            for seq in ranked:
                if active + len(admitted) >= self.max_num_seqs:
                    break
                initial_blocks = initial_prefixes[seq]
                cached_tokens = (
                    seq.num_cached_tokens
                    + len(initial_blocks) * block_manager.block_size
                )
                remaining = len(seq) - cached_tokens
                assert remaining > 0
                if not self.enable_chunked and remaining > token_budget:
                    break
                num_new_tokens = min(remaining, token_budget)
                if num_new_tokens <= 0:
                    break

                newly_protected = {
                    block_id
                    for block_id in initial_blocks
                    if block_id in block_manager.free_block_ids
                    and block_id not in protected_free_ids
                }
                new_blocks = (
                    num_new_tokens + block_manager.block_size - 1
                ) // block_manager.block_size
                if len(newly_protected) + new_blocks > free_budget:
                    break

                context_end = cached_tokens + num_new_tokens
                admitted.append(seq)
                admission[seq] = (initial_blocks, context_end)
                protected_free_ids.update(newly_protected)
                free_budget -= len(newly_protected) + new_blocks
                token_budget -= num_new_tokens
                if num_new_tokens < remaining:
                    break

            for seq in admitted:
                block_manager.claim_prefix(seq, admission[seq][0])
            waiting_prefills = []

        if (
            scheduled_chunked is not None
            and not self.enable_same_step_prefix_reuse
        ):
            scheduled_chunked.num_new_tokens = chunked_tokens
            block_manager.may_append(scheduled_chunked)
            prefill_seqs.append(scheduled_chunked)

        if not self.enable_same_step_prefix_reuse:
            for seq in admitted:
                initial_blocks, context_end = admission[seq]
                seq.num_new_tokens = context_end - seq.num_cached_tokens
                assert seq.num_new_tokens > 0
                block_manager.allocate_new(seq)
                self._record_prefix_hits(
                    seq, len(initial_blocks), len(initial_blocks)
                )
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                waiting_prefills.append(seq)
        prefill_seqs.extend(waiting_prefills)

        for seq in decode_seqs:
            assert len(seq) - seq.num_cached_tokens == 1
            seq.num_new_tokens = 1
            block_manager.may_append(seq)

        scheduled = prefill_seqs + decode_seqs
        if not scheduled:
            if not self.enable_chunked and ranked:
                first = ranked[0]
                if self.enable_same_step_prefix_reuse:
                    prefix_blocks = block_manager.match_prefix(first)
                else:
                    prefix_blocks = initial_prefixes[first]
                cached_tokens = (
                    first.num_cached_tokens
                    + len(prefix_blocks) * block_manager.block_size
                )
                if len(first) - cached_tokens > self.max_num_batched_tokens:
                    raise ValueError(
                        "uncached prompt tokens exceed "
                        "max_num_batched_tokens; enable chunked_prefill"
                    )
            raise RuntimeError(
                "scheduler cannot make progress with the current token/KV budget"
            )
        self.num_scheduled_prefill_seqs = len(prefill_seqs)
        self.num_scheduled_prefill_tokens = sum(
            seq.num_new_tokens for seq in prefill_seqs
        )
        return scheduled

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def _finish(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        if self.chunked_req is seq:
            self.chunked_req = None
        if seq in self.running:
            self.running.remove(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        seq_need_compute_logits,
    ) -> None:
        logits_indices = (
            seq_need_compute_logits.tolist()
            if hasattr(seq_need_compute_logits, "tolist")
            else list(seq_need_compute_logits)
        )
        assert len(token_ids) == len(logits_indices)
        for seq_index, token_id in zip(logits_indices, token_ids):
            seq = seqs[int(seq_index)]
            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens >= seq.max_tokens
                or len(seq) >= self.max_model_len
            ):
                if len(seq) >= self.max_model_len:
                    print(
                        f"Sequence {seq.seq_id} reached max_model_len "
                        f"{self.max_model_len}."
                    )
                self._finish(seq)

        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED:
                seq.num_cached_tokens += seq.num_new_tokens
                seq.num_new_tokens = 0

        prefill_seqs = seqs[: self.num_scheduled_prefill_seqs]
        partial_prefills = []
        for seq in prefill_seqs:
            if seq.status == SequenceStatus.FINISHED:
                continue
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                partial_prefills.append(seq)
            elif seq not in self.running:
                self.running.append(seq)

        assert len(partial_prefills) <= 1
        if prefill_seqs:
            self.chunked_req = (
                partial_prefills[0] if partial_prefills else None
            )
