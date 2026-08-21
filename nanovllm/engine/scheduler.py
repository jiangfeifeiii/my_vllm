from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = 32
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = 32


class Scheduler:

    def __init__(self, config: Config):
        self.enable_chunked = config.chunked_prefill
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_lpm = getattr(config, "enable_lpm", True)
        self.enable_in_batch_prefix_deprioritization = getattr(
            config,
            "enable_in_batch_prefix_deprioritization",
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
        self.temporary_prefix_index: dict[int, tuple[int, int]] = {}
        self.temporary_deprioritized: set[int] = set()

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

    def _detect_temporary_prefixes(
        self,
        waiting: list[Sequence],
        real_prefixes: dict[Sequence, list[int]],
    ) -> set[Sequence]:
        block_size = self.block_manager.block_size
        index: dict[int, tuple[Sequence, int, list[int]]] = {}
        deprioritized: set[Sequence] = set()

        # Detection intentionally walks the original FCFS queue. The first
        # request that exposes a full-block prefix is its implicit leader.
        for seq in waiting:
            persistent_tokens = len(real_prefixes[seq]) * block_size
            if persistent_tokens > IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                continue

            prefix_hash = -1
            matching_temporary_prefix = True
            matched_blocks = 0
            eligible_blocks: list[tuple[int, int, list[int]]] = []
            for block_index in range(seq.num_blocks):
                token_ids = seq.block(block_index)
                if len(token_ids) != block_size:
                    break
                prefix_hash = self.block_manager.compute_hash(
                    token_ids,
                    prefix_hash,
                )
                eligible_blocks.append(
                    (prefix_hash, block_index + 1, token_ids)
                )
                if matching_temporary_prefix:
                    prior = index.get(prefix_hash)
                    if prior is not None and prior[2] == token_ids:
                        matched_blocks = block_index + 1
                    else:
                        matching_temporary_prefix = False

            temporary_tokens = matched_blocks * block_size
            if (
                temporary_tokens >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
            ):
                deprioritized.add(seq)
                continue

            # A follower is never inserted into the temporary index. Requests
            # below the threshold become implicit leaders for later requests.
            for block_hash, num_blocks, token_ids in eligible_blocks:
                index.setdefault(
                    block_hash,
                    (seq, num_blocks, token_ids),
                )

        self.temporary_prefix_index = {
            block_hash: (seq.seq_id, num_blocks)
            for block_hash, (seq, num_blocks, _) in index.items()
        }
        self.temporary_deprioritized = {
            seq.seq_id for seq in deprioritized
        }
        return deprioritized

    def _rank_waiting(
        self,
        waiting: list[Sequence],
        real_prefixes: dict[Sequence, list[int]],
        temporary_deprioritized: set[Sequence],
    ) -> list[Sequence]:
        # Python's sort is stable, so requests with the same priority preserve
        # their original FCFS order. Temporary matches only affect priority;
        # they never count as computed cache in admission or allocation.
        def priority(seq: Sequence) -> tuple[int, int]:
            if seq in temporary_deprioritized:
                return (1, 0)
            persistent_priority = (
                -len(real_prefixes[seq]) if self.enable_lpm else 0
            )
            return (0, persistent_priority)

        return sorted(waiting, key=priority)

    def schedule(self) -> list[Sequence]:
        block_manager = self.block_manager
        self.num_scheduled_prefill_seqs = 0
        self.num_scheduled_prefill_tokens = 0
        self.temporary_prefix_index = {}
        self.temporary_deprioritized = set()

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
        real_prefixes: dict[Sequence, list[int]] = {}
        if allow_waiting and self.waiting and token_budget > 0:
            waiting, real_prefixes = self._match_waiting()
            temporary_deprioritized = (
                self._detect_temporary_prefixes(waiting, real_prefixes)
                if (
                    self.enable_lpm
                    and self.enable_in_batch_prefix_deprioritization
                )
                else set()
            )
            ranked = self._rank_waiting(
                waiting,
                real_prefixes,
                temporary_deprioritized,
            )

        admitted: list[Sequence] = []
        admission: dict[Sequence, tuple[list[int], int]] = {}
        protected_free_ids: set[int] = set()
        active = len(decode_seqs) + (self.chunked_req is not None)

        for seq in ranked:
            if active + len(admitted) >= self.max_num_seqs:
                break
            real_blocks = real_prefixes[seq]
            cached_tokens = len(real_blocks) * block_manager.block_size
            remaining = len(seq) - cached_tokens
            assert remaining > 0
            if not self.enable_chunked and remaining > token_budget:
                if remaining > self.max_num_batched_tokens:
                    raise ValueError(
                        "uncached prompt tokens exceed "
                        "max_num_batched_tokens; enable chunked_prefill"
                    )
                break
            num_new_tokens = min(remaining, token_budget)
            if num_new_tokens <= 0:
                break

            newly_protected = {
                block_id
                for block_id in real_blocks
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
            admission[seq] = (real_blocks, context_end)
            protected_free_ids.update(newly_protected)
            free_budget -= len(newly_protected) + new_blocks
            token_budget -= num_new_tokens
            if num_new_tokens < remaining:
                break

        # Protect every real cached prefix before any operation can reset a
        # cached-free block selected by the planning pass.
        for seq in admitted:
            block_manager.claim_prefix(seq, admission[seq][0])

        prefill_seqs: list[Sequence] = []
        if scheduled_chunked is not None:
            scheduled_chunked.num_new_tokens = chunked_tokens
            block_manager.may_append(scheduled_chunked)
            prefill_seqs.append(scheduled_chunked)

        # Allocation consumes only the persistent prefix plan protected above.
        # Temporary matches never alter cached tokens or block tables.
        for seq in admitted:
            _, context_end = admission[seq]
            seq.num_new_tokens = context_end - seq.num_cached_tokens
            assert seq.num_new_tokens > 0
            block_manager.allocate_new(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.remove(seq)
            prefill_seqs.append(seq)

        for seq in decode_seqs:
            assert len(seq) - seq.num_cached_tokens == 1
            seq.num_new_tokens = 1
            block_manager.may_append(seq)

        scheduled = prefill_seqs + decode_seqs
        if not scheduled:
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
