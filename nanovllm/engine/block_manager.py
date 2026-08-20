from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    Blocks (or tokens) layout:
    
    ----------------------------------------------------------------------
    | < computed > | < new_computed > |       < new >       |
    ----------------------------------------------------------------------
    |     < Prefix-cached tokens >    |  < to be computed > |
    ----------------------------------------------------------------------
                                      | < to be allocated > |
    ----------------------------------------------------------------------
                                      |   < to be cached >  |
    ----------------------------------------------------------------------
    
    """
    
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        assert block_size > 0
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _assert_block_size(self, seq: Sequence) -> None:
        assert seq.block_size == self.block_size, (
            f"sequence block size {seq.block_size} does not match "
            f"manager block size {self.block_size}"
        )

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if self.hash_to_block_id.get(block.hash) == block_id:
            self.hash_to_block_id.pop(block.hash, None)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, num_tokens: int) -> bool:
        """
        Only for seq in the waiting queue.
        """
        return len(self.free_block_ids) >= (num_tokens + self.block_size - 1) // self.block_size

    def _validate_claimed_prefix(self, seq: Sequence) -> int:
        assert seq.num_cached_tokens == len(seq.block_table) * self.block_size
        prefix_hash = -1
        for block_index, block_id in enumerate(seq.block_table):
            assert 0 <= block_id < len(self.blocks)
            token_ids = seq.block(block_index)
            assert len(token_ids) == self.block_size
            prefix_hash = self.compute_hash(token_ids, prefix_hash)
            block = self.blocks[block_id]
            assert block_id in self.used_block_ids
            assert block_id not in self.free_block_ids
            assert block.ref_count > 0
            assert block.hash == prefix_hash
            assert block.token_ids == token_ids
        return prefix_hash

    def match_prefix(
        self,
        seq: Sequence,
        max_blocks: int | None = None,
    ) -> list[int]:
        self._assert_block_size(seq)
        if max_blocks is not None:
            assert max_blocks >= 0

        prefix_hash = self._validate_claimed_prefix(seq)
        start = len(seq.block_table)
        stop = max(start, seq.num_blocks - 1)
        if max_blocks is not None:
            stop = min(stop, start + max_blocks)

        matched = []
        for block_index in range(start, stop):
            token_ids = seq.block(block_index)
            if len(token_ids) != self.block_size:
                break
            block_hash = self.compute_hash(token_ids, prefix_hash)
            block_id = self.hash_to_block_id.get(block_hash, -1)
            if not 0 <= block_id < len(self.blocks):
                break
            block = self.blocks[block_id]
            is_free = block_id in self.free_block_ids
            is_used = block_id in self.used_block_ids
            if (
                block.hash != block_hash
                or block.token_ids != token_ids
                or is_free == is_used
                or (is_free and block.ref_count != 0)
                or (is_used and block.ref_count <= 0)
            ):
                break
            matched.append(block_id)
            prefix_hash = block_hash
        return matched

    def claim_prefix(self, seq: Sequence, block_ids: list[int]) -> None:
        self._assert_block_size(seq)
        prefix_hash = self._validate_claimed_prefix(seq)
        start = len(seq.block_table)
        planned = list(block_ids)
        assert len(set(planned)) == len(planned)

        claims = []
        for offset, block_id in enumerate(planned):
            block_index = start + offset
            assert block_index < seq.num_blocks - 1
            assert 0 <= block_id < len(self.blocks)
            token_ids = seq.block(block_index)
            assert len(token_ids) == self.block_size
            block_hash = self.compute_hash(token_ids, prefix_hash)
            assert self.hash_to_block_id.get(block_hash) == block_id, (
                "prefix plan is stale: hash mapping changed"
            )
            block = self.blocks[block_id]
            is_free = block_id in self.free_block_ids
            is_used = block_id in self.used_block_ids
            assert is_free != is_used, "prefix plan is stale: block state changed"
            assert block.hash == block_hash, "prefix plan is stale: hash changed"
            assert block.token_ids == token_ids, (
                "prefix plan is stale: tokens changed"
            )
            assert (is_free and block.ref_count == 0) or (
                is_used and block.ref_count > 0
            )
            claims.append((block_id, block_hash, token_ids, is_free))
            prefix_hash = block_hash

        for block_id, block_hash, token_ids, is_free in claims:
            if is_free:
                block = self._allocate_block(block_id)
            else:
                block = self.blocks[block_id]
                block.ref_count += 1
            block.update(block_hash, token_ids)
            self.hash_to_block_id[block_hash] = block_id
            seq.block_table.append(block_id)
            seq.num_cached_tokens += self.block_size

    def get_token_layout(self, seq: Sequence):
        """Compatibility wrapper for the legacy waiting scheduler."""
        block_ids = self.match_prefix(seq)
        num_used_tokens = sum(
            self.block_size
            for block_id in block_ids
            if block_id in self.used_block_ids
        )
        num_free_tokens = len(block_ids) * self.block_size - num_used_tokens
        num_new_tokens = (
            len(seq) - seq.num_cached_tokens - len(block_ids) * self.block_size
        )
        assert num_new_tokens >= 0
        return num_used_tokens, num_free_tokens, num_new_tokens

    def allocate_new(self, seq: Sequence) -> None:
        """Allocate only the uncached suffix selected for this step."""
        self._assert_block_size(seq)
        h = self._validate_claimed_prefix(seq)
        assert seq.num_new_tokens >= 0
        context_end = seq.num_cached_tokens + seq.num_new_tokens
        assert context_end <= len(seq)
        required_blocks = (
            seq.num_new_tokens + self.block_size - 1
        ) // self.block_size
        assert required_blocks <= len(self.free_block_ids)
        
        # Hash only tokens committed to this scheduling step. A partial block
        # must remain unpublished until may_append observes it as complete.
        for i in range(seq.num_cached_tokens, context_end, self.block_size):
            token_ids = seq[i: min(i + self.block_size, context_end)]
            block_hash = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            if block_hash != -1:
                block.update(block_hash, token_ids)
                self.hash_to_block_id[block_hash] = block_id
            seq.block_table.append(block_id)
            h = block_hash

    def allocate(self, seq: Sequence) -> None:
        """Compatibility wrapper for the legacy waiting scheduler."""
        block_ids = self.match_prefix(seq)
        self.claim_prefix(seq, block_ids)
        self.allocate_new(seq)



    def deallocate(self, seq: Sequence):
        """
        For finished seq or preempted seq in the running queue.
        """
        self._assert_block_size(seq)
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_new_tokens = 0
        seq.block_table.clear()

    def num_blocks_to_append(
        self, seq: Sequence, num_new_tokens: int
    ) -> int:
        self._assert_block_size(seq)
        assert num_new_tokens >= 0
        target_blocks = (
            seq.num_cached_tokens + num_new_tokens + self.block_size - 1
        ) // self.block_size
        return max(target_blocks - len(seq.block_table), 0)

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        """
        Only for seq in the running queue.
        """
        return self.num_blocks_to_append(
            seq, num_new_tokens
        ) <= len(self.free_block_ids)

    def may_append(self, seq: Sequence):
        """
        Only for seq in the running queue.
        """
        self._assert_block_size(seq)
        for i in range(
            seq.num_cached_blocks * self.block_size, 
            seq.num_cached_tokens + seq.num_new_tokens, 
            self.block_size
        ):  
            token_ids = seq[i: min(i + self.block_size, seq.num_cached_tokens + seq.num_new_tokens)]
            current_block_id = seq.block_table[i // self.block_size] \
                    if i // self.block_size < len(seq.block_table) else -1
            if current_block_id != -1:
                current_block = self.blocks[current_block_id]
                assert current_block.hash == -1
            if len(token_ids) % self.block_size == 0:
                previous_block_id = seq.block_table[i // self.block_size - 1] if i >= self.block_size else -1
                prefix = self.blocks[previous_block_id].hash if previous_block_id != -1 else -1
                h = self.compute_hash(token_ids, prefix)
                if current_block_id == -1:
                    block_id = self.free_block_ids[0]
                    current_block = self._allocate_block(block_id)
                    seq.block_table.append(block_id)
                current_block.update(h, token_ids)
                self.hash_to_block_id[h] = current_block.block_id
            elif current_block_id == -1:
                    block_id = self.free_block_ids[0]
                    self._allocate_block(block_id)
                    seq.block_table.append(block_id)
