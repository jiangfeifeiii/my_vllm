from copy import copy
from enum import Enum, auto
from itertools import count

import numpy as np
import xxhash

from nanovllm.sampling_params import SamplingParams


def compute_block_hash(token_ids: list[int], prefix: int = -1) -> int:
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        block_size: int = 16,
    ):
        assert block_size > 0
        self.seq_id = next(Sequence.counter)
        self._block_size = block_size
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.block_hashes = self._build_block_hashes()
        self.num_cached_tokens = 0
        self.num_new_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def block_size(self) -> int:
        """The cache page size is immutable for a sequence's lifetime."""
        return self._block_size

    def _build_block_hashes(self) -> list[int]:
        block_hashes: list[int] = []
        prefix_hash = -1
        for offset in range(0, len(self.token_ids), self.block_size):
            token_ids = self.token_ids[offset : offset + self.block_size]
            if len(token_ids) != self.block_size:
                break
            prefix_hash = compute_block_hash(token_ids, prefix_hash)
            block_hashes.append(prefix_hash)
        return block_hashes

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens
    
    @property
    def num_context_tokens(self):
        return self.num_cached_tokens + self.num_new_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_current_blocks(self):
        assert (self.num_cached_tokens + self.num_new_tokens + self.block_size - 1) // self.block_size == len(self.block_table)
        return len(self.block_table)

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # @property
    # def last_block_num_tokens(self):
    #     return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        assert self.num_tokens == len(self.token_ids)
        next_num_tokens = self.num_tokens + 1
        next_block_hash = None
        if next_num_tokens % self.block_size == 0:
            block_index = next_num_tokens // self.block_size - 1
            assert block_index == len(self.block_hashes)
            block_start = block_index * self.block_size
            token_ids = self.token_ids[block_start:] + [token_id]
            assert len(token_ids) == self.block_size
            prefix_hash = self.block_hashes[-1] if self.block_hashes else -1
            next_block_hash = compute_block_hash(token_ids, prefix_hash)

        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens = next_num_tokens
        assert self.num_tokens == len(self.token_ids)
        if next_block_hash is not None:
            self.block_hashes.append(next_block_hash)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        restored = state.copy()
        if "_block_size" not in restored:
            restored["_block_size"] = restored.pop("block_size")
        self.__dict__.update(restored)
        if "block_hashes" not in restored:
            self.block_hashes = self._build_block_hashes()
