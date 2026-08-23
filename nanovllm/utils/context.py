from dataclasses import dataclass
from enum import Enum
from typing import Any
import torch


class BatchType(str, Enum):
    PURE_PREFILL = "pure_prefill"
    PURE_DECODE = "pure_decode"
    MIXED = "mixed"


class RuntimeExecutionMode(str, Enum):
    EAGER = "eager"
    FULL_GRAPH = "full_graph"


@dataclass(frozen=True)
class CommonAttentionMetadata:
    """Backend-neutral metadata produced once for one scheduled batch."""

    num_prefill_seqs: int
    num_decode_seqs: int
    num_prefill_tokens: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor | None = None
    max_q_len: int = 0
    max_kv_len: int = 0
    block_counts: tuple[int, ...] = ()
    num_kv_blocks: int = 0
    num_prefill_kv_blocks: int = 0
    trusted: bool = False

    @property
    def num_seqs(self) -> int:
        return self.num_prefill_seqs + self.num_decode_seqs

    @property
    def num_query_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens


@dataclass
class Context:
    attention_metadata: CommonAttentionMetadata | None = None
    attention_plan: Any = None
    seq_need_compute_logits: torch.Tensor | None = None
    runtime_mode: RuntimeExecutionMode = RuntimeExecutionMode.EAGER

    @property
    def slot_mapping(self) -> torch.Tensor | None:
        metadata = self.attention_metadata
        return None if metadata is None else metadata.slot_mapping

    @property
    def cu_seqlens_q(self) -> torch.Tensor | None:
        """Compatibility alias for the backend-neutral query offsets."""
        metadata = self.attention_metadata
        return None if metadata is None else metadata.query_start_loc

    @property
    def max_seqlen_q(self) -> int:
        metadata = self.attention_metadata
        return 0 if metadata is None else metadata.max_q_len

    @property
    def batch_type(self) -> BatchType | None:
        plan = self.attention_plan
        return None if plan is None else plan.batch_type


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    attention_metadata: CommonAttentionMetadata | None = None,
    attention_plan: Any = None,
    seq_need_compute_logits=None,
    runtime_mode=RuntimeExecutionMode.EAGER,
):
    global _CONTEXT
    _CONTEXT = Context(
        attention_metadata=attention_metadata,
        attention_plan=attention_plan,
        seq_need_compute_logits=seq_need_compute_logits,
        runtime_mode=runtime_mode,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
