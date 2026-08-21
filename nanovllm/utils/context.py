from dataclasses import dataclass
from enum import Enum
import torch


class BatchType(str, Enum):
    PURE_PREFILL = "pure_prefill"
    PURE_DECODE = "pure_decode"
    MIXED = "mixed"


class RuntimeExecutionMode(str, Enum):
    EAGER = "eager"
    FULL_GRAPH = "full_graph"


@dataclass
class Context:
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    seq_need_compute_logits: torch.Tensor | None = None
    page_q_indptr: torch.Tensor | None = None
    page_kv_indptr: torch.Tensor | None = None
    page_indices: torch.Tensor | None = None
    page_last_page_len: torch.Tensor | None = None
    num_prefill_seqs: int | None = None
    num_prefill_tokens: int | None = None
    num_decode_tokens: int | None = None
    batch_type: BatchType | None = None
    runtime_mode: RuntimeExecutionMode = RuntimeExecutionMode.EAGER


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    seq_need_compute_logits=None,
    page_q_indptr=None,
    page_kv_indptr=None,
    page_indices=None,
    page_last_page_len=None,
    num_prefill_seqs=None,
    num_prefill_tokens=None,
    num_decode_tokens=None,
    batch_type=None,
    runtime_mode=RuntimeExecutionMode.EAGER,
):
    global _CONTEXT
    _CONTEXT = Context(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        seq_need_compute_logits=seq_need_compute_logits,
        page_q_indptr=page_q_indptr,
        page_kv_indptr=page_kv_indptr,
        page_indices=page_indices,
        page_last_page_len=page_last_page_len,
        num_prefill_seqs=num_prefill_seqs,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        batch_type=batch_type,
        runtime_mode=runtime_mode,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
