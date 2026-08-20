from dataclasses import dataclass
import torch


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
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
