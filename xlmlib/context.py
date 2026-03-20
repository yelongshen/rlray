from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import torch
from typing import List, Optional, Tuple, Union, Any, Dict


class ContextStatus(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"

@dataclass
class Context:
    cu_seqlens_q : Optional[torch.Tensor] = None # | None = None
    cu_seqlens_k : Optional[torch.Tensor] = None # | None = None
    max_seqlen_q : int = 0
    max_seqlen_k : int = 0
    slot_mapping : Optional[torch.Tensor] = None # | None = None
    context_lens : Optional[torch.Tensor] = None # | None = None
    block_tables : Optional[torch.Tensor] = None # | None = None
    linear_slots : Optional[torch.Tensor] = None # | None = None

    @property
    def status(self) -> ContextStatus:
        has_prefill = self.cu_seqlens_q is not None
        has_decode = (
            self.slot_mapping is not None
            or self.context_lens is not None
            or self.block_tables is not None
            or self.linear_slots is not None
        )

        if has_prefill and has_decode:
            return ContextStatus.MIXED
        if has_prefill:
            return ContextStatus.PREFILL
        return ContextStatus.DECODE

    @property
    def is_prefill(self) -> bool:
        return self.status in (ContextStatus.PREFILL, ContextStatus.MIXED)

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, linear_slots=None):
    global _CONTEXT
    _CONTEXT = Context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, linear_slots)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

