from contextlib import contextmanager
from dataclasses import dataclass
import torch
from typing import List, Optional, Tuple, Union, Any, Dict

@dataclass
class Context:
    is_prefill : bool = False
    cu_seqlens_q : Optional[torch.Tensor] = None # | None = None
    cu_seqlens_k : Optional[torch.Tensor] = None # | None = None
    max_seqlen_q : int = 0
    max_seqlen_k : int = 0
    slot_mapping : Optional[torch.Tensor] = None # | None = None
    context_lens : Optional[torch.Tensor] = None # | None = None
    block_tables : Optional[torch.Tensor] = None # | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, ):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

