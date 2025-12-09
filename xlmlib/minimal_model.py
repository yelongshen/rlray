"""
Minimal Transformer Model with Tensor Parallel Support
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ====================== Tensor Parallel Utilities ======================

def get_tensor_model_parallel_group():
    """Get the tensor model parallel group."""
    from parallel_initialization import get_model_parallel_group
    return get_model_parallel_group()


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    from parallel_initialization import get_model_parallel_world_size
    return get_model_parallel_world_size()


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    from parallel_initialization import get_model_parallel_rank
    return get_model_parallel_rank()


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # All-reduce across tensor parallel group
        group = get_tensor_model_parallel_group()
        if dist.get_world_size(group) > 1:
            dist.all_reduce(grad_output, group=group)
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        # All-reduce across tensor parallel group
        group = get_tensor_model_parallel_group()
        if dist.get_world_size(group) > 1:
            dist.all_reduce(input_, group=group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chunk to the rank."""

    @staticmethod
    def forward(ctx, input_):
        group = get_tensor_model_parallel_group()
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return input_

        # Split along last dimension.
        rank = get_tensor_model_parallel_rank()
        last_dim_size = input_.size()[-1]
        assert last_dim_size % world_size == 0
        per_partition_size = last_dim_size // world_size
        
        # Split along last dimension
        input_list = torch.split(input_, per_partition_size, dim=-1)
        output = input_list[rank].contiguous()
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # All-gather across tensor parallel group
        group = get_tensor_model_parallel_group()
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return grad_output

        # Size and dimension.
        last_dim = grad_output.dim() - 1
        rank = get_tensor_model_parallel_rank()

        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        tensor_list[rank] = grad_output
        dist.all_gather(tensor_list, grad_output, group=group)

        # Concatenate.
        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        return output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_):
        group = get_tensor_model_parallel_group()
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return input_

        # Size and dimension.
        last_dim = input_.dim() - 1
        rank = get_tensor_model_parallel_rank()

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        dist.all_gather(tensor_list, input_, group=group)

        # Concatenate.
        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Split along last dimension.
        group = get_tensor_model_parallel_group()
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return grad_output

        rank = get_tensor_model_parallel_rank()
        last_dim_size = grad_output.size()[-1]
        assert last_dim_size % world_size == 0
        per_partition_size = last_dim_size // world_size
        
        # Split
        grad_list = torch.split(grad_output, per_partition_size, dim=-1)
        output = grad_list[rank].contiguous()

        return output


# Helper functions
def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


# ====================== Tensor Parallel Layers ======================

class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.
    
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: str = "normal",
        std: float = 0.02,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = output_size // world_size

        # Parameters.
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            self.input_size,
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition
            ))
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        if init_method == "normal":
            nn.init.normal_(self.weight, mean=0.0, std=std)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Set up backprop all-reduce.
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        
        return output


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    
    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: str = "normal",
        std: float = 0.02,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel

        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = input_size // world_size

        # Parameters.
        self.weight = nn.Parameter(torch.empty(
            self.output_size,
            self.input_size_per_partition,
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        if init_method == "normal":
            nn.init.normal_(self.weight, mean=0.0, std=std)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        
        return output


class VocabParallelEmbedding(nn.Module):
    """Embedding parallelized in the vocabulary dimension.
    
    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        init_method: str = "normal",
        std: float = 0.02,
    ):
        super(VocabParallelEmbedding, self).__init__()
        
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Divide the weight matrix along the vocab dimension.
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        
        self.vocab_start_idx = rank * num_embeddings // world_size
        self.vocab_end_idx = (rank + 1) * num_embeddings // world_size
        self.num_embeddings_per_partition = self.vocab_end_idx - self.vocab_start_idx
        
        # Allocate weights.
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition,
            self.embedding_dim
        ))
        
        # Initialize weight.
        if init_method == "normal":
            nn.init.normal_(self.weight, mean=0.0, std=std)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_idx) | (input_ >= self.vocab_end_idx)
        
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_idx
        masked_input[input_mask] = 0
        
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, None, 2.0, False, False)
        
        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.0
        
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        
        return output


# ====================== Model Components ======================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin.unsqueeze(1)  # [seq_len, 1, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLP(nn.Module):
    """MLP with tensor parallelism support"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_tensor_parallel: bool = False,
    ):
        super().__init__()
        self.use_tensor_parallel = use_tensor_parallel
        
        if use_tensor_parallel:
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
            )
            self.up_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                bias=False,
                gather_output=False,
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                input_is_parallel=True,
            )
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down


class Attention(nn.Module):
    """Multi-head attention with tensor parallelism support"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        use_tensor_parallel: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.use_tensor_parallel = use_tensor_parallel
        
        if use_tensor_parallel:
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                num_attention_heads * self.head_dim,
                bias=False,
                gather_output=False,
            )
            self.k_proj = ColumnParallelLinear(
                hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
            )
            self.o_proj = RowParallelLinear(
                num_attention_heads * self.head_dim,
                hidden_size,
                bias=False,
                input_is_parallel=True,
            )
            
            # Adjust for tensor parallel
            world_size = get_tensor_model_parallel_world_size()
            self.num_attention_heads = num_attention_heads // world_size
            self.num_key_value_heads = self.num_key_value_heads // world_size
        else:
            self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat k/v heads if necessary (for GQA)
        if self.num_key_value_heads != self.num_attention_heads:
            key_states = key_states.repeat_interleave(
                self.num_attention_heads // self.num_key_value_heads, dim=1
            )
            value_states = value_states.repeat_interleave(
                self.num_attention_heads // self.num_key_value_heads, dim=1
            )

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class TransformerBlock(nn.Module):
    """Transformer decoder block"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        use_tensor_parallel: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            use_tensor_parallel=use_tensor_parallel,
        )
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_tensor_parallel=use_tensor_parallel,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MinimalTransformer(nn.Module):
    """Minimal Transformer model with DP/TP support"""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        use_tensor_parallel: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_tensor_parallel = use_tensor_parallel

        # Embedding
        if use_tensor_parallel:
            self.embed_tokens = VocabParallelEmbedding(
                vocab_size,
                hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                use_tensor_parallel=use_tensor_parallel,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])

        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # LM head
        if use_tensor_parallel:
            self.lm_head = ColumnParallelLinear(
                hidden_size,
                vocab_size,
                bias=False,
                gather_output=True,
            )
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask
        if attention_mask is None:
            seq_length = input_ids.shape[1]
            attention_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=input_ids.device),
                diagonal=1
            )
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            loss = loss_fct(shift_logits, shift_labels)

        return logits, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), subtract position and token embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params
