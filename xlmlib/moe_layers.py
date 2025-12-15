"""
Mixture of Experts (MoE) Layers with Expert Parallel and No-Drop Tokens

This module implements MoE layers with:
- Expert Parallel: Distribute experts across GPUs
- No-Drop Tokens: All tokens are routed to experts (no capacity dropping)
- Load balancing loss
- Top-K routing with auxiliary loss
"""
import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def get_expert_parallel_group():
    """Get the expert parallel group."""
    from parallel_initialization import get_model_parallel_group
    # For now, use model parallel group for expert parallel
    # In production, you might want a separate EP group
    return get_model_parallel_group()


def get_expert_parallel_world_size():
    """Return world size for the expert parallel group."""
    try:
        from parallel_initialization import get_model_parallel_world_size
        return get_model_parallel_world_size()
    except:
        return 1


def get_expert_parallel_rank():
    """Return my rank for the expert parallel group."""
    try:
        from parallel_initialization import get_model_parallel_rank
        return get_model_parallel_rank()
    except:
        return 0


class TopKRouter(nn.Module):
    """
    Top-K Router for MoE with load balancing.
    
    Routes each token to top-k experts with no token dropping.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        
        # Router weights
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            router_logits: [batch_size * seq_len, num_experts]
            top_k_indices: [batch_size * seq_len, num_experts_per_token]
            top_k_weights: [batch_size * seq_len, num_experts_per_token]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape to [batch_size * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, hidden_size)
        
        # Compute router logits: [batch_size * seq_len, num_experts]
        router_logits = self.router(hidden_states)
        
        # Get top-k experts for each token
        top_k_logits, top_k_indices = torch.topk(
            router_logits, 
            k=self.num_experts_per_token, 
            dim=-1
        )
        
        # Compute routing weights with softmax over top-k
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        return router_logits, top_k_indices, top_k_weights
    
    def compute_aux_loss(
        self, 
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute auxiliary losses for load balancing.
        
        Args:
            router_logits: [num_tokens, num_experts]
            top_k_indices: [num_tokens, num_experts_per_token]
            
        Returns:
            aux_loss: Load balancing loss
            z_loss: Router z-loss for stability
        """
        # Load balancing loss
        # Encourages uniform distribution of tokens across experts
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Expert usage frequency
        num_tokens = router_logits.shape[0]
        expert_mask = torch.zeros_like(router_logits)
        expert_mask.scatter_(1, top_k_indices, 1.0)
        
        # Fraction of tokens assigned to each expert
        tokens_per_expert = expert_mask.sum(dim=0) / num_tokens / self.num_experts_per_token
        
        # Average router probability per expert
        router_prob_per_expert = router_probs.mean(dim=0)
        
        # Load balancing loss: encourages balance
        aux_loss = self.num_experts * torch.sum(
            tokens_per_expert * router_prob_per_expert
        )
        
        # Router z-loss: encourages router logits to stay small
        z_loss = torch.logsumexp(router_logits, dim=-1).mean()
        
        return aux_loss * self.router_aux_loss_coef, z_loss * self.router_z_loss_coef


class MoEExpert(nn.Module):
    """
    Single Expert in MoE (FFN layer).
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Args:
            x: [num_tokens, hidden_size]
            
        Returns:
            output: [num_tokens, hidden_size]
        """
        if self.activation == "silu":
            gate = F.silu(self.gate_proj(x))
        elif self.activation == "gelu":
            gate = F.gelu(self.gate_proj(x))
        else:
            gate = self.gate_proj(x)
        
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts layer with Expert Parallel and no token dropping.
    
    Features:
    - Expert Parallel: Experts distributed across EP group
    - No token dropping: All tokens routed to top-k experts
    - Load balancing with auxiliary loss
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        use_expert_parallel: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.use_expert_parallel = use_expert_parallel
        
        # Expert parallel setup
        if use_expert_parallel:
            self.ep_size = get_expert_parallel_world_size()
            self.ep_rank = get_expert_parallel_rank()
            self.num_local_experts = num_experts // self.ep_size
            self.expert_start_idx = self.ep_rank * self.num_local_experts
            self.expert_end_idx = (self.ep_rank + 1) * self.num_local_experts
        else:
            self.ep_size = 1
            self.ep_rank = 0
            self.num_local_experts = num_experts
            self.expert_start_idx = 0
            self.expert_end_idx = num_experts
        
        # Router
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            router_aux_loss_coef=router_aux_loss_coef,
            router_z_loss_coef=router_z_loss_coef,
        )
        
        # Local experts
        self.experts = nn.ModuleList([
            MoEExpert(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            for _ in range(self.num_local_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing decisions
        router_logits, top_k_indices, top_k_weights = self.router(hidden_states)
        # router_logits: [batch_size * seq_len, num_experts]
        # top_k_indices: [batch_size * seq_len, num_experts_per_token]
        # top_k_weights: [batch_size * seq_len, num_experts_per_token]
        
        # Compute auxiliary loss
        aux_loss, z_loss = self.router.compute_aux_loss(router_logits, top_k_indices)
        total_aux_loss = aux_loss + z_loss
        
        # Reshape hidden states
        hidden_states = hidden_states.view(-1, hidden_size)  # [num_tokens, hidden_size]
        
        if self.use_expert_parallel:
            # Expert parallel: route tokens across GPUs
            output = self._forward_expert_parallel(
                hidden_states, top_k_indices, top_k_weights
            )
        else:
            # Local: all experts on same GPU
            output = self._forward_local(
                hidden_states, top_k_indices, top_k_weights
            )
        
        # Reshape output
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, total_aux_loss
    
    def _forward_local(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with all experts local (no EP).
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            top_k_indices: [num_tokens, num_experts_per_token]
            top_k_weights: [num_tokens, num_experts_per_token]
            
        Returns:
            output: [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]
        output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            token_mask = (top_k_indices == expert_idx)  # [num_tokens, num_experts_per_token]
            token_indices = torch.any(token_mask, dim=-1).nonzero(as_tuple=True)[0]
            
            if token_indices.numel() == 0:
                continue
            
            # Get weights for this expert
            expert_weights = top_k_weights[token_mask].unsqueeze(-1)  # [num_selected, 1]
            
            # Get tokens for this expert
            expert_inputs = hidden_states[token_indices]  # [num_selected, hidden_size]
            
            # Forward through expert
            expert_output = self.experts[expert_idx](expert_inputs)  # [num_selected, hidden_size]
            
            # Weight and accumulate
            output[token_indices] += expert_weights * expert_output
        
        return output
    
    def _forward_expert_parallel(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with expert parallel.
        
        Each GPU has a subset of experts. Tokens are routed across GPUs.
        No tokens are dropped - all tokens are processed by their top-k experts.
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            top_k_indices: [num_tokens, num_experts_per_token]
            top_k_weights: [num_tokens, num_experts_per_token]
            
        Returns:
            output: [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]
        output = torch.zeros_like(hidden_states)
        
        # Process local experts
        for local_expert_idx in range(self.num_local_experts):
            global_expert_idx = self.expert_start_idx + local_expert_idx
            
            # Find tokens routed to this expert
            token_mask = (top_k_indices == global_expert_idx)
            token_indices = torch.any(token_mask, dim=-1).nonzero(as_tuple=True)[0]
            
            if token_indices.numel() == 0:
                continue
            
            # Get weights for this expert
            expert_weights = top_k_weights[token_mask].unsqueeze(-1)
            
            # Get tokens for this expert
            expert_inputs = hidden_states[token_indices]
            
            # Forward through expert
            expert_output = self.experts[local_expert_idx](expert_inputs)
            
            # Weight and accumulate
            output[token_indices] += expert_weights * expert_output
        
        # AllReduce across expert parallel group to combine outputs
        if self.ep_size > 1:
            group = get_expert_parallel_group()
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        
        return output


class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE FFN layer.
    
    Replaces the standard MLP with a Sparse MoE layer.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        num_key_value_heads: Optional[int] = None,
        use_tensor_parallel: bool = False,
        use_expert_parallel: bool = False,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        # Import here to avoid circular dependency
        from minimal_model import Attention, RMSNorm
        
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            use_tensor_parallel=use_tensor_parallel,
        )
        
        # MoE FFN instead of regular MLP
        self.moe = SparseMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            router_aux_loss_coef=router_aux_loss_coef,
            router_z_loss_coef=router_z_loss_coef,
            use_expert_parallel=use_expert_parallel,
        )
        
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE transformer block.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
            aux_loss: MoE auxiliary loss
        """
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        
        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, aux_loss


class MoETransformer(nn.Module):
    """
    Transformer model with Mixture of Experts layers.
    
    Supports both dense and sparse (MoE) layers.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        moe_layer_interval: int = 2,  # MoE every N layers (0 = all layers)
        use_tensor_parallel: bool = False,
        use_expert_parallel: bool = False,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_tensor_parallel = use_tensor_parallel
        self.use_expert_parallel = use_expert_parallel
        self.num_experts = num_experts
        self.moe_layer_interval = moe_layer_interval
        
        # Import here to avoid circular dependency
        from minimal_model import RMSNorm, VocabParallelEmbedding, ColumnParallelLinear, TransformerBlock
        
        # Embedding
        if use_tensor_parallel:
            self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        else:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers (mix of dense and MoE)
        self.layers = nn.ModuleList()
        for layer_idx in range(num_hidden_layers):
            # Determine if this layer should be MoE
            if moe_layer_interval > 0 and (layer_idx + 1) % moe_layer_interval == 0:
                # MoE layer
                layer = MoETransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    use_tensor_parallel=use_tensor_parallel,
                    use_expert_parallel=use_expert_parallel,
                    router_aux_loss_coef=router_aux_loss_coef,
                    router_z_loss_coef=router_z_loss_coef,
                    rms_norm_eps=rms_norm_eps,
                )
            else:
                # Dense layer
                layer = TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    use_tensor_parallel=use_tensor_parallel,
                    rms_norm_eps=rms_norm_eps,
                )
            self.layers.append(layer)
        
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through MoE transformer.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: Optional language modeling loss
            aux_loss: Optional MoE auxiliary loss
        """
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
        total_aux_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, MoETransformerBlock):
                hidden_states, aux_loss = layer(hidden_states, attention_mask=attention_mask)
                total_aux_loss += aux_loss
            else:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            lm_loss = loss_fct(shift_logits, shift_labels)
            
            # Add auxiliary loss
            loss = lm_loss + total_aux_loss
        
        return logits, loss, total_aux_loss
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params
