"""
Qwen3-Next Model Adapter for In-house LLM Engine

This file adapts the HuggingFace Qwen3-Next model to work with the in-house
llm_engine.py for fast inference with paged attention.

The in-house LLMEngine expects:
- model(input_ids=..., position_ids=..., inference_mode=True, logits_to_keep=...) -> (_, logits, _, _)
- Attention layers with k_cache and v_cache attributes for paged attention

Usage:
    from qwen3_next_engine import Qwen3NextForLLMEngine, load_qwen3_next_for_engine
    
    model, tokenizer, config = load_qwen3_next_for_engine("Qwen/Qwen3-Coder-Next")
    engine = LLMEngine(model, config, "cuda")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, paged attention will not work")

try:
    from context import set_context, get_context, reset_context
except ImportError:
    print("Warning: context module not found, using dummy implementation")
    class DummyContext:
        is_prefill = False
        slot_mapping = None
        context_lens = None
        block_tables = None
        cu_seqlens_q = None
        cu_seqlens_k = None
        max_seqlen_q = 0
        max_seqlen_k = 0
    _context = DummyContext()
    def get_context(): return _context
    def set_context(*args, **kwargs): pass
    def reset_context(): pass


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.jit.script
def store_kvcache(key_states: torch.Tensor, value_states: torch.Tensor, 
                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                  slot_mapping: torch.Tensor):
    """Store key/value states into paged KV cache."""
    k_cache.index_copy_(0, slot_mapping, key_states)
    v_cache.index_copy_(0, slot_mapping, value_states)


class Qwen3NextRMSNorm(nn.Module):
    """RMSNorm for Qwen3-Next."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3NextRotaryEmbedding(nn.Module):
    """Rotary position embedding for Qwen3-Next."""
    def __init__(self, dim, max_position_embeddings=131072, base=1000000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3NextAttentionForEngine(nn.Module):
    """
    Qwen3-Next Attention adapted for in-house LLM Engine.
    
    Key changes from HuggingFace version:
    - Uses paged attention with external k_cache/v_cache
    - Supports inference_mode parameter
    - Uses flash_attn_varlen_func and flash_attn_with_kvcache
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Projections (Qwen3-Next uses combined q_proj that outputs q and gate)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # QK norm (Qwen3-Next specific)
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # KV cache for paged attention (will be set by LLMEngine)
        self.k_cache = None
        self.v_cache = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        inference_mode: bool = False,
        max_generation: int = 0,
        cur_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        
        # Compute Q, K, V (Qwen3-Next has q_proj output q and gate concatenated)
        qg = self.q_proj(hidden_states)
        query_states, gate = torch.chunk(qg, 2, dim=-1)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        # Apply QK norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # Transpose for attention: [bsz, num_heads, q_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        
        # Apply rotary embeddings
        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Transpose back for flash attention: [bsz, q_len, num_heads, head_dim]
        query_states = query_states.transpose(1, 2).to(hidden_states.dtype)
        key_states = key_states.transpose(1, 2).to(hidden_states.dtype)
        
        key_cache = None
        value_cache = None
        page_attention = False
        
        if not inference_mode:
            # Training mode - no paged attention
            key_cache = key_states
            value_cache = value_states
        elif inference_mode and self.k_cache is not None and self.v_cache is not None:
            # Inference mode with paged attention
            key_cache = None
            value_cache = None
            context = get_context()
            
            query_states = query_states.view(-1, self.num_heads, self.head_dim).contiguous()
            key_states = key_states.view(-1, self.num_key_value_heads, self.head_dim).contiguous()
            value_states = value_states.view(-1, self.num_key_value_heads, self.head_dim).contiguous()
            
            store_kvcache(key_states, value_states, self.k_cache, self.v_cache, context.slot_mapping)
            page_attention = True
        else:
            # Fallback: simple KV cache
            key_cache = key_states
            value_cache = value_states
        
        if page_attention and FLASH_ATTN_AVAILABLE:
            context = get_context()
            if context.is_prefill:
                attn_output = flash_attn_varlen_func(
                    query_states, self.k_cache, self.v_cache,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    causal=True,
                    block_table=context.block_tables
                )
            else:
                attn_output = flash_attn_with_kvcache(
                    query_states.unsqueeze(1),
                    self.k_cache, self.v_cache,
                    cache_seqlens=context.context_lens,
                    causal=True,
                    block_table=context.block_tables
                )
        else:
            # Fallback: standard attention
            key_states_expanded = key_cache.repeat_interleave(self.num_key_value_groups, dim=2)
            value_states_expanded = value_cache.repeat_interleave(self.num_key_value_groups, dim=2)
            
            # [bsz, num_heads, q_len, head_dim] @ [bsz, num_heads, head_dim, kv_len]
            query_for_attn = query_states.transpose(1, 2) if not page_attention else query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_for_attn = key_states_expanded.transpose(1, 2)
            value_for_attn = value_states_expanded.transpose(1, 2)
            
            attn_weights = torch.matmul(query_for_attn, key_for_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_for_attn.dtype)
            attn_output = torch.matmul(attn_weights, value_for_attn)
            attn_output = attn_output.transpose(1, 2)
        
        # Reshape and apply gate
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, key_cache, value_cache


class Qwen3NextMLPForEngine(nn.Module):
    """MLP for Qwen3-Next."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3NextDecoderLayerForEngine(nn.Module):
    """Decoder layer adapted for LLM Engine (full attention only)."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3NextAttentionForEngine(config, layer_idx)
        self.mlp = Qwen3NextMLPForEngine(config)
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        inference_mode: bool = False,
        max_generation: int = 0,
        cur_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, k_cache, v_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            inference_mode=inference_mode,
            max_generation=max_generation,
            cur_pos=cur_pos,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, k_cache, v_cache


class Qwen3NextModelForEngine(nn.Module):
    """Qwen3-Next Model backbone adapted for LLM Engine."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3NextDecoderLayerForEngine(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3NextRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inference_mode: bool = False,
        max_generation: int = 0,
        cur_pos: int = 0,
    ) -> Tuple[torch.Tensor, List]:
        batch_size, seq_length = input_ids.shape[:2]
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = self.embed_tokens(input_ids)
        
        # Compute rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        
        next_cache = []
        for layer in self.layers:
            hidden_states, k_cache, v_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                inference_mode=inference_mode,
                max_generation=max_generation,
                cur_pos=cur_pos,
            )
            next_cache.append((k_cache, v_cache))
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache


class Qwen3NextForLLMEngine(nn.Module):
    """
    Qwen3-Next For Causal LM adapted for in-house LLM Engine.
    
    Key interface requirement for LLMEngine:
    - forward(input_ids, position_ids, inference_mode=True, logits_to_keep=...) -> (_, logits, _, _)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3NextModelForEngine(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inference_mode: bool = False,
        max_generation: int = 0,
        cur_pos: int = 0,
        logits_to_keep: Optional[torch.Tensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Tuple[None, torch.Tensor, None, List]:
        """
        Forward pass compatible with LLMEngine.
        
        Returns: (_, logits, _, past_key_values)
        """
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inference_mode=inference_mode,
            max_generation=max_generation,
            cur_pos=cur_pos,
        )
        
        # Handle logits_to_keep for efficient prefill
        if logits_to_keep is not None:
            hidden_states = hidden_states.squeeze(0)[logits_to_keep]
            hidden_states = hidden_states.unsqueeze(0)
        elif num_logits_to_keep > 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        
        logits = self.lm_head(hidden_states)
        
        # Return format expected by LLMEngine: (_, logits, _, cache)
        return None, logits, None, next_cache


def load_qwen3_next_for_engine(model_path: str, device: str = "cuda", torch_dtype=torch.bfloat16):
    """
    Load Qwen3-Next model and convert to LLMEngine-compatible format.
    
    This loads the HuggingFace model and copies weights to our custom model.
    
    Returns:
        model: Qwen3NextForLLMEngine instance
        tokenizer: AutoTokenizer instance
        llm_config: Config object for LLMEngine
    """
    print(f"Loading Qwen3-Next from {model_path}...")
    
    # Load HuggingFace model
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",  # Load to CPU first for weight transfer
        trust_remote_code=True
    )
    
    # Create our engine-compatible model
    model = Qwen3NextForLLMEngine(hf_config)
    
    # Copy weights from HuggingFace model
    print("Copying weights to engine-compatible model...")
    _copy_weights(hf_model, model, hf_config)
    
    # Move to device
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    
    # Create LLMEngine-compatible config
    class LLMConfig:
        pass
    
    llm_config = LLMConfig()
    llm_config.hidden_size = hf_config.hidden_size
    llm_config.num_attention_heads = hf_config.num_attention_heads
    llm_config.num_key_value_heads = hf_config.num_key_value_heads
    llm_config.num_hidden_layers = hf_config.num_hidden_layers
    llm_config.eos_token_id = tokenizer.eos_token_id
    
    # Clean up HF model
    del hf_model
    torch.cuda.empty_cache()
    
    print("Model loaded and converted successfully!")
    return model, tokenizer, llm_config


def _copy_weights(hf_model, engine_model, config):
    """Copy weights from HuggingFace model to our engine-compatible model."""
    
    # Copy embeddings
    engine_model.model.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
    
    # Copy lm_head
    engine_model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
    
    # Copy final norm
    engine_model.model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
    
    # Copy layers
    for layer_idx in range(config.num_hidden_layers):
        hf_layer = hf_model.model.layers[layer_idx]
        engine_layer = engine_model.model.layers[layer_idx]
        
        # Check layer type - only copy full attention layers
        layer_type = config.layer_types[layer_idx] if hasattr(config, 'layer_types') else "full_attention"
        
        if layer_type == "full_attention" and hasattr(hf_layer, 'self_attn'):
            # Attention weights
            engine_layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
            engine_layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
            engine_layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
            engine_layer.self_attn.o_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)
            
            # QK norm
            if hasattr(hf_layer.self_attn, 'q_norm'):
                engine_layer.self_attn.q_norm.weight.data.copy_(hf_layer.self_attn.q_norm.weight.data)
            if hasattr(hf_layer.self_attn, 'k_norm'):
                engine_layer.self_attn.k_norm.weight.data.copy_(hf_layer.self_attn.k_norm.weight.data)
        
        # MLP weights (for non-MoE layers)
        if hasattr(hf_layer, 'mlp') and hasattr(hf_layer.mlp, 'gate_proj'):
            engine_layer.mlp.gate_proj.weight.data.copy_(hf_layer.mlp.gate_proj.weight.data)
            engine_layer.mlp.up_proj.weight.data.copy_(hf_layer.mlp.up_proj.weight.data)
            engine_layer.mlp.down_proj.weight.data.copy_(hf_layer.mlp.down_proj.weight.data)
        
        # Layer norms
        engine_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
        engine_layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)


# Simple test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Coder-Next")
    parser.add_argument("--test_engine", action="store_true", help="Test with LLMEngine")
    parser.add_argument("--prompt", type=str, default="What is 2+2?")
    args = parser.parse_args()
    
    print(f"Testing load_qwen3_next_for_engine with {args.model_path}")
    model, tokenizer, config = load_qwen3_next_for_engine(args.model_path)
    print(f"Model loaded: {type(model)}")
    print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    # Test 1: Direct forward pass
    print("\n=== Test 1: Direct Forward Pass ===")
    test_input = tokenizer.encode("Hello", return_tensors="pt").cuda()
    with torch.no_grad():
        _, logits, _, _ = model(test_input, inference_mode=False)
    print(f"Input shape: {test_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print("Direct forward pass: OK")
    
    # Test 2: With LLMEngine
    if args.test_engine:
        print("\n=== Test 2: LLMEngine Generation ===")
        from llm_engine import LLMEngine
        
        config.eos_token_id = tokenizer.eos_token_id
        engine = LLMEngine(model, config, "cuda")
        
        # Prepare prompt with chat template
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(text)
        
        print(f"Prompt: {args.prompt}")
        print(f"Input tokens: {len(input_ids)}")
        
        import time
        start = time.time()
        output_ids = engine.generate([input_ids])[0]
        elapsed = time.time() - start
        
        new_tokens = len(output_ids) - len(input_ids)
        response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
        
        print(f"Generated {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")
        print(f"Response: {response}")

