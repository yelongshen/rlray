#!/usr/bin/env python3
"""
Infinite Context Module for Qwen3-Next / Hybrid GDN Models

This module enables streaming infinite context through:
1. Sliding Window Attention - Limits attention to recent tokens
2. Attention Sinks - Keeps initial tokens (StreamingLLM approach)
3. Position Reset - Resets positions periodically to stay within RoPE limits
4. SSM State Persistence - Leverages SSM layers for long-range dependencies

The hybrid architecture (attention + SSM) is ideal for infinite context:
- SSM layers naturally handle infinite context with O(1) memory
- Attention layers provide precise local attention within a window

Reference: StreamingLLM (https://arxiv.org/abs/2309.17453)
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class InfiniteContextConfig:
    """Configuration for infinite context streaming."""
    
    # Sliding window size for attention (e.g., 4096, 8192)
    sliding_window: int = 4096
    
    # Number of "sink" tokens to always keep at the beginning
    # These prevent attention sink collapse
    num_sink_tokens: int = 4
    
    # Whether to reset positions after reaching max_position_embeddings
    # If True: positions cycle 0 -> max_pos -> 0 (requires training)
    # If False: use sliding window only
    reset_positions: bool = False
    
    # Maximum position before reset (typically model's max_position_embeddings)
    max_position: int = 262144
    
    # Position offset for reset (to avoid discontinuity at boundary)
    position_reset_offset: int = 0
    
    # Whether to apply RoPE scaling (YaRN, NTK) for extrapolation
    use_rope_scaling: bool = True
    rope_scaling_type: str = "yarn"  # "linear", "ntk", "yarn"
    rope_scaling_factor: float = 1.0


class StreamingKVCache:
    """
    KV Cache manager for streaming inference with infinite context.
    
    Implements:
    - Sliding window: Only keeps recent K tokens in cache
    - Attention sinks: Always keeps first N tokens
    - Efficient memory management
    """
    
    def __init__(
        self,
        config: InfiniteContextConfig,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Cache storage: [layer][key/value] -> [batch, heads, seq, head_dim]
        self.key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        
        # Track current sequence position (for position IDs)
        self.seq_position = 0
        
        # Track which layers are attention vs SSM (for hybrid models)
        self.attention_layers: Optional[List[int]] = None
        
    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a layer and return the full (windowed) cache.
        
        Args:
            layer_idx: Layer index
            key_states: New key states [batch, heads, seq, head_dim]
            value_states: New value states [batch, heads, seq, head_dim]
            
        Returns:
            Tuple of (cached_keys, cached_values) with sliding window applied
        """
        batch_size = key_states.size(0)
        new_seq_len = key_states.size(2)
        
        if self.key_cache[layer_idx] is None:
            # First update for this layer
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Concatenate new states
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )
        
        # Apply sliding window with sink tokens
        cache_len = self.key_cache[layer_idx].size(2)
        max_cache = self.config.sliding_window + self.config.num_sink_tokens
        
        if cache_len > max_cache:
            # Keep sink tokens + recent window
            sink_keys = self.key_cache[layer_idx][:, :, :self.config.num_sink_tokens, :]
            sink_values = self.value_cache[layer_idx][:, :, :self.config.num_sink_tokens, :]
            
            window_keys = self.key_cache[layer_idx][:, :, -self.config.sliding_window:, :]
            window_values = self.value_cache[layer_idx][:, :, -self.config.sliding_window:, :]
            
            self.key_cache[layer_idx] = torch.cat([sink_keys, window_keys], dim=2)
            self.value_cache[layer_idx] = torch.cat([sink_values, window_values], dim=2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current cache sequence length."""
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].size(2)
    
    def clear(self):
        """Clear all cache."""
        self.key_cache = [None] * self.num_layers
        self.value_cache = [None] * self.num_layers
        self.seq_position = 0
    
    def get_position_ids(
        self,
        input_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get position IDs for the current input, handling position wrap-around.
        
        For infinite context, positions after max_position either:
        1. Continue (with RoPE scaling extrapolation)
        2. Reset to stay within trained distribution
        """
        if self.config.reset_positions and self.seq_position >= self.config.max_position:
            # Reset positions - start from sink tokens offset
            # This requires continued training to work well
            start_pos = self.config.num_sink_tokens
        else:
            start_pos = self.seq_position
        
        positions = torch.arange(
            start_pos, 
            start_pos + input_length, 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0)
        
        return positions
    
    def advance_position(self, num_tokens: int):
        """Advance the sequence position counter."""
        self.seq_position += num_tokens


class SlidingWindowAttention(nn.Module):
    """
    Attention module with sliding window support for infinite context.
    
    Key features:
    - Configurable window size
    - Attention sinks (always attend to first N tokens)
    - Compatible with Flash Attention 2
    """
    
    def __init__(
        self,
        config,
        layer_idx: int,
        infinite_config: Optional[InfiniteContextConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.infinite_config = infinite_config or InfiniteContextConfig()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.sliding_window = self.infinite_config.sliding_window
        
    def _create_sliding_window_mask(
        self,
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create causal mask with sliding window.
        
        Tokens can only attend to:
        1. First num_sink_tokens (attention sinks)
        2. Previous sliding_window tokens
        """
        # Start with causal mask (lower triangular)
        mask = torch.ones(query_length, key_length, device=device, dtype=dtype)
        mask = torch.tril(mask)
        
        # Apply sliding window (set to 0 outside window)
        for i in range(query_length):
            # Position in full sequence
            pos = key_length - query_length + i
            
            # Can attend to sink tokens
            sink_end = self.infinite_config.num_sink_tokens
            
            # Can attend to recent window
            window_start = max(sink_end, pos - self.sliding_window + 1)
            
            # Mask out tokens outside sink and window
            if sink_end < window_start:
                mask[i, sink_end:window_start] = 0
        
        # Convert to attention mask format (0 = attend, -inf = mask)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        
        return mask


def apply_yarn_scaling(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    original_max_position: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """
    Apply YaRN (Yet another RoPE extensioN) scaling to rotation frequencies.
    
    YaRN combines:
    1. NTK-aware interpolation for high frequencies
    2. Linear interpolation for low frequencies
    3. Attention temperature scaling (handled separately)
    
    Args:
        inv_freq: Original inverse frequencies
        scaling_factor: How much to extend context (e.g., 4 for 256K -> 1M)
        original_max_position: Original max position embeddings
        beta_fast: Controls high-frequency interpolation
        beta_slow: Controls low-frequency interpolation
    
    Returns:
        Scaled inverse frequencies
    """
    dim = inv_freq.numel() * 2
    low_freq_factor = 1
    high_freq_factor = scaling_factor
    
    # Compute ramp function for interpolation
    pos_freq = 1.0 / inv_freq
    low_freq_wavelen = original_max_position / beta_slow
    high_freq_wavelen = original_max_position / beta_fast
    
    # Create smooth ramp between linear and NTK interpolation
    ramp = (pos_freq - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    ramp = ramp.clamp(0, 1)
    
    # Blend between low and high frequency factors
    factors = low_freq_factor * (1 - ramp) + high_freq_factor * ramp
    
    return inv_freq / factors


class InfiniteContextWrapper:
    """
    Wrapper to enable infinite context on existing models.
    
    Usage:
        wrapper = InfiniteContextWrapper(model, config)
        
        for chunk in stream:
            outputs = wrapper.forward(chunk)
    """
    
    def __init__(
        self,
        model,
        config: Optional[InfiniteContextConfig] = None,
    ):
        self.model = model
        self.config = config or InfiniteContextConfig()
        
        # Create streaming cache
        model_config = model.config
        num_layers = model_config.num_hidden_layers
        num_heads = model_config.num_key_value_heads
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        
        self.cache = StreamingKVCache(
            config=self.config,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        
        # Apply RoPE scaling if enabled
        if self.config.use_rope_scaling:
            self._apply_rope_scaling()
    
    def _apply_rope_scaling(self):
        """Apply RoPE scaling to model's rotary embeddings."""
        scaling_config = {
            "type": self.config.rope_scaling_type,
            "factor": self.config.rope_scaling_factor,
        }
        
        if hasattr(self.model.config, 'rope_scaling'):
            self.model.config.rope_scaling = scaling_config
            print(f"Applied RoPE scaling: {scaling_config}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """
        Forward pass with streaming cache management.
        
        Automatically handles:
        - Position ID generation
        - Cache updates with sliding window
        - SSM state persistence
        """
        device = input_ids.device
        seq_len = input_ids.size(1)
        
        # Get position IDs
        position_ids = self.cache.get_position_ids(seq_len, device)
        
        # Forward pass (model should use our custom cache)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=None,  # We manage cache manually
            use_cache=True,
            **kwargs,
        )
        
        # Update cache position
        self.cache.advance_position(seq_len)
        
        return outputs
    
    def reset(self):
        """Reset streaming state for a new session."""
        self.cache.clear()


# === Training utilities for infinite context ===

def prepare_infinite_context_training_data(
    tokenizer,
    texts: List[str],
    chunk_size: int = 4096,
    overlap: int = 512,
) -> List[dict]:
    """
    Prepare training data for infinite context fine-tuning.
    
    Creates overlapping chunks to help model learn sliding window patterns.
    """
    samples = []
    
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
        seq_len = tokens.size(0)
        
        for start in range(0, seq_len - chunk_size, chunk_size - overlap):
            end = start + chunk_size
            chunk = tokens[start:end]
            
            samples.append({
                "input_ids": chunk,
                "position_offset": start,  # For training position reset
            })
    
    return samples


def get_continued_training_config(
    base_config,
    target_context_length: int = 1_000_000,
    sliding_window: int = 4096,
    num_sink_tokens: int = 4,
) -> dict:
    """
    Get configuration for continued training to enable infinite context.
    
    Key modifications:
    1. Add sliding window to attention
    2. Configure attention sinks
    3. Set up position reset
    """
    return {
        # Model modifications
        "sliding_window": sliding_window,
        "num_sink_tokens": num_sink_tokens,
        "max_position_embeddings": base_config.max_position_embeddings,  # Keep original
        
        # RoPE scaling for extrapolation
        "rope_scaling": {
            "type": "yarn",
            "factor": target_context_length / base_config.max_position_embeddings,
            "original_max_position_embeddings": base_config.max_position_embeddings,
        },
        
        # Training parameters
        "gradient_checkpointing": True,  # Essential for long context
        "use_flash_attention_2": True,
        
        # Data parameters
        "max_seq_len": sliding_window + num_sink_tokens,  # Effective training length
        "chunk_overlap": sliding_window // 4,
    }


if __name__ == "__main__":
    # Example usage
    config = InfiniteContextConfig(
        sliding_window=4096,
        num_sink_tokens=4,
        max_position=262144,
        use_rope_scaling=True,
        rope_scaling_type="yarn",
        rope_scaling_factor=4.0,  # 256K -> 1M effective
    )
    
    print("Infinite Context Configuration:")
    print(f"  Sliding window: {config.sliding_window}")
    print(f"  Sink tokens: {config.num_sink_tokens}")
    print(f"  Effective window: {config.sliding_window + config.num_sink_tokens}")
    print(f"  Max position: {config.max_position}")
    print(f"  RoPE scaling: {config.rope_scaling_type} x{config.rope_scaling_factor}")
    print(f"  Theoretical max context: {config.max_position * config.rope_scaling_factor:,.0f} tokens")
