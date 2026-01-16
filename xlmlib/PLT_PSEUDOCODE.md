# Parallel Loop Transformer (PLT) - Pseudocode

**Paper:** "Parallel Loop Transformer for Efficient Test-Time Computation Scaling"  
**arXiv:** 2510.24824

## Overview

The Parallel Loop Transformer (PLT) enables test-time computation scaling by reusing transformer weights across multiple loops while maintaining low latency through parallel execution.

**Key Innovations:**
1. **Cross-Loop Parallelism (CLP)**: Computes different loops for different tokens simultaneously in a single pass
2. **Gated Sliding-Window Attention (G-SWA)**: Efficiently combines global and local information
3. **Shared KV Cache**: Memory-efficient representation by sharing the first loop's KV cache

---

## 1. Standard Looped Transformer (Baseline)

```python
# Traditional sequential looped transformer
def looped_transformer_sequential(
    input_ids,          # [batch, seq_len]
    num_loops,          # Number of times to reuse weights
    transformer_block   # Single transformer block with shared weights
):
    """
    Sequential looped transformer - slow due to sequential dependency
    """
    # Initial embedding
    x = embedding(input_ids)  # [batch, seq_len, hidden_dim]
    
    # Sequential loops - each depends on previous
    for loop_idx in range(num_loops):
        # Apply same transformer block repeatedly
        x = transformer_block(x)  # [batch, seq_len, hidden_dim]
        
        # Cannot parallelize - must wait for previous loop to complete
    
    # Output projection
    logits = output_projection(x)  # [batch, seq_len, vocab_size]
    return logits

# Problem: Latency = num_loops × single_block_latency
# Problem: KV cache size = num_loops × base_kv_cache_size
```

---

## 2. Cross-Loop Parallelism (CLP)

```python
def cross_loop_parallelism(
    input_ids,          # [batch, seq_len]
    num_loops,          # Number of loops
    transformer_block   # Shared transformer block
):
    """
    Core innovation: Compute different loops for different tokens in parallel
    
    Key Idea: Token i at loop j can be computed while token i+1 is at loop j-1
    This breaks the sequential dependency across loops.
    """
    batch_size, seq_len = input_ids.shape
    
    # Step 1: Create loop assignment for each token position
    # Token at position i will execute loop (i mod num_loops)
    loop_assignments = create_loop_schedule(seq_len, num_loops)
    # Example for seq_len=8, num_loops=3:
    # Position: [0, 1, 2, 3, 4, 5, 6, 7]
    # Loop:     [0, 1, 2, 0, 1, 2, 0, 1]
    
    # Step 2: Embedding
    x = embedding(input_ids)  # [batch, seq_len, hidden_dim]
    
    # Step 3: Parallel execution with loop routing
    # Each token goes through its assigned loop in a single forward pass
    x_parallel = parallel_loop_forward(
        x, 
        loop_assignments,
        transformer_block,
        num_loops
    )
    
    # Step 4: Output
    logits = output_projection(x_parallel)
    return logits


def create_loop_schedule(seq_len, num_loops):
    """
    Assign loop index to each token position
    Different tokens compute different loops simultaneously
    """
    loop_schedule = []
    for pos in range(seq_len):
        loop_idx = pos % num_loops
        loop_schedule.append(loop_idx)
    return loop_schedule  # [seq_len]


def parallel_loop_forward(x, loop_assignments, transformer_block, num_loops):
    """
    Execute different loops for different tokens in parallel
    """
    batch_size, seq_len, hidden_dim = x.shape
    
    # Create loop-specific embeddings/features
    loop_embeddings = create_loop_embeddings(num_loops, hidden_dim)
    
    # Add loop information to each token
    for pos in range(seq_len):
        loop_idx = loop_assignments[pos]
        x[:, pos, :] += loop_embeddings[loop_idx]  # Add loop-specific signal
    
    # Single transformer forward pass
    # Attention mechanism naturally handles different loops per token
    x_out = transformer_block(x)  # [batch, seq_len, hidden_dim]
    
    return x_out

# Advantage: Latency = single_block_latency (independent of num_loops!)
```

---

## 3. Gated Sliding-Window Attention (G-SWA)

```python
def gated_sliding_window_attention(
    query,              # [batch, seq_len, hidden_dim]
    key,                # [batch, seq_len, hidden_dim]
    value,              # [batch, seq_len, hidden_dim]
    global_kv_cache,    # Shared KV cache from first loop
    window_size,        # Local attention window size
    num_heads
):
    """
    Combines global context (from shared first-loop KV cache) 
    with local context (sliding window) using learned gates
    """
    batch_size, seq_len, hidden_dim = query.shape
    head_dim = hidden_dim // num_heads
    
    # Reshape for multi-head attention
    Q = rearrange(query, 'b s (h d) -> b h s d', h=num_heads)
    K = rearrange(key, 'b s (h d) -> b h s d', h=num_heads)
    V = rearrange(value, 'b s (h d) -> b h s d', h=num_heads)
    
    # --- Global Attention Component ---
    # Use shared KV cache from first loop (memory efficient)
    K_global, V_global = global_kv_cache  # [batch, num_heads, cache_len, head_dim]
    
    # Compute global attention scores
    scores_global = einsum('bhqd,bhkd->bhqk', Q, K_global) / sqrt(head_dim)
    attn_global = softmax(scores_global, dim=-1)
    global_out = einsum('bhqk,bhkd->bhqd', attn_global, V_global)
    
    # --- Local Sliding-Window Attention ---
    # Apply sliding window mask for local context
    local_mask = create_sliding_window_mask(seq_len, window_size)
    # local_mask[i, j] = 1 if |i - j| <= window_size, else 0
    
    scores_local = einsum('bhqd,bhkd->bhqk', Q, K) / sqrt(head_dim)
    scores_local = scores_local.masked_fill(local_mask == 0, float('-inf'))
    attn_local = softmax(scores_local, dim=-1)
    local_out = einsum('bhqk,bhkd->bhqd', attn_local, V)
    
    # --- Gated Combination ---
    # Learn to balance global vs local information per token
    gate_input = concat([global_out, local_out], dim=-1)
    gate = sigmoid(gate_projection(gate_input))  # [batch, num_heads, seq_len, 1]
    
    # Combine with gating
    output = gate * global_out + (1 - gate) * local_out
    
    # Reshape back
    output = rearrange(output, 'b h s d -> b s (h d)')
    return output


def create_sliding_window_mask(seq_len, window_size):
    """
    Create causal sliding window attention mask
    Token at position i can attend to [i-window_size, i]
    """
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size)
        mask[i, start:i+1] = 1
    return mask  # [seq_len, seq_len]
```

---

## 4. Complete PLT Architecture

```python
class ParallelLoopTransformer:
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_heads,
        num_loops,
        window_size,
        num_layers
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_loops = num_loops
        self.window_size = window_size
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embedding = Embedding(vocab_size, hidden_dim)
        self.loop_embeddings = Parameter(torch.randn(num_loops, hidden_dim))
        
        # Shared transformer blocks (weight reuse across loops)
        self.transformer_blocks = [
            PLTBlock(hidden_dim, num_heads, window_size)
            for _ in range(num_layers)
        ]
        
        # Output projection
        self.output_proj = Linear(hidden_dim, vocab_size)
        
        # Global KV cache (shared across loops)
        self.global_kv_cache = None
    
    def forward(self, input_ids):
        """
        Forward pass with cross-loop parallelism
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, hidden_dim]
        
        # Step 2: Create loop schedule
        loop_schedule = self.create_loop_schedule(seq_len)
        
        # Step 3: Add loop-specific embeddings
        for pos in range(seq_len):
            loop_idx = loop_schedule[pos]
            x[:, pos, :] += self.loop_embeddings[loop_idx]
        
        # Step 4: First pass to generate global KV cache
        with torch.no_grad():
            self.global_kv_cache = self.compute_global_kv_cache(x)
        
        # Step 5: Apply transformer blocks with G-SWA
        for block in self.transformer_blocks:
            x = block(x, self.global_kv_cache, loop_schedule)
        
        # Step 6: Output projection
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def create_loop_schedule(self, seq_len):
        """Assign loop index to each position"""
        return [i % self.num_loops for i in range(seq_len)]
    
    def compute_global_kv_cache(self, x):
        """
        Compute KV cache from first loop (loop 0 tokens)
        This is shared across all loops for memory efficiency
        """
        # Extract tokens assigned to loop 0
        loop_0_mask = (torch.arange(x.size(1)) % self.num_loops) == 0
        x_loop_0 = x[:, loop_0_mask, :]  # [batch, seq_len/num_loops, hidden_dim]
        
        # Compute K, V for global attention
        K_global = self.compute_keys(x_loop_0)
        V_global = self.compute_values(x_loop_0)
        
        return (K_global, V_global)


class PLTBlock:
    """Single transformer block with G-SWA"""
    
    def __init__(self, hidden_dim, num_heads, window_size):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Attention projections
        self.q_proj = Linear(hidden_dim, hidden_dim)
        self.k_proj = Linear(hidden_dim, hidden_dim)
        self.v_proj = Linear(hidden_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)
        
        # Gate projection for G-SWA
        self.gate_proj = Linear(hidden_dim * 2, hidden_dim)
        
        # FFN
        self.ffn = FeedForward(hidden_dim)
        
        # LayerNorms
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
    
    def forward(self, x, global_kv_cache, loop_schedule):
        """
        Forward pass with gated sliding-window attention
        """
        # Pre-norm
        x_norm = self.norm1(x)
        
        # Project to Q, K, V
        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        
        # Apply G-SWA
        attn_out = gated_sliding_window_attention(
            Q, K, V,
            global_kv_cache,
            self.window_size,
            self.num_heads
        )
        
        # Residual connection
        x = x + self.out_proj(attn_out)
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x
```

---

## 5. Training Algorithm

```python
def train_parallel_loop_transformer(
    model,
    train_loader,
    num_epochs,
    learning_rate
):
    """
    Training procedure for PLT
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids = batch['input_ids']  # [batch, seq_len]
            labels = batch['labels']        # [batch, seq_len]
            
            # Forward pass
            logits = model(input_ids)  # [batch, seq_len, vocab_size]
            
            # Compute loss
            # Standard cross-entropy for language modeling
            loss = cross_entropy_loss(
                logits.view(-1, model.vocab_size),
                labels.view(-1)
            )
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Optional: Add auxiliary loss for loop balancing
            # Encourage different loops to contribute equally
            loop_usage_loss = compute_loop_balance_loss(model)
            total_loss = loss + 0.1 * loop_usage_loss
    
    return model


def compute_loop_balance_loss(model):
    """
    Auxiliary loss to encourage balanced usage of different loops
    Prevents model from ignoring certain loops
    """
    # Compute entropy of loop embedding norms
    loop_norms = torch.norm(model.loop_embeddings, dim=-1)  # [num_loops]
    loop_probs = softmax(loop_norms, dim=0)
    
    # Maximize entropy (uniform distribution is best)
    entropy = -torch.sum(loop_probs * torch.log(loop_probs + 1e-10))
    
    # Loss encourages high entropy (balanced usage)
    target_entropy = torch.log(torch.tensor(model.num_loops))
    balance_loss = (target_entropy - entropy) ** 2
    
    return balance_loss
```

---

## 6. Inference with Test-Time Computation Scaling

```python
def inference_with_computation_scaling(
    model,
    input_ids,
    num_loops_inference  # Can be different from training!
):
    """
    Inference with adjustable computation budget
    Key advantage: Can increase num_loops at test time for harder problems
    """
    # Temporarily adjust num_loops
    original_num_loops = model.num_loops
    model.num_loops = num_loops_inference
    
    # If we need more loop embeddings, interpolate from trained ones
    if num_loops_inference > original_num_loops:
        model.loop_embeddings = interpolate_loop_embeddings(
            model.loop_embeddings,
            num_loops_inference
        )
    
    # Run inference
    with torch.no_grad():
        logits = model(input_ids)
        predictions = torch.argmax(logits, dim=-1)
    
    # Restore original configuration
    model.num_loops = original_num_loops
    
    return predictions


def interpolate_loop_embeddings(trained_embeddings, new_num_loops):
    """
    Interpolate loop embeddings when scaling to more loops at test time
    """
    # Simple linear interpolation
    old_num_loops, hidden_dim = trained_embeddings.shape
    
    new_embeddings = torch.zeros(new_num_loops, hidden_dim)
    
    for i in range(new_num_loops):
        # Map new index to old index space
        old_idx_float = i * (old_num_loops - 1) / (new_num_loops - 1)
        
        # Interpolate between floor and ceil
        idx_floor = int(old_idx_float)
        idx_ceil = min(idx_floor + 1, old_num_loops - 1)
        weight = old_idx_float - idx_floor
        
        new_embeddings[i] = (
            (1 - weight) * trained_embeddings[idx_floor] +
            weight * trained_embeddings[idx_ceil]
        )
    
    return new_embeddings
```

---

## 7. Memory and Latency Analysis

```python
def complexity_analysis():
    """
    Comparison of PLT vs Standard Looped Transformer
    """
    
    # Standard Sequential Looped Transformer:
    # ----------------------------------------
    # Latency: O(num_loops × seq_len × hidden_dim²)
    # KV Cache Memory: O(num_loops × seq_len × hidden_dim)
    # Problem: Linear scaling with num_loops
    
    standard_latency = lambda L, S, D: L * S * D**2
    standard_memory = lambda L, S, D: L * S * D
    
    # Parallel Loop Transformer (PLT):
    # --------------------------------
    # Latency: O(seq_len × hidden_dim²)  [independent of num_loops!]
    # KV Cache Memory: O(seq_len × hidden_dim)  [shared cache]
    # Advantage: Constant latency and memory regardless of num_loops
    
    plt_latency = lambda L, S, D: S * D**2  # L doesn't matter!
    plt_memory = lambda L, S, D: S * D      # L doesn't matter!
    
    # Example: num_loops=4, seq_len=2048, hidden_dim=4096
    L, S, D = 4, 2048, 4096
    
    print(f"Standard Latency: {standard_latency(L, S, D) / 1e9:.2f}B ops")
    print(f"PLT Latency: {plt_latency(L, S, D) / 1e9:.2f}B ops")
    print(f"Speedup: {standard_latency(L, S, D) / plt_latency(L, S, D):.1f}x")
    
    print(f"\nStandard Memory: {standard_memory(L, S, D) / 1e6:.1f}M params")
    print(f"PLT Memory: {plt_memory(L, S, D) / 1e6:.1f}M params")
    print(f"Memory Reduction: {standard_memory(L, S, D) / plt_memory(L, S, D):.1f}x")

# Output:
# Standard Latency: 137.44B ops
# PLT Latency: 34.36B ops
# Speedup: 4.0x
#
# Standard Memory: 33.6M params
# PLT Memory: 8.4M params
# Memory Reduction: 4.0x
```

---

## 8. Key Techniques Summary

```python
# 1. Cross-Loop Parallelism (CLP)
# --------------------------------
# Break sequential dependency by assigning different loops to different tokens
loop_schedule[position] = position % num_loops

# 2. Shared Global KV Cache
# --------------------------
# All loops share KV cache from loop 0 tokens to save memory
global_kv = compute_kv_cache(x[loop_schedule == 0])

# 3. Gated Sliding-Window Attention (G-SWA)
# ------------------------------------------
# Combine global context (from shared cache) with local context (sliding window)
output = gate * global_attention(Q, global_kv) + (1 - gate) * local_attention(Q, K, V, window_mask)

# 4. Loop Embeddings
# -------------------
# Add learnable embeddings to distinguish which loop each token is in
x[pos] += loop_embeddings[loop_schedule[pos]]

# 5. Test-Time Scaling
# ---------------------
# Adjust num_loops at inference for computation scaling
# More loops = more computation = better accuracy (for hard problems)
predictions = model(input_ids, num_loops=inference_loops)
```

---

## 9. Practical Implementation Tips

```python
def implementation_tips():
    """
    Tips for implementing PLT in practice
    """
    
    # Tip 1: Start with small num_loops during development
    # ----------------------------------------------------
    num_loops = 2  # Easier to debug than num_loops=8
    
    # Tip 2: Use efficient attention implementation
    # ----------------------------------------------
    # Use FlashAttention or similar for both global and local attention
    from flash_attn import flash_attn_func
    
    # Tip 3: Careful with loop embedding initialization
    # --------------------------------------------------
    # Initialize loop embeddings with small values
    loop_embeddings = torch.randn(num_loops, hidden_dim) * 0.01
    
    # Tip 4: Monitor loop usage during training
    # ------------------------------------------
    # Log which loops are being used most frequently
    def log_loop_usage(loop_schedule):
        unique, counts = torch.unique(loop_schedule, return_counts=True)
        print(f"Loop usage: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    # Tip 5: Experiment with window size
    # -----------------------------------
    # Smaller windows = less computation, but may hurt quality
    # Typical values: window_size in [128, 512, 1024]
    window_size = 256
    
    # Tip 6: Gradient checkpointing for memory efficiency
    # ---------------------------------------------------
    # Use gradient checkpointing on transformer blocks
    from torch.utils.checkpoint import checkpoint
    x = checkpoint(transformer_block, x)
    
    # Tip 7: Test-time loop scaling strategy
    # ---------------------------------------
    # Use fewer loops for easy prompts, more loops for hard ones
    def adaptive_num_loops(prompt_difficulty):
        if prompt_difficulty == "easy":
            return 2
        elif prompt_difficulty == "medium":
            return 4
        else:  # hard
            return 8
```

---

## Summary

**Parallel Loop Transformer (PLT)** achieves test-time computation scaling without the latency/memory penalties of sequential looping:

1. **Cross-Loop Parallelism**: Different tokens compute different loops simultaneously
2. **Shared KV Cache**: Memory efficient by sharing first loop's cache
3. **Gated Sliding-Window Attention**: Balances global and local context
4. **Benefits**: O(1) latency and memory w.r.t. num_loops instead of O(num_loops)

This enables flexible test-time computation scaling: use more loops for harder problems while maintaining fast inference for easier ones.
