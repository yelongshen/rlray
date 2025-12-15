# Mixture of Experts (MoE) Implementation Guide

## Overview

This implementation adds **Mixture of Experts (MoE)** support to the minimal pretraining codebase with:

✅ **Expert Parallel (EP)**: Distribute experts across GPUs  
✅ **No-Drop Tokens**: All tokens are routed to top-k experts (no capacity dropping)  
✅ **Load Balancing**: Auxiliary loss to encourage uniform expert usage  
✅ **Flexible Architecture**: Mix MoE and dense layers  
✅ **Compatible**: Works with DP, TP, and PP  

## Key Features

### 1. Expert Parallel (EP)
- Distribute experts across multiple GPUs
- Each GPU holds a subset of experts
- Tokens routed across GPUs via AllReduce
- Reduces per-GPU memory for models with many experts

### 2. No-Drop Token Routing
- **All tokens are processed** by their top-k experts
- No capacity limits or token dropping
- Ensures consistent training signal
- Better quality than capacity-based routing

### 3. Load Balancing
- Auxiliary loss encourages uniform expert usage
- Prevents expert collapse (all tokens to one expert)
- Two loss components:
  - **Aux Loss**: Balances token distribution
  - **Z Loss**: Stabilizes router logits

## Architecture

### MoE Layer Structure

```
Input [batch, seq_len, hidden]
    ↓
Router (Top-K Selection)
    ↓
┌──────────────────────────────────┐
│  Expert 0  │  Expert 1  │  ...   │
│  (FFN)     │  (FFN)     │  (FFN) │
└──────────────────────────────────┘
    ↓
Weighted Combination
    ↓
Output [batch, seq_len, hidden]
```

### Expert Parallel Distribution

```
Example: 8 Experts, 4 GPUs (EP=4)

GPU 0: Expert 0, 1
GPU 1: Expert 2, 3
GPU 2: Expert 4, 5
GPU 3: Expert 6, 7

Token routing:
- Each token selects top-k experts (e.g., k=2)
- Token processed on GPU(s) with selected experts
- Results AllReduced across EP group
```

## Usage

### Basic MoE Training

```bash
# Single GPU with MoE
python minimal_pretrain.py \
    --num-experts 8 \
    --num-experts-per-token 2 \
    --moe-layer-interval 2 \
    --max-steps 1000
```

### Multi-GPU with Expert Parallel

```bash
# 8 GPUs: 4 DP × 2 EP
torchrun --nproc_per_node=8 minimal_pretrain.py \
    --num-experts 8 \
    --num-experts-per-token 2 \
    --moe-layer-interval 2 \
    --tensor-parallel-size 2 \
    --max-steps 1000
```

**Note**: Expert parallel size automatically matches tensor parallel size.

### Combined Parallelism

```bash
# 16 GPUs: 2 DP × 2 TP/EP × 4 PP
torchrun --nproc_per_node=16 minimal_pretrain.py \
    --num-experts 16 \
    --num-experts-per-token 2 \
    --moe-layer-interval 2 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 4 \
    --num-microbatches 16 \
    --max-steps 10000
```

## Configuration Arguments

### MoE Arguments

```bash
--num-experts               # Number of experts (0 = dense model)
--num-experts-per-token     # Top-k experts per token (default: 2)
--moe-layer-interval        # MoE every N layers (default: 2)
--router-aux-loss-coef      # Load balancing loss weight (default: 0.01)
--router-z-loss-coef        # Router stability loss weight (default: 0.001)
--expert-parallel-size      # EP size (auto-set to match TP)
```

### Example Configurations

#### Small MoE (Testing)
```bash
--num-experts 4 \
--num-experts-per-token 2 \
--hidden-size 512 \
--num-layers 8
```

#### Medium MoE (Development)
```bash
--num-experts 8 \
--num-experts-per-token 2 \
--hidden-size 1024 \
--num-layers 24 \
--moe-layer-interval 2
```

#### Large MoE (Production)
```bash
--num-experts 16 \
--num-experts-per-token 2 \
--hidden-size 2048 \
--num-layers 48 \
--moe-layer-interval 2 \
--tensor-parallel-size 4
```

## MoE Layer Patterns

### Pattern 1: All MoE Layers
```bash
--moe-layer-interval 1  # Every layer is MoE
```

### Pattern 2: Alternating (Recommended)
```bash
--moe-layer-interval 2  # Every 2nd layer is MoE
```
```
Layer 0: Dense
Layer 1: Dense
Layer 2: MoE    ← Every 2nd
Layer 3: Dense
Layer 4: MoE    ← Every 2nd
...
```

### Pattern 3: Sparse MoE
```bash
--moe-layer-interval 4  # Every 4th layer is MoE
```

### Pattern 4: Dense Model
```bash
--num-experts 0  # No MoE, pure dense model
```

## Load Balancing

### Understanding Auxiliary Loss

The router adds two auxiliary losses:

1. **Load Balancing Loss** (aux_loss):
   - Encourages uniform token distribution across experts
   - Formula: `num_experts * sum(f_i * P_i)`
   - Where `f_i` = fraction of tokens to expert i
   - And `P_i` = average router probability for expert i

2. **Router Z-Loss** (z_loss):
   - Prevents router logits from growing too large
   - Formula: `log(sum(exp(logits)))`
   - Improves training stability

### Tuning Load Balancing

```bash
# Stronger load balancing (more uniform, possibly lower quality)
--router-aux-loss-coef 0.1

# Weaker load balancing (less uniform, possibly higher quality)
--router-aux-loss-coef 0.001

# Default (recommended)
--router-aux-loss-coef 0.01
```

### Monitoring Load Balance

The training logs show auxiliary loss:
```
Step 100 | Loss: 3.2415 | Aux Loss: 0.0234 | LR: 3.00e-04
```

Good auxiliary loss: 0.01 - 0.05 (depends on coefficient)

## Expert Parallel Strategies

### Strategy 1: No Expert Parallel (Small Models)
```bash
# All experts on each GPU
torchrun --nproc_per_node=4 minimal_pretrain.py \
    --num-experts 8
```

Result: 4× data parallel, each GPU has all 8 experts

### Strategy 2: Expert Parallel (Large Models)
```bash
# Experts distributed across GPUs
torchrun --nproc_per_node=8 minimal_pretrain.py \
    --num-experts 16 \
    --tensor-parallel-size 2
```

Result: 4× data parallel, 2× expert parallel  
- Each GPU pair has all 16 experts (8 per GPU)

### Strategy 3: Combined DP + EP + PP
```bash
# Maximum scaling
torchrun --nproc_per_node=16 minimal_pretrain.py \
    --num-experts 32 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

Result: 2× data parallel, 4× expert parallel, 2× pipeline parallel

## Performance Tips

### Memory Optimization

1. **Expert Parallel**: Reduce memory per GPU
   ```bash
   --tensor-parallel-size 4  # 4-way EP
   --num-experts 32          # 8 experts per GPU
   ```

2. **Fewer Experts Per Token**: Less computation
   ```bash
   --num-experts-per-token 1  # Top-1 routing
   ```

3. **Sparse MoE**: Mix dense and MoE layers
   ```bash
   --moe-layer-interval 4  # 25% MoE layers
   ```

### Quality Optimization

1. **More Experts Per Token**: Better quality
   ```bash
   --num-experts-per-token 2  # Top-2 routing (recommended)
   --num-experts-per-token 3  # Top-3 routing (higher quality)
   ```

2. **More Experts**: Larger capacity
   ```bash
   --num-experts 16  # More experts = more capacity
   ```

3. **Balanced Load**: Tune auxiliary loss
   ```bash
   --router-aux-loss-coef 0.01  # Standard
   ```

## Testing

### Run MoE Tests
```bash
python test_moe.py
```

Expected output:
```
✓ TopK Router
✓ MoE Expert
✓ Sparse MoE Layer
✓ MoE Transformer Block
✓ MoE Transformer
✓ No-Drop Routing
✓ Load Balancing
✓ Training Step
```

### Quick Validation
```bash
# Test single GPU
python minimal_pretrain.py \
    --num-experts 4 \
    --max-steps 10 \
    --log-interval 5

# Test multi-GPU
torchrun --nproc_per_node=2 minimal_pretrain.py \
    --num-experts 4 \
    --tensor-parallel-size 2 \
    --max-steps 10 \
    --log-interval 5
```

## Model Size Examples

### Small MoE (~200M params)
```bash
--vocab-size 32000 \
--hidden-size 768 \
--num-layers 12 \
--num-attention-heads 12 \
--num-experts 8 \
--num-experts-per-token 2 \
--moe-layer-interval 2
```

Active params per token: ~125M (dense) + ~25M (2 experts)

### Medium MoE (~1B params)
```bash
--vocab-size 50257 \
--hidden-size 1280 \
--num-layers 32 \
--num-attention-heads 20 \
--num-experts 16 \
--num-experts-per-token 2 \
--moe-layer-interval 2
```

Active params per token: ~700M (dense) + ~100M (2 experts)

### Large MoE (~8B params)
```bash
--vocab-size 50257 \
--hidden-size 2560 \
--num-layers 48 \
--num-attention-heads 40 \
--num-experts 32 \
--num-experts-per-token 2 \
--moe-layer-interval 2 \
--tensor-parallel-size 8
```

Active params per token: ~5B (dense) + ~600M (2 experts)

## Comparison: Dense vs MoE

| Metric | Dense | MoE (8 experts, top-2) |
|--------|-------|------------------------|
| Total Params | 1B | 3B |
| Active Params | 1B | 1.2B |
| Memory per GPU | 4GB | 1.5GB (with EP=4) |
| Compute | 100% | 25% per expert |
| Capacity | Fixed | 4× larger |

## Common Issues

### Issue 1: Load Imbalance

**Symptom**: Some experts never used
```
Tokens per expert: [0, 5, 120, 8, 0, 2, 95, 0]
```

**Solution**: Increase auxiliary loss coefficient
```bash
--router-aux-loss-coef 0.05  # Higher = more balancing
```

### Issue 2: High Auxiliary Loss

**Symptom**: Auxiliary loss dominates total loss
```
Step 100 | Loss: 0.8234 | Aux Loss: 1.2345
```

**Solution**: Decrease auxiliary loss coefficient
```bash
--router-aux-loss-coef 0.001  # Lower = less balancing
```

### Issue 3: OOM with Many Experts

**Symptom**: Out of memory error

**Solution**: Use expert parallel
```bash
--tensor-parallel-size 4  # 4-way EP
--num-experts 32          # 8 experts per GPU
```

### Issue 4: Slow Training

**Symptom**: Training much slower than expected

**Solution**: Check expert parallel setup
```bash
# Make sure EP size matches TP size
--tensor-parallel-size 2  # Automatically sets EP=2
```

## Advanced: Custom Routing

The router can be customized in `moe_layers.py`:

```python
class CustomRouter(TopKRouter):
    def forward(self, hidden_states):
        # Custom routing logic
        router_logits = self.router(hidden_states)
        
        # Example: Add noise for exploration
        noise = torch.randn_like(router_logits) * 0.1
        router_logits = router_logits + noise
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(...)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        return router_logits, top_k_indices, top_k_weights
```

## References

1. **Switch Transformers** (Fedus et al., 2021): No-drop token routing
2. **GLaM** (Du et al., 2021): Expert parallel
3. **ST-MoE** (Zoph et al., 2022): Load balancing techniques
4. **Mixtral** (Mistral AI, 2023): Modern MoE architecture

## Summary

✅ **Expert Parallel**: Scale to many experts  
✅ **No-Drop Tokens**: All tokens processed  
✅ **Load Balancing**: Uniform expert usage  
✅ **Flexible**: Mix MoE and dense layers  
✅ **Compatible**: Works with DP, TP, PP  

Start with:
```bash
python test_moe.py  # Verify installation
python minimal_pretrain.py --num-experts 8 --max-steps 100  # Quick test
torchrun --nproc_per_node=4 minimal_pretrain.py --num-experts 8  # Scale up
```
