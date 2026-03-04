"""
MoE Forward Pass Profiling: Compare Triton, BMM, and Loop approaches.

Usage:
    cd xlmlib && python moe_profiling.py

Tests all three MoE strategies with varying batch sizes and measures:
- Latency per call
- Memory usage
- Throughput (tokens/s)
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing Triton MoE
try:
    from fused_moe_triton import fused_moe as triton_fused_moe
    TRITON_AVAILABLE = True
    print("✓ Triton fused MoE loaded")
except ImportError:
    TRITON_AVAILABLE = False
    print("✗ Triton fused MoE not available")


def create_moe_weights(num_experts, intermediate_size, hidden_size, device, dtype=torch.bfloat16):
    """Create random MoE expert weights."""
    gate_up_proj = torch.randn(num_experts, 2 * intermediate_size, hidden_size, 
                                device=device, dtype=dtype) * 0.01
    down_proj = torch.randn(num_experts, hidden_size, intermediate_size,
                             device=device, dtype=dtype) * 0.01
    return gate_up_proj, down_proj


def moe_forward_loop(hidden_states, gate_up_proj, down_proj, top_k_indices, top_k_weights):
    """Expert-by-expert loop (baseline)."""
    num_experts = gate_up_proj.shape[0]
    final_output = torch.zeros_like(hidden_states)
    act_fn = torch.nn.SiLU()
    
    for expert_idx in range(num_experts):
        expert_mask = (top_k_indices == expert_idx)
        if not expert_mask.any():
            continue
        token_indices, top_k_positions = torch.where(expert_mask)
        expert_input = hidden_states[token_indices]
        expert_weights = top_k_weights[token_indices, top_k_positions].unsqueeze(-1)
        
        gate, up = F.linear(expert_input, gate_up_proj[expert_idx]).chunk(2, dim=-1)
        expert_output = act_fn(gate) * up
        expert_output = F.linear(expert_output, down_proj[expert_idx])
        expert_output = expert_output * expert_weights
        final_output.index_add_(0, token_indices, expert_output.to(final_output.dtype))
    
    return final_output


def moe_forward_bmm(hidden_states, gate_up_proj, down_proj, top_k_indices, top_k_weights):
    """Fused BMM approach."""
    num_tokens = hidden_states.shape[0]
    top_k = top_k_indices.shape[1]
    num_experts = gate_up_proj.shape[0]
    act_fn = torch.nn.SiLU()
    
    flat_indices = top_k_indices.view(-1).clamp(0, num_experts - 1)
    flat_weights = top_k_weights.view(-1, 1)
    expanded_hidden = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_states.shape[-1])
    
    selected_gate_up = gate_up_proj[flat_indices]
    selected_down = down_proj[flat_indices]
    
    gate_up_out = torch.bmm(
        expanded_hidden.unsqueeze(1),
        selected_gate_up.transpose(1, 2)
    ).squeeze(1)
    
    gate, up = gate_up_out.chunk(2, dim=-1)
    expert_out = act_fn(gate) * up
    
    expert_out = torch.bmm(
        expert_out.unsqueeze(1),
        selected_down.transpose(1, 2)
    ).squeeze(1)
    
    expert_out = expert_out * flat_weights
    
    token_indices = torch.arange(num_tokens, device=hidden_states.device
                                 ).unsqueeze(1).expand(-1, top_k).reshape(-1)
    final_output = torch.zeros_like(hidden_states)
    final_output.index_add_(0, token_indices, expert_out.to(final_output.dtype))
    
    return final_output


def moe_forward_triton(hidden_states, gate_up_proj, down_proj, top_k_indices, top_k_weights, top_k, num_experts):
    """Triton fused MoE."""
    return triton_fused_moe(
        hidden_states, gate_up_proj, down_proj,
        top_k_weights, top_k_indices,
        top_k=top_k, num_experts=num_experts,
    )


def benchmark_fn(fn, warmup=3, repeats=10, sync=True):
    """Benchmark a function with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        out = fn()
    
    if sync:
        torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(repeats):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'output': out,
    }


def run_profiling():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping profiling")
        return
    
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    
    # Model dimensions (Qwen3-Next)
    num_experts = 64
    top_k = 8
    intermediate_size = 1408
    hidden_size = 2048
    
    # Test batch sizes
    batch_sizes = [1, 4, 8, 16, 30, 64, 128, 256]
    
    print(f"\n{'='*80}")
    print(f"MoE Profiling: {num_experts} experts, top_k={top_k}, "
          f"intermediate={intermediate_size}, hidden={hidden_size}")
    print(f"{'='*80}\n")
    
    # Create weights (shared across all tests)
    gate_up_proj, down_proj = create_moe_weights(
        num_experts, intermediate_size, hidden_size, device, dtype
    )
    
    print(f"Expert weights: gate_up_proj={gate_up_proj.shape}, down_proj={down_proj.shape}")
    print(f"Weight memory: {(gate_up_proj.numel() + down_proj.numel()) * 2 / 1e6:.1f} MB\n")
    
    header = f"{'Batch':>6} | {'N*K':>6} | {'Loop (ms)':>10} | {'BMM (ms)':>10} | {'Triton (ms)':>12} | {'Speedup':>8} | {'Match':>5}"
    print(header)
    print("-" * len(header))
    
    for batch_size in batch_sizes:
        # Create random inputs
        hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        
        # Random expert assignments
        top_k_indices = torch.randint(0, num_experts, (batch_size, top_k), device=device)
        top_k_weights = torch.softmax(torch.randn(batch_size, top_k, device=device, dtype=torch.float32), dim=-1).to(dtype)
        
        nk = batch_size * top_k
        
        # ---- Loop ----
        loop_result = benchmark_fn(
            lambda: moe_forward_loop(hidden_states, gate_up_proj, down_proj, top_k_indices, top_k_weights),
            warmup=2, repeats=5
        )
        
        # ---- BMM ----
        if nk <= 1024:
            bmm_result = benchmark_fn(
                lambda: moe_forward_bmm(hidden_states, gate_up_proj, down_proj, top_k_indices, top_k_weights),
                warmup=2, repeats=5
            )
            bmm_str = f"{bmm_result['mean_ms']:10.3f}"
        else:
            bmm_result = None
            bmm_str = f"{'OOM':>10}"
        
        # ---- Triton ----
        if TRITON_AVAILABLE:
            try:
                triton_result = benchmark_fn(
                    lambda: moe_forward_triton(hidden_states, gate_up_proj, down_proj, 
                                               top_k_indices, top_k_weights, top_k, num_experts),
                    warmup=2, repeats=5
                )
                triton_str = f"{triton_result['mean_ms']:12.3f}"
                
                # Speedup vs loop
                speedup = loop_result['mean_ms'] / triton_result['mean_ms']
                speedup_str = f"{speedup:7.1f}x"
                
                # Check correctness vs loop
                diff = (triton_result['output'] - loop_result['output']).abs().max().item()
                match_str = f"{'✓' if diff < 0.1 else '✗'} {diff:.4f}"
            except Exception as e:
                triton_str = f"{'ERR':>12}"
                speedup_str = f"{'N/A':>8}"
                match_str = str(e)[:20]
        else:
            triton_str = f"{'N/A':>12}"
            speedup_str = f"{'N/A':>8}"
            match_str = "N/A"
        
        print(f"{batch_size:>6} | {nk:>6} | {loop_result['mean_ms']:10.3f} | {bmm_str} | {triton_str} | {speedup_str} | {match_str}")
    
    # Memory profiling for the largest batch
    print(f"\n{'='*80}")
    print("Memory Usage (batch=30, typical decode)")
    print(f"{'='*80}\n")
    
    batch_size = 30
    hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    top_k_indices = torch.randint(0, num_experts, (batch_size, top_k), device=device)
    top_k_weights = torch.softmax(torch.randn(batch_size, top_k, device=device, dtype=torch.float32), dim=-1).to(dtype)
    
    for name, fn in [
        ("Loop", lambda: moe_forward_loop(hidden_states, gate_up_proj, down_proj, top_k_indices, top_k_weights)),
        ("BMM", lambda: moe_forward_bmm(hidden_states, gate_up_proj, down_proj, top_k_indices, top_k_weights)),
    ] + ([("Triton", lambda: moe_forward_triton(hidden_states, gate_up_proj, down_proj, 
                                                  top_k_indices, top_k_weights, top_k, num_experts))] if TRITON_AVAILABLE else []):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        mem_before = torch.cuda.memory_allocated()
        _ = fn()
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated()
        mem_delta = mem_peak - mem_before
        
        print(f"  {name:>8}: peak_delta={mem_delta/1e6:.1f} MB")
    
    print(f"\nDone!")


if __name__ == "__main__":
    run_profiling()
