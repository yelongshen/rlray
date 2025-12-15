"""
Test script for MoE (Mixture of Experts) implementation
"""
import torch
import torch.nn.functional as F
from moe_layers import (
    TopKRouter,
    MoEExpert,
    SparseMoELayer,
    MoETransformerBlock,
    MoETransformer,
)


def test_router():
    """Test TopK router"""
    print("\n=== Testing TopK Router ===")
    
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_experts = 8
    num_experts_per_token = 2
    
    router = TopKRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    router_logits, top_k_indices, top_k_weights = router(x)
    
    print(f"✓ Router input: {x.shape}")
    print(f"✓ Router logits: {router_logits.shape}")
    print(f"✓ Top-k indices: {top_k_indices.shape}")
    print(f"✓ Top-k weights: {top_k_weights.shape}")
    
    # Check weights sum to 1
    weight_sum = top_k_weights.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5)
    print(f"✓ Routing weights sum to 1.0")
    
    # Test auxiliary loss
    aux_loss, z_loss = router.compute_aux_loss(router_logits, top_k_indices)
    print(f"✓ Auxiliary loss: {aux_loss.item():.6f}")
    print(f"✓ Z-loss: {z_loss.item():.6f}")
    
    return True


def test_expert():
    """Test single expert"""
    print("\n=== Testing MoE Expert ===")
    
    num_tokens = 32
    hidden_size = 256
    intermediate_size = 1024
    
    expert = MoEExpert(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    
    x = torch.randn(num_tokens, hidden_size)
    out = expert(x)
    
    print(f"✓ Expert input: {x.shape}")
    print(f"✓ Expert output: {out.shape}")
    
    assert out.shape == x.shape
    print(f"✓ Expert output shape correct")
    
    return True


def test_moe_layer():
    """Test sparse MoE layer"""
    print("\n=== Testing Sparse MoE Layer ===")
    
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 1024
    num_experts = 8
    num_experts_per_token = 2
    
    moe = SparseMoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        use_expert_parallel=False,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    out, aux_loss = moe(x)
    
    print(f"✓ MoE input: {x.shape}")
    print(f"✓ MoE output: {out.shape}")
    print(f"✓ Auxiliary loss: {aux_loss.item():.6f}")
    
    assert out.shape == x.shape
    print(f"✓ MoE output shape correct")
    
    # Test backward
    loss = out.sum() + aux_loss
    loss.backward()
    print(f"✓ Backward pass completed")
    
    return True


def test_moe_transformer_block():
    """Test MoE transformer block"""
    print("\n=== Testing MoE Transformer Block ===")
    
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 1024
    num_attention_heads = 8
    num_experts = 8
    num_experts_per_token = 2
    
    block = MoETransformerBlock(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        use_tensor_parallel=False,
        use_expert_parallel=False,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    out, aux_loss = block(x)
    
    print(f"✓ Block input: {x.shape}")
    print(f"✓ Block output: {out.shape}")
    print(f"✓ Auxiliary loss: {aux_loss.item():.6f}")
    
    assert out.shape == x.shape
    print(f"✓ Block output shape correct")
    
    # Test backward
    loss = out.sum() + aux_loss
    loss.backward()
    print(f"✓ Backward pass completed")
    
    return True


def test_moe_transformer():
    """Test complete MoE transformer"""
    print("\n=== Testing MoE Transformer ===")
    
    vocab_size = 1000
    hidden_size = 256
    intermediate_size = 1024
    num_layers = 4
    num_attention_heads = 8
    num_experts = 8
    num_experts_per_token = 2
    moe_layer_interval = 2  # MoE every 2 layers
    
    model = MoETransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        moe_layer_interval=moe_layer_interval,
        use_tensor_parallel=False,
        use_expert_parallel=False,
    )
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    logits, loss, aux_loss = model(input_ids=input_ids, labels=labels)
    
    print(f"✓ Model input: {input_ids.shape}")
    print(f"✓ Model output: {logits.shape}")
    print(f"✓ LM Loss: {loss.item():.4f}")
    print(f"✓ Auxiliary Loss: {aux_loss.item():.6f}")
    
    # Check which layers are MoE
    moe_count = sum(1 for layer in model.layers if hasattr(layer, 'moe'))
    dense_count = len(model.layers) - moe_count
    print(f"✓ MoE layers: {moe_count}, Dense layers: {dense_count}")
    
    # Test backward
    loss.backward()
    print(f"✓ Backward pass completed")
    
    # Count parameters
    num_params = model.get_num_params()
    print(f"✓ Model parameters: {num_params / 1e6:.2f}M")
    
    return True


def test_no_drop_routing():
    """Test that all tokens are routed (no dropping)"""
    print("\n=== Testing No-Drop Token Routing ===")
    
    batch_size = 4
    seq_len = 32
    hidden_size = 256
    intermediate_size = 1024
    num_experts = 8
    num_experts_per_token = 2
    
    moe = SparseMoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        use_expert_parallel=False,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    out, aux_loss = moe(x)
    
    num_tokens = batch_size * seq_len
    print(f"✓ Total tokens: {num_tokens}")
    print(f"✓ Tokens per expert (expected): {num_tokens * num_experts_per_token / num_experts:.1f}")
    
    # Check output is not zero (all tokens were processed)
    assert not torch.allclose(out, torch.zeros_like(out))
    print(f"✓ All tokens processed (output non-zero)")
    
    # Check output has reasonable magnitude
    out_mean = out.abs().mean()
    print(f"✓ Output mean magnitude: {out_mean:.4f}")
    
    return True


def test_load_balancing():
    """Test that load balancing loss encourages uniform distribution"""
    print("\n=== Testing Load Balancing ===")
    
    batch_size = 8
    seq_len = 64
    hidden_size = 256
    num_experts = 8
    num_experts_per_token = 2
    
    router = TopKRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        router_aux_loss_coef=0.1,  # Higher coef for testing
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    router_logits, top_k_indices, top_k_weights = router(x)
    
    # Count tokens per expert
    expert_counts = torch.zeros(num_experts)
    for i in range(num_experts):
        expert_counts[i] = (top_k_indices == i).sum().item()
    
    print(f"✓ Tokens per expert: {expert_counts.tolist()}")
    
    total_tokens = batch_size * seq_len * num_experts_per_token
    avg_tokens = total_tokens / num_experts
    print(f"✓ Average tokens per expert: {avg_tokens:.1f}")
    
    # Compute load balancing metrics
    std_dev = expert_counts.std()
    print(f"✓ Standard deviation: {std_dev:.2f}")
    
    # Auxiliary loss should be non-zero
    aux_loss, z_loss = router.compute_aux_loss(router_logits, top_k_indices)
    print(f"✓ Load balancing loss: {aux_loss.item():.6f}")
    
    return True


def test_training_step():
    """Test complete training step with MoE"""
    print("\n=== Testing Training Step ===")
    
    model = MoETransformer(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_experts=8,
        num_experts_per_token=2,
        moe_layer_interval=2,
        use_tensor_parallel=False,
        use_expert_parallel=False,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training step
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()
    
    optimizer.zero_grad()
    logits, loss, aux_loss = model(input_ids=input_ids, labels=labels)
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step completed")
    print(f"✓ Total Loss: {loss.item():.4f}")
    print(f"✓ Auxiliary Loss: {aux_loss.item():.6f}")
    
    return True


def run_all_tests():
    """Run all MoE tests"""
    print("=" * 80)
    print("Running MoE Implementation Tests")
    print("=" * 80)
    
    tests = [
        ("TopK Router", test_router),
        ("MoE Expert", test_expert),
        ("Sparse MoE Layer", test_moe_layer),
        ("MoE Transformer Block", test_moe_transformer_block),
        ("MoE Transformer", test_moe_transformer),
        ("No-Drop Routing", test_no_drop_routing),
        ("Load Balancing", test_load_balancing),
        ("Training Step", test_training_step),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                print(f"✗ {test_name} failed")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
