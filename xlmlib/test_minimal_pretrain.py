"""
Test script for minimal pretraining implementation
Verifies model creation, forward/backward pass, and basic parallel setup
"""
import torch
import torch.distributed as dist
from minimal_model import (
    MinimalTransformer,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


def test_model_creation():
    """Test basic model creation and forward pass"""
    print("\n=== Testing Model Creation ===")
    
    model = MinimalTransformer(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=128,
        use_tensor_parallel=False,
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()
    
    logits, loss = model(input_ids=input_ids, labels=labels)
    
    print(f"✓ Model created successfully")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Model params: {model.get_num_params() / 1e6:.2f}M")
    
    # Test backward
    loss.backward()
    print(f"✓ Backward pass completed")
    
    return True


def test_tensor_parallel_layers():
    """Test tensor parallel layers (without distributed init)"""
    print("\n=== Testing Tensor Parallel Layers (Single GPU) ===")
    
    # Note: These will work in single-GPU mode by falling back to regular layers
    hidden_size = 256
    intermediate_size = 1024
    batch_size = 2
    seq_len = 16
    
    # Test ColumnParallelLinear
    col_linear = ColumnParallelLinear(
        input_size=hidden_size,
        output_size=intermediate_size,
        bias=True,
        gather_output=True,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    y = col_linear(x)
    print(f"✓ ColumnParallelLinear: {x.shape} -> {y.shape}")
    
    # Test RowParallelLinear
    row_linear = RowParallelLinear(
        input_size=intermediate_size,
        output_size=hidden_size,
        bias=True,
        input_is_parallel=False,
    )
    
    y2 = row_linear(y)
    print(f"✓ RowParallelLinear: {y.shape} -> {y2.shape}")
    
    # Test VocabParallelEmbedding
    vocab_size = 1000
    emb = VocabParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    emb_out = emb(input_ids)
    print(f"✓ VocabParallelEmbedding: {input_ids.shape} -> {emb_out.shape}")
    
    return True


def test_attention():
    """Test attention mechanism"""
    print("\n=== Testing Attention ===")
    
    from minimal_model import Attention
    
    batch_size = 2
    seq_len = 32
    hidden_size = 256
    num_heads = 8
    
    attn = Attention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        use_tensor_parallel=False,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    out = attn(x)
    
    print(f"✓ Attention: {x.shape} -> {out.shape}")
    
    # Test with GQA
    attn_gqa = Attention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=2,  # 2 KV heads for GQA
        use_tensor_parallel=False,
    )
    
    out_gqa = attn_gqa(x)
    print(f"✓ Grouped Query Attention: {x.shape} -> {out_gqa.shape}")
    
    return True


def test_mlp():
    """Test MLP layer"""
    print("\n=== Testing MLP ===")
    
    from minimal_model import MLP
    
    batch_size = 2
    seq_len = 32
    hidden_size = 256
    intermediate_size = 1024
    
    mlp = MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        use_tensor_parallel=False,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    out = mlp(x)
    
    print(f"✓ MLP: {x.shape} -> {out.shape}")
    
    return True


def test_transformer_block():
    """Test transformer block"""
    print("\n=== Testing Transformer Block ===")
    
    from minimal_model import TransformerBlock
    
    batch_size = 2
    seq_len = 32
    hidden_size = 256
    intermediate_size = 1024
    num_heads = 8
    
    block = TransformerBlock(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        use_tensor_parallel=False,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    out = block(x)
    
    print(f"✓ Transformer Block: {x.shape} -> {out.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in block.parameters())
    print(f"  Block params: {num_params / 1e6:.2f}M")
    
    return True


def test_training_step():
    """Test a complete training step"""
    print("\n=== Testing Training Step ===")
    
    model = MinimalTransformer(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=128,
        use_tensor_parallel=False,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training step
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()
    
    optimizer.zero_grad()
    logits, loss = model(input_ids=input_ids, labels=labels)
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step completed")
    print(f"  Loss: {loss.item():.4f}")
    
    return True


def test_checkpointing():
    """Test model save and load"""
    print("\n=== Testing Checkpointing ===")
    
    model = MinimalTransformer(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        use_tensor_parallel=False,
    )
    
    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': 1000,
            'hidden_size': 256,
            'num_layers': 4,
        }
    }
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(checkpoint, f.name)
        print(f"✓ Checkpoint saved to {f.name}")
        
        # Load checkpoint
        loaded = torch.load(f.name)
        model.load_state_dict(loaded['model_state_dict'])
        print(f"✓ Checkpoint loaded successfully")
        
        import os
        os.unlink(f.name)
    
    return True


def test_different_model_sizes():
    """Test different model configurations"""
    print("\n=== Testing Different Model Sizes ===")
    
    configs = [
        ("Tiny", 128, 512, 4, 4),
        ("Small", 768, 3072, 12, 12),
        ("Medium", 1024, 4096, 24, 16),
    ]
    
    for name, hidden, intermediate, layers, heads in configs:
        model = MinimalTransformer(
            vocab_size=32000,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            use_tensor_parallel=False,
        )
        
        num_params = model.get_num_params()
        print(f"✓ {name:8s}: {num_params / 1e6:6.1f}M params "
              f"(h={hidden}, l={layers}, heads={heads})")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("Running Minimal Pretraining Tests")
    print("=" * 80)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Tensor Parallel Layers", test_tensor_parallel_layers),
        ("Attention", test_attention),
        ("MLP", test_mlp),
        ("Transformer Block", test_transformer_block),
        ("Training Step", test_training_step),
        ("Checkpointing", test_checkpointing),
        ("Different Model Sizes", test_different_model_sizes),
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
