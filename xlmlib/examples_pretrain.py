"""
Example: Quick start guide for minimal pretraining

This script demonstrates how to use the minimal pretraining codebase
for different parallelism configurations.
"""

import os
import subprocess
import sys


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def example_single_gpu():
    """Example 1: Single GPU training"""
    print_section("Example 1: Single GPU Training (Data Parallel)")
    
    cmd = """
python minimal_pretrain.py \\
    --hidden-size 512 \\
    --num-layers 8 \\
    --num-attention-heads 8 \\
    --micro-batch-size 4 \\
    --max-steps 100 \\
    --log-interval 10
"""
    
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- Single GPU training")
    print("- Small model (512 hidden, 8 layers)")
    print("- Fast test run (100 steps)")


def example_data_parallel():
    """Example 2: Multi-GPU data parallel"""
    print_section("Example 2: Multi-GPU Data Parallel (4 GPUs)")
    
    cmd = """
torchrun --nproc_per_node=4 minimal_pretrain.py \\
    --hidden-size 768 \\
    --num-layers 12 \\
    --num-attention-heads 12 \\
    --micro-batch-size 4 \\
    --max-steps 1000 \\
    --learning-rate 3e-4
"""
    
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- 4 GPUs with data parallel")
    print("- Model replicated 4 times")
    print("- Effective batch size = 4 * 4 = 16")
    print("- Good for: Model fits on single GPU, want faster training")


def example_tensor_parallel():
    """Example 3: Tensor parallel"""
    print_section("Example 3: Tensor Parallel (4 GPUs)")
    
    cmd = """
torchrun --nproc_per_node=4 minimal_pretrain.py \\
    --hidden-size 2048 \\
    --num-layers 24 \\
    --num-attention-heads 16 \\
    --tensor-parallel-size 4 \\
    --micro-batch-size 2 \\
    --max-steps 1000
"""
    
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- 4 GPUs with tensor parallel")
    print("- Model split across 4 GPUs")
    print("- Each GPU has 1/4 of attention heads and MLP")
    print("- Good for: Large model that doesn't fit on single GPU")
    print("- Requires: NVLink or high bandwidth interconnect")


def example_pipeline_parallel():
    """Example 4: Pipeline parallel"""
    print_section("Example 4: Pipeline Parallel (4 GPUs)")
    
    cmd = """
torchrun --nproc_per_node=4 minimal_pretrain.py \\
    --hidden-size 1024 \\
    --num-layers 24 \\
    --num-attention-heads 16 \\
    --pipeline-parallel-size 4 \\
    --num-microbatches 16 \\
    --pipeline-schedule 1f1b \\
    --micro-batch-size 2 \\
    --max-steps 1000
"""
    
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- 4 GPUs with pipeline parallel")
    print("- Model layers split across 4 stages")
    print("- 16 microbatches for pipeline efficiency")
    print("- Using 1F1B schedule (recommended)")
    print("- Good for: Very deep models, multi-node training")


def example_combined():
    """Example 5: Combined DP + TP + PP"""
    print_section("Example 5: Combined Parallelism (16 GPUs)")
    
    cmd = """
torchrun --nproc_per_node=16 minimal_pretrain.py \\
    --hidden-size 2048 \\
    --num-layers 48 \\
    --num-attention-heads 32 \\
    --tensor-parallel-size 2 \\
    --pipeline-parallel-size 4 \\
    --num-microbatches 16 \\
    --micro-batch-size 2 \\
    --max-steps 10000
"""
    
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- 16 GPUs total")
    print("- 2 Data Parallel replicas")
    print("- 2 Tensor Parallel per replica")
    print("- 4 Pipeline Parallel stages")
    print("- Calculation: 2 (DP) × 2 (TP) × 4 (PP) = 16 GPUs")
    print("- Good for: Very large models, maximum scaling")


def example_custom_config():
    """Example 6: Custom configuration"""
    print_section("Example 6: Custom Model Configuration")
    
    cmd = """
torchrun --nproc_per_node=8 minimal_pretrain.py \\
    --vocab-size 50257 \\
    --hidden-size 1536 \\
    --intermediate-size 6144 \\
    --num-layers 32 \\
    --num-attention-heads 24 \\
    --num-key-value-heads 4 \\
    --max-seq-length 1024 \\
    --tensor-parallel-size 4 \\
    --micro-batch-size 2 \\
    --learning-rate 2e-4 \\
    --weight-decay 0.1 \\
    --warmup-ratio 0.1 \\
    --max-steps 50000 \\
    --save-interval 5000
"""
    
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- Custom model architecture")
    print("- Grouped Query Attention (24 heads, 4 KV heads)")
    print("- Longer sequences (1024 tokens)")
    print("- Custom training hyperparameters")
    print("- Regular checkpointing every 5000 steps")


def example_with_real_data():
    """Example 7: Training with real data"""
    print_section("Example 7: Training with Real Data")
    
    cmd = """
torchrun --nproc_per_node=8 minimal_pretrain.py \\
    --data-dir /path/to/packed/data \\
    --hidden-size 1024 \\
    --num-layers 24 \\
    --num-attention-heads 16 \\
    --tensor-parallel-size 2 \\
    --pipeline-parallel-size 2 \\
    --micro-batch-size 4 \\
    --global-batch-size 256 \\
    --max-steps 100000 \\
    --learning-rate 3e-4 \\
    --output-dir ./checkpoints/my_model
"""
    
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- Using real training data from --data-dir")
    print("- Combined 2×TP and 2×PP for 2×DP replicas")
    print("- Global batch size of 256")
    print("- Checkpoints saved to custom directory")


def example_multi_node():
    """Example 8: Multi-node training"""
    print_section("Example 8: Multi-Node Training (2 Nodes, 8 GPUs each)")
    
    print("Node 0 (Master):")
    cmd1 = """
torchrun \\
    --nnodes=2 \\
    --nproc_per_node=8 \\
    --node_rank=0 \\
    --master_addr="192.168.1.100" \\
    --master_port=29500 \\
    minimal_pretrain.py \\
    --hidden-size 2048 \\
    --num-layers 48 \\
    --tensor-parallel-size 2 \\
    --pipeline-parallel-size 4 \\
    --num-microbatches 16 \\
    --micro-batch-size 2
"""
    print(cmd1)
    
    print("\nNode 1 (Worker):")
    cmd2 = """
torchrun \\
    --nnodes=2 \\
    --nproc_per_node=8 \\
    --node_rank=1 \\
    --master_addr="192.168.1.100" \\
    --master_port=29500 \\
    minimal_pretrain.py \\
    --hidden-size 2048 \\
    --num-layers 48 \\
    --tensor-parallel-size 2 \\
    --pipeline-parallel-size 4 \\
    --num-microbatches 16 \\
    --micro-batch-size 2
"""
    print(cmd2)
    
    print("\nDescription:")
    print("- 2 nodes with 8 GPUs each = 16 total GPUs")
    print("- 2 DP × 2 TP × 4 PP = 16 GPUs")
    print("- Master node at 192.168.1.100")
    print("- Same command on both nodes, only node_rank differs")


def show_usage_script():
    """Example 9: Using environment variables with launch script"""
    print_section("Example 9: Using Launch Scripts")
    
    print("Linux/Mac (launch_pretrain.sh):")
    bash_cmd = """
export NPROC_PER_NODE=8
export TP_SIZE=2
export PP_SIZE=2
export NUM_MICROBATCHES=8
export MICRO_BATCH_SIZE=4
export MAX_STEPS=10000

bash launch_pretrain.sh
"""
    print(bash_cmd)
    
    print("\nWindows (launch_pretrain.ps1):")
    ps_cmd = """
$env:NPROC_PER_NODE=8
$env:TP_SIZE=2
$env:PP_SIZE=2
$env:NUM_MICROBATCHES=8
$env:MICRO_BATCH_SIZE=4
$env:MAX_STEPS=10000

.\\launch_pretrain.ps1
"""
    print(ps_cmd)
    
    print("\nDescription:")
    print("- Use environment variables for configuration")
    print("- Easy to script and automate")
    print("- Launch scripts handle torchrun setup")


def print_tips():
    """Print tips and best practices"""
    print_section("Tips and Best Practices")
    
    tips = [
        ("Memory", "If OOM, reduce micro_batch_size or increase TP/PP size"),
        ("Speed", "Use TP within node (NVLink), PP across nodes"),
        ("Pipeline", "num_microbatches >= 4 × pipeline_parallel_size"),
        ("Convergence", "Start with smaller model to verify setup"),
        ("Debugging", "Use --log-interval 1 for detailed logging"),
        ("Testing", "Run test_minimal_pretrain.py first to verify setup"),
        ("Checkpoints", "Set --save-interval to save regularly"),
        ("Data", "Use PackedDataset for efficient data loading"),
    ]
    
    for category, tip in tips:
        print(f"• {category:12s}: {tip}")


def main():
    print("\n" + "=" * 80)
    print("  MINIMAL LANGUAGE MODEL PRETRAINING - EXAMPLES")
    print("=" * 80)
    
    examples = [
        example_single_gpu,
        example_data_parallel,
        example_tensor_parallel,
        example_pipeline_parallel,
        example_combined,
        example_custom_config,
        example_with_real_data,
        example_multi_node,
        show_usage_script,
    ]
    
    for i, example_fn in enumerate(examples, 1):
        example_fn()
        if i < len(examples):
            input("\nPress Enter for next example...")
    
    print_tips()
    
    print("\n" + "=" * 80)
    print("  Quick Test")
    print("=" * 80)
    print("\nTo verify your setup, run the test suite:")
    print("  python test_minimal_pretrain.py")
    print("\nTo start training immediately (single GPU):")
    print("  python minimal_pretrain.py --max-steps 100 --log-interval 10")
    print("\nFor more information, see README_PRETRAIN.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
