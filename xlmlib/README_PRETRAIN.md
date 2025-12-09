# Minimal Language Model Pretraining

A minimal yet comprehensive implementation of transformer language model pretraining with support for:
- **Data Parallel (DP)**: Distribute data across multiple GPUs
- **Tensor Parallel (TP)**: Split model tensors across GPUs
- **Pipeline Parallel (PP)**: Split model layers across GPUs

## Features

✅ **Complete Parallelism Support**
- Data Parallel with DistributedDataParallel (DDP)
- Tensor Parallel for attention and MLP layers
- Pipeline Parallel with GPipe and 1F1B schedules

✅ **Production-Ready Components**
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- RMSNorm normalization
- SwiGLU activation in MLP

✅ **Flexible Configuration**
- Predefined model configs (Small, Medium, Large, XL)
- Easy-to-use launch scripts
- Checkpointing and logging

## File Structure

```
xlmlib/
├── minimal_model.py           # Transformer model with TP support
├── pipeline_parallel.py       # Pipeline parallel implementation
├── parallel_initialization.py # Process group initialization
├── minimal_pretrain.py        # Main training script
├── pretrain_config.py         # Configuration dataclasses
├── launch_pretrain.sh         # Bash launch script
├── launch_pretrain.ps1        # PowerShell launch script
└── README_PRETRAIN.md         # This file
```

## Installation

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate
```

## Quick Start

### Single GPU Training (Data Parallel Only)

```bash
# Using Python directly
python minimal_pretrain.py \
    --hidden-size 768 \
    --num-layers 12 \
    --num-attention-heads 12 \
    --micro-batch-size 4 \
    --max-steps 10000

# Using torchrun (recommended)
torchrun --nproc_per_node=1 minimal_pretrain.py \
    --hidden-size 768 \
    --num-layers 12 \
    --num-attention-heads 12 \
    --micro-batch-size 4 \
    --max-steps 10000
```

### Multi-GPU Training

#### Data Parallel (4 GPUs)

```bash
torchrun --nproc_per_node=4 minimal_pretrain.py \
    --hidden-size 768 \
    --num-layers 12 \
    --num-attention-heads 12 \
    --micro-batch-size 4 \
    --max-steps 10000
```

#### Tensor Parallel (2 GPUs)

```bash
torchrun --nproc_per_node=2 minimal_pretrain.py \
    --hidden-size 768 \
    --num-layers 12 \
    --num-attention-heads 12 \
    --tensor-parallel-size 2 \
    --micro-batch-size 4 \
    --max-steps 10000
```

#### Pipeline Parallel (4 GPUs)

```bash
torchrun --nproc_per_node=4 minimal_pretrain.py \
    --hidden-size 768 \
    --num-layers 12 \
    --num-attention-heads 12 \
    --pipeline-parallel-size 4 \
    --num-microbatches 4 \
    --micro-batch-size 4 \
    --max-steps 10000
```

#### Combined: DP + TP + PP (8 GPUs)

```bash
# 2 DP x 2 TP x 2 PP = 8 GPUs
torchrun --nproc_per_node=8 minimal_pretrain.py \
    --hidden-size 1024 \
    --num-layers 24 \
    --num-attention-heads 16 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --num-microbatches 4 \
    --micro-batch-size 2 \
    --max-steps 10000
```

## Using Launch Scripts

### Linux/Mac (Bash)

```bash
# Set environment variables
export NPROC_PER_NODE=8
export TP_SIZE=2
export PP_SIZE=2
export MICRO_BATCH_SIZE=4
export MAX_STEPS=10000

# Run
bash launch_pretrain.sh
```

### Windows (PowerShell)

```powershell
# Set environment variables
$env:NPROC_PER_NODE=8
$env:TP_SIZE=2
$env:PP_SIZE=2
$env:MICRO_BATCH_SIZE=4
$env:MAX_STEPS=10000

# Run
.\launch_pretrain.ps1
```

## Configuration

### Model Configurations

Predefined configurations in `pretrain_config.py`:

| Config | Params | Hidden | Layers | Heads | TP | PP |
|--------|--------|--------|--------|-------|----|----|
| Small  | ~125M  | 768    | 12     | 12    | 1  | 1  |
| Medium | ~350M  | 1024   | 24     | 16    | 2  | 1  |
| Large  | ~774M  | 1280   | 36     | 20    | 4  | 2  |
| XL     | ~1.5B  | 1600   | 48     | 25    | 4  | 4  |

### Command Line Arguments

#### Model Arguments
```
--vocab-size              Vocabulary size (default: 32000)
--hidden-size             Hidden dimension (default: 768)
--intermediate-size       FFN intermediate size (default: 3072)
--num-layers              Number of transformer layers (default: 12)
--num-attention-heads     Number of attention heads (default: 12)
--num-key-value-heads     Number of KV heads for GQA (default: None)
--max-seq-length          Maximum sequence length (default: 512)
```

#### Training Arguments
```
--micro-batch-size        Batch size per GPU (default: 4)
--global-batch-size       Total batch size across all GPUs (default: 32)
--max-steps               Maximum training steps (default: 10000)
--learning-rate           Peak learning rate (default: 3e-4)
--min-lr                  Minimum learning rate (default: 3e-5)
--weight-decay            Weight decay coefficient (default: 0.01)
--grad-clip               Gradient clipping value (default: 1.0)
--warmup-ratio            Warmup ratio (default: 0.05)
```

#### Parallelism Arguments
```
--tensor-parallel-size      Number of GPUs for tensor parallel (default: 1)
--pipeline-parallel-size    Number of GPUs for pipeline parallel (default: 1)
--num-microbatches          Number of microbatches for PP (default: 1)
--pipeline-schedule         Pipeline schedule: gpipe or 1f1b (default: 1f1b)
```

#### Data Arguments
```
--data-dir                Path to training data directory
--num-train-samples       Number of training samples for dummy data (default: 10000)
```

#### Logging Arguments
```
--output-dir              Output directory for checkpoints (default: ./checkpoints)
--log-interval            Log every N steps (default: 10)
--save-interval           Save checkpoint every N steps (default: 1000)
```

## Parallelism Strategies

### When to Use Each Strategy

#### Data Parallel (DP)
- ✅ Model fits on single GPU
- ✅ Want to scale training throughput
- ✅ Simple to use, no code changes needed
- ❌ Memory limited by single GPU

#### Tensor Parallel (TP)
- ✅ Model too large for single GPU
- ✅ Need to split attention and MLP layers
- ✅ Good for scaling up to 8 GPUs per node
- ❌ Requires high bandwidth between GPUs (NVLink)

#### Pipeline Parallel (PP)
- ✅ Very deep models (many layers)
- ✅ Can work across nodes
- ✅ Less bandwidth intensive than TP
- ❌ Pipeline bubble overhead
- ❌ More complex to tune (microbatches)

#### Combined (DP + TP + PP)
- ✅ Very large models (billions of parameters)
- ✅ Maximum scaling across many GPUs
- ✅ Balance memory and communication
- ❌ Most complex to configure

### Parallelism Math

```
Total GPUs = Data Parallel × Tensor Parallel × Pipeline Parallel
           = DP × TP × PP

Example: 64 GPUs = 4 (DP) × 4 (TP) × 4 (PP)
```

## Pipeline Schedules

### GPipe
- Simple: All forward passes, then all backward passes
- High pipeline bubble overhead
- Good for small number of microbatches

### 1F1B (One Forward One Backward)
- Interleaved: Forward and backward alternate
- Lower pipeline bubble overhead
- Recommended for most cases

## Performance Tips

1. **Tensor Parallel Size**
   - Use within a single node (8 GPUs max)
   - Requires NVLink for good performance
   - Powers of 2 work best: 2, 4, 8

2. **Pipeline Parallel Size**
   - Number of microbatches ≥ 4 × pipeline parallel size
   - Use 1F1B schedule for better efficiency
   - Good for scaling across nodes

3. **Microbatch Size**
   - Balance between memory and throughput
   - Smaller = less memory, more pipeline efficiency
   - Try: `num_microbatches = pipeline_parallel_size × 4`

4. **Gradient Accumulation**
   - Effective batch size = micro_batch_size × gradient_accumulation_steps × data_parallel_size
   - Use to achieve larger batch sizes

## Examples

### Example 1: 4 GPUs with DP

```bash
torchrun --nproc_per_node=4 minimal_pretrain.py \
    --hidden-size 768 \
    --num-layers 12 \
    --micro-batch-size 8 \
    --max-steps 10000
```
Result: 4-way data parallel, model replicated 4 times

### Example 2: 4 GPUs with TP

```bash
torchrun --nproc_per_node=4 minimal_pretrain.py \
    --hidden-size 2048 \
    --num-layers 24 \
    --tensor-parallel-size 4 \
    --micro-batch-size 4 \
    --max-steps 10000
```
Result: Model split across 4 GPUs using tensor parallel

### Example 3: 8 GPUs with TP + DP

```bash
torchrun --nproc_per_node=8 minimal_pretrain.py \
    --hidden-size 1536 \
    --num-layers 24 \
    --tensor-parallel-size 4 \
    --micro-batch-size 4 \
    --max-steps 10000
```
Result: 2 DP replicas, each using 4-way TP

### Example 4: 16 GPUs with DP + TP + PP

```bash
torchrun --nproc_per_node=16 minimal_pretrain.py \
    --hidden-size 2048 \
    --num-layers 48 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 4 \
    --num-microbatches 16 \
    --micro-batch-size 2 \
    --max-steps 10000
```
Result: 2 DP × 2 TP × 4 PP = 16 GPUs

## Multi-Node Training

### Node 0 (Master)

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    minimal_pretrain.py \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --micro-batch-size 4
```

### Node 1 (Worker)

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    minimal_pretrain.py \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --micro-batch-size 4
```

## Monitoring Training

The training script logs:
- Loss every `--log-interval` steps
- Learning rate
- Throughput (samples/second)
- Parallel configuration

Example output:
```
========================================
Training Configuration
========================================
World Size: 8
Data Parallel Size: 2
Tensor Parallel Size: 2
Pipeline Parallel Size: 2
Micro Batch Size: 4
Global Batch Size: 32
...
========================================
Rank 0 | DP: 0/2 | TP: 0/2 | PP: 0/2
Step 10 | Loss: 10.5432 | LR: 1.20e-04 | Throughput: 123.45 samples/s
Step 20 | Loss: 10.1234 | LR: 2.40e-04 | Throughput: 125.67 samples/s
```

## Troubleshooting

### Out of Memory
- Decrease `--micro-batch-size`
- Increase `--tensor-parallel-size` or `--pipeline-parallel-size`
- Decrease `--hidden-size` or `--num-layers`

### Slow Training
- Check TP size matches GPU topology (use NVLink)
- Increase `--num-microbatches` for pipeline parallel
- Use 1F1B schedule instead of GPipe

### Convergence Issues
- Adjust learning rate
- Check global batch size is reasonable
- Verify data is being loaded correctly

## Custom Data

To use your own data, provide a `--data-dir` pointing to packed dataset files:

```bash
torchrun --nproc_per_node=4 minimal_pretrain.py \
    --data-dir /path/to/data \
    --micro-batch-size 4 \
    --max-steps 100000
```

The code expects `PackedDataset` format. Without `--data-dir`, it uses dummy random data.

## Checkpointing

Checkpoints are saved to `--output-dir` every `--save-interval` steps:

```
checkpoints/
├── checkpoint-1000.pt
├── checkpoint-2000.pt
└── checkpoint-3000.pt
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training step
- Configuration

## License

This is a minimal educational implementation. For production use, consider frameworks like:
- Megatron-LM
- DeepSpeed
- PyTorch FSDP

## References

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): Original tensor and pipeline parallel implementation
- [GPipe](https://arxiv.org/abs/1811.06965): Pipeline parallel training
- [1F1B Schedule](https://arxiv.org/abs/2104.04473): Efficient pipeline schedule
