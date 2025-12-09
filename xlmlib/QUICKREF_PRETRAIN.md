# Quick Reference Guide - Minimal Pretraining

## Files Overview

| File | Purpose |
|------|---------|
| `minimal_model.py` | Transformer model with tensor parallel support |
| `pipeline_parallel.py` | Pipeline parallel wrapper and schedules |
| `parallel_initialization.py` | Process group initialization (existing) |
| `minimal_pretrain.py` | Main training script |
| `pretrain_config.py` | Configuration dataclasses and presets |
| `test_minimal_pretrain.py` | Unit tests |
| `examples_pretrain.py` | Usage examples |
| `launch_pretrain.sh` | Bash launch script |
| `launch_pretrain.ps1` | PowerShell launch script |
| `README_PRETRAIN.md` | Full documentation |

## Quick Commands

### Run Tests
```bash
python test_minimal_pretrain.py
```

### Single GPU
```bash
python minimal_pretrain.py --max-steps 100
```

### 4 GPUs (Data Parallel)
```bash
torchrun --nproc_per_node=4 minimal_pretrain.py --max-steps 1000
```

### 4 GPUs (Tensor Parallel)
```bash
torchrun --nproc_per_node=4 minimal_pretrain.py \
    --tensor-parallel-size 4 \
    --hidden-size 2048 \
    --max-steps 1000
```

### 8 GPUs (2 DP × 2 TP × 2 PP)
```bash
torchrun --nproc_per_node=8 minimal_pretrain.py \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --num-microbatches 8 \
    --max-steps 1000
```

## Key Arguments

### Model Size
- `--hidden-size`: Model dimension (768, 1024, 2048...)
- `--num-layers`: Number of transformer layers (12, 24, 48...)
- `--num-attention-heads`: Number of attention heads

### Training
- `--micro-batch-size`: Batch size per GPU
- `--max-steps`: Training steps
- `--learning-rate`: Peak LR (3e-4 typical)

### Parallelism
- `--tensor-parallel-size`: TP size (1, 2, 4, 8)
- `--pipeline-parallel-size`: PP size (1, 2, 4...)
- `--num-microbatches`: Microbatches for PP (≥4×PP)

## Parallelism Decision Tree

```
Start
  ├─ Model fits on 1 GPU?
  │   ├─ Yes → Use Data Parallel (DP)
  │   │         torchrun --nproc_per_node=N
  │   └─ No → Model too large?
  │       ├─ Wide model → Use Tensor Parallel (TP)
  │       │               --tensor-parallel-size 2/4/8
  │       └─ Deep model → Use Pipeline Parallel (PP)
  │                       --pipeline-parallel-size 2/4/8
  │
  └─ Very large model?
      └─ Yes → Combine DP + TP + PP
                DP × TP × PP = Total GPUs
```

## Common Issues

### Out of Memory
```bash
# Reduce batch size
--micro-batch-size 2

# Or increase parallelism
--tensor-parallel-size 4
```

### Slow Training
```bash
# For PP, increase microbatches
--num-microbatches 16

# Use 1F1B schedule
--pipeline-schedule 1f1b
```

### Convergence Issues
```bash
# Adjust learning rate
--learning-rate 2e-4

# Increase warmup
--warmup-ratio 0.1
```

## Model Size Calculator

| Hidden | Layers | Heads | Params | Min GPUs (TP) |
|--------|--------|-------|--------|---------------|
| 768    | 12     | 12    | ~125M  | 1             |
| 1024   | 24     | 16    | ~350M  | 1             |
| 1280   | 36     | 20    | ~774M  | 1-2           |
| 1600   | 48     | 25    | ~1.5B  | 2-4           |
| 2048   | 48     | 32    | ~2.7B  | 4-8           |
| 4096   | 40     | 32    | ~7B    | 8-16          |

## Environment Variables

### For Bash Script
```bash
export NPROC_PER_NODE=8
export TP_SIZE=2
export PP_SIZE=2
export MICRO_BATCH_SIZE=4
bash launch_pretrain.sh
```

### For PowerShell Script
```powershell
$env:NPROC_PER_NODE=8
$env:TP_SIZE=2
$env:PP_SIZE=2
$env:MICRO_BATCH_SIZE=4
.\launch_pretrain.ps1
```

## Monitoring

### Training Logs
```
Step 100 | Loss: 10.5432 | LR: 1.20e-04 | Throughput: 123.45 samples/s
```

### GPU Usage
```bash
# Watch GPU memory and utilization
nvidia-smi -l 1

# Or use
watch -n 1 nvidia-smi
```

### Checkpoints
Saved to `--output-dir` (default: `./checkpoints/`)
- `checkpoint-1000.pt`
- `checkpoint-2000.pt`
- ...

## Best Practices

1. **Start Small**: Test with smaller model first
2. **Verify Setup**: Run `test_minimal_pretrain.py`
3. **TP Topology**: Use TP within node (NVLink required)
4. **PP Efficiency**: `num_microbatches ≥ 4 × pipeline_parallel_size`
5. **Batch Size**: Start small, increase gradually
6. **Checkpointing**: Save regularly with `--save-interval`
7. **Logging**: Use `--log-interval 10` for frequent updates

## Getting Help

1. Read `README_PRETRAIN.md` for full documentation
2. Run `python examples_pretrain.py` for interactive examples
3. Check `test_minimal_pretrain.py` for code usage patterns
4. Review existing code: `mini_pretrain_example1.py`

## Next Steps

1. ✅ Run tests: `python test_minimal_pretrain.py`
2. ✅ Try single GPU: `python minimal_pretrain.py --max-steps 100`
3. ✅ Scale to multi-GPU: `torchrun --nproc_per_node=4 ...`
4. ✅ Add your data: `--data-dir /path/to/data`
5. ✅ Experiment with parallelism configurations
