# PowerShell script for launching distributed training on Windows

# Set these variables
$NNODES = if ($env:NNODES) { $env:NNODES } else { 1 }
$NPROC_PER_NODE = if ($env:NPROC_PER_NODE) { $env:NPROC_PER_NODE } else { 4 }
$NODE_RANK = if ($env:NODE_RANK) { $env:NODE_RANK } else { 0 }
$MASTER_ADDR = if ($env:MASTER_ADDR) { $env:MASTER_ADDR } else { "localhost" }
$MASTER_PORT = if ($env:MASTER_PORT) { $env:MASTER_PORT } else { 29500 }

# Model configuration
$CONFIG = if ($env:CONFIG) { $env:CONFIG } else { "small" }
$DATA_DIR = $env:DATA_DIR

# Parallelism settings
$TP_SIZE = if ($env:TP_SIZE) { $env:TP_SIZE } else { 1 }
$PP_SIZE = if ($env:PP_SIZE) { $env:PP_SIZE } else { 1 }
$NUM_MICROBATCHES = if ($env:NUM_MICROBATCHES) { $env:NUM_MICROBATCHES } else { 1 }

# Training hyperparameters
$MICRO_BATCH_SIZE = if ($env:MICRO_BATCH_SIZE) { $env:MICRO_BATCH_SIZE } else { 4 }
$GLOBAL_BATCH_SIZE = if ($env:GLOBAL_BATCH_SIZE) { $env:GLOBAL_BATCH_SIZE } else { 32 }
$MAX_STEPS = if ($env:MAX_STEPS) { $env:MAX_STEPS } else { 10000 }
$LR = if ($env:LR) { $env:LR } else { "3e-4" }

# Paths
$OUTPUT_DIR = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\checkpoints" }

# Calculate world size
$WORLD_SIZE = $NNODES * $NPROC_PER_NODE

Write-Host "========================================"
Write-Host "Launching distributed training"
Write-Host "========================================"
Write-Host "Nodes: $NNODES"
Write-Host "GPUs per node: $NPROC_PER_NODE"
Write-Host "Total GPUs: $WORLD_SIZE"
Write-Host "Config: $CONFIG"
Write-Host "Tensor Parallel: $TP_SIZE"
Write-Host "Pipeline Parallel: $PP_SIZE"
Write-Host "Data Parallel: $($WORLD_SIZE / ($TP_SIZE * $PP_SIZE))"
Write-Host "Micro batch size: $MICRO_BATCH_SIZE"
Write-Host "Global batch size: $GLOBAL_BATCH_SIZE"
Write-Host "========================================"

# Build command arguments
$args = @(
    "--nnodes=$NNODES",
    "--nproc_per_node=$NPROC_PER_NODE",
    "--node_rank=$NODE_RANK",
    "--master_addr=$MASTER_ADDR",
    "--master_port=$MASTER_PORT",
    "minimal_pretrain.py",
    "--tensor-parallel-size", $TP_SIZE,
    "--pipeline-parallel-size", $PP_SIZE,
    "--num-microbatches", $NUM_MICROBATCHES,
    "--micro-batch-size", $MICRO_BATCH_SIZE,
    "--global-batch-size", $GLOBAL_BATCH_SIZE,
    "--max-steps", $MAX_STEPS,
    "--learning-rate", $LR,
    "--output-dir", $OUTPUT_DIR
)

if ($DATA_DIR) {
    $args += "--data-dir"
    $args += $DATA_DIR
}

# Launch with torchrun
& torchrun $args
