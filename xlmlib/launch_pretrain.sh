#!/bin/bash
# Launch script for distributed training with torchrun

# Set these variables
NNODES=${NNODES:-1}              # Number of nodes
NPROC_PER_NODE=${NPROC_PER_NODE:-8}  # GPUs per node
NODE_RANK=${NODE_RANK:-0}        # Rank of this node
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

# Model configuration
CONFIG=${CONFIG:-"small"}        # small, medium, large, xl
DATA_DIR=${DATA_DIR:-""}         # Path to training data

# Parallelism settings
TP_SIZE=${TP_SIZE:-1}            # Tensor parallel size
PP_SIZE=${PP_SIZE:-1}            # Pipeline parallel size
NUM_MICROBATCHES=${NUM_MICROBATCHES:-1}

# Training hyperparameters
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MAX_STEPS=${MAX_STEPS:-10000}
LR=${LR:-3e-4}

# Paths
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints"}

# Calculate world size
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

echo "========================================"
echo "Launching distributed training"
echo "========================================"
echo "Nodes: $NNODES"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Total GPUs: $WORLD_SIZE"
echo "Config: $CONFIG"
echo "Tensor Parallel: $TP_SIZE"
echo "Pipeline Parallel: $PP_SIZE"
echo "Data Parallel: $((WORLD_SIZE / (TP_SIZE * PP_SIZE)))"
echo "Micro batch size: $MICRO_BATCH_SIZE"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "========================================"

# Launch with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    minimal_pretrain.py \
    --tensor-parallel-size $TP_SIZE \
    --pipeline-parallel-size $PP_SIZE \
    --num-microbatches $NUM_MICROBATCHES \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --max-steps $MAX_STEPS \
    --learning-rate $LR \
    --output-dir $OUTPUT_DIR \
    ${DATA_DIR:+--data-dir $DATA_DIR}
