#!/bin/bash
# Evaluate Qwen3-Next on AIME 2024 and AIME 2025 benchmarks
#
# Usage:
#   bash eval/run_aime_eval.sh <model_path> [tp_size] [n_rollout] [max_tokens]
#
# Examples:
#   # Greedy eval on AIME24 + AIME25 (single GPU)
#   bash eval/run_aime_eval.sh ./models/Qwen_Qwen3-Coder-Next/
#
#   # TP=2, greedy
#   bash eval/run_aime_eval.sh ./models/Qwen_Qwen3-Coder-Next/ 2
#
#   # TP=2, 8 rollouts with sampling (pass@8)
#   bash eval/run_aime_eval.sh ./models/Qwen_Qwen3-Coder-Next/ 2 8
#
#   # TP=2, 16 rollouts, 32K max tokens
#   bash eval/run_aime_eval.sh ./models/Qwen_Qwen3-Coder-Next/ 2 16 32768

set -e

MODEL_PATH="${1:?Usage: $0 <model_path> [tp_size] [n_rollout] [max_tokens]}"
TP_SIZE="${2:-1}"
N_ROLLOUT="${3:-1}"
MAX_TOKENS="${4:-4096}"

# Sampling config: greedy if n_rollout=1, else temperature sampling with top-k
if [ "$N_ROLLOUT" -eq 1 ]; then
    TEMPERATURE=0.0
    TOP_K=0
else
    TEMPERATURE=0.7
    TOP_K=50
fi

# Max batch size: pack multiple problems per batch
MAX_BATCH_SIZE=64

# Output directory
OUTPUT_DIR="eval/results"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build launch command
if [ "$TP_SIZE" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=$TP_SIZE"
else
    LAUNCHER="python"
fi

COMMON_ARGS="--model_path $MODEL_PATH \
    --tensor_parallel $TP_SIZE \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --max_tokens $MAX_TOKENS \
    --max_batch_size $MAX_BATCH_SIZE \
    --n_rollout $N_ROLLOUT \
    --prompt_type v17"

echo "============================================"
echo "AIME Evaluation Configuration"
echo "============================================"
echo "Model:         $MODEL_PATH"
echo "TP size:       $TP_SIZE"
echo "N rollout:     $N_ROLLOUT"
echo "Max tokens:    $MAX_TOKENS"
echo "Temperature:   $TEMPERATURE"
echo "Top-k:         $TOP_K"
echo "Max batch:     $MAX_BATCH_SIZE"
echo "Launcher:      $LAUNCHER"
echo "============================================"
echo ""

# ---- AIME 2024 ----
echo ">>> Evaluating on AIME 2024..."
$LAUNCHER eval/eval_qwen3_next.py \
    $COMMON_ARGS \
    --dataset aime24 \
    --output "$OUTPUT_DIR/aime24_tp${TP_SIZE}_r${N_ROLLOUT}_${TIMESTAMP}.json"
echo ""

# ---- AIME 2025 ----
echo ">>> Evaluating on AIME 2025..."
$LAUNCHER eval/eval_qwen3_next.py \
    $COMMON_ARGS \
    --dataset aime25 \
    --output "$OUTPUT_DIR/aime25_tp${TP_SIZE}_r${N_ROLLOUT}_${TIMESTAMP}.json"
echo ""

echo "============================================"
echo "Done! Results saved to $OUTPUT_DIR/"
echo "============================================"
