#!/bin/bash
# Run all GSM8K scaling ladder experiments sequentially
# Each experiment: 50 steps of GRPO on GSM8K

set -e

export TINKER_API_KEY="${TINKER_API_KEY:?Set TINKER_API_KEY first}"

EXPERIMENTS=(
    "configs/gsm8k_llama_3b.yaml"
    "configs/gsm8k_llama_8b.yaml"
    "configs/gsm8k_qwen_8b.yaml"
    "configs/gsm8k_qwen_30b_moe.yaml"
)

echo "============================================"
echo "  GRPO Scaling Ladder - All Experiments"
echo "  Models: Llama-3B, Llama-8B, Qwen-8B, Qwen-30B-MoE"
echo "  Task: GSM8K, 50 steps each"
echo "============================================"
echo ""

RESULTS_FILE="./logs/scaling_results.txt"
mkdir -p ./logs
echo "GRPO Scaling Ladder Results - $(date)" > "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"

for config in "${EXPERIMENTS[@]}"; do
    name=$(basename "$config" .yaml)
    echo ""
    echo ">>> Starting experiment: $name"
    echo ">>> Config: $config"
    echo ""

    START_TIME=$(date +%s)

    if ./run_experiment.sh "$config"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "PASS: $name (${DURATION}s)" >> "$RESULTS_FILE"
        echo ">>> COMPLETED: $name in ${DURATION}s"
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "FAIL: $name (${DURATION}s)" >> "$RESULTS_FILE"
        echo ">>> FAILED: $name after ${DURATION}s"
        echo ">>> Check logs/$(basename "$config" .yaml)/ for details"
    fi

    echo ""
    echo "--- Waiting 10s before next experiment ---"
    sleep 10
done

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Results: $RESULTS_FILE"
echo "============================================"
cat "$RESULTS_FILE"
