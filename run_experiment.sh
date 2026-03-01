#!/bin/bash
# Run a single GSM8K GRPO experiment end-to-end
# Usage: ./run_experiment.sh configs/gsm8k_llama_3b.yaml
#
# This script starts all 3 required processes:
#   1. Atropos API server (run-api)
#   2. Tinker trainer (launch_training.py)
#   3. GSM8K environment (gsm8k_tinker.py serve)
#
# And waits for the trainer to finish, then cleans up.

set -e

CONFIG="${1:?Usage: ./run_experiment.sh <config.yaml>}"
EXPERIMENT_NAME=$(basename "$CONFIG" .yaml)
LOG_DIR="./logs/${EXPERIMENT_NAME}"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Tinker RL Experiment: $EXPERIMENT_NAME"
echo "  Config: $CONFIG"
echo "  Logs: $LOG_DIR/"
echo "============================================"

# Ensure TINKER_API_KEY is set
if [ -z "$TINKER_API_KEY" ]; then
    echo "ERROR: TINKER_API_KEY not set"
    echo "  export TINKER_API_KEY='your-key-here'"
    exit 1
fi

# Activate venv
source .venv/bin/activate

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up processes..."
    [ -n "$API_PID" ] && kill $API_PID 2>/dev/null && echo "  Stopped Atropos API (PID $API_PID)"
    [ -n "$ENV_PID" ] && kill $ENV_PID 2>/dev/null && echo "  Stopped Environment (PID $ENV_PID)"
    [ -n "$TRAINER_PID" ] && kill $TRAINER_PID 2>/dev/null && echo "  Stopped Trainer (PID $TRAINER_PID)"
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT

# 1. Start Atropos API server
echo ""
echo "[1/3] Starting Atropos API server..."
run-api > "$LOG_DIR/api.log" 2>&1 &
API_PID=$!
sleep 3
echo "  Atropos API running (PID $API_PID)"

# 2. Start trainer
echo "[2/3] Starting Tinker trainer..."
python launch_training.py --config "$CONFIG" > "$LOG_DIR/trainer.log" 2>&1 &
TRAINER_PID=$!
sleep 5
echo "  Trainer running (PID $TRAINER_PID)"

# 3. Start GSM8K environment
echo "[3/3] Starting GSM8K environment..."

# Update CONFIG_PATH in the environment file for this run
export GSM8K_CONFIG_PATH="$CONFIG"
python tinker_atropos/environments/gsm8k_tinker.py serve --config "$CONFIG" > "$LOG_DIR/env.log" 2>&1 &
ENV_PID=$!
echo "  Environment running (PID $ENV_PID)"

echo ""
echo "============================================"
echo "  All processes started!"
echo "  Trainer PID: $TRAINER_PID"
echo "  Tailing trainer log..."
echo "============================================"
echo ""

# Wait for the trainer to finish
wait $TRAINER_PID
TRAINER_EXIT=$?

echo ""
echo "============================================"
echo "  Trainer finished (exit code: $TRAINER_EXIT)"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Logs saved to: $LOG_DIR/"
echo "============================================"

exit $TRAINER_EXIT
