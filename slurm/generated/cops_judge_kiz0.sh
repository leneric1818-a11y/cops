#!/bin/bash
#SBATCH --job-name=cops_judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --qos=basic
#SBATCH --output=logs/cops_judge_%j.out
#SBATCH --error=logs/cops_judge_%j.err

set -euo pipefail
SUBMIT_DIR="${CLUSTER_HOME}/llm_ft_comparison"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${CLUSTER_HOME}/.conda_envs/llm_ft_comparison}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
cd "$SUBMIT_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r
export PYTHONPATH="$SUBMIT_DIR/src:$SUBMIT_DIR/scripts:${PYTHONPATH:-}"

# Load API key from .env (expects OPENAI_API_KEY)
if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a
    source "$SUBMIT_DIR/.env"
    set +a
fi

MODEL="${JUDGE_MODEL:-gpt-5.4-mini}"
PROVIDER="${JUDGE_PROVIDER:-openai}"
WORKERS="${WORKERS:-6}"
SINGLE_FILE="${SINGLE_FILE:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
EVAL_DIR="${EVAL_DIR:-}"
OUT_DIR="${OUT_DIR:-}"

echo "=== COPS judge at $(date)  provider=$PROVIDER model=$MODEL effort=$REASONING_EFFORT workers=$WORKERS single=$SINGLE_FILE ==="

ARGS="--provider $PROVIDER --model $MODEL --workers $WORKERS"
if [ -n "$SINGLE_FILE" ]; then
    ARGS="$ARGS --single-file $SINGLE_FILE"
fi
if [ -n "$EVAL_DIR" ]; then
    ARGS="$ARGS --eval-dir $EVAL_DIR"
fi
if [ -n "$OUT_DIR" ]; then
    ARGS="$ARGS --out-dir $OUT_DIR"
fi
if [ -n "$MAX_EXAMPLES" ]; then
    ARGS="$ARGS --max-examples-per-file $MAX_EXAMPLES"
fi
if [ -n "$REASONING_EFFORT" ]; then
    ARGS="$ARGS --reasoning-effort $REASONING_EFFORT"
fi

python -u -B scripts/cops_judge_steering_eval.py $ARGS

echo "Done at $(date)"
