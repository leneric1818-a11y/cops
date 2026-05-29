#!/bin/bash
#SBATCH --job-name=cops_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --qos=basic
#SBATCH --output=logs/cops_score_%j.out
#SBATCH --error=logs/cops_score_%j.err

set -euo pipefail
SUBMIT_DIR="${CLUSTER_HOME}/llm_ft_comparison"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${CLUSTER_HOME}/.conda_envs/llm_ft_comparison}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
cd "$SUBMIT_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r
export PYTHONPATH="$SUBMIT_DIR/src:${PYTHONPATH:-}"

echo "=== Scoring COPS steering eval at $(date) ==="
python -u -B scripts/cops_score_steering_eval.py
echo "Done at $(date)"
