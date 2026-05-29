#!/bin/bash
#SBATCH --job-name=cops_per_persona
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --qos=gpubasic
#SBATCH --output=logs/cops_per_persona_%j.out
#SBATCH --error=logs/cops_per_persona_%j.err

set -euo pipefail
SUBMIT_DIR="${CLUSTER_HOME}/llm_ft_comparison"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${CLUSTER_HOME}/.conda_envs/llm_ft_comparison}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
cd "$SUBMIT_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r
export PYTHONPATH="$SUBMIT_DIR/src:${PYTHONPATH:-}"

echo "=== Per-persona v_open analysis at $(date) ==="

echo ""
echo "=========================================="
echo "CASE-BLIND"
echo "=========================================="
python -u -B scripts/cops_per_persona_v_open.py \
    --cache-file outputs/metrics/cops_hidden_cache/v2_n1000_seed42_layers1-5-9-13-18.npz \
    --pairs-path outputs/metrics/persona_pairs_openness_gpt54mini_1000.jsonl \
    --output-path outputs/metrics/cops_per_persona_caseblind.json

echo ""
echo "=========================================="
echo "CASE-AWARE"
echo "=========================================="
python -u -B scripts/cops_per_persona_v_open.py \
    --cache-file outputs/metrics/cops_hidden_cache/v2_persona_pairs_openness_case_aware_1000_n1000_seed42_layers1-5-9-13-18.npz \
    --pairs-path outputs/metrics/persona_pairs_openness_case_aware_1000.jsonl \
    --output-path outputs/metrics/cops_per_persona_caseaware.json

echo "Done at $(date)"
