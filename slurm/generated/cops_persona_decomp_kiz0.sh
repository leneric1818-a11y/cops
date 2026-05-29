#!/bin/bash
#SBATCH --job-name=cops_decomp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --qos=gpubasic
#SBATCH --output=logs/cops_decomp_%j.out
#SBATCH --error=logs/cops_decomp_%j.err

set -euo pipefail
echo "=== COPS persona decomposition at $(date) ==="

SUBMIT_DIR="${CLUSTER_HOME}/llm_ft_comparison"
SCRATCH_ROOT="${SCRATCH_ROOT:-${CLUSTER_HOME}}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$SCRATCH_ROOT/.conda_envs/llm_ft_comparison}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
cd "$SUBMIT_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r
export PYTHONPATH="$SUBMIT_DIR/src:${PYTHONPATH:-}"

echo ""
echo "=========================================="
echo "CASE-BLIND (gpt54mini_1000)"
echo "=========================================="
python -u -B scripts/cops_persona_decomposition.py \
    --cache-file outputs/metrics/cops_hidden_cache/v2_n1000_seed42_layers1-5-9-13-18.npz \
    --pairs-path outputs/metrics/persona_pairs_openness_gpt54mini_1000.jsonl \
    --output-path outputs/metrics/cops_persona_decomp_caseblind.json

echo ""
echo "=========================================="
echo "CASE-AWARE"
echo "=========================================="
python -u -B scripts/cops_persona_decomposition.py \
    --cache-file outputs/metrics/cops_hidden_cache/v2_persona_pairs_openness_case_aware_1000_n1000_seed42_layers1-5-9-13-18.npz \
    --pairs-path outputs/metrics/persona_pairs_openness_case_aware_1000.jsonl \
    --output-path outputs/metrics/cops_persona_decomp_caseaware.json

echo ""
echo "Done at $(date)"
