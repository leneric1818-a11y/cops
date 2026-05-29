#!/bin/bash
#SBATCH --job-name=cops_gemma_ana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --qos=gpubasic
#SBATCH --output=logs/cops_gemma_ana_%j.out
#SBATCH --error=logs/cops_gemma_ana_%j.err

set -euo pipefail
SUBMIT_DIR="${CLUSTER_HOME}/llm_ft_comparison"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${CLUSTER_HOME}/.conda_envs/llm_ft_comparison}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
cd "$SUBMIT_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r
export PYTHONPATH="$SUBMIT_DIR/src:${PYTHONPATH:-}"

LAYERS="1,3,6,10,15,20"

echo "=== Gemma: per-persona + PCA analysis at $(date) ==="

for TAG in caseblind caseaware; do
  if [ "$TAG" = caseblind ]; then
    CACHE=outputs/metrics/cops_hidden_cache/v2_persona_pairs_openness_gpt54mini_1000_n1000_seed42_layers1-3-6-10-15-20.npz
    PAIRS=outputs/metrics/persona_pairs_openness_gpt54mini_1000.jsonl
  else
    CACHE=outputs/metrics/cops_hidden_cache/v2_persona_pairs_openness_case_aware_1000_n1000_seed42_layers1-3-6-10-15-20.npz
    PAIRS=outputs/metrics/persona_pairs_openness_case_aware_1000.jsonl
  fi

  echo ""
  echo "=== $TAG — per-persona decomposition ==="
  python -u -B scripts/cops_persona_decomposition.py \
      --cache-file "$CACHE" \
      --pairs-path "$PAIRS" \
      --layers "$LAYERS" \
      --output-path outputs/metrics/cops_persona_decomp_gemma_${TAG}.json

  echo ""
  echo "=== $TAG — per-persona v_open + PCA ==="
  python -u -B scripts/cops_per_persona_v_open.py \
      --cache-file "$CACHE" \
      --pairs-path "$PAIRS" \
      --layers "$LAYERS" \
      --output-path outputs/metrics/cops_per_persona_gemma_${TAG}.json

  echo ""
  echo "=== $TAG — style-core PCA ==="
  python -u -B scripts/cops_style_core_pca.py \
      --cache-file "$CACHE" \
      --pairs-path "$PAIRS" \
      --layers "$LAYERS" \
      --output-path outputs/metrics/cops_style_core_gemma_${TAG}.json \
      --save-vectors-dir outputs/metrics/cops_style_core_vectors_gemma_${TAG}
done

echo "Done at $(date)"
