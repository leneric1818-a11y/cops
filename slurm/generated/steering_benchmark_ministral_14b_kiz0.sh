#!/bin/bash
#SBATCH --job-name=bench_ministral14b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpubasic

# Usage:
#   AXIS=openness sbatch slurm/generated/steering_benchmark_ministral_14b_kiz0.sh
#
# Runs the full benchmark (prompt_diff + paired_dense + reft_loreft) for
# mistralai/Mistral-7B-Instruct-v0.3 on the specified axis.
#
# Prerequisites:
#   1. Layerwise separability must be run first (layerwise_separability_ministral_14b_kiz0.sh)
#   2. Update the layer in configs/benchmarks/model_matrix/persona_{axis}_v1__ministral_14b.json
#      to match the best_by_test_auc layer from the separability output.
#   3. rsync the updated configs back to the cluster before running this job.

set -euo pipefail

echo "=================================================================="
echo "Ministral-3-14B Steering Benchmark at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "=================================================================="

module purge 2>/dev/null || true
module load cuda/cuda-12.6 2>/dev/null || true
module load python/anaconda3 2>/dev/null || true

PROJECT_DIR="${PROJECT_DIR:-${CLUSTER_HOME}/llm_ft_comparison}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${CLUSTER_HOME}}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH_ROOT/.cache}"
CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-$SCRATCH_ROOT/.conda_envs}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$CONDA_ENVS_DIRS/llm_ft_comparison}"
TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp}"

export HF_HOME="${HF_HOME:-$CACHE_DIR/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_DIR/triton}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=disabled

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
         "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TMPDIR"

cd "$PROJECT_DIR" || exit 1

export CONDA_PREFIX="$CONDA_ENV_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
hash -r

echo "Python:       $(which python)"

python - <<'PY'
import torch
print(f"Torch:        {torch.__version__}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA not available.")
PY

AXIS="${AXIS:-openness}"
RUN_SPEC_PATH="$PROJECT_DIR/configs/benchmarks/model_matrix/persona_${AXIS}_v1__ministral_14b.json"

if [ ! -f "$RUN_SPEC_PATH" ]; then
  echo "ERROR: Run spec not found: $RUN_SPEC_PATH"
  exit 1
fi

echo "Axis:         $AXIS"
echo "Run spec:     $RUN_SPEC_PATH"
echo ""

python -B scripts/run_steering_benchmark.py --mode run-spec --run-spec "$RUN_SPEC_PATH"

echo ""
echo "Job completed at $(date)"
