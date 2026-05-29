#!/bin/bash
#SBATCH --job-name=projection_monitor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpubasic

set -euo pipefail

echo "=================================================================="
echo "Persona Projection Monitor at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
else
  echo "GPU: <none>"
fi
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

if [ ! -x "$CONDA_ENV_DIR/bin/python" ] || [ ! -x "$CONDA_ENV_DIR/bin/pip" ]; then
  echo "ERROR: prepared env not found: $CONDA_ENV_DIR"
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_DIR"
else
  source "$CONDA_ENV_DIR/bin/activate"
fi

export PATH="$CONDA_ENV_DIR/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
hash -r

echo "Python:       $(which python)"
echo "Pip:          $(which pip)"
echo "Conda env:    ${CONDA_PREFIX:-<none>}"

if [ "$(which python)" != "$CONDA_ENV_DIR/bin/python" ]; then
  echo "ERROR: python is not from the activated conda env."
  exit 1
fi

if [ -z "${PROJECTION_SPEC_PATH:-}" ]; then
  echo "ERROR: PROJECTION_SPEC_PATH must be set."
  exit 1
fi

python - <<'PY'
import torch
print(f"Torch:        {torch.__version__}")
print(f"Torch CUDA:   {torch.version.cuda}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:  {torch.cuda.get_device_name(0)}")
PY

echo "Projection spec: $PROJECTION_SPEC_PATH"
echo ""

python -B scripts/run_steering_benchmark.py \
  --mode score-projection-spec \
  --projection-spec "$PROJECTION_SPEC_PATH"

echo ""
echo "Job completed at $(date)"
