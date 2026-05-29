#!/bin/bash
#SBATCH --job-name=layersep_initiative
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p0
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=preemptible

set -euo pipefail

echo "=================================================================="
echo "Qwen 3 Layerwise Separability at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "=================================================================="

module purge 2>/dev/null || true
module load cuda/cuda-12.6 2>/dev/null || true

SUBMIT_DIR="${CLUSTER_HOME}/llm_ft_comparison"
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

cd "$SUBMIT_DIR" || exit 1

export CONDA_PREFIX="$CONDA_ENV_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r

echo "Python:       $(which python)"
echo "Pip:          $(which pip)"
echo "Pip version:  $(pip --version)"

python - <<'PY'
import torch
print(f"Torch:        {torch.__version__}")
print(f"Torch CUDA:   {torch.version.cuda}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:  {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("ERROR: torch.cuda.is_available() is false; refusing to run on CPU.")
PY

export PYTHONPATH="$SUBMIT_DIR/src:${PYTHONPATH:-}"

DATA_PATH="${DATA_PATH:-outputs/metrics/persona_pairs_initiative_gpt54mini_1000_flat.jsonl}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
LAYERS="${LAYERS:-all}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_FRACTION="${TEST_FRACTION:-0.25}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MAX_EXAMPLES_PER_STYLE="${MAX_EXAMPLES_PER_STYLE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/metrics/layerwise_separability_initiative_gpt54mini_1000}"
STYLES="${STYLES:-reactive,explorative}"

echo "Project dir:  $SUBMIT_DIR"
echo "Data:         $DATA_PATH"
echo "Model:        $MODEL_PATH"
echo "Styles:       $STYLES"
echo "Layers:       $LAYERS"
echo "Batch size:   $BATCH_SIZE"
echo "Test frac:    $TEST_FRACTION"
echo "Output dir:   $OUTPUT_DIR"
echo ""

python -B scripts/compute_layerwise_separability.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --styles "$STYLES" \
  --layers "$LAYERS" \
  --batch-size "$BATCH_SIZE" \
  --test-fraction "$TEST_FRACTION" \
  --split-seed "$SPLIT_SEED" \
  --max-examples-per-style "$MAX_EXAMPLES_PER_STYLE"

echo ""
echo "Job completed at $(date)"
