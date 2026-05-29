#!/bin/bash
#SBATCH --job-name=layersep_ministral14b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpubasic

set -euo pipefail

echo "=================================================================="
echo "Ministral-3-14B Layerwise Separability at $(date)"
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

MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3"
LAYERS="all"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEST_FRACTION="${TEST_FRACTION:-0.25}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MAX_EXAMPLES_PER_STYLE="${MAX_EXAMPLES_PER_STYLE:-0}"

AXIS="${AXIS:-openness}"

case "$AXIS" in
  openness)
    DATA_PATH="outputs/metrics/persona_pairs_openness_gpt54mini_1000_flat.jsonl"
    STYLES="defensive,open"
    OUTPUT_DIR="outputs/metrics/layerwise_separability_openness_ministral_14b_gpt54mini_1000"
    ;;
  initiative)
    DATA_PATH="outputs/metrics/persona_pairs_initiative_gpt54mini_1000_flat.jsonl"
    STYLES="reactive,explorative"
    OUTPUT_DIR="outputs/metrics/layerwise_separability_initiative_ministral_14b_gpt54mini_1000"
    ;;
  cooperation)
    DATA_PATH="outputs/metrics/persona_pairs_cooperation_gpt54mini_1000_flat.jsonl"
    STYLES="resistant,cooperative"
    OUTPUT_DIR="outputs/metrics/layerwise_separability_cooperation_ministral_14b_gpt54mini_1000"
    ;;
  hopefulness)
    DATA_PATH="outputs/metrics/persona_pairs_hopefulness_gpt54mini_1000_flat.jsonl"
    STYLES="resigned,hopeful"
    OUTPUT_DIR="outputs/metrics/layerwise_separability_hopefulness_ministral_14b_gpt54mini_1000"
    ;;
  *)
    echo "ERROR: Unknown AXIS='$AXIS'. Must be one of: openness, initiative, cooperation, hopefulness."
    exit 1
    ;;
esac

echo "Project dir:  $SUBMIT_DIR"
echo "Axis:         $AXIS"
echo "Data:         $DATA_PATH"
echo "Model:        $MODEL_PATH"
echo "Styles:       $STYLES"
echo "Layers:       $LAYERS"
echo "Batch size:   $BATCH_SIZE"
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
  --max-examples-per-style "$MAX_EXAMPLES_PER_STYLE" \
  --trust-remote-code

echo ""
echo "Job completed at $(date)"
