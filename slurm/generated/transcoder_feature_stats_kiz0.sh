#!/bin/bash
#SBATCH --job-name=qwen3_featstats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpubasic

set -euo pipefail

echo "=================================================================="
echo "Qwen 3 Transcoder Feature Stats at $(date)"
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

DATA_PATH="${DATA_PATH:-outputs/metrics/steering_contrast_pairs_gpt54mini_200_flat.jsonl}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
LAYERS="${LAYERS:-18}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOP_K="${TOP_K:-200}"
MAX_EXAMPLES_PER_STYLE="${MAX_EXAMPLES_PER_STYLE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/metrics/transcoder_feature_stats_gpt54mini_200_layer18}"
TRANSCODER_REPO="${TRANSCODER_REPO:-mwhanna/qwen3-4b-transcoders}"

echo "Project dir:  $SUBMIT_DIR"
echo "Data:         $DATA_PATH"
echo "Model:        $MODEL_PATH"
echo "Layers:       $LAYERS"
echo "Batch size:   $BATCH_SIZE"
echo "Top K:        $TOP_K"
echo "Output dir:   $OUTPUT_DIR"
echo ""

python -B scripts/compute_transcoder_feature_stats.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --layers "$LAYERS" \
  --batch-size "$BATCH_SIZE" \
  --top-k "$TOP_K" \
  --max-examples-per-style "$MAX_EXAMPLES_PER_STYLE" \
  --transcoder-repo "$TRANSCODER_REPO"

echo ""
echo "Job completed at $(date)"
