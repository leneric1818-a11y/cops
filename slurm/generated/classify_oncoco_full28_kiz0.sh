#!/bin/bash
#SBATCH --job-name=oncoco_classify_full28
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpubasic
#SBATCH --output=logs/oncoco_classify_full28_%j.out
#SBATCH --error=logs/oncoco_classify_full28_%j.err

set -euo pipefail

echo "=================================================================="
echo "OnCoCo response classification (full 28-class probs) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo 'no gpu detected')"
echo "=================================================================="

module purge 2>/dev/null || true
module load cuda/cuda-12.6 2>/dev/null || true

SUBMIT_DIR="${CLUSTER_HOME}/llm_ft_comparison"
SCRATCH_ROOT="${SCRATCH_ROOT:-${CLUSTER_HOME}}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH_ROOT/.cache}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$SCRATCH_ROOT/.conda_envs/llm_ft_comparison}"
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
         "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TMPDIR" \
         "$SUBMIT_DIR/logs" \
         "$SUBMIT_DIR/outputs/steering_eval"

cd "$SUBMIT_DIR"

export CONDA_PREFIX="$CONDA_ENV_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r
export PYTHONPATH="$SUBMIT_DIR/src:${PYTHONPATH:-}"

echo "Python: $(which python)"
python - <<'PY'
import torch
print(f"Torch: {torch.__version__}  CUDA avail: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: no CUDA — refusing to run on CPU.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

# ---------------------------------------------------------------------------
# Config (override via sbatch --export=ALL,VAR=val)
MODEL_PATH="${MODEL_PATH:-models/xlm-roberta-large-OnCoCo-DE-EN}"
INPUT_DIRS="${INPUT_DIRS:-outputs/steering_eval/paper_v2 outputs/steering_eval/paper_v2_negalpha}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/steering_eval/oncoco_labels_full28.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TOP_K="${TOP_K:-28}"
SKIP_GLOB="${SKIP_GLOB:-*_all.jsonl}"

echo "Model:      $MODEL_PATH"
echo "Input dirs: $INPUT_DIRS"
echo "Output:     $OUTPUT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Top-k:      $TOP_K"

# ---------------------------------------------------------------------------
# shellcheck disable=SC2086
python -u -B scripts/classify_responses_oncoco.py \
    --model-path "$MODEL_PATH" \
    --input-dirs $INPUT_DIRS \
    --output "$OUTPUT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --top-k "$TOP_K" \
    --skip-glob "$SKIP_GLOB"

echo ""
echo "Done at $(date)."
echo "Output: $OUTPUT_PATH"
echo "Lines:  $(wc -l < "$OUTPUT_PATH")"
