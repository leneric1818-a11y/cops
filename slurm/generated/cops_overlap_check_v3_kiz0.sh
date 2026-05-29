#!/bin/bash
#SBATCH --job-name=cops_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpubasic
#SBATCH --output=logs/cops_v3_%j.out
#SBATCH --error=logs/cops_v3_%j.err

set -euo pipefail

echo "=== COPS overlap v3 (response-mean pooling) at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"

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

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TMPDIR" "$SUBMIT_DIR/logs" "$SUBMIT_DIR/outputs/metrics"
cd "$SUBMIT_DIR"
export CONDA_PREFIX="$CONDA_ENV_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r
export PYTHONPATH="$SUBMIT_DIR/src:${PYTHONPATH:-}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA")
print(f"Torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
PAIRS_PATH="${PAIRS_PATH:-outputs/metrics/persona_pairs_openness_gpt54mini_1000.jsonl}"
V_OPEN_DIR="${V_OPEN_DIR:-outputs/metrics/layerwise_separability_openness_qwen3_4b_gpt54mini_1000}"
LAYERS="${LAYERS:-1}"
LAYERS="${LAYERS//;/,}"
N_PAIRS="${N_PAIRS:-1000}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/metrics/cops_overlap_check_v3_${SLURM_JOB_ID:-manual}.json}"

echo "MODEL: $MODEL_PATH  LAYERS: $LAYERS  N: $N_PAIRS  OUT: $OUTPUT_PATH"

python -u -B scripts/cops_overlap_check_v3.py \
    --model-path "$MODEL_PATH" \
    --pairs-path "$PAIRS_PATH" \
    --v-open-dir "$V_OPEN_DIR" \
    --layers "$LAYERS" \
    --n-pairs "$N_PAIRS" \
    --output-path "$OUTPUT_PATH"

echo "Done at $(date). Output: $OUTPUT_PATH"
