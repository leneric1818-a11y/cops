#!/bin/bash
#SBATCH --job-name=cops_overlap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpubasic
#SBATCH --output=logs/cops_overlap_%j.out
#SBATCH --error=logs/cops_overlap_%j.err

set -euo pipefail

echo "=================================================================="
echo "COPS overlap check (v_open vs content-probe directions) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo 'no gpu detected')"
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
         "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TMPDIR" \
         "$SUBMIT_DIR/logs" "$SUBMIT_DIR/outputs/metrics"

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
# Run config
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
CONTEXTS_PATH="${CONTEXTS_PATH:-data/processed/cops_contexts.jsonl}"
V_OPEN_DIR="${V_OPEN_DIR:-outputs/metrics/layerwise_separability_openness_qwen3_4b_gpt54mini_1000}"
LAYERS="${LAYERS:-1,5,9,13,18}"
N_CONTEXTS="${N_CONTEXTS:-300}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/metrics/cops_overlap_check_${SLURM_JOB_ID:-manual}.json}"

echo "MODEL_PATH:    $MODEL_PATH"
echo "CONTEXTS_PATH: $CONTEXTS_PATH"
echo "V_OPEN_DIR:    $V_OPEN_DIR"
echo "LAYERS:        $LAYERS"
echo "N_CONTEXTS:    $N_CONTEXTS"
echo "OUTPUT_PATH:   $OUTPUT_PATH"

# ---------------------------------------------------------------------------
# Work
python -u -B scripts/cops_overlap_check.py \
    --model-path "$MODEL_PATH" \
    --contexts-path "$CONTEXTS_PATH" \
    --v-open-dir "$V_OPEN_DIR" \
    --layers "$LAYERS" \
    --n-contexts "$N_CONTEXTS" \
    --output-path "$OUTPUT_PATH"

echo ""
echo "Done at $(date). Output: $OUTPUT_PATH"
