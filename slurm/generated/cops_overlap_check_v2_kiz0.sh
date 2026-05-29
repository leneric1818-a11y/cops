#!/bin/bash
#SBATCH --job-name=cops_overlap_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpubasic
#SBATCH --output=logs/cops_overlap_v2_%j.out
#SBATCH --error=logs/cops_overlap_v2_%j.err

set -euo pipefail

echo "=================================================================="
echo "COPS overlap check v2 (response-token probes) at $(date)"
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
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
PAIRS_PATH="${PAIRS_PATH:-outputs/metrics/persona_pairs_openness_gpt54mini_1000.jsonl}"
NCP_PATH="${NCP_PATH:-data/processed/ncp_eval.jsonl}"
V_OPEN_DIR="${V_OPEN_DIR:-outputs/metrics/layerwise_separability_openness_qwen3_4b_gpt54mini_1000}"
LAYERS="${LAYERS:-1,5,9,13,18}"
# Allow semicolon/colon delimiters (commas confuse sbatch --export)
LAYERS="${LAYERS//;/,}"
LAYERS="${LAYERS//:/,}"
N_PAIRS="${N_PAIRS:-200}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/metrics/cops_overlap_check_v2_${SLURM_JOB_ID:-manual}.json}"

echo "MODEL_PATH:  $MODEL_PATH"
echo "PAIRS_PATH:  $PAIRS_PATH"
echo "NCP_PATH:    $NCP_PATH"
echo "LAYERS:      $LAYERS"
echo "N_PAIRS:     $N_PAIRS"
echo "OUTPUT_PATH: $OUTPUT_PATH"

python -u -B scripts/cops_overlap_check_v2.py \
    --model-path "$MODEL_PATH" \
    --pairs-path "$PAIRS_PATH" \
    --ncp-path "$NCP_PATH" \
    --v-open-dir "$V_OPEN_DIR" \
    --layers "$LAYERS" \
    --n-pairs "$N_PAIRS" \
    --output-path "$OUTPUT_PATH"

echo ""
echo "Done at $(date). Output: $OUTPUT_PATH"
