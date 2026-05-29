#!/bin/bash
#SBATCH --job-name=cops_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpubasic
#SBATCH --output=logs/cops_eval_%j.out
#SBATCH --error=logs/cops_eval_%j.err

set -euo pipefail
echo "=== COPS steering eval at $(date)  Job ${SLURM_JOB_ID:-?}  Node $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"

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
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TMPDIR" "$SUBMIT_DIR/logs" "$SUBMIT_DIR/outputs/steering_eval"
cd "$SUBMIT_DIR"
export CONDA_PREFIX="$CONDA_ENV_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r
export PYTHONPATH="$SUBMIT_DIR/src:$SUBMIT_DIR/scripts:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:?must set}"
LAYER="${LAYER:?must set}"
AXIS="${AXIS:?must set}"
V_OPEN_DIR="${V_OPEN_DIR:?must set}"
VARIANTS="${VARIANTS:-baseline,global_blind,global_aware,P_matched,P_mismatched}"
ALPHAS="${ALPHAS:-0.5,1.5,3.0}"
SEEDS="${SEEDS:-42}"
# Allow semicolon delimiter (commas confuse sbatch --export)
VARIANTS="${VARIANTS//;/,}"
ALPHAS="${ALPHAS//;/,}"
SEEDS="${SEEDS//;/,}"
HELD_OUT_COUNT="${HELD_OUT_COUNT:-200}"
HELD_OUT_START="${HELD_OUT_START:-200}"
OUTPUT_PATH="${OUTPUT_PATH:?must set}"
CASE_AWARE_CACHE="${CASE_AWARE_CACHE:-}"
CASE_AWARE_PAIRS="${CASE_AWARE_PAIRS:-}"
PERSONA_INSTRUCTION="${PERSONA_INSTRUCTION:-}"

echo "MODEL=$MODEL_PATH LAYER=$LAYER AXIS=$AXIS SEEDS=$SEEDS"
echo "VARIANTS=$VARIANTS ALPHAS=$ALPHAS"
echo "OUT=$OUTPUT_PATH"

python -u -B scripts/cops_evaluate_steering.py \
    --model-path "$MODEL_PATH" \
    --layer "$LAYER" \
    --axis "$AXIS" \
    --v-open-dir "$V_OPEN_DIR" \
    --variants="$VARIANTS" \
    --alphas="$ALPHAS" \
    --seeds="$SEEDS" \
    --held-out-count "$HELD_OUT_COUNT" \
    --held-out-start "$HELD_OUT_START" \
    --case-aware-cache "$CASE_AWARE_CACHE" \
    --case-aware-pairs "$CASE_AWARE_PAIRS" \
    --persona-instruction "$PERSONA_INSTRUCTION" \
    --output-path "$OUTPUT_PATH"

echo "Done at $(date). Output: $OUTPUT_PATH"
