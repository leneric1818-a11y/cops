#!/bin/bash
#SBATCH --job-name=qwen3_steering
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
echo "Qwen 3 Steering Vector Sweep at $(date)"
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
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SCRATCH_ROOT/.conda_pkgs}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$CONDA_ENVS_DIRS/llm_ft_comparison}"
ENV_READY_FILE="${ENV_READY_FILE:-$CONDA_ENV_DIR/.llm_ft_comparison_env_ready}"
TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_DIR/pip}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$SUBMIT_DIR/requirements-steering.txt}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

export CONDA_ENVS_DIRS
export CONDA_PKGS_DIRS
export HF_HOME="${HF_HOME:-$CACHE_DIR/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_DIR/triton}"
export PIP_CACHE_DIR
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=0
export PIP_NO_USER=0
export PIP_DISABLE_PIP_VERSION_CHECK=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=disabled

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" \
         "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" \
         "$CONDA_ENVS_DIRS" "$CONDA_PKGS_DIRS"

if [ ! -d "$SUBMIT_DIR" ]; then
  echo "ERROR: hardcoded project dir does not exist: $SUBMIT_DIR"
  exit 1
fi

cd "$SUBMIT_DIR" || exit 1

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "ERROR: requirements file not found: $REQUIREMENTS_FILE"
  exit 1
fi

if [ ! -x "$CONDA_ENV_DIR/bin/python" ] || [ ! -x "$CONDA_ENV_DIR/bin/pip" ]; then
  echo "ERROR: prepared env not found: $CONDA_ENV_DIR"
  exit 1
fi

export CONDA_PREFIX="$CONDA_ENV_DIR"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
hash -r

echo "Python:       $(which python)"
echo "Pip:          $(which pip)"
echo "Pip version:  $(pip --version)"

if [ "$(which pip)" != "$CONDA_ENV_DIR/bin/pip" ]; then
  echo "ERROR: pip is not from the activated conda env."
  exit 1
fi

if [ ! -f "$ENV_READY_FILE" ]; then
  echo "Installing repo dependencies into $CONDA_ENV_DIR"
  pip install --force-reinstall --no-cache-dir "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"
  pip install -r "$REQUIREMENTS_FILE"
  touch "$ENV_READY_FILE"
else
  echo "Using existing prepared conda env at $CONDA_ENV_DIR"
fi

python - <<'PY'
import sys
import torch

print(f"Torch:        {torch.__version__}")
print(f"Torch CUDA:   {torch.version.cuda}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"CUDA device:  {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("ERROR: torch.cuda.is_available() is false; refusing to run on CPU.")
PY

export PYTHONPATH="$SUBMIT_DIR/src:${PYTHONPATH:-}"

DATA_PATH="${DATA_PATH:-data/processed/ncp_eval.jsonl}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
CONTEXT_FIELD="${CONTEXT_FIELD:-}"
POSITIVE_INSTRUCTION="${POSITIVE_INSTRUCTION:-Der Klient reagiert eher defensiv und hält sich eher zurück.}"
NEGATIVE_INSTRUCTION="${NEGATIVE_INSTRUCTION:-Der Klient reagiert eher offen und spricht frei über Gedanken und Gefühle.}"
NEUTRAL_INSTRUCTION="${NEUTRAL_INSTRUCTION:-Der Klient antwortet auf die letzte Aussage des Beraters.}"
TRAIN_LIMIT="${TRAIN_LIMIT:-64}"
EVAL_LIMIT="${EVAL_LIMIT:-50}"
EVAL_OFFSET="${EVAL_OFFSET:-$TRAIN_LIMIT}"
STEERING_MODE="${STEERING_MODE:-gaussian}"
GAUSSIAN_CENTER="${GAUSSIAN_CENTER:-18}"
GAUSSIAN_SIGMA="${GAUSSIAN_SIGMA:-3.0}"
GAUSSIAN_MIN_WEIGHT="${GAUSSIAN_MIN_WEIGHT:-0.05}"
ALPHAS="${ALPHAS:-1.0,1.5,2.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
DO_SAMPLE="${DO_SAMPLE:-1}"
SEED="${SEED:-42}"
LAYERS="${LAYERS:-12,15,18,21,24}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/metrics/steering_gaussian_defensive_open_${SLURM_JOB_ID:-manual}.jsonl}"
SAVE_VECTOR_DIR="${SAVE_VECTOR_DIR:-outputs/metrics/steering_vectors_gaussian_defensive_open_${SLURM_JOB_ID:-manual}}"

echo "Project dir:  $SUBMIT_DIR"
echo "Data:         $DATA_PATH"
echo "Model:        $MODEL_PATH"
echo "Adapter:      ${ADAPTER_PATH:-<none>}"
echo "Layers:       $LAYERS"
echo "Mode:         $STEERING_MODE"
echo "Gaussian:     center=$GAUSSIAN_CENTER sigma=$GAUSSIAN_SIGMA min=$GAUSSIAN_MIN_WEIGHT"
echo "Alphas:       $ALPHAS"
echo "Train/Eval:   $TRAIN_LIMIT / $EVAL_LIMIT"
echo "Output:       $OUTPUT_PATH"
echo ""

cmd=(
  python -B scripts/steering_vector_experiment.py
  --data-path "$DATA_PATH"
  --model-path "$MODEL_PATH"
  --positive-instruction "$POSITIVE_INSTRUCTION"
  --negative-instruction "$NEGATIVE_INSTRUCTION"
  --neutral-instruction "$NEUTRAL_INSTRUCTION"
  --train-limit "$TRAIN_LIMIT"
  --eval-limit "$EVAL_LIMIT"
  --eval-offset "$EVAL_OFFSET"
  --require-last-speaker Berater
  --min-turns 3
  --max-context-chars 1000
  --steering-mode "$STEERING_MODE"
  --layers "$LAYERS"
  --gaussian-center "$GAUSSIAN_CENTER"
  --gaussian-sigma "$GAUSSIAN_SIGMA"
  --gaussian-min-weight "$GAUSSIAN_MIN_WEIGHT"
  --alphas "$ALPHAS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --repetition-penalty "$REPETITION_PENALTY"
  --seed "$SEED"
  --save-vector-dir "$SAVE_VECTOR_DIR"
  --output-path "$OUTPUT_PATH"
  --trim-to-first-utterance
)

if [ -n "$ADAPTER_PATH" ]; then
  cmd+=(--adapter-path "$ADAPTER_PATH")
fi

if [ -n "$CONTEXT_FIELD" ]; then
  cmd+=(--context-field "$CONTEXT_FIELD")
fi

if [ "$DO_SAMPLE" -eq 1 ]; then
  cmd+=(--do-sample)
else
  cmd+=(--no-do-sample)
fi

"${cmd[@]}"

echo ""
echo "Job completed at $(date)"
