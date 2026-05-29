#!/bin/bash
# Layer-Sweep for 2 new models: Qwen3.5-9B + Mistral-7B-Instruct-v0.3
# 2 models × 4 axes × 5 layers × α=1.5 × baseline+global_blind × N=50 = 40 jobs
set -euo pipefail
cd ${CLUSTER_HOME}/llm_ft_comparison
mkdir -p outputs/steering_eval/layer_sweep logs

SCRIPT=slurm/generated/cops_evaluate_steering_kiz0.sh
N=50
QOS_LIST=(basic advanced gpubasic preemptible)
qi=0

submit() {
    local tag="$1" env="$2"
    local qos="${QOS_LIST[$((qi % 4))]}"
    qi=$((qi+1))
    echo ">>> $tag qos=$qos"
    sbatch --qos=$qos --partition=p6,p1,p2,p4 --export="ALL,$env" "$SCRIPT"
}

run_sweep() {
    local model_key=$1 model_path=$2 axis=$3 v_dir=$4 layers_csv=$5
    IFS=',' read -ra LAYERS <<< "$layers_csv"
    for L in "${LAYERS[@]}"; do
        local out="outputs/steering_eval/layer_sweep/${model_key}_${axis}_L${L}.jsonl"
        submit "${model_key}_${axis}_L${L}" "MODEL_PATH=${model_path},LAYER=${L},AXIS=${axis},V_OPEN_DIR=${v_dir},VARIANTS=baseline;global_blind,ALPHAS=1.5,SEEDS=42,HELD_OUT_COUNT=${N},HELD_OUT_START=400,OUTPUT_PATH=${out}"
    done
}

# Qwen3.5-9B (Qwen/Qwen3.5-9B)
run_sweep qwen35_9b Qwen/Qwen3.5-9B openness     outputs/metrics/layerwise_separability_openness_qwen3_5_9b_gpt54mini_1000     "5,10,15,20,25"
run_sweep qwen35_9b Qwen/Qwen3.5-9B initiative   outputs/metrics/layerwise_separability_initiative_qwen3_5_9b_gpt54mini_1000   "5,10,15,20,25"
run_sweep qwen35_9b Qwen/Qwen3.5-9B cooperation  outputs/metrics/layerwise_separability_cooperation_qwen3_5_9b_gpt54mini_1000  "5,10,15,20,25"
run_sweep qwen35_9b Qwen/Qwen3.5-9B hopefulness  outputs/metrics/layerwise_separability_hopefulness_qwen3_5_9b_gpt54mini_1000  "5,10,15,20,25"

# Mistral-7B-Instruct-v0.3 (mistralai/Mistral-7B-Instruct-v0.3) — uses ministral_14b folder names
run_sweep mistral7b mistralai/Mistral-7B-Instruct-v0.3 openness     outputs/metrics/layerwise_separability_openness_ministral_14b_gpt54mini_1000     "5,10,15,20,25"
run_sweep mistral7b mistralai/Mistral-7B-Instruct-v0.3 initiative   outputs/metrics/layerwise_separability_initiative_ministral_14b_gpt54mini_1000   "5,10,15,20,25"
run_sweep mistral7b mistralai/Mistral-7B-Instruct-v0.3 cooperation  outputs/metrics/layerwise_separability_cooperation_ministral_14b_gpt54mini_1000  "5,10,15,20,25"
run_sweep mistral7b mistralai/Mistral-7B-Instruct-v0.3 hopefulness  outputs/metrics/layerwise_separability_hopefulness_ministral_14b_gpt54mini_1000  "5,10,15,20,25"

echo ""
echo "Layer-sweep newmodels submitted (40 jobs)"
squeue -u $USER -h | wc -l
