#!/bin/bash
# Layer-Sweep for COPS: empirical steering-effect per layer (not AUC-based).
# 2 models × 4 axes × 5 layers × α=1.5 × (baseline + global_blind) × N=50 ctx × 1 seed
set -euo pipefail
cd ${CLUSTER_HOME}/llm_ft_comparison
mkdir -p outputs/steering_eval/layer_sweep logs

SCRIPT=slurm/generated/cops_evaluate_steering_kiz0.sh
N=50
QOS_LIST=(basic advanced gpubasic preemptible)
qi=0

submit_job() {
    local tag="$1" env="$2"
    local qos="${QOS_LIST[$((qi % ${#QOS_LIST[@]}))]}"
    qi=$((qi+1))
    echo ">>> $tag  qos=$qos"
    sbatch --qos="$qos" --export="ALL,$env" "$SCRIPT"
}

# Matrix: model_key model_path axis v_dir layers
run_matrix() {
    local model_key=$1 model_path=$2 axis=$3 v_dir=$4 layers_csv=$5
    IFS=',' read -ra LAYERS <<< "$layers_csv"
    for L in "${LAYERS[@]}"; do
        local out="outputs/steering_eval/layer_sweep/${model_key}_${axis}_L${L}.jsonl"
        submit_job "${model_key}_${axis}_L${L}" "MODEL_PATH=${model_path},LAYER=${L},AXIS=${axis},V_OPEN_DIR=${v_dir},VARIANTS=baseline;global_blind,ALPHAS=1.5,SEEDS=42,HELD_OUT_COUNT=${N},HELD_OUT_START=400,OUTPUT_PATH=${out}"
    done
}

# Qwen3-4B
run_matrix qwen Qwen/Qwen3-4B openness     outputs/metrics/layerwise_separability_openness_qwen3_4b_gpt54mini_1000     "1,5,10,15,20"
run_matrix qwen Qwen/Qwen3-4B initiative   outputs/metrics/layerwise_separability_initiative_qwen3_4b_gpt54mini_1000   "5,9,14,20,25"
run_matrix qwen Qwen/Qwen3-4B cooperation  outputs/metrics/layerwise_separability_cooperation_qwen3_4b_gpt54mini_1000  "5,9,14,20,25"
run_matrix qwen Qwen/Qwen3-4B hopefulness  outputs/metrics/layerwise_separability_hopefulness_qwen3_4b_gpt54mini_1000  "3,7,12,18,25"

# Gemma-4-E4B-it
run_matrix gemma google/gemma-4-E4B-it openness     outputs/metrics/layerwise_separability_openness_gemma_4_e4b_it_gpt54mini_1000     "3,8,13,20,28"
run_matrix gemma google/gemma-4-E4B-it initiative   outputs/metrics/layerwise_separability_initiative_gemma_4_e4b_it_gpt54mini_1000   "3,8,13,20,28"
run_matrix gemma google/gemma-4-E4B-it cooperation  outputs/metrics/layerwise_separability_cooperation_gemma_4_e4b_it_gpt54mini_1000  "8,15,23,28,32"
run_matrix gemma google/gemma-4-E4B-it hopefulness  outputs/metrics/layerwise_separability_hopefulness_gemma_4_e4b_it_gpt54mini_1000  "5,13,20,25,32"

echo ""
echo "Layer-sweep jobs submitted (40 total)."
squeue -u $USER | wc -l
