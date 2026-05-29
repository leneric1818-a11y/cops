#!/bin/bash
# Negative-α paper_v2 matrix at canonical layers — counseling-relevant defensive/resistant steering.
# Covers all 4 models × 4 axes × α=-1.5,-3.0 × 5 variants × N=200 = 16 jobs
set -euo pipefail
cd ${CLUSTER_HOME}/llm_ft_comparison
mkdir -p outputs/steering_eval/paper_v2_negalpha logs

SCRIPT=slurm/generated/cops_evaluate_steering_kiz0.sh
N=200
ALPHAS="-1.5;-3.0"
VARIANTS="baseline;global_blind;global_aware;P_matched;P_mismatched"
QOS_LIST=(basic advanced gpubasic preemptible)
qi=0

submit() {
    local tag="$1" env="$2"
    local qos="${QOS_LIST[$((qi % 4))]}"
    qi=$((qi+1))
    echo ">>> $tag qos=$qos"
    sbatch --qos=$qos --partition=p6,p1,p2,p4 --export="ALL,$env" "$SCRIPT"
}

PAIRS_O=outputs/metrics/persona_pairs_openness_case_aware_1000.jsonl
PAIRS_I=outputs/metrics/persona_pairs_initiative_case_aware_1000.jsonl
PAIRS_C=outputs/metrics/persona_pairs_cooperation_case_aware_1000.jsonl
PAIRS_H=outputs/metrics/persona_pairs_hopefulness_case_aware_1000.jsonl

# === Qwen3-4B (canonical paper_v2 layers) ===
submit "qwen_openness_L15_neg" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=15,AXIS=openness,V_OPEN_DIR=outputs/metrics/layerwise_separability_openness_qwen3_4b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3-4B_persona_pairs_openness_case_aware_1000_n1000_seed42_layers15.npz,CASE_AWARE_PAIRS=${PAIRS_O},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/qwen_openness_L15_neg.jsonl"
submit "qwen_initiative_L9_neg" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=9,AXIS=initiative,V_OPEN_DIR=outputs/metrics/layerwise_separability_initiative_qwen3_4b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3-4B_persona_pairs_initiative_case_aware_1000_n1000_seed42_layers9.npz,CASE_AWARE_PAIRS=${PAIRS_I},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/qwen_initiative_L9_neg.jsonl"
submit "qwen_cooperation_L20_neg" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=20,AXIS=cooperation,V_OPEN_DIR=outputs/metrics/layerwise_separability_cooperation_qwen3_4b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3-4B_persona_pairs_cooperation_case_aware_1000_n999_seed42_layers20.npz,CASE_AWARE_PAIRS=${PAIRS_C},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/qwen_cooperation_L20_neg.jsonl"
submit "qwen_hopefulness_L18_neg" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=18,AXIS=hopefulness,V_OPEN_DIR=outputs/metrics/layerwise_separability_hopefulness_qwen3_4b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3-4B_persona_pairs_hopefulness_case_aware_1000_n1000_seed42_layers18.npz,CASE_AWARE_PAIRS=${PAIRS_H},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/qwen_hopefulness_L18_neg.jsonl"

# === Gemma-4-E4B-it ===
submit "gemma_openness_L3_neg" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=3,AXIS=openness,V_OPEN_DIR=outputs/metrics/layerwise_separability_openness_gemma_4_e4b_it_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_google_gemma-4-E4B-it_persona_pairs_openness_case_aware_1000_n1000_seed42_layers3.npz,CASE_AWARE_PAIRS=${PAIRS_O},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/gemma_openness_L3_neg.jsonl"
submit "gemma_initiative_L13_neg" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=13,AXIS=initiative,V_OPEN_DIR=outputs/metrics/layerwise_separability_initiative_gemma_4_e4b_it_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_google_gemma-4-E4B-it_persona_pairs_initiative_case_aware_1000_n1000_seed42_layers13.npz,CASE_AWARE_PAIRS=${PAIRS_I},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/gemma_initiative_L13_neg.jsonl"
submit "gemma_cooperation_L23_neg" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=23,AXIS=cooperation,V_OPEN_DIR=outputs/metrics/layerwise_separability_cooperation_gemma_4_e4b_it_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_google_gemma-4-E4B-it_persona_pairs_cooperation_case_aware_1000_n999_seed42_layers23.npz,CASE_AWARE_PAIRS=${PAIRS_C},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/gemma_cooperation_L23_neg.jsonl"
submit "gemma_hopefulness_L25_neg" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=25,AXIS=hopefulness,V_OPEN_DIR=outputs/metrics/layerwise_separability_hopefulness_gemma_4_e4b_it_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_google_gemma-4-E4B-it_persona_pairs_hopefulness_case_aware_1000_n1000_seed42_layers25.npz,CASE_AWARE_PAIRS=${PAIRS_H},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/gemma_hopefulness_L25_neg.jsonl"

# === Qwen3.5-9B ===
submit "q35_9b_openness_L20_neg" "MODEL_PATH=Qwen/Qwen3.5-9B,LAYER=20,AXIS=openness,V_OPEN_DIR=outputs/metrics/layerwise_separability_openness_qwen3_5_9b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3.5-9B_persona_pairs_openness_case_aware_1000_n1000_seed42_layers20.npz,CASE_AWARE_PAIRS=${PAIRS_O},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/qwen35_9b_openness_L20_neg.jsonl"
submit "q35_9b_initiative_L20_neg" "MODEL_PATH=Qwen/Qwen3.5-9B,LAYER=20,AXIS=initiative,V_OPEN_DIR=outputs/metrics/layerwise_separability_initiative_qwen3_5_9b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3.5-9B_persona_pairs_initiative_case_aware_1000_n1000_seed42_layers20.npz,CASE_AWARE_PAIRS=${PAIRS_I},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/qwen35_9b_initiative_L20_neg.jsonl"
submit "q35_9b_cooperation_L15_neg" "MODEL_PATH=Qwen/Qwen3.5-9B,LAYER=15,AXIS=cooperation,V_OPEN_DIR=outputs/metrics/layerwise_separability_cooperation_qwen3_5_9b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3.5-9B_persona_pairs_cooperation_case_aware_1000_n999_seed42_layers15.npz,CASE_AWARE_PAIRS=${PAIRS_C},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/qwen35_9b_cooperation_L15_neg.jsonl"
submit "q35_9b_hopefulness_L15_neg" "MODEL_PATH=Qwen/Qwen3.5-9B,LAYER=15,AXIS=hopefulness,V_OPEN_DIR=outputs/metrics/layerwise_separability_hopefulness_qwen3_5_9b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3.5-9B_persona_pairs_hopefulness_case_aware_1000_n1000_seed42_layers15.npz,CASE_AWARE_PAIRS=${PAIRS_H},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/qwen35_9b_hopefulness_L15_neg.jsonl"

# === Mistral-7B-Instruct-v0.3 ===
submit "m7b_openness_L15_neg" "MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3,LAYER=15,AXIS=openness,V_OPEN_DIR=outputs/metrics/layerwise_separability_openness_ministral_14b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_mistralai_Mistral-7B-Instruct-v0.3_persona_pairs_openness_case_aware_1000_n1000_seed42_layers15.npz,CASE_AWARE_PAIRS=${PAIRS_O},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/mistral7b_openness_L15_neg.jsonl"
submit "m7b_initiative_L5_neg" "MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3,LAYER=5,AXIS=initiative,V_OPEN_DIR=outputs/metrics/layerwise_separability_initiative_ministral_14b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_mistralai_Mistral-7B-Instruct-v0.3_persona_pairs_initiative_case_aware_1000_n1000_seed42_layers5.npz,CASE_AWARE_PAIRS=${PAIRS_I},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/mistral7b_initiative_L5_neg.jsonl"
submit "m7b_cooperation_L15_neg" "MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3,LAYER=15,AXIS=cooperation,V_OPEN_DIR=outputs/metrics/layerwise_separability_cooperation_ministral_14b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_mistralai_Mistral-7B-Instruct-v0.3_persona_pairs_cooperation_case_aware_1000_n999_seed42_layers15.npz,CASE_AWARE_PAIRS=${PAIRS_C},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/mistral7b_cooperation_L15_neg.jsonl"
submit "m7b_hopefulness_L20_neg" "MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3,LAYER=20,AXIS=hopefulness,V_OPEN_DIR=outputs/metrics/layerwise_separability_hopefulness_ministral_14b_gpt54mini_1000,VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=outputs/metrics/cops_hidden_cache_v3/v3_mistralai_Mistral-7B-Instruct-v0.3_persona_pairs_hopefulness_case_aware_1000_n1000_seed42_layers20.npz,CASE_AWARE_PAIRS=${PAIRS_H},OUTPUT_PATH=outputs/steering_eval/paper_v2_negalpha/mistral7b_hopefulness_L20_neg.jsonl"

echo ""
echo "negα paper_v2 matrix submitted (16 jobs, all 4 models)"
squeue -u $USER -h | wc -l
