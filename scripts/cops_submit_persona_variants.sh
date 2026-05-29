#!/bin/bash
# Test alternative persona vector constructions A/B/C as steering vectors.
# A: orthogonal residual to v_global; B: cross-persona contrast; C: interaction term.
set -euo pipefail
cd ${CLUSTER_HOME}/llm_ft_comparison
mkdir -p outputs/steering_eval/persona_variants logs

SCRIPT=slurm/generated/cops_evaluate_steering_kiz0.sh
N=200
ALPHAS="3.0"
VARIANTS="baseline;P_matched;P_mismatched;P_matched_A;P_mismatched_A;P_matched_B;P_mismatched_B;P_matched_C;P_mismatched_C"
QOS_LIST=(basic advanced gpubasic preemptible)
qi=0
submit() {
    local tag="$1" env="$2"
    local qos="${QOS_LIST[$((qi % 4))]}"
    qi=$((qi+1))
    echo ">>> $tag qos=$qos"
    sbatch --qos=$qos --partition=p6,p1,p2 --export="ALL,$env" "$SCRIPT"
}

PAIRS_O=outputs/metrics/persona_pairs_openness_case_aware_1000.jsonl
PAIRS_I=outputs/metrics/persona_pairs_initiative_case_aware_1000.jsonl
PAIRS_C=outputs/metrics/persona_pairs_cooperation_case_aware_1000.jsonl
PAIRS_H=outputs/metrics/persona_pairs_hopefulness_case_aware_1000.jsonl

CACHE_Q_O15=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3-4B_persona_pairs_openness_case_aware_1000_n1000_seed42_layers15.npz
CACHE_Q_I9=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3-4B_persona_pairs_initiative_case_aware_1000_n1000_seed42_layers9.npz
CACHE_Q_C20=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3-4B_persona_pairs_cooperation_case_aware_1000_n999_seed42_layers20.npz
CACHE_Q_H18=outputs/metrics/cops_hidden_cache_v3/v3_Qwen_Qwen3-4B_persona_pairs_hopefulness_case_aware_1000_n1000_seed42_layers18.npz
CACHE_G_I13=outputs/metrics/cops_hidden_cache_v3/v3_google_gemma-4-E4B-it_persona_pairs_initiative_case_aware_1000_n1000_seed42_layers13.npz
CACHE_G_C23=outputs/metrics/cops_hidden_cache_v3/v3_google_gemma-4-E4B-it_persona_pairs_cooperation_case_aware_1000_n999_seed42_layers23.npz
CACHE_G_H25=outputs/metrics/cops_hidden_cache_v3/v3_google_gemma-4-E4B-it_persona_pairs_hopefulness_case_aware_1000_n1000_seed42_layers25.npz

V_Q_O=outputs/metrics/layerwise_separability_openness_qwen3_4b_gpt54mini_1000
V_Q_I=outputs/metrics/layerwise_separability_initiative_qwen3_4b_gpt54mini_1000
V_Q_C=outputs/metrics/layerwise_separability_cooperation_qwen3_4b_gpt54mini_1000
V_Q_H=outputs/metrics/layerwise_separability_hopefulness_qwen3_4b_gpt54mini_1000
V_G_I=outputs/metrics/layerwise_separability_initiative_gemma_4_e4b_it_gpt54mini_1000
V_G_C=outputs/metrics/layerwise_separability_cooperation_gemma_4_e4b_it_gpt54mini_1000
V_G_H=outputs/metrics/layerwise_separability_hopefulness_gemma_4_e4b_it_gpt54mini_1000

submit "qwen_openness_L15_pv" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=15,AXIS=openness,V_OPEN_DIR=${V_Q_O},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_Q_O15},CASE_AWARE_PAIRS=${PAIRS_O},OUTPUT_PATH=outputs/steering_eval/persona_variants/qwen_openness_L15_pv.jsonl"
submit "qwen_initiative_L9_pv" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=9,AXIS=initiative,V_OPEN_DIR=${V_Q_I},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_Q_I9},CASE_AWARE_PAIRS=${PAIRS_I},OUTPUT_PATH=outputs/steering_eval/persona_variants/qwen_initiative_L9_pv.jsonl"
submit "qwen_cooperation_L20_pv" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=20,AXIS=cooperation,V_OPEN_DIR=${V_Q_C},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_Q_C20},CASE_AWARE_PAIRS=${PAIRS_C},OUTPUT_PATH=outputs/steering_eval/persona_variants/qwen_cooperation_L20_pv.jsonl"
submit "qwen_hopefulness_L18_pv" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=18,AXIS=hopefulness,V_OPEN_DIR=${V_Q_H},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_Q_H18},CASE_AWARE_PAIRS=${PAIRS_H},OUTPUT_PATH=outputs/steering_eval/persona_variants/qwen_hopefulness_L18_pv.jsonl"
submit "gemma_initiative_L13_pv" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=13,AXIS=initiative,V_OPEN_DIR=${V_G_I},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_G_I13},CASE_AWARE_PAIRS=${PAIRS_I},OUTPUT_PATH=outputs/steering_eval/persona_variants/gemma_initiative_L13_pv.jsonl"
submit "gemma_cooperation_L23_pv" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=23,AXIS=cooperation,V_OPEN_DIR=${V_G_C},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_G_C23},CASE_AWARE_PAIRS=${PAIRS_C},OUTPUT_PATH=outputs/steering_eval/persona_variants/gemma_cooperation_L23_pv.jsonl"
submit "gemma_hopefulness_L25_pv" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=25,AXIS=hopefulness,V_OPEN_DIR=${V_G_H},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_G_H25},CASE_AWARE_PAIRS=${PAIRS_H},OUTPUT_PATH=outputs/steering_eval/persona_variants/gemma_hopefulness_L25_pv.jsonl"

echo ""
echo "persona_variants matrix submitted (7 jobs)"
squeue -u $USER -h | wc -l
