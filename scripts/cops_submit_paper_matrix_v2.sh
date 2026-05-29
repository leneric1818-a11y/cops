#!/bin/bash
# Full-matrix COPS paper v2: new canonical layers from layer-sweep, case-aware for all 4 axes.
set -euo pipefail
cd ${CLUSTER_HOME}/llm_ft_comparison
mkdir -p outputs/steering_eval/paper_v2 logs

SCRIPT=slurm/generated/cops_evaluate_steering_kiz0.sh
N=200
ALPHAS="1.5;3.0"
VARIANTS="baseline;global_blind;global_aware;P_matched;P_mismatched"
QOS_LIST=(basic advanced gpubasic preemptible)
qi=0
submit() {
    local tag="$1" env="$2"
    local qos="${QOS_LIST[$((qi % 4))]}"
    qi=$((qi+1))
    echo ">>> $tag qos=$qos"
    sbatch --qos=$qos --partition=p6,p1,p2 --export="ALL,$env" "$SCRIPT"
}

PAIRS_AWARE_OPENNESS=outputs/metrics/persona_pairs_openness_case_aware_1000.jsonl
PAIRS_AWARE_INITIATIVE=outputs/metrics/persona_pairs_initiative_case_aware_1000.jsonl
PAIRS_AWARE_COOPERATION=outputs/metrics/persona_pairs_cooperation_case_aware_1000.jsonl
PAIRS_AWARE_HOPEFULNESS=outputs/metrics/persona_pairs_hopefulness_case_aware_1000.jsonl

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

# Qwen openness L15: 3 seeds (parallelized as 3 jobs)
for SEED in 42 123 2024; do
    submit "qwen_openness_L15_seed${SEED}" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=15,AXIS=openness,V_OPEN_DIR=${V_Q_O},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=${SEED},HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_Q_O15},CASE_AWARE_PAIRS=${PAIRS_AWARE_OPENNESS},OUTPUT_PATH=outputs/steering_eval/paper_v2/qwen_openness_L15_seed${SEED}.jsonl"
done

# Qwen cross-axis, 1 seed each
submit "qwen_initiative_L9_v2" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=9,AXIS=initiative,V_OPEN_DIR=${V_Q_I},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_Q_I9},CASE_AWARE_PAIRS=${PAIRS_AWARE_INITIATIVE},OUTPUT_PATH=outputs/steering_eval/paper_v2/qwen_initiative_L9.jsonl"
submit "qwen_cooperation_L20_v2" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=20,AXIS=cooperation,V_OPEN_DIR=${V_Q_C},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_Q_C20},CASE_AWARE_PAIRS=${PAIRS_AWARE_COOPERATION},OUTPUT_PATH=outputs/steering_eval/paper_v2/qwen_cooperation_L20.jsonl"
submit "qwen_hopefulness_L18_v2" "MODEL_PATH=Qwen/Qwen3-4B,LAYER=18,AXIS=hopefulness,V_OPEN_DIR=${V_Q_H},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_Q_H18},CASE_AWARE_PAIRS=${PAIRS_AWARE_HOPEFULNESS},OUTPUT_PATH=outputs/steering_eval/paper_v2/qwen_hopefulness_L18.jsonl"

# Gemma cross-axis
submit "gemma_initiative_L13_v2" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=13,AXIS=initiative,V_OPEN_DIR=${V_G_I},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_G_I13},CASE_AWARE_PAIRS=${PAIRS_AWARE_INITIATIVE},OUTPUT_PATH=outputs/steering_eval/paper_v2/gemma_initiative_L13.jsonl"
submit "gemma_cooperation_L23_v2" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=23,AXIS=cooperation,V_OPEN_DIR=${V_G_C},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_G_C23},CASE_AWARE_PAIRS=${PAIRS_AWARE_COOPERATION},OUTPUT_PATH=outputs/steering_eval/paper_v2/gemma_cooperation_L23.jsonl"
submit "gemma_hopefulness_L25_v2" "MODEL_PATH=google/gemma-4-E4B-it,LAYER=25,AXIS=hopefulness,V_OPEN_DIR=${V_G_H},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${CACHE_G_H25},CASE_AWARE_PAIRS=${PAIRS_AWARE_HOPEFULNESS},OUTPUT_PATH=outputs/steering_eval/paper_v2/gemma_hopefulness_L25.jsonl"

echo ""
echo "paper_v2 matrix submitted (9 jobs)"
squeue -u $USER -h | wc -l
