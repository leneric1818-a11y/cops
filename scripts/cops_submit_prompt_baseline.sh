#!/bin/bash
# Prompt-baseline ablation v2: each file has baseline+global_blind so judge can pair.
# For each headline stem, run ONE file:
#   - WITH persona instruction (baseline, global_blind α=-3.0)
# The WITHOUT-prompt condition is recycled from outputs/steering_eval/paper_v2_negalpha/
# (so we save ~50% of GPU time). Pair-comparison gives the 4-cell ablation:
#   noprompt-baseline    = paper_v2_negalpha/<stem>_neg.jsonl  (variant=baseline,    α=0)
#   noprompt-steered     = paper_v2_negalpha/<stem>_neg.jsonl  (variant=global_blind, α=-3.0)
#   prompt-baseline      = prompt_baseline/<stem>_withprompt.jsonl (variant=baseline,    α=0)
#   prompt-steered       = prompt_baseline/<stem>_withprompt.jsonl (variant=global_blind, α=-3.0)
#
# Updated for EMNLP rebuttal: extended from 3 representative stems to all 16
# (4 models × 4 axes) so the prompt-vs-steering comparison spans the full headline matrix.
set -euo pipefail
cd ${CLUSTER_HOME}/llm_ft_comparison
mkdir -p outputs/steering_eval/prompt_baseline logs

SCRIPT=slurm/generated/cops_evaluate_steering_kiz0.sh
N=200
ALPHAS="-3.0"
VARIANTS="baseline;global_blind"
QOS_LIST=(basic advanced gpubasic preemptible)
qi=0
submit() {
    local tag="$1" env="$2"
    local qos="${QOS_LIST[$((qi % 4))]}"
    qi=$((qi+1))
    echo ">>> $tag qos=$qos"
    sbatch --qos=$qos --partition=p6,p1,p2,p4 --export="ALL,$env" "$SCRIPT"
}

# Per-axis pair files (case-aware) and German negative-pole persona instructions
PAIRS_O=outputs/metrics/persona_pairs_openness_case_aware_1000.jsonl
PAIRS_I=outputs/metrics/persona_pairs_initiative_case_aware_1000.jsonl
PAIRS_C=outputs/metrics/persona_pairs_cooperation_case_aware_1000.jsonl
PAIRS_H=outputs/metrics/persona_pairs_hopefulness_case_aware_1000.jsonl

INSTR_DEFENSIVE="Du bist ein defensiver Klient, der zurückhaltend, selbstschützend, ausweichend oder rechtfertigend antwortet."
INSTR_REACTIVE="Du bist ein reaktiver Klient, der vor allem auf die letzte Berateraussage antwortet und wenig eigene Gesprächsinitiative zeigt."
INSTR_RESISTANT="Du bist ein widerständiger, ablehnender Klient, der Therapievorschläge zurückweist."
INSTR_HOPELESS="Du bist ein resignierter, hoffnungsloser Klient, der nicht mehr an Veränderung glaubt."

# Helper: submit one stem.  Resolves the model-specific cache-file glob at sbatch time
# (the n-pair counter is 999 or 1000 depending on axis filtering — both forms exist on KIZ0).
run_stem() {
    local tag="$1" model_path="$2" layer="$3" axis="$4" instr="$5" pairs="$6" \
          model_slug="$7" v_dir_slug="$8"
    local v_dir="outputs/metrics/layerwise_separability_${axis}_${v_dir_slug}_gpt54mini_1000"
    # Cache glob — let the eval script resolve n=999 or n=1000 by trying both
    local cache_glob="outputs/metrics/cops_hidden_cache_v3/v3_${model_slug}_persona_pairs_${axis}_case_aware_1000_n*_seed42_layers${layer}.npz"
    local cache_file
    cache_file=$(ls -t $cache_glob 2>/dev/null | head -1)
    if [[ -z "$cache_file" ]]; then
        echo "!!! MISSING CACHE for $tag: glob=$cache_glob"
        return 1
    fi
    local out="outputs/steering_eval/prompt_baseline/${tag}_withprompt.jsonl"
    submit "$tag" "MODEL_PATH=${model_path},LAYER=${layer},AXIS=${axis},V_OPEN_DIR=${v_dir},VARIANTS=${VARIANTS},ALPHAS=${ALPHAS},SEEDS=42,HELD_OUT_COUNT=${N},CASE_AWARE_CACHE=${cache_file},CASE_AWARE_PAIRS=${pairs},PERSONA_INSTRUCTION=${instr},OUTPUT_PATH=${out}"
}

# ----- Qwen3-4B (4 axes) -----
run_stem qwen_openness_L15    Qwen/Qwen3-4B          15 openness    "$INSTR_DEFENSIVE" "$PAIRS_O" Qwen_Qwen3-4B          qwen3_4b
run_stem qwen_initiative_L9   Qwen/Qwen3-4B           9 initiative  "$INSTR_REACTIVE"  "$PAIRS_I" Qwen_Qwen3-4B          qwen3_4b
run_stem qwen_cooperation_L20 Qwen/Qwen3-4B          20 cooperation "$INSTR_RESISTANT" "$PAIRS_C" Qwen_Qwen3-4B          qwen3_4b
run_stem qwen_hopefulness_L18 Qwen/Qwen3-4B          18 hopefulness "$INSTR_HOPELESS"  "$PAIRS_H" Qwen_Qwen3-4B          qwen3_4b

# ----- Gemma-4-E4B-it (4 axes) -----
run_stem gemma_openness_L3    google/gemma-4-E4B-it   3 openness    "$INSTR_DEFENSIVE" "$PAIRS_O" google_gemma-4-E4B-it  gemma_4_e4b_it
run_stem gemma_initiative_L13 google/gemma-4-E4B-it  13 initiative  "$INSTR_REACTIVE"  "$PAIRS_I" google_gemma-4-E4B-it  gemma_4_e4b_it
run_stem gemma_cooperation_L23 google/gemma-4-E4B-it 23 cooperation "$INSTR_RESISTANT" "$PAIRS_C" google_gemma-4-E4B-it  gemma_4_e4b_it
run_stem gemma_hopefulness_L25 google/gemma-4-E4B-it 25 hopefulness "$INSTR_HOPELESS"  "$PAIRS_H" google_gemma-4-E4B-it  gemma_4_e4b_it

# ----- Qwen3.5-9B (4 axes) -----
run_stem qwen35_9b_openness_L20    Qwen/Qwen3.5-9B 20 openness    "$INSTR_DEFENSIVE" "$PAIRS_O" Qwen_Qwen3.5-9B qwen3_5_9b
run_stem qwen35_9b_initiative_L20  Qwen/Qwen3.5-9B 20 initiative  "$INSTR_REACTIVE"  "$PAIRS_I" Qwen_Qwen3.5-9B qwen3_5_9b
run_stem qwen35_9b_cooperation_L15 Qwen/Qwen3.5-9B 15 cooperation "$INSTR_RESISTANT" "$PAIRS_C" Qwen_Qwen3.5-9B qwen3_5_9b
run_stem qwen35_9b_hopefulness_L15 Qwen/Qwen3.5-9B 15 hopefulness "$INSTR_HOPELESS"  "$PAIRS_H" Qwen_Qwen3.5-9B qwen3_5_9b

# ----- Mistral-7B-Instruct-v0.3 (4 axes) -----
run_stem mistral7b_openness_L15    mistralai/Mistral-7B-Instruct-v0.3 15 openness    "$INSTR_DEFENSIVE" "$PAIRS_O" mistralai_Mistral-7B-Instruct-v0.3 ministral_14b
run_stem mistral7b_initiative_L5   mistralai/Mistral-7B-Instruct-v0.3  5 initiative  "$INSTR_REACTIVE"  "$PAIRS_I" mistralai_Mistral-7B-Instruct-v0.3 ministral_14b
run_stem mistral7b_cooperation_L15 mistralai/Mistral-7B-Instruct-v0.3 15 cooperation "$INSTR_RESISTANT" "$PAIRS_C" mistralai_Mistral-7B-Instruct-v0.3 ministral_14b
run_stem mistral7b_hopefulness_L20 mistralai/Mistral-7B-Instruct-v0.3 20 hopefulness "$INSTR_HOPELESS"  "$PAIRS_H" mistralai_Mistral-7B-Instruct-v0.3 ministral_14b

echo ""
echo "Prompt-baseline ablation v3 (full 16-stem matrix): jobs submitted."
echo "Outputs land in outputs/steering_eval/prompt_baseline/<stem>_withprompt.jsonl"
echo "Pair against paper_v2_negalpha/<stem>_neg.jsonl for the 4-cell ablation"
echo "(see scripts/cops_judge_prompt_baseline_v2.py)."
squeue -u $USER -h | wc -l
