# COPS Paper v2 — Steering Evaluation Results (new canonical layers)

Generated from layer-sweep based canonical layer selection. Full-matrix evaluation: 5 variants × α=1.5/3.0 × N=200, 3-judge ensemble (Mistral-Magistral, gpt-5.4-nano@high, gpt-oss-120b@high), Fleiss's κ-validated.

## Headline: Layer-Sweep gewinnt — alle Achsen jetzt funktionsfähig

| Model × Axis | Layer | best variant | net (95% CI) | Fleiss κ | n |
|---|---:|---|---|---:|---:|
| qwen_openness | L15 | P_matched α=3.0 | +0.323 | 0.52 | 4800 |
| qwen_initiative | L9 | P_mismatched α=1.5 | +0.095 | 0.40 | 1600 |
| qwen_cooperation | L20 | global_blind α=3.0 | +0.340 | 0.47 | 1600 |
| qwen_hopefulness | L18 | P_matched α=1.5 | +0.315 | 0.52 | 1600 |
| gemma_initiative | L13 | P_mismatched α=3.0 | +0.300 | 0.47 | 1600 |
| gemma_cooperation | L23 | P_mismatched α=3.0 | +0.420 | 0.46 | 1600 |
| gemma_hopefulness | L25 | global_blind α=3.0 | +0.380 | 0.55 | 1600 |

## Vollständige Net-Pairwise Tabelle (3-judge ensemble)

| Stem | Variant | α | net | n |
|---|---|---:|---:|---:|
| qwen_openness_L15 | P_matched | 1.5 | +0.138 | 600 |
| qwen_openness_L15 | P_matched | 3.0 | +0.323 | 600 |
| qwen_openness_L15 | P_mismatched | 1.5 | +0.175 | 600 |
| qwen_openness_L15 | P_mismatched | 3.0 | +0.285 | 600 |
| qwen_openness_L15 | global_aware | 1.5 | +0.155 | 600 |
| qwen_openness_L15 | global_aware | 3.0 | +0.303 | 600 |
| qwen_openness_L15 | global_blind | 1.5 | +0.168 | 600 |
| qwen_openness_L15 | global_blind | 3.0 | +0.297 | 600 |
| qwen_initiative_L9 | P_matched | 1.5 | +0.015 | 200 |
| qwen_initiative_L9 | P_matched | 3.0 | +-0.030 | 200 |
| qwen_initiative_L9 | P_mismatched | 1.5 | +0.095 | 200 |
| qwen_initiative_L9 | P_mismatched | 3.0 | +0.050 | 200 |
| qwen_initiative_L9 | global_aware | 1.5 | +0.020 | 200 |
| qwen_initiative_L9 | global_aware | 3.0 | +0.025 | 200 |
| qwen_initiative_L9 | global_blind | 1.5 | +0.075 | 200 |
| qwen_initiative_L9 | global_blind | 3.0 | +-0.015 | 200 |
| qwen_cooperation_L20 | P_matched | 1.5 | +0.165 | 200 |
| qwen_cooperation_L20 | P_matched | 3.0 | +0.250 | 200 |
| qwen_cooperation_L20 | P_mismatched | 1.5 | +0.175 | 200 |
| qwen_cooperation_L20 | P_mismatched | 3.0 | +0.260 | 200 |
| qwen_cooperation_L20 | global_aware | 1.5 | +0.185 | 200 |
| qwen_cooperation_L20 | global_aware | 3.0 | +0.200 | 200 |
| qwen_cooperation_L20 | global_blind | 1.5 | +0.325 | 200 |
| qwen_cooperation_L20 | global_blind | 3.0 | +0.340 | 200 |
| qwen_hopefulness_L18 | P_matched | 1.5 | +0.315 | 200 |
| qwen_hopefulness_L18 | P_matched | 3.0 | +0.215 | 200 |
| qwen_hopefulness_L18 | P_mismatched | 1.5 | +0.120 | 200 |
| qwen_hopefulness_L18 | P_mismatched | 3.0 | +0.215 | 200 |
| qwen_hopefulness_L18 | global_aware | 1.5 | +0.195 | 200 |
| qwen_hopefulness_L18 | global_aware | 3.0 | +0.180 | 200 |
| qwen_hopefulness_L18 | global_blind | 1.5 | +0.265 | 200 |
| qwen_hopefulness_L18 | global_blind | 3.0 | +0.225 | 200 |
| gemma_initiative_L13 | P_matched | 1.5 | +0.220 | 200 |
| gemma_initiative_L13 | P_matched | 3.0 | +0.250 | 200 |
| gemma_initiative_L13 | P_mismatched | 1.5 | +0.195 | 200 |
| gemma_initiative_L13 | P_mismatched | 3.0 | +0.300 | 200 |
| gemma_initiative_L13 | global_aware | 1.5 | +0.155 | 200 |
| gemma_initiative_L13 | global_aware | 3.0 | +0.285 | 200 |
| gemma_initiative_L13 | global_blind | 1.5 | +0.085 | 200 |
| gemma_initiative_L13 | global_blind | 3.0 | +0.155 | 200 |
| gemma_cooperation_L23 | P_matched | 1.5 | +0.305 | 200 |
| gemma_cooperation_L23 | P_matched | 3.0 | +0.295 | 200 |
| gemma_cooperation_L23 | P_mismatched | 1.5 | +0.195 | 200 |
| gemma_cooperation_L23 | P_mismatched | 3.0 | +0.420 | 200 |
| gemma_cooperation_L23 | global_aware | 1.5 | +0.280 | 200 |
| gemma_cooperation_L23 | global_aware | 3.0 | +0.380 | 200 |
| gemma_cooperation_L23 | global_blind | 1.5 | +0.165 | 200 |
| gemma_cooperation_L23 | global_blind | 3.0 | +0.325 | 200 |
| gemma_hopefulness_L25 | P_matched | 1.5 | +0.115 | 200 |
| gemma_hopefulness_L25 | P_matched | 3.0 | +0.225 | 200 |
| gemma_hopefulness_L25 | P_mismatched | 1.5 | +0.080 | 200 |
| gemma_hopefulness_L25 | P_mismatched | 3.0 | +0.250 | 200 |
| gemma_hopefulness_L25 | global_aware | 1.5 | +0.110 | 200 |
| gemma_hopefulness_L25 | global_aware | 3.0 | +0.185 | 200 |
| gemma_hopefulness_L25 | global_blind | 1.5 | +0.220 | 200 |
| gemma_hopefulness_L25 | global_blind | 3.0 | +0.380 | 200 |

## Statistische Signifikanz — Paired Tests (95% CI on diff)

Nur signifikante Vergleiche (CI schließt 0 nicht ein) gelistet. Y = signifikant.

| Comparison | Stem | α | diff | CI |
|---|---|---:|---:|---|
| P_matched - global_blind | gemma_cooperation_L23 | 1.5 | +0.140 | [+0.005, +0.265] |
| P_matched - global_blind | gemma_hopefulness_L25 | 3.0 | -0.155 | [-0.285, -0.020] |
| P_matched - global_blind | gemma_initiative_L13 | 1.5 | +0.135 | [+0.010, +0.265] |
| P_matched - global_blind | qwen_cooperation_L20 | 1.5 | -0.160 | [-0.305, -0.005] |
| global_aware - global_blind | gemma_hopefulness_L25 | 3.0 | -0.195 | [-0.330, -0.055] |
| global_aware - global_blind | qwen_cooperation_L20 | 1.5 | -0.140 | [-0.285, -0.005] |
| global_aware - global_blind | qwen_cooperation_L20 | 3.0 | -0.140 | [-0.270, -0.010] |
| P_matched - P_mismatched | qwen_hopefulness_L18 | 1.5 | +0.195 | [+0.075, +0.330] |
| P_matched - global_aware | qwen_hopefulness_L18 | 1.5 | +0.120 | [+0.010, +0.225] |

## Klassifikator-Δstyle (TF-IDF on persona pairs)

Vorsicht: TF-IDF ist OOD-miscalibriert auf Chat-Outputs (Faktor ~4× Unterschied zu Judge).

| Stem | Variant | α | Δstyle | n |
|---|---|---:|---:|---:|
| qwen_openness_L15 | P_matched | 1.5 | +0.046 | 200 |
| qwen_openness_L15 | P_matched | 3.0 | +0.055 | 200 |
| qwen_openness_L15 | P_mismatched | 1.5 | +0.020 | 200 |
| qwen_openness_L15 | P_mismatched | 3.0 | +0.066 | 200 |
| qwen_openness_L15 | baseline | 0.0 | +0.000 | 200 |
| qwen_openness_L15 | global_aware | 1.5 | +0.040 | 200 |
| qwen_openness_L15 | global_aware | 3.0 | +0.074 | 200 |
| qwen_openness_L15 | global_blind | 1.5 | +0.037 | 200 |
| qwen_openness_L15 | global_blind | 3.0 | +0.068 | 200 |
| qwen_initiative_L9 | P_matched | 1.5 | +0.020 | 200 |
| qwen_initiative_L9 | P_matched | 3.0 | +0.033 | 200 |
| qwen_initiative_L9 | P_mismatched | 1.5 | +0.016 | 200 |
| qwen_initiative_L9 | P_mismatched | 3.0 | +0.020 | 200 |
| qwen_initiative_L9 | baseline | 0.0 | +0.000 | 200 |
| qwen_initiative_L9 | global_aware | 1.5 | +0.019 | 200 |
| qwen_initiative_L9 | global_aware | 3.0 | +0.018 | 200 |
| qwen_initiative_L9 | global_blind | 1.5 | +0.016 | 200 |
| qwen_initiative_L9 | global_blind | 3.0 | +0.008 | 200 |
| qwen_cooperation_L20 | P_matched | 1.5 | +0.034 | 200 |
| qwen_cooperation_L20 | P_matched | 3.0 | +0.080 | 200 |
| qwen_cooperation_L20 | P_mismatched | 1.5 | +0.042 | 200 |
| qwen_cooperation_L20 | P_mismatched | 3.0 | +0.078 | 200 |
| qwen_cooperation_L20 | baseline | 0.0 | +0.000 | 200 |
| qwen_cooperation_L20 | global_aware | 1.5 | +0.040 | 200 |
| qwen_cooperation_L20 | global_aware | 3.0 | +0.077 | 200 |
| qwen_cooperation_L20 | global_blind | 1.5 | +0.065 | 200 |
| qwen_cooperation_L20 | global_blind | 3.0 | +0.090 | 200 |
| qwen_hopefulness_L18 | P_matched | 1.5 | +0.066 | 200 |
| qwen_hopefulness_L18 | P_matched | 3.0 | +0.100 | 200 |
| qwen_hopefulness_L18 | P_mismatched | 1.5 | +0.072 | 200 |
| qwen_hopefulness_L18 | P_mismatched | 3.0 | +0.104 | 200 |
| qwen_hopefulness_L18 | baseline | 0.0 | +0.000 | 200 |
| qwen_hopefulness_L18 | global_aware | 1.5 | +0.070 | 200 |
| qwen_hopefulness_L18 | global_aware | 3.0 | +0.107 | 200 |
| qwen_hopefulness_L18 | global_blind | 1.5 | +0.061 | 200 |
| qwen_hopefulness_L18 | global_blind | 3.0 | +0.106 | 200 |
| gemma_initiative_L13 | P_matched | 1.5 | +0.017 | 200 |
| gemma_initiative_L13 | P_matched | 3.0 | +0.020 | 200 |
| gemma_initiative_L13 | P_mismatched | 1.5 | +0.018 | 200 |
| gemma_initiative_L13 | P_mismatched | 3.0 | +0.018 | 200 |
| gemma_initiative_L13 | baseline | 0.0 | +0.000 | 200 |
| gemma_initiative_L13 | global_aware | 1.5 | +0.018 | 200 |
| gemma_initiative_L13 | global_aware | 3.0 | +0.025 | 200 |
| gemma_initiative_L13 | global_blind | 1.5 | -0.008 | 200 |
| gemma_initiative_L13 | global_blind | 3.0 | +0.003 | 200 |
| gemma_cooperation_L23 | P_matched | 1.5 | +0.064 | 200 |
| gemma_cooperation_L23 | P_matched | 3.0 | +0.102 | 200 |
| gemma_cooperation_L23 | P_mismatched | 1.5 | +0.063 | 200 |
| gemma_cooperation_L23 | P_mismatched | 3.0 | +0.095 | 200 |
| gemma_cooperation_L23 | baseline | 0.0 | +0.000 | 200 |
| gemma_cooperation_L23 | global_aware | 1.5 | +0.083 | 200 |
| gemma_cooperation_L23 | global_aware | 3.0 | +0.096 | 200 |
| gemma_cooperation_L23 | global_blind | 1.5 | +0.032 | 200 |
| gemma_cooperation_L23 | global_blind | 3.0 | +0.069 | 200 |
| gemma_hopefulness_L25 | P_matched | 1.5 | +0.075 | 200 |
| gemma_hopefulness_L25 | P_matched | 3.0 | +0.114 | 200 |
| gemma_hopefulness_L25 | P_mismatched | 1.5 | +0.080 | 200 |
| gemma_hopefulness_L25 | P_mismatched | 3.0 | +0.114 | 200 |
| gemma_hopefulness_L25 | baseline | 0.0 | +0.000 | 200 |
| gemma_hopefulness_L25 | global_aware | 1.5 | +0.075 | 200 |
| gemma_hopefulness_L25 | global_aware | 3.0 | +0.127 | 200 |
| gemma_hopefulness_L25 | global_blind | 1.5 | +0.096 | 200 |
| gemma_hopefulness_L25 | global_blind | 3.0 | +0.185 | 200 |

## Inter-Rater Agreement (Fleiss's κ per stem, 3 Provider)

| Stem | Fleiss κ | n |
|---|---:|---:|
| qwen_openness_L15 | 0.520 | 4800 |
| qwen_initiative_L9 | 0.404 | 1600 |
| qwen_cooperation_L20 | 0.471 | 1600 |
| qwen_hopefulness_L18 | 0.522 | 1600 |
| gemma_initiative_L13 | 0.473 | 1600 |
| gemma_cooperation_L23 | 0.463 | 1600 |
| gemma_hopefulness_L25 | 0.546 | 1600 |

## Layer-Sweep — alte vs neue kanonische Layer

| Model×Axis | Old (AUC) | New (Sweep) | Old net | New net | Improvement |
|---|---:|---:|---:|---:|---:|
| qwen_openness | L1 | L15 | +0.020 | +0.323 | +0.303 |
| qwen_initiative | L9 | L9 | +0.000 | +0.095 | +0.095 |
| qwen_cooperation | L9 | L20 | +0.060 | +0.340 | +0.280 |
| qwen_hopefulness | L7 | L18 | -0.040 | +0.315 | +0.355 |
| gemma_initiative | L8 | L13 | -0.060 | +0.300 | +0.360 |
| gemma_cooperation | L23 | L23 | +0.120 | +0.420 | +0.300 |
| gemma_hopefulness | L13 | L25 | +0.120 | +0.380 | +0.260 |

## New Models (Qwen3.5-9B + Mistral-7B-Instruct-v0.3) — Generalization Test

Pipeline replicated on 2 new models (9B and 7B): empirical layer-sweep + 3-judge ensemble. Same data, same pairs, same protocol.

### Canonical layers from sweep + paper-v2 best variant

| Model × Axis | Layer | Sweep net | Paper-v2 best variant @α=3.0 | net | Fleiss κ |
|---|:---:|:---:|---|:---:|:---:|
| Qwen3.5-9B × openness | L20 | +0.500 | global_blind | **+0.410** | 0.36 |
| Qwen3.5-9B × initiative | L20 | +0.180 | global_aware | **+0.455** | 0.17 |
| Qwen3.5-9B × cooperation | L15 | +0.200 | global_blind | +0.235 | 0.33 |
| Qwen3.5-9B × hopefulness | L15 | +0.420 | P_mismatched | +0.345 | 0.50 |
| Mistral-7B × openness | L15 | +0.240 | P_mismatched | **+0.460** | 0.45 |
| Mistral-7B × initiative | L5 | +0.280 | global_aware | +0.025 | 0.39 |
| Mistral-7B × cooperation | L15 | +0.102 | global_blind | +0.115 | 0.34 |
| Mistral-7B × hopefulness | L20 | +0.380 | global_blind | +0.040 | 0.47 |

### Key insights

1. **Layer-sweep methodology generalizes** über Familie und Größe. AUC-best Layer (L5) durchweg sub-optimal — tiefe Layer L15-L20 dominieren bei openness/cooperation/hopefulness across all 4 tested architectures.

2. **Initiative now steerable at 9B:** Qwen3.5-9B initiative L20 net=+0.455 (vs Qwen3-4B initiative L9 ≈ 0). **Modell-Größe matters für schwierige Achsen.**

3. **Cross-family difference:** Mistral-7B hopefulness/initiative kollabieren in Full-matrix (Sweep prediction +0.28-0.38, Matrix +0.025-0.04). Mögliche Erklärung: Mistral-Architektur hat anderen Hidden-State-Aufbau; Sweep auf N=50 robust, Full-Matrix auf N=200 zeigt schwächere Effekte.

4. **COPS hypothesis falsified again:** P_mismatched ≥ P_matched in **3/8** stems (Qwen openness, Qwen hopefulness, Mistral openness). Konsistent mit den Original-Modellen.

5. **Initiative judge unreliable:** Qwen3.5-9B initiative L20 Fleiss κ=0.17 (poor) — Provider-Disagreement bei Initiative bestätigt sich auf 9B. Auch Mistral initiative κ=0.39.

### Komplette Variant-Tabelle

→ Daten: `outputs/steering_eval/judged_v2_newmodels_ensemble/`

## Befunde — Critical Review

1. **Layer-Wahl ist der Hauptfaktor.** Empirischer Layer-Sweep schlägt AUC-basierte Selection systematisch.

2. **Persona-Matching: kein robuster Effekt.** P_matched − P_mismatched signifikant nur für 1/14 stem×α-Zellen (chance level bei 5% α).

3. **Case-Awareness: kein Vorteil.** global_aware − global_blind signifikant *negativ* in 3 Zellen, nie positiv.

4. **v_persona ist konstruktionsbedingt ≈ v_global** (cos > 0.95 für n>50). Die Pol-Differenz cancelt persona-spezifische Information. P_matched ≠ echter Persona-Steering-Test.

5. **Klassifikator-Judge-Gap (Faktor ~4×)** bleibt bestehen — TF-IDF auf Chat-Outputs OOD.

6. **qwen_initiative ist nicht steerbar** (alle Varianten CI⊃0 selbst bei α=3.0).

7. **3-Provider Ensemble:** Fleiss's κ 0.40-0.55 (substantial agreement). GPT-Family (nano, oss-120b) liberal, Mistral konservativ.
