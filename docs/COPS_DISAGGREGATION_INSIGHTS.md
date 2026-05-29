# COPS — Disaggregation Insights

Disaggregierte Analyse der paper_v2 Ensemble-Daten, um aggregations-bedingte Verschleierung aufzulösen und neue Erkenntnisse zu identifizieren.

## 1. Kern-Befund: Variant-Effekte statistisch ununterscheidbar

**Frage:** Wie oft ist jede Variant "best" auf einer (Stem × Persona) Zelle?

| Variant | Wins (von 70 Zellen) | Anteil | vs Zufallsbaseline (25%) |
|---|---:|---:|---:|
| global_blind | 19 | 27.1% | +2.1 pp |
| P_matched | 19 | 27.1% | +2.1 pp |
| P_mismatched | 17 | 24.3% | −0.7 pp |
| global_aware | 15 | 21.4% | −3.6 pp |

**Alle 4 Varianten gewinnen ~25% — keine ist systematisch besser.**

Konsequenz: Die "spezialisierten" Varianten (P_matched, P_mismatched, global_aware) bringen **keinen messbaren Mehrwert** über naives globales Steering.

## 2. Alpha-Instabilität

**Best variant @α=1.5 vs @α=3.0:** stabil nur 3/7 Stems (43%).

| Stem | best @α=1.5 | best @α=3.0 | stabil |
|---|---|---|:---:|
| qwen_openness_L15 | P_mismatched | P_matched | × |
| qwen_cooperation_L20 | global_blind | global_blind | ✓ |
| qwen_hopefulness_L18 | P_matched | global_blind | × |
| qwen_initiative_L9 | P_mismatched | P_mismatched | ✓ |
| gemma_cooperation_L23 | P_matched | P_mismatched | × |
| gemma_hopefulness_L25 | global_blind | global_blind | ✓ |
| gemma_initiative_L13 | P_matched | P_mismatched | × |

→ "Best variant" ist nicht robust gegen α-Wahl.

## 3. Seed-Instabilität (qwen_openness_L15)

3 Seeds × 2 α = 6 Rankings, alle unterschiedlich:

| Seed | α=1.5 best | α=3.0 best |
|---|---|---|
| 42 | global_blind | global_aware |
| 123 | P_mismatched | global_blind |
| 2024 | P_matched | P_matched |

→ "Best variant" ist Stichproben-Varianz, kein Signal.

## 4. Context-Polarization (neuer Befund)

α=3.0: pro Kontext zählen wie viele der 4 Varianten "steered" gewinnen.

| Stem | 0 wins | 1 | 2 | 3 | 4 wins |
|---|---:|---:|---:|---:|---:|
| qwen_openness_L15 | 14.5% | 12.5% | 12.5% | 24.0% | **36.5%** |
| qwen_cooperation_L20 | 14.0% | 20.0% | 12.5% | 16.5% | **37.0%** |
| gemma_cooperation_L23 | 10.0% | 16.5% | 17.5% | 25.0% | **31.0%** |
| gemma_hopefulness_L25 | 19.0% | 14.0% | 15.5% | 14.5% | **37.0%** |
| qwen_initiative_L9 | 27.5% | 18.5% | 20.5% | 18.0% | 15.5% |

**Bimodale Verteilung** in den starken Stems: ~30-40% der Kontexte werden klar gesteuert (alle Varianten), ~10-20% gar nicht (keine Variante). Mittel ist klein. Steering ist im Kontext **binär** — manche Konversationen sind steerbar, andere nicht.

qwen_initiative_L9 ist der einzige Stem ohne Bimodalität (uniformere Verteilung) — bestätigt dass dieser Stem schwach ist.

## 5. Provider-Disagreement bei schwachen Stems

**qwen_initiative_L9 global_blind α=3.0 axis_alignment delta:**

| Provider | Δaxis_alignment |
|---|---:|
| cluster_mistral | **−0.255** |
| ionos (oss-120b) | +0.115 |
| openai (nano) | +0.145 |

Mistral sagt: Steering **schadet**. GPT-Family: leicht positiv. Vorzeichen-Disagreement → Majority Vote masked die Heterogenität. Aggregat-Net war ~0, verbirgt aber dass 1/3 Providern klaren negativen Effekt sieht.

**Full-Agreement-Rate per (variant, α):** 52–65% Kontext. 35–48% Cases mit mind. einem Dissenter.

## 6. Per-Dimension Provider-Kalibrierung

**qwen_openness_L15 P_matched α=3.0 — axis_alignment Δ:**
- cluster_mistral: +0.430
- openai (nano): +0.575
- ionos (oss-120b): +0.652

Mistral systematisch ~30% kleiner als GPT-Family. Aber Vorzeichen-konsistent.

**case_fidelity** über alle Stems × Provider: meistens +0.05 bis +0.20. **Steering bricht nicht systematisch case fidelity.**

**client_role_fidelity:** nahe 0 — Modell bleibt konsistent als Klient.

## 7. Tie Rate als Reliability-Signal

| Stem | best Tie Rate (α=3.0) | Stem-Stärke |
|---|---:|---|
| qwen_hopefulness_L18 | 3.5–5.5% | stark |
| qwen_cooperation_L20 | 4.0–6.0% | stark |
| qwen_openness_L15 | 4.0–5.3% | stark |
| gemma_cooperation_L23 | 8.0–13.5% | mittel |
| gemma_hopefulness_L25 | 6.0–9.5% | stark |
| gemma_initiative_L13 | 13.0–27.5% | mittel |
| qwen_initiative_L9 | 10.5–15.0% | schwach |
| gemma_initiative_L13 (g_b α=1.5) | **44.5%** | extrem ambig |

→ Niedriger Tie-Rate ist ein Quality-Indikator pro Stem.

## 8. Per-Persona × Variant Cross-Tab

Pro Stem × Persona: welcher Variant wins?

Daten in `outputs/metrics/persona_stem_matrix.json`. Korrelationen über Stems pro Persona-Pair (10×10 Matrix) zeigen: **keine Persona ist systematisch über Achsen hinweg "gut" oder "schlecht"** für P_matched.

Stem×Stem Korrelationen (über Personas):
- gemma_cooperation_L23 ↔ gemma_hopefulness_L25: r = +0.68 (innerhalb-Modell konsistent)
- gemma_initiative_L13 ↔ qwen_hopefulness_L18: r = −0.52 (cross-Modell anti-konsistent)
- qwen_cooperation_L20 ↔ qwen_initiative_L9: r = −0.27

→ Persona-Effekte sind **modell-spezifisch**, nicht universell.

## 🎯 A/B/C Variants Ergebnis (Update 2026-04-25)

Eigenständiger Eval-Run mit Variante A (orthogonal residual), B (cross-persona contrast), C (interaction term) als steering vectors statt v_persona:

| Stem | P_matched (orig) | P_matched_A | P_matched_B | P_matched_C |
|---|---:|---:|---:|---:|
| qwen_openness_L15 | +0.250 | −0.105 | −0.125 | −0.060 |
| gemma_cooperation_L23 | +0.350 | +0.040 | −0.020 | +0.020 |
| gemma_initiative_L13 | +0.290 | −0.005 | +0.040 | −0.015 |
| gemma_hopefulness_L25 | +0.245 | −0.015 | −0.020 | −0.035 |
| qwen_cooperation_L20 | +0.250 | +0.040 | +0.080 | +0.080 |
| qwen_hopefulness_L18 | +0.275 | +0.045 | −0.015 | +0.005 |
| qwen_initiative_L9 | −0.005 | +0.055 | −0.020 | +0.035 |

**A/B/C steuern nicht** — net pairwise ≈ 0. Der frühere "P_matched"-Effekt kommt fast ausschließlich vom v_global-Anteil; isoliert man die persona-spezifische Komponente, kollabiert das Signal auf Rauschen.

**Konstruktions-Hypothese empirisch validiert:** v_persona = v_axis + Sample-Rauschen, Persona-Information cancelt durch Pol-Differenz.

## Konsolidierte neue Insights

1. **Variant-Wahl ist kein robustes Signal** — alle 4 Varianten gewinnen ~25% (Zufallsbaseline)
2. **Best variant flippt** zwischen α-Werten, Seeds, Personas — pure Stichproben-Varianz
3. **Kontext-Bimodalität** — Steering ist context-binär: ~35% Kontexte robust steerbar, ~15% gar nicht
4. **Provider-Disagreement** auf schwachen Stems sogar im Vorzeichen
5. **case_fidelity wird durch Steering nicht systematisch geschädigt** (alle Δ < +0.30)
6. **Tie Rate als Reliability-Metrik** — neue Stem-Stärke-Heuristik
7. **Persona-Effekte sind modell-spezifisch** — kein universeller "guter Persona für Steering"

## Implikationen für Paper-Story

Statt "wir finden P_matched bringt etwas in einigen Fällen" → **"alle 4 Varianten sind statistisch ununterscheidbar; ein einziger globaler Vektor reicht"**.

**Vereinfachte methodische Empfehlung für Praxis:**
1. Pick Layer durch empirischen Sweep (~40 Generation-Jobs)
2. Globaler Vektor v_axis = mean(h_pos) − mean(h_neg) auf 1000 Paaren
3. Steering bei α=3.0
4. Multi-Provider Judge Ensemble (3+ provider) mit majority vote + Fleiss's κ als reliability check
5. Persona-Conditioning: NICHT nötig, aktuelle Konstruktion liefert kein Mehrwert

Daten:
- `outputs/metrics/paper_v2_stat_tests.json` — bootstrap CIs + paired tests
- `outputs/metrics/persona_diagnostics.json` — v_persona Vektor-Eigenschaften
- `outputs/metrics/persona_variants_comparison.json` — A/B/C Vergleich
- `outputs/metrics/persona_stem_matrix.json` — Persona × Stem Matrix
- `outputs/steering_eval/judged_ensemble_v2/` — vollständige Ensemble-JSONLs
- `outputs/steering_eval/judged/` — per-Provider judgments
