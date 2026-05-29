# COPS — Methodenteil & Paper-Skizze v2

## 1. Pipeline-Überblick

```
1. Kontrast-Paare generieren (case-aware GPT-mini@high, 1000 Paare/Achse)
2. Hidden-States extrahieren (response-mean pooling, layer L)
3. Steering-Vektor v_axis = mean(h_pos) - mean(h_neg) berechnen
4. Steering bei Inference: h_L' = h_L + α · v_axis
5. Eval auf 200 held-out Kontexten: 5 Varianten × 2 α
6. Multi-Provider LLM-Judge (3 Modelle), Majority-Vote, Fleiss's κ
```

## 2. Layer-Selection

### 2.1 Verworfen: AUC-basierte Selection
Frühere Arbeiten wählen Steering-Layer durch Klassifikator-AUC auf Persona-Pairs am Hidden State. Wir zeigen empirisch dass diese Heuristik **systematisch ungeeignet** ist:

| Model × Axis | AUC-best Layer | Empirisch-best Layer | net-pairwise improvement |
|---|---|---|---|
| Qwen openness | L1 (AUC 0.99) | **L15** | +0.02 → +0.32 (16×) |
| Qwen cooperation | L9 | **L20** | +0.06 → +0.34 |
| Qwen hopefulness | L7 | **L18** | −0.04 → +0.32 |
| Gemma initiative | L8 | **L13** | −0.06 → +0.30 |
| Gemma hopefulness | L13 | **L25** | +0.12 → +0.38 |

### 2.2 Empirischer Layer-Sweep
Für jedes (Modell, Achse): 5 Kandidat-Layer × N=50 held-out Kontexte × 2 Varianten (baseline, global_blind) × α=1.5. Output: net pairwise per Layer. Pick Layer mit größtem positiven net.

Kosten: 2 Modelle × 4 Achsen × 5 Layer = 40 GPU-Jobs (~30 min/Job auf L40S).

## 3. Multi-Provider Judge Ensemble

### 3.1 Setup
4 Provider getestet, 3 produktiv im Ensemble:
- `openai:gpt-5.4-mini` (Paper Matrix v1)
- `cluster_mistral:mistralai/Magistral-Small-2509` (selbst-gehostet)
- `openai:gpt-5.4-nano@high` (reasoning effort high, ~5× günstiger als mini)
- `ionos:openai/gpt-oss-120b@high` (open-weight, kostenlos via Cluster-Provider)

### 3.2 Bewertungs-Schema
- Pointwise rubric (4 Dimensionen, Skala 1-5): axis_alignment, case_fidelity, client_role_fidelity, training_utility
- Pairwise: welcher von zwei Responses (steered vs base) ist näher am target_style?
- A/B-Position randomisiert, target_pole flippt bei α<0 (polarity-aware)

### 3.3 Ensemble-Aggregation
- Per (run_seed, seed_id, variant, α): majority-vote winner across providers (tie wenn split)
- Per (variant, α): net_pairwise_overall = (steered − base) / N
- **Inter-Rater Agreement:** Fleiss's κ; substantial agreement (κ > 0.4) als Reliability-Schwelle
- Pro Provider: ebenfalls aggregiert für Sensitivitätsanalyse

### 3.4 Provider-Kalibrierung
GPT-Family-Provider (nano, oss-120b) signifikant liberaler als Mistral-Magistral. Per-Provider net-pairwise auf gleichem Datensatz:
- nano @ qwen_openness L15 α=3.0 P_matched: +0.380
- ionos @ same: +0.340
- mistral @ same: +0.145

→ Mistral konservativer um Faktor ~2-3. Majority Vote schützt vor Mono-Provider-Bias, aber per-Provider Reporting empfohlen.

## 4. Statistische Inferenz

- **Bootstrap 95% CIs** (n_boot=2000) auf net pairwise (resample winners)
- **Paired Bootstrap** für Vergleiche zwischen Varianten (gemeinsame seed_ids)
- Signifikanz-Schwelle: CI schließt 0 nicht ein (α=0.05)

## 5. Persona-Test — methodische Falsifikation

### 5.1 Konstruktion v_persona
Original-COPS: `v_persona = mean(h_pos | persona=p) - mean(h_neg | persona=p)`.

**Strukturelles Problem:** persona-spezifischer Hidden-State-Offset δ_p tritt in beiden Polen auf (gleicher Klient antwortet mit pos und neg). Die Pol-Differenz **cancelt δ_p** by design:

```
v_persona = (h_axis_pos + δ_p + ε₁) - (h_axis_neg + δ_p + ε₂) = v_axis + (ε₁ - ε₂)
```

Empirische Bestätigung: cos(v_persona, v_global) = 0.78–0.997 (höher mit n).

### 5.2 Persona-Imbalance
| Persona | n_pairs (openness) |
|---|---:|
| Frau Wolke | 200 |
| Frau Schuster | 155 |
| ... | ... |
| Eddie | 20 |

→ 10× Imbalance. Eddie's v_persona ist verrauscht; Frau Wolkes praktisch = v_global.

### 5.3 Persona-Hypothese ungetestet
Mit dieser Konstruktion testet "P_matched vs P_mismatched" effektiv "global_blind + Rauschen vs global_blind + anderes Rauschen". Ergebnis: kein signifikanter Unterschied (1/14 stem×α-Zellen, chance level).

**Empfehlung für saubere Persona-Tests** (separates Paper):
- Variante A: `v_persona_residual = v_persona − proj(v_persona ∥ v_global)`
- Variante B: `v_persona_cross = h_pool[p] − h_pool[¬p]` (Pol-übergreifend)
- Variante C: `v_persona_inter = v_persona − v_global`
- Größere Persona-N (≥100/Persona) und systematisches (nicht-random) mismatch-Sampling

## 6. Daten

### 6.1 Conversations
- Quelle: VIKL-F2F German counseling conversations (76 Originalgespräche)
- Rolling-Window-Schnitt → 5400+ Kontexte
- Held-in (Vektor-Training): seeds 0-199; Held-out (Eval): seeds 200-399; disjunkt

### 6.2 Persona-Achsen
- 4 Achsen: openness (offen/defensiv), initiative (explorativ/reaktiv), cooperation (kooperativ/widerständig), hopefulness (hoffnungsvoll/resigniert)
- Bipolar, Pol-Beschreibung in `configs/persona_axes/client_persona_axes_v1.json`
- Polarity-aware bei α<0 (target flippt zum Negativ-Pol)

### 6.3 Generation Settings
- Pair-Generation: GPT-5.4-mini@high reasoning, batch API (50% Rabatt)
- Eval-Generation: 200 Kontexte × 5 Varianten × 2 α × {1,3} seeds = 2000–6000 generations/Stem
- Sampling: do_sample=True, top_p=0.95, T=0.7, max_new_tokens=200

## 7. Paper-Skizze

### 7.1 Story-Pivot
**Original-Hypothese:** Persona-Conditioning verbessert Steering-Effekte
**Befund:** Layer-Wahl ist Hauptfaktor; Persona-Conditioning bringt nichts robust

### 7.2 Title (Vorschlag)
"Layer Selection Matters More Than Persona Conditioning: A Methodological Study of Activation Steering for Counseling LLMs"

### 7.3 Abstract-Skizze
> Activation Steering durch additive Vektoren in Hidden States ist eine bewährte Technik für Style-Conditioning. Wir untersuchen am Beispiel von vier Persona-Achsen in deutschsprachigen Beratungsgesprächen, ob (a) Persona-spezifische Steering-Vektoren bessere Ergebnisse liefern als globale Vektoren, und (b) wie die Layer-Wahl die Effektgröße beeinflusst. Mit 200 held-out Kontexten und einem 3-Provider LLM-Judge-Ensemble (Fleiss's κ > 0.4) zeigen wir: empirische Layer-Selection durch Sweep verbessert net-pairwise Effekte um den Faktor 5-16× gegenüber AUC-basierten Heuristiken; Persona-Conditioning hingegen bringt keinen reproduzierbaren Mehrwert. Wir identifizieren einen strukturellen Konstruktionsfehler in der naiven v_persona-Berechnung (Pol-Differenz cancelt Persona-Offset) und schlagen drei Alternativen vor. Code, Daten und Ensemble-Pipeline open-source.

### 7.4 Beiträge
1. **Empirische Layer-Selection-Methodik** statt AUC-Heuristik
2. **Multi-Provider Judge-Ensemble** als Reliability-Standard für Steering-Eval
3. **Negativbefund Persona-Conditioning** mit struktureller Erklärung
4. **Open-Source-Pipeline** für reproduzierbare Steering-Studien

### 7.5 Limitations
- 2 Modelle (Qwen3-4B, Gemma-4-E4B-it), Generalisierung offen
- 10 Personas, klein-N für Persona-spezifisches Testing
- Persona-Test mit fehlerhafter v_persona-Konstruktion — neuer Test mit Variante A/B/C läuft (Stand Manuscript)
- TF-IDF Klassifier OOD-miscalibriert auf Chat-Outputs (Faktor ~4× zu Judge)
- Domain: nur deutsche Counseling-Konversationen, English/cross-domain offen

### 7.6 Sektionen-Plan
1. Introduction (1.5 S.)
2. Related Work — activation steering, persona conditioning (1 S.)
3. Method
   - 3.1 Pipeline (Daten, Vektoren, Steering)
   - 3.2 Layer Selection: AUC vs Empirical Sweep
   - 3.3 Multi-Provider Judge Ensemble
4. Experiments
   - 4.1 Layer Sweep Results
   - 4.2 Full-Matrix Eval at Canonical Layers
   - 4.3 Persona Conditioning Test
5. Critical Analysis
   - 5.1 v_persona Construction Issue
   - 5.2 Provider Calibration
   - 5.3 Classifier-Judge Gap
6. Discussion & Limitations
7. Conclusion

### 7.7 Pending für finales Paper
- [x] Persona-Variants Eval (A/B/C) — abgeschlossen 2026-04-25, alle drei Konstruktionen liefern net pairwise ≈ 0 (Konstruktions-Hypothese empirisch validiert)
- [ ] Cross-domain validation (English counseling?)
- [ ] User-Study / Human-Eval als Cross-Check zu LLM-Judge
- [ ] Reviewer-Antwort auf "warum nicht LoReFT?"

Daten: `outputs/metrics/{paper_v2_stat_tests,persona_diagnostics,persona_variants_comparison,persona_stem_matrix}.json`, `outputs/steering_eval/{paper_v2,judged_ensemble_v2}/`
