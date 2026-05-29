# COPS: Case-Orthogonal Persona Steering

**Arbeitstitel:** *Persona-Steering Entanglement is an Architectural Property: A Cross-Model Analysis*

**Zentrale Forschungsfragen:**
1. Läuft die Style-Steering-Richtung `v_open` durch den Persona-/Case-Subspace?
2. Wenn ja — ist das ein Datenartefakt oder eine architekturale Eigenschaft?
3. Wie unterscheiden sich Modellfamilien in ihrer Style-Identity-Disentanglement?

**Empirischer Befund (Best-Layer v_open openness, response-mean pooling = deployed formulation):**

| Modell | Layer | Sanity cos(fresh, stored) | Persona proj CB | Persona proj CA | Case-aware Δ |
|---|---|---|---|---|---|
| Qwen3-4B | L1 | 0.977 | **0.241** | 0.194 | −20% |
| Gemma-4-E4B-it | L3 | 0.797 | **0.151** | 0.152 | ~0% |

Gemma hat **~1.6× weniger** Persona-Entanglement als Qwen am Best-Layer. Case-aware Trainingsdaten reduzieren Qwen's Entanglement messbar (20%), bei Gemma null Effekt weil Vektor schon sauber.

**Methodischer Hinweis:** Eine frühere Version dieser Analyse verwendete Last-Token-Pooling und überschätzte die Entanglement systematisch (Qwen: 0.558, Gemma: 0.166). Nach Korrektur auf Response-Mean-Pooling (wie im deployten v_open tatsächlich berechnet) sind die Werte moderater. Die **qualitativen Befunde bleiben bestehen**: Qwen ist entangled, Gemma ist sauberer, case-aware hilft Qwen.

**Vollständige Gemma-Schicht-Landschaft (alle 42 Layer analysiert):**

| Layer | Persona-Proj CB | Persona-Proj CA | Probe test_acc |
|---|---|---|---|
| 1 | 0.166 | 0.135 | 0.48 / 0.55 |
| 3 | 0.089 | 0.101 | 0.47 / 0.62 |
| 6 | 0.070 | 0.100 | 0.75 / 0.82 |
| 10 | 0.058 | 0.081 | 0.61 / 0.71 |
| 15 | 0.054 | 0.068 | 0.78 / 0.83 |
| **20** | **0.038** | 0.046 | 0.87 / 0.87 |
| 25 | 0.061 | 0.078 | 0.87 / 0.88 |
| 30 | 0.058 | 0.075 | 0.88 / 0.86 |
| 35 | 0.055 | **0.149** | 0.95 / 0.93 |
| 40 | 0.076 | **0.121** | 0.95 / 0.95 |

**Drei Struktur-Befunde:**

1. **Gemma ist über alle 42 Layer sauber** (max Persona-Proj 0.17 bei L1). Es gibt keinen Layer der auch nur annähernd Qwen's L1 Wert (0.56) erreicht.
2. **Sweet Spot für Steering: L15–L20** (niedrigste Entanglement 0.04–0.07, gleichzeitig starke Probe 0.78–0.87). Paper's Best-AUC-Layer L3 ist suboptimal aus Fidelity-Sicht.
3. **Case-aware wirkt modellspezifisch:** reduziert Qwen-Entanglement messbar, bei Gemma nur marginal und bei Deep-Layers (L35/40) sogar leicht verschlechternd.

---

## Experimentelle Befunde (Stand 2026-04-22)

### Was NICHT funktioniert (Null-Resultate)

- **Lineare Orthogonalisierung:** Within-Persona-Centering ist mathematisch identisch zu v_global (cos = 1.0000)
- **PCA über per-persona Vektoren:** PC1 ist nicht die gemeinsame Style-Richtung, sondern die größte Abweichungsrichtung (meist Outlier-Persona). Reduziert Persona-Entanglement nicht systematisch
- **Per-persona v_P:** Reduziert Persona-Entanglement nicht messbar — per-persona Vektoren haben selbe Projection auf Persona-Subspace wie v_global

### Was funktioniert (positive Befunde)

- **Case-aware Trainingsdaten:** Reduziert Persona-Entanglement bei Qwen3-4B (Layer 1: 0.56 → 0.38, −32%). Bei Gemma kleiner Effekt (0.17 → 0.14, −19%) weil Gemma schon sauber ist
- **Architekturwahl:** Gemma's Style-Richtung ist strukturell disentangled von Persona-Identität — das ist der stärkste Einzeleffekt

### Die wichtige Nuance

Persona-Entanglement ist **nicht komplett** ein entfernbarer Confound. Ein Teil ist die intrinsische Semantik der Achse:
> *Offenheit bedeutet case-spezifischen Inhalt preisgeben. Das erhöht zwangsläufig persona-diskriminative Information.*

Was entfernbar wäre ist die **Richtungsverzerrung** (v_open zieht systematisch zu bestimmten Personas, z.B. Susi +0.234, Lina −0.212 bei Qwen L1). Das ist ein echtes Artefakt des Trainingsdatensatzes.

---

## 0. Terminologie (verbindlich für das ganze Dokument)

| Begriff | Definition | Rolle |
|---|---|---|
| **Case** | `hauptanliegen` + `steckbrief` + `nebenanliegen` aus dem Dataset | Ground-Truth-Fakten, **Invariante** unter Steering |
| **History** | Bisheriger Dialog (Berater + Klient Turns bis inkl. letzter Berater-Turn) | **Kontext** für die Generierung |
| **Response** | Nächster Klient-Turn | **Ziel** der Generierung, wird entlang der Persona-Achse gesteuert |
| **Persona-Achse** | openness / initiative / cooperation / hopefulness | **Richtung** der Intervention |
| **Case-Richtung** | Vektor im Hidden-Space der case-diskriminativ ist (Normalenvektor einer Case-Probe) | Basis für den Case-Subspace `U_case` |
| **Case-Probe** | Linearer Klassifikator der auf Case-Merkmalen trainiert ist | Werkzeug um Case-Richtungen zu identifizieren |
| **Invarianzmenge** | Teilmenge des Case die unter Steering stabil bleiben soll | formalisiert über `U_case` |

### Kernbeziehung

```
Case                      (statisch, aus Fallbeschreibung)
  └── wird teilweise enthüllt in →
        History           (dynamisch, wächst turn für turn)
              └── bestimmt Erwartung für →
                    Response   (generiert, entlang Persona-Achse steuerbar
                                — aber Case bleibt invariant)
```

**Wichtig:** Der Case ist vollständig definiert bevor die History beginnt. Steering darf nicht implizit zu einem anderen Case driften, nur weil die History noch nicht alle Case-Fakten enthüllt hat.

---

## 1. Kernhypothese

Persona-Stil und Case-Fakten liegen in den Hidden States eines Sprachmodells partiell trennbar. Wenn Case-tragende Richtungen explizit identifiziert und bei der Aktivierungsintervention ausgeklammert werden, verschiebt sich die Pareto-Front zwischen Persona-Kontrolle und Case-Fidelity zugunsten beider Ziele.

Das Paper formuliert Steering als **constrained intervention problem**:

> *Steuere entlang der Persona-Achse — aber nur in dem Teilraum der Hidden States, der Case-Fakten nicht enkodiert.*

---

## 2. Abgrenzung zum aktuellen Benchmark-Paper

Das bestehende Paper (`persona_vectors_emnlp.pdf`) zeigt empirisch:

- Monitoring (Klassifikatoren, Layerwise Separability) ist stabiler als Kontrolle
- Stärkere Interventionen produzieren systematisch mehr Drift
- Automatische Metriken und Judge-Präferenzen sind stark antikorreliert (r = −0.81)
- Openness ist besonders anfällig für Drift durch kontextlizenzierte Elaboration

COPS adressiert direkt die dort identifizierte Limitation: Drift wird aktuell heuristisch gemessen (Jaccard, Novelty, Drift-Flag), weil keine Case-Referenz vorhanden ist. Mit strukturierten Fallbeschreibungen wird Fidelity gegen Ground-Truth-Fakten messbar.

---

## 3. Datenbasis

### ViKl-F2F Dataset

- 76 echte Beratungsgespräche (Rollenspiele), davon 73 mit Case-Attributen
- 10 eindeutige Fälle (`problemfall` 1–10), bis zu 20 Gespräche pro Fall
- Gespräche werden als Rolling-Window-Kontexte (5 Nachrichten) verarbeitet

### Case-Struktur pro Gespräch

```json
{
  "problemfall": 1,
  "persona": {
    "name": "Elke",
    "steckbrief": {
      "Alter": 37,
      "Familienstand": "verheiratet, mehrere Kinder",
      "Geschlecht": "weiblich",
      "Job": "Hausfrau"
    },
    "hauptanliegen": "Freitext-Narrativ mit konkreten Fakten: Namen, Substanzen, Ereignisse, Zeitbezüge, bisherige Lösungsversuche, Ängste...",
    "nebenanliegen": ["Gefährdung der Beziehung zum Sohn", "..."]
  }
}
```

### Invarianzmenge (was Steering nicht verändern darf)

Das `hauptanliegen` definiert die harte Invarianzmenge:
- Personennamen (z.B. Max, Ehemann)
- Substanzen und Konsummuster (Cannabis, 2–3 Joints/Tag)
- Zeitbezüge (seit 7 Jahren, nächsten Monat 30)
- Bisherige Lösungsversuche und deren Ergebnis
- Explizite Ziele und Ängste

**Wichtig:** Das `hauptanliegen` löst nur die harte Invarianzmenge. Eine dritte Drift-Kategorie bleibt epistemisch unterbestimmt:

| Drift-Typ | Beispiel | Lösbar mit hauptanliegen? |
|---|---|---|
| Faktenwiderspruch | "Max hat aufgehört" (falsch) | Ja |
| Faktische Erfindung | Neues unplausibles Ereignis | Teilweise |
| Plausible Elaboration | Passt zum Case, steht aber nicht drin | Nein — Judge bleibt hier das bessere Signal |

---

## 4. Methode: Fünf Module

### Modul 1 — Case-State als Invarianzmenge

Die `hauptanliegen`-Felder aus dem ViKl-F2F-Datensatz liefern Ground-Truth-Fakten pro Gespräch. Diese werden nicht zur Laufzeit in den Prompt gegeben (das würde das Steering mit Prompting konfundieren), sondern offline für Probe-Training und Evaluation verwendet.

**Wichtig:** Der Case erscheint an drei Stellen — aber *nicht* im Generierungs-Prompt:

| Verwendungsort | Zweck | Konfundiert Steering? |
|---|---|---|
| Probe-Training (offline) | U_case aufbauen | Nein |
| Vektorkonstruktion (offline) | Case-Richtungen orthogonalisieren | Nein |
| Evaluation (Referenz) | Fidelity gegen Ground Truth messen | Nein |
| Generierungs-Prompt | — | Ja — wird NICHT gemacht |

### Modul 2 — Persona-Subspace

Pro Persona-Achse (open↔defensive, explorative↔reactive, cooperative↔resistant, hopeful↔resigned) wird ein Steuerungsvektor aus kontrastiven Paaren extrahiert:

```
v_persona(a) = mean(h_pos) - mean(h_neg)
```

- Kontrastive Paare: 1.000 synthetische Paar-Kontinuationen pro Achse (GPT-5.4-mini, gleiche Beratungskontexte, Invarianzbedingungen)
- Layer-Auswahl per Layerwise Separability (AUC auf Grouped Holdout)
- Bestehende Experimente aus dem aktuellen Paper können direkt wiederverwendet werden

### Modul 3 — Case-Subspace U_case(x)

**Dies ist die methodische Mitte des Papers.**

Für jeden Kontext x werden die Richtungen in den Hidden States identifiziert, auf die Case-Probes sensitiv sind:

1. Trainiere binäre Probes auf `hauptanliegen`-Fakten:
   - "Wird Personenname N in der Antwort erwähnt?"
   - "Bleibt die genannte Substanz erhalten?"
   - "Werden Zeitbezüge aus dem Case reproduziert?"
2. Extrahiere Gradienten oder Probe-Gewichtsvektoren als Case-Richtungen
3. Fasse sie zu `U_case(x)` zusammen (kontextabhängiger Case-Subspace)

Die Case-Subspace-Dimensionen variieren je nach Fall: bei Fall 1 (Elke/Max) encodiert U_case andere Richtungen als bei Fall 5 (Susi/Ehemann).

### Modul 4 — Orthogonalisierte Intervention

Zur Laufzeit wird nur der Anteil des Persona-Vektors angewendet, der orthogonal zum Case-Subspace ist:

```
Δh = α · P⊥(U_case(x)) · v_persona(a)

wobei: P⊥(U) = I - U · U^T
```

**Was das bewirkt (Beispiel Openness-Achse):**

- `v_open` enthält sowohl "mehr selbstoffenbaren" als auch "mehr elaborieren" — und Elaboration korreliert mit Faktenneuheit
- `P⊥(U_case) · v_open` entfernt den Anteil, der Namen/Substanzen/Fakten gemeinsam verschiebt
- Übrig bleibt: Gesprächsstil, Selbstoffenbarungsbereitschaft, affektive Rahmung

### Modul 5 — Token-Gating (COPS-train Variante)

Optional: Intervention nicht gleichmäßig über alle Tokens, sondern positions-sensitiv:

- Stark: Tokens mit affektiver, metakognitiver, dialogpragmatischer Funktion
- Schwach/keine Intervention: Tokens mit hoher Faktensensitivität (Namen, Zahlen, Substanzen)

Gate-Bestimmung: durch Token-Klassifikation auf den ViKl-F2F-Kategorieannotationen (jede Nachricht ist bereits segmentweise annotiert mit Labels wie `Eigene Gefühlsdarstellung`, `Problemdefinition`, `Preisgeben persönlicher Daten`).

---

## 5. Zwei Varianten

### COPS-lite (training-light)

- Keine neuen trainierten Komponenten außer Case-Probes
- Orthogonalisierung: `P⊥(U_case) · v_persona`
- Schnell testbar auf bestehenden Benchmark-Daten
- Ablationsfreundliche Basis-Variante

### COPS-train (Full)

Gelernter Low-Rank-Controller mit Fidelity-Loss:

```
L = L_style
  + λ1 · L_case_consistency
  + λ2 · L_role_fidelity
  + λ3 · L_entity_time_stability
  + λ4 · ||Δh||²
```

- `L_style`: Klassifikator-Score auf Zielachse soll steigen
- `L_case_consistency`: Rekonstruierter Case-State soll stabil bleiben (gegen hauptanliegen gemessen)
- `L_role_fidelity`: Antwort soll weiterhin wie Klient klingen
- `L_entity_time_stability`: Explizite Strafe für Änderung von Entitäten/Zeitreferenzen aus dem Case
- `||Δh||²`: Verhindert aggressive Rewrites

---

## 6. Evaluation

### Primärmetriken

| Metrik | Messung |
|---|---|
| Style shift | Klassifikator-Delta (bestehend) |
| Hard fidelity | Entitäts-/Faktenübereinstimmung gegen hauptanliegen (neu) |
| Soft fidelity | Content preservation score (bestehend: Jaccard, Novelty, Drift) |
| Role fidelity | LLM-Judge (bestehend) |
| Training utility | LLM-Judge (bestehend) |

### Drei Drift-Kategorien (empirisch trennbar)

1. **Faktenwiderspruch** — messbar gegen hauptanliegen, sollte durch COPS reduziert werden
2. **Faktische Erfindung** — teilweise messbar, teilweise unterbestimmt
3. **Plausible Elaboration** — bleibt epistemisch offen; Judge bleibt hier das bessere Signal

**Zentraler empirischer Beitrag:** Zeige, welche Drift-Typen durch Case-Orthogonalisierung lösbar sind und welche nicht — das ist ehrlicher und informativer als ein einzelner Fidelity-Score.

### Ablationen

- Ohne Orthogonalisierung (= aktuelles Paired Dense)
- Ohne Case-Probes (globaler statt kontextueller Subspace)
- Mit Case im Prompt statt in U_case (Konfundierungscheck)
- Verschiedene Intervention Sites (Layer-Auswahl)
- COPS-lite vs. COPS-train

### Modell-Matrix

Bestehende sechs Modelle übernehmen: Qwen3-4B, Qwen3.5-2B/4B/9B, Gemma-4-E2B/E4B-it

---

## 7. Erster empirischer Check (vor vollem Methodenaufbau)

**Frage:** Wie viel Overlap haben `v_open` und die Case-Probe-Gradienten für Personennamen und Substanznennung?

Wenn der Overlap hoch ist → Orthogonalisierung ist der Kern des Beitrags.
Wenn er niedrig ist → das Problem liegt woanders, Methode muss angepasst werden.

**Aufwand:** Ein halber Tag. Beantwortet ob die Grundidee trägt, bevor die volle Methode implementiert wird.

---

## 8. Paper-Claim in einem Satz

> *We introduce case-faithful persona steering: a constrained activation intervention that separates stylistic directions from case-critical content directions in model representations, and characterize empirically which types of content drift this resolves and which remain inherently underdetermined.*

---

## 9. Interne Planungsnotizen (nicht für Paper)

> **Nur für Entwicklungsphase — vor Submission entfernen.**

- Detection-Control-Gap aus Vorarbeiten → motiviert constrained intervention
- r = −0.81 zwischen automatischen Metriken und Judge → motiviert Trennung der Drift-Typen
- Openness als härteste Achse → wird zum Hauptbeispiel für Case-Orthogonalisierung
- Bestehende Benchmark-Infrastruktur (shared eval manifests, Judge-Protokoll) kann direkt wiederverwendet werden

---

## 10. Offene Fragen

- Wie stabil ist U_case über verschiedene Gespräche desselben Falls?
- Skaliert die Case-Probe-Qualität mit der Anzahl Gespräche pro Fall (Fall 1: 20, Fall 5: 3)?
- Ist Token-Gating über ViKl-F2F-Kategorien ausreichend oder braucht es eigene Token-Annotation?
- Verknüpfung Rolling-Window-Kontexte → problemfall: wird dieser Identifier beim Sampling mitgeführt?

---

## 11. Praktische Steering-Evaluation (nächster Schritt)

Das Entanglement-Maß ist ein Proxy. Der entscheidende Test ist: **produziert ein weniger entangletes v_open auch weniger Drift bei der Generierung?**

### Experimental-Design

**Held-out Kontexte:** 50 COPS-Kontexte die nicht im Generation-Set waren (restliche 1073 nach Abzug der 200 für case-aware Paare).

**Steering-Varianten pro Kontext:**
1. **Baseline** — keine Intervention
2. **v_global_blind** — aktuelles gespeichertes v_open aus dem Benchmark
3. **v_global_aware** — frisches v_open aus case-aware Paaren
4. **v_P_matched** — persona-spezifischer Vektor, gleiche Persona wie Kontext
5. **v_P_mismatched** — persona-spezifischer Vektor, andere Persona (Kontrolle)

**Modelle:** Qwen3-4B + Gemma-4-E4B-it (beide Varianten pro Modell)

**Metriken:**
- **Directional shift** (gewollt): Klassifikator-Score defensive→open
- **Case-Fidelity** (ungewollt bei Drift): Hauptanliegen-Keyword-Retention
- **Persona-Drift** (ungewollt bei Drift): bleibt Persona-Probe-Label am gesteuerten Output der Ground-Truth-Persona zugeordnet?
- **Lexikalische Treue** (ungewollt bei Drift): Jaccard/Novelty gegen Baseline

### Erwartete Befunde

Wenn Hypothese trägt:
- **Qwen**: case-aware v_open reduziert Drift messbar vs case-blind v_open
- **Gemma**: case-aware macht kaum Unterschied (Basis schon sauber)
- **v_P_mismatched** produziert deutlich mehr Persona-Drift als v_P_matched

Wenn Hypothese nicht trägt (Negativresultat):
- Alle v_open-Varianten verhalten sich praktisch gleich
- Entanglement-Maß ist kein guter Prädiktor für tatsächlichen Drift
- Paper muss sich auf die architektonische Beobachtung (Qwen vs Gemma) konzentrieren

---

## 12. Paper-Architektur (Cross-Model Version)

1. **Introduction:** Steering-Paper beobachten Drift unterschiedlich zwischen Modellfamilien — bisher unerklärt
2. **Method:** Entanglement-Metrik (Projection auf Persona-Subspace) + case-aware Baseline
3. **Results:**
   - Qwen: hohe Persona-Entanglement, v_open läuft durch Persona-Subspace
   - Gemma: niedrige Persona-Entanglement, v_open disentangled
   - Case-aware Daten reduzieren Entanglement bei Qwen, nicht bei Gemma (kein Effekt weil Basis sauber)
4. **Validation:** Praktische Steering-Evaluation (Abschnitt 11) — korreliert Entanglement mit Drift?
5. **Discussion:** Implikation für Modellwahl und Dateenkuration in fidelity-sensitiven Anwendungen
