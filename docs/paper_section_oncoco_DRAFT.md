# DRAFT — Paper section for review (not yet integrated)

Proposed placement: new §7 between current §6 (IRR) and current §7 Discussion (which becomes §8). Title: **"Content Validation via OnCoCo Sentence Classification"**.

---

## §7 Content Validation via OnCoCo Sentence Classification

### 7.1 Motivation and Setup

The judge ensemble in §5 establishes that steering works on cooperation and hopefulness and is weak on openness and initiative, but the judge rubric scores *axis alignment* directly — the same construct the prompt is conditioning on, leaving open whether the effect is genuine content shift or judge-perceived stylistic noise. We run a fully independent content classifier on the generated responses to obtain converging evidence.

We use the OnCoCo XLM-RoBERTa-large sentence classifier of \citet{schmid2025oncoco}, trained on 8.5k labelled utterances from German online counseling chat, with a 28-class taxonomy of client utterance types (e.g.\ *Eigene Gefühlsdarstellung* — self-disclosure of feelings; *Bericht Umsetzung* — progress report; *Zustimmung* — agreement). Each generated response is split into sentences (German rule-based splitter with abbreviation masking), prepended with the `[CL]` role token used during OnCoCo fine-tuning, and classified with output masking (counselor-only logits suppressed before softmax, per the original protocol). This yields 86,716 sentence-level classifications across the four-axis × four-model × variant × $\alpha$ matrix.

### 7.2 Distributional Significance

For each (axis, pole) cell we test whether the predicted-label distribution differs between baseline and steered responses, using a 28-class $\chi^2$ contingency test with a **cluster-robust permutation correction** (200 permutations shuffling the baseline/steered assignment at the *response* level, since sentences from the same response are not independent).

\begin{table}[h]
\centering\small
\begin{tabular}{llrrr}
\toprule
\textbf{Axis} & \textbf{Pole} & $\chi^2(27)$ & \textbf{cluster $p$} & \textbf{Cramér's $V$} \\
\midrule
hopefulness & negative & 246.6 & 0.005 & 0.091 \\
hopefulness & positive &  90.4 & 0.005 & 0.084 \\
cooperation & negative & 163.9 & 0.005 & 0.074 \\
cooperation & positive &  62.2 & 0.005 & 0.070 \\
openness    & positive &  63.9 & 0.100 & 0.063 \\
openness    & negative &  34.3 & 0.249 & 0.034 \\
initiative  & negative &  28.7 & 0.483 & 0.031 \\
initiative  & positive &  19.5 & 0.846 & 0.039 \\
\bottomrule
\end{tabular}
\caption{OnCoCo distribution shift per (axis, pole). Cluster-robust permutation $p$ shuffles at the response level to control for within-response sentence dependence. Naïve asymptotic $\chi^2$ inflates significance roughly seven-fold.}
\label{tab:oncoco-global}
\end{table}

The pattern from the judge ensemble (§5) is **independently reproduced**: cooperation and hopefulness yield highly significant content shifts in both directions; openness and initiative do not change the predicted-label distribution. Effect sizes are modest in absolute terms (Cramér's $V \leq 0.09$) but are bounded above by the 28-class granularity — even a 25--30\% relative change in a 5\%-baseline class produces only a 1--2 percentage point absolute shift.

### 7.3 Which Categories Move

For the four significant cells, we report two-proportion $z$-tests per label with Benjamini--Hochberg FDR correction (28 comparisons). All shifts are semantically aligned with the steering target.

\begin{table}[h]
\centering\small
\begin{tabular}{lrrl}
\toprule
\textbf{Label} & \textbf{base\%} & \textbf{steer\%} & $\Delta$ \\
\midrule
\multicolumn{4}{l}{\textit{Hopefulness $\to$ resigned}} \\
Eigene Gefühlsdarst.    &  5.0 & 10.2 & +5.2*** \\
Misserfolg              &  1.1 &  3.7 & +2.6*** \\
Problemdarstellung      &  6.4 &  9.3 & +2.8*** \\
Bericht Umsetzung       &  8.6 &  5.4 & $-$3.2*** \\
Einwand                 &  4.7 &  2.5 & $-$2.2*** \\
Zustimmung              &  9.1 &  6.8 & $-$2.3** \\
\midrule
\multicolumn{4}{l}{\textit{Cooperation $\to$ resistant}} \\
Neg.\ RM zu Empfehlung  &  3.1 &  5.9 & +2.8*** \\
Misserfolg              &  1.1 &  2.6 & +1.5*** \\
Zustimmung              &  9.3 &  6.4 & $-$2.9*** \\
Allg.\ pos.\ Rückmeldung &  4.8 &  2.9 & $-$1.8*** \\
Pos.\ RM zu Empfehlung  &  3.6 &  2.1 & $-$1.5*** \\
Formales Abschluss      &  3.8 &  2.0 & $-$1.8*** \\
\bottomrule
\end{tabular}
\caption{Top per-label shifts after BH-FDR correction (negative pole, all models pooled, $|\alpha|{=}3.0$). *** $q<0.001$, ** $q<0.01$. Shifts at the positive pole are sign-symmetric. Full per-cell tables in Appendix~\ref{app:oncoco-full}.}
\label{tab:oncoco-perlabel}
\end{table}

The categories that increase under negative-pole steering are exactly the ones a clinically informed reader would predict: under \emph{resigned} the model produces more self-disclosure of distress (\emph{Eigene Gefühlsdarstellung}, \emph{Misserfolg}) and fewer progress reports or agreements; under \emph{resistant} it produces more negative feedback to recommendations and fewer agreements, polite acknowledgements, or formal closures.

### 7.4 Word-Level Effects on the Non-Significant Axes

The OnCoCo null on openness and initiative does **not** imply that steering fails on these axes; rather, the 28-class top-1 argmax is too coarse for the lexical changes that do occur. A Monroe log-odds analysis \citep{monroe2008fightin} of unigrams between baseline and $\alpha=-3.0$ responses produces stable, model-replicated signatures for all four axes:

| Axis (negative pole) | Top words ↑ steered | Top words ↓ steered |
|---|---|---|
| openness/defensive | hab, einfach, okay, viel, alles, melde | fühle, fühlt, angst, sprechen, verstehen, allein |
| cooperation/resistant | weiß, einfach, will, sicher, schaffe | können, gemeinsam, danke, gerne, klingt, gut |
| hopefulness/resigned | weiß, soll, mehr, alles, fühle, glaube | vielleicht, manchmal, hoffe, versuche, danke |
| initiative/reactive | okay, will, soll, nie, könnte | verstehen, ändern, andere, kontrollieren, selbst |

For openness specifically, paired per-seed metrics (Qwen3-4B, $\alpha=-3.0$, $n=200$) show a **−1.0 PP drop in emotion-word frequency** (relative −33\%), a **−13.5 PP drop in the share of responses containing any emotion word**, and a **−1.4-word reduction in mean response length**. The lexical signal is unambiguous; what is missing is enough mass for OnCoCo's argmax to flip categories. The same observation explains the asymmetry in §5: when the underlying construct shift is borderline, both an LLM-as-judge rubric and a fine-grained classifier may register it weakly, even if individual word distributions move.

### 7.5 Convergent Validity Summary

OnCoCo, an external classifier trained on real (non-LLM) German counseling chat data, replicates the headline pattern from §5: strong, semantically directed effects on cooperation and hopefulness; subthreshold effects on openness and initiative under top-1 argmax, but a clear lexical signature still present at the word level. We treat this as content validation that the judge ensemble in §5 is detecting genuine construct movement rather than rubric artefacts.

---

## Appendix additions

### A.x OnCoCo Classification Pipeline

- **Model:** `xlm-roberta-large-OnCoCo-DE-EN`, 68 labels (40 CO-, 28 CL-).
- **Inference:** sentence-level. Each generated response is split with a German rule-based splitter that masks 14 common abbreviations (`Dr.`, `z.B.`, `d.h.`, ...) before splitting on `[.!?…]\s+(?=[A-ZÄÖÜ])` to avoid false breaks; we always emit at least the original text. 86{,}716 sentences from 45{,}000 responses (median 2 sentences/response, 54\% multi-sentence).
- **Role conditioning:** the special token `[CL]` is prepended to every input, matching the OnCoCo training format. Output masking suppresses all `CO-*` logits to $-\infty$ before softmax \citep{schmid2025oncoco}. After masking, 0\% of predictions fall in counselor categories (vs. 26.8\% without masking, an internal sanity check).
- **Compute:** ~12 minutes on a single A40 (Slurm partition `p6`, batch size 256). Output: one JSONL row per sentence with top-5 labels and probabilities. Reproduction: `slurm/generated/classify_oncoco_kiz0.sh`.

### A.y Cluster-Robust Permutation Procedure

For each (axis, pole) cell we:
1. Compute the observed $\chi^2_{27}$ statistic for the 28×2 contingency table of CL-* labels (baseline vs steered).
2. Build a response-level table where each response is a unit with its sentence-level label list.
3. Permute the baseline/steered group assignment **at the response level** 200 times. For each permutation, recompute the contingency table by aggregating sentences within the permuted groups, then recompute $\chi^2$.
4. Report $p_\text{cluster} = \tfrac{|\{ \chi^2_\text{perm} \geq \chi^2_\text{obs} \}| + 1}{200 + 1}$.

The cluster-robust $p$ is materially larger than the asymptotic $p$ on three of eight cells (e.g.\ openness/positive: $7.8\!\times\!10^{-5} \to 0.10$). We use the cluster-robust value for inference throughout §7.2.

### A.z Full Per-Cell OnCoCo Tables and Word-Level Statistics

[Reference: `outputs/figures/oncoco_full_dist_negative_all.pdf`, `oncoco_full_dist_positive_all.pdf` — grouped horizontal bar charts per axis with all CL-* labels at $\geq 0.5\%$ in either group.]

[Per-model Monroe log-odds tables, $z$-thresholded at $|z|\geq 1.5$, in the released artefact under `outputs/figures/oncoco_lexical_signatures.csv` — table to be assembled if accepted.]

### A.w Qualitative Paired Examples

(Three to five paired baseline/steered examples per axis, picked from clear-shift seeds, with German source and translation. Already collected; can be inserted as a small TikZ-styled box if space allows.)

---

## Open questions for you to confirm

1. **Placement.** New section between current §6 IRR and §7 Discussion (renumber), or fold as a subsection of §6 ("Content Validation")? I prefer a new section because the analysis is methodologically distinct (independent classifier, not human/LLM judge).

2. **Length.** ~1.0 column as drafted. If we need to compress for ACL page budget I can drop §7.4 (word-level Monroe analysis) into the appendix and keep just §7.1–7.3 + 7.5 in main text.

3. **OnCoCo citation key.** I used `\citep{schmid2025oncoco}` as a placeholder — what is the actual bib key? If the paper isn't in `paper/bib/` yet I can add it.

4. **Discussion section.** Should I also add 1–2 paragraphs to §7 Discussion (now §8) tying the OnCoCo null on openness/initiative to the limitations narrative? The natural framing is: "fine-grained classifiers may still register weak axes if used continuously rather than top-1, suggesting future work."

5. **Tables in main text.** Two small tables as drafted (8 rows × 5 cols + 12 rows × 4 cols) — fits one column. OK?

6. **Anything to remove?** The qualitative examples (Appendix A.w) are nice but optional. Likewise the per-model word-level table. Tell me if either stays out.
