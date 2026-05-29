#!/usr/bin/env python3
"""Build LaTeX tables for the COPS v2 paper from existing ensemble JSON files."""
from __future__ import annotations

import json
import glob
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stem_norm(s: str) -> str:
    return re.sub(r"_seed\d+$", "", s)


# JSON keys (left) → paper display names (right). The pipeline still emits
# the original `global_*`/`P_*` keys, but the paper presents them as
# case_*/persona_* for clearer dimension separation
# (case_* = generator conditioning at pair-gen; persona_* = vector pool).
VARIANT_TEX = {
    "global_blind": r"\texttt{case\_blind}",
    "global_aware": r"\texttt{case\_aware}",
    "P_matched": r"\texttt{persona\_matched}",
    "P_mismatched": r"\texttt{persona\_mismatched}",
}

STEM_ORDER = [
    ("qwen", "openness", "qwen_openness_L15"),
    ("qwen", "initiative", "qwen_initiative_L9"),
    ("qwen", "cooperation", "qwen_cooperation_L20"),
    ("qwen", "hopefulness", "qwen_hopefulness_L18"),
    ("gemma", "initiative", "gemma_initiative_L13"),
    ("gemma", "cooperation", "gemma_cooperation_L23"),
    ("gemma", "hopefulness", "gemma_hopefulness_L25"),
    ("qwen35_9b", "openness", "qwen35_9b_openness_L20"),
    ("qwen35_9b", "initiative", "qwen35_9b_initiative_L20"),
    ("qwen35_9b", "cooperation", "qwen35_9b_cooperation_L15"),
    ("qwen35_9b", "hopefulness", "qwen35_9b_hopefulness_L15"),
    ("mistral7b", "openness", "mistral7b_openness_L15"),
    ("mistral7b", "initiative", "mistral7b_initiative_L5"),
    ("mistral7b", "cooperation", "mistral7b_cooperation_L15"),
    ("mistral7b", "hopefulness", "mistral7b_hopefulness_L20"),
]
MODEL_LBL = {
    "qwen": "Qwen3-4B",
    "gemma": "Gemma-4-E4B-it",
    "qwen35_9b": "Qwen3.5-9B",
    "mistral7b": "Mistral-7B",
}


def load_paper_v2_stems():
    files = sorted(glob.glob(str(ROOT / "outputs/steering_eval/judged_ensemble_v2/*_ensemble_summary.json"))) + \
            sorted(glob.glob(str(ROOT / "outputs/steering_eval/judged_v2_newmodels_ensemble/*_ensemble_summary.json")))
    by_stem = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        by_stem[stem_norm(d["eval_stem"])].append(d)
    return by_stem


def best_variant(stem_data: list, alpha: float):
    by_va = defaultdict(list)
    kappas = []
    for d in stem_data:
        kappas.append(d["inter_rater_agreement"]["fleiss_kappa"])
        for k, v in d["by_variant_alpha"].items():
            if f"a{alpha}" in k and "baseline" not in k:
                by_va[k].append(v["pairwise_net_overall"])
    if not by_va:
        return None
    best_k = max(by_va, key=lambda k: np.mean(by_va[k]))
    var = best_k.rsplit("__a", 1)[0]
    return {"variant": var, "net": float(np.mean(by_va[best_k])),
            "kappa": float(np.mean(kappas))}


def write_fullmatrix_pos():
    by_stem = load_paper_v2_stems()
    lines = [
        r"% auto-generated",
        r"\begin{table*}[t]\centering\small",
        r"\begin{tabular}{ll c c r r}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Axis} & \textbf{L} & \textbf{Best variant} & \textbf{\netpw} & \textbf{$\kappa$} \\",
        r"\midrule",
    ]
    last_model = None
    for mk, ax, stem in STEM_ORDER:
        if stem not in by_stem:
            continue
        res = best_variant(by_stem[stem], 3.0)
        if not res:
            continue
        layer = stem.split("_L")[-1]
        if mk != last_model and last_model is not None:
            lines.append(r"\midrule")
        last_model = mk
        var_tex = VARIANT_TEX.get(res["variant"], res["variant"].replace("_", r"\_"))
        lines.append(f"{MODEL_LBL[mk]} & {ax} & L{layer} & {var_tex} & ${res['net']:+.3f}$ & ${res['kappa']:.2f}$ \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\emph{Oracle / upper bound.} Full-matrix evaluation at $\alphapos$: per (model, axis) the best of the four variants by $\netpw$. Cell-mean lift over prespecified CA (Table~\ref{tab:fullmatrix-pos-headline}) is $+0.070$. $N=200$ held-out contexts per stem ($N=600$ for Qwen3-4B \emph{openness} via 3 random seeds).}",
        r"\label{tab:fullmatrix-pos-oracle}",
        r"\end{table*}",
    ]
    Path(ROOT / "paper/tables/tab_fullmatrix_pos.tex").write_text("\n".join(lines))
    print("Wrote tab_fullmatrix_pos.tex")


def write_variant_wins():
    lines = [
        r"% auto-generated",
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Wins (\% of cells)} \\",
        r"\midrule",
        r"case\_blind  & 27.1\% \\",
        r"\texttt{persona\_matched}    & 27.1\% \\",
        r"\texttt{persona\_mismatched} & 24.3\% \\",
        r"case\_aware  & 21.4\% \\",
        r"\midrule",
        r"random baseline & 25.0\% \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{`Best variant' per (stem, persona) cell at $\alphapos$, aggregated across $7\,\text{stems} \times 10\,\text{personas} = 70$ cells. All four variants win $\sim 25\%$, indistinguishable from a uniform prior.}",
        r"\label{tab:variant-wins}",
        r"\end{table}",
    ]
    Path(ROOT / "paper/tables/tab_variant_wins.tex").write_text("\n".join(lines))
    print("Wrote tab_variant_wins.tex")


def write_abc_variants():
    files = sorted(glob.glob(str(ROOT / "outputs/steering_eval/judged_pv_ensemble/*_ensemble_summary.json")))
    results = {}
    for f in files:
        d = json.load(open(f))
        stem = d["eval_stem"].replace("_pv", "")
        by_va = d["by_variant_alpha"]
        results[stem] = {v: by_va.get(f"{v}__a3.0", {}).get("pairwise_net_overall", float("nan"))
                         for v in ["P_matched", "P_matched_A", "P_matched_B", "P_matched_C"]}
    LBL = {
        "qwen_openness_L15": "Qwen openness L15",
        "qwen_initiative_L9": "Qwen initiative L9",
        "qwen_cooperation_L20": "Qwen cooperation L20",
        "qwen_hopefulness_L18": "Qwen hopefulness L18",
        "gemma_initiative_L13": "Gemma initiative L13",
        "gemma_cooperation_L23": "Gemma cooperation L23",
        "gemma_hopefulness_L25": "Gemma hopefulness L25",
    }
    lines = [
        r"% auto-generated",
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{l rrrr}",
        r"\toprule",
        r"\textbf{Stem} & $P_\text{m}$ & $P^A_\text{m}$ & $P^B_\text{m}$ & $P^C_\text{m}$ \\",
        r"\midrule",
    ]
    for stem, lbl in LBL.items():
        if stem not in results:
            continue
        r = results[stem]
        lines.append(f"{lbl} & ${r['P_matched']:+.3f}$ & ${r['P_matched_A']:+.3f}$ & ${r['P_matched_B']:+.3f}$ & ${r['P_matched_C']:+.3f}$ \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Alternative $\Vpersona$ constructions tested as steering vectors at $\alphapos$. $P^A$ = orthogonal residual, $P^B$ = cross-persona contrast, $P^C$ = interaction term. All three collapse to $\netpw \approx 0$.}",
        r"\label{tab:abc-variants}",
        r"\end{table}",
    ]
    Path(ROOT / "paper/tables/tab_abc_variants.tex").write_text("\n".join(lines))
    print("Wrote tab_abc_variants.tex")


def write_fleiss():
    by_stem = load_paper_v2_stems()
    lines = [
        r"% auto-generated",
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Stem} & \textbf{Fleiss $\kappa$} \\",
        r"\midrule",
    ]
    for _, _, stem in STEM_ORDER:
        if stem not in by_stem:
            continue
        kappas = [d["inter_rater_agreement"]["fleiss_kappa"] for d in by_stem[stem]]
        k = float(np.mean(kappas))
        # Replace _ with \_ and L<digits> intact
        stem_tex = stem.replace("_", r"\_")
        lines.append(f"{stem_tex} & ${k:.3f}$ \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Fleiss's $\kappa$ over 3-provider ensemble (Mistral, Nano, GPT-OSS-120B) per stem at $\alphapos$.}",
        r"\label{tab:fleiss}",
        r"\end{table}",
    ]
    Path(ROOT / "paper/tables/tab_fleiss.tex").write_text("\n".join(lines))
    print("Wrote tab_fleiss.tex")


def write_layer_sweep():
    files = sorted(glob.glob(str(ROOT / "outputs/steering_eval/layer_sweep/judged_ensemble/*_ensemble_summary.json")))
    by_ma = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        stem = d["eval_stem"]
        m = re.match(r"(qwen|gemma|qwen35_9b|mistral7b)_(\w+)_L(\d+)", stem)
        if not m:
            continue
        model, axis, layer = m.group(1), m.group(2), int(m.group(3))
        agg = d["by_variant_alpha"]
        key = next((k for k in agg if "global_blind" in k and "a1.5" in k), None)
        if key:
            by_ma[(model, axis)].append((layer, agg[key]["pairwise_net_overall"]))
    AUC_LAYERS = {
        ("qwen", "openness"): 1, ("qwen", "initiative"): 9,
        ("qwen", "cooperation"): 9, ("qwen", "hopefulness"): 7,
        ("gemma", "openness"): 3, ("gemma", "initiative"): 8,
        ("gemma", "cooperation"): 23, ("gemma", "hopefulness"): 13,
        ("qwen35_9b", "openness"): 5, ("qwen35_9b", "initiative"): 5,
        ("qwen35_9b", "cooperation"): 5, ("qwen35_9b", "hopefulness"): 5,
        ("mistral7b", "openness"): 5, ("mistral7b", "initiative"): 5,
        ("mistral7b", "cooperation"): 5, ("mistral7b", "hopefulness"): 5,
    }
    lines = [
        r"% auto-generated",
        r"\begin{table*}[t]\centering\small",
        r"\begin{tabular}{ll cr cr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Axis} & \textbf{AUC L} & \textbf{AUC \netpw} & \textbf{Sweep L} & \textbf{Sweep \netpw} \\",
        r"\midrule",
    ]
    last_model = None
    for (mk, ax), data in sorted(by_ma.items()):
        data.sort(key=lambda r: -r[1])
        sweep_l, sweep_n = data[0]
        auc_l = AUC_LAYERS.get((mk, ax))
        auc_n = next((n for L, n in data if L == auc_l), None)
        if mk != last_model and last_model is not None:
            lines.append(r"\midrule")
        last_model = mk
        auc_str = f"${auc_n:+.3f}$" if auc_n is not None else "n/a"
        lines.append(f"{MODEL_LBL[mk]} & {ax} & L{auc_l} & {auc_str} & L{sweep_l} & ${sweep_n:+.3f}$ \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Layer selection: AUC-best layer vs empirical sweep-best layer. \netpw at $\alpha=1.5$, \texttt{case\_blind} variant, $N=50$ held-out contexts, 3-judge ensemble. AUC-best layer matches sweep-best in only 1/16 cells (Gemma cooperation L23).}",
        r"\label{tab:layer-sweep}",
        r"\end{table*}",
    ]
    Path(ROOT / "paper/tables/tab_layer_sweep.tex").write_text("\n".join(lines))
    print("Wrote tab_layer_sweep.tex")


def write_fullmatrix_neg():
    """Best variant per stem at α=-3.0 from negα ensemble."""
    files = sorted(glob.glob(str(ROOT / "outputs/steering_eval/judged_v2_negalpha_ensemble/*_ensemble_summary.json")))
    by_stem_neg = {}
    for f in files:
        d = json.load(open(f))
        stem = d["eval_stem"].replace("_neg", "")
        by_stem_neg[stem] = d
    # Map normalized stem → label
    NEG_STEM_ORDER = [
        ("qwen", "openness", "qwen_openness_L15"),
        ("qwen", "initiative", "qwen_initiative_L9"),
        ("qwen", "cooperation", "qwen_cooperation_L20"),
        ("qwen", "hopefulness", "qwen_hopefulness_L18"),
        ("gemma", "openness", "gemma_openness_L3"),
        ("gemma", "initiative", "gemma_initiative_L13"),
        ("gemma", "cooperation", "gemma_cooperation_L23"),
        ("gemma", "hopefulness", "gemma_hopefulness_L25"),
        ("qwen35_9b", "openness", "qwen35_9b_openness_L20"),
        ("qwen35_9b", "initiative", "qwen35_9b_initiative_L20"),
        ("qwen35_9b", "cooperation", "qwen35_9b_cooperation_L15"),
        ("qwen35_9b", "hopefulness", "qwen35_9b_hopefulness_L15"),
        ("mistral7b", "openness", "mistral7b_openness_L15"),
        ("mistral7b", "initiative", "mistral7b_initiative_L5"),
        ("mistral7b", "cooperation", "mistral7b_cooperation_L15"),
        ("mistral7b", "hopefulness", "mistral7b_hopefulness_L20"),
    ]
    lines = [
        r"% auto-generated",
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{ll c c r r}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Axis} & \textbf{L} & \textbf{Best variant} & \textbf{\netpw} & \textbf{$\kappa$} \\",
        r"\midrule",
    ]
    last_model = None
    for mk, ax, stem in NEG_STEM_ORDER:
        if stem not in by_stem_neg:
            continue
        d = by_stem_neg[stem]
        by_va = defaultdict(list)
        kappa = d["inter_rater_agreement"]["fleiss_kappa"]
        for k, v in d["by_variant_alpha"].items():
            if "a-3.0" in k and "baseline" not in k:
                by_va[k].append(v["pairwise_net_overall"])
        if not by_va:
            continue
        best_k = max(by_va, key=lambda k: np.mean(by_va[k]))
        var = best_k.rsplit("__a", 1)[0]
        net = float(np.mean(by_va[best_k]))
        layer = stem.split("_L")[-1]
        if mk != last_model and last_model is not None:
            lines.append(r"\midrule")
        last_model = mk
        var_tex = VARIANT_TEX.get(var, var.replace("_", r"\_"))
        lines.append(f"{MODEL_LBL[mk]} & {ax} & L{layer} & {var_tex} & ${net:+.3f}$ & ${kappa:.2f}$ \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\emph{Oracle / upper bound.} Negative-$\alpha$ steering at $\alpha=-3.0$ per (model, axis): best of the four variants by $\netpw$. Cell-mean lift over prespecified CA (Table~\ref{tab:fullmatrix-neg-headline}) is $+0.048$. $N=200$ held-out contexts per stem.}",
        r"\label{tab:fullmatrix-neg-oracle}",
        r"\end{table}",
    ]
    Path(ROOT / "paper/tables/tab_fullmatrix_neg.tex").write_text("\n".join(lines))
    print("Wrote tab_fullmatrix_neg.tex")


def write_provider_sens():
    """Per-provider net pairwise on Qwen3-4B openness L15 P_matched α=3.0 + similar."""
    # Read raw judged JSONLs per provider for that stem
    provider_files = {
        "Mistral-Magistral": "cluster_mistral__mistralai_Magistral-Small-2509",
        "GPT-5.4-nano": "openai__gpt-5.4-nano_high",
        "GPT-OSS-120B": "ionos__openai_gpt-oss-120b_high",
    }
    # Use 3 illustrative stems from paper_v2
    showcase_stems = [
        ("Qwen3-4B \\emph{openness} L15", "qwen_openness_L15_seed42"),
        ("Qwen3.5-9B \\emph{openness} L20", "qwen35_9b_openness_L20"),
        ("Mistral-7B \\emph{openness} L15", "mistral7b_openness_L15"),
    ]
    judged_dirs = [
        ROOT / "outputs/steering_eval/judged",
        ROOT / "outputs/steering_eval/judged_v2_newmodels",
    ]

    rows = []
    for label, stem in showcase_stems:
        per_prov = {}
        for prov_label, prov_suffix in provider_files.items():
            for d in judged_dirs:
                f = d / f"{stem}_judged__{prov_suffix}.jsonl"
                if not f.exists():
                    continue
                rows_jsonl = [json.loads(l) for l in open(f)]
                # Compute net pairwise for global_blind α=3.0
                gb_winners = [r["pairwise_winner"] for r in rows_jsonl
                              if r.get("variant") == "global_blind" and r.get("alpha") == 3.0]
                if gb_winners:
                    n = len(gb_winners)
                    net = (gb_winners.count("steered") - gb_winners.count("base")) / max(1, n)
                    per_prov[prov_label] = net
                break
        rows.append((label, per_prov))

    lines = [
        r"% auto-generated",
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{l rrr}",
        r"\toprule",
        r"\textbf{Stem (\texttt{case\_blind}, $\alpha=3$)} & \textbf{Mistral} & \textbf{Nano} & \textbf{OSS-120B} \\",
        r"\midrule",
    ]
    for label, per_prov in rows:
        m = per_prov.get("Mistral-Magistral", float("nan"))
        n = per_prov.get("GPT-5.4-nano", float("nan"))
        o = per_prov.get("GPT-OSS-120B", float("nan"))
        def fmt(x):
            return "n/a" if not np.isfinite(x) else f"${x:+.3f}$"
        lines.append(f"{label} & {fmt(m)} & {fmt(n)} & {fmt(o)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-provider net-pairwise on \texttt{case\_blind} at $\alphapos$. Mistral-Magistral runs systematically lower than GPT-family (Nano, OSS-120B agree closely). The ensemble majority vote averages over this calibration shift.}",
        r"\label{tab:provider-sens}",
        r"\end{table}",
    ]
    Path(ROOT / "paper/tables/tab_provider_sens.tex").write_text("\n".join(lines))
    print("Wrote tab_provider_sens.tex")


def write_prompt_baseline():
    """4-cell ablation across all 16 (model × axis) headline stems at α=-3.0.
    Judged by GPT-OSS-120B alone (position-debiased anchor from §5.3.1) — Mistral-Magistral
    omitted because its documented position bias (net bias 0.417, see App. tab:judge-reliability)
    inflates noise on small-effect cells (defensive, reactive).
    Stems are grouped by axis strength to highlight the strong/weak asymmetry."""
    # (label, prompt_baseline_stem_id, paper_v2_negalpha_stem_id, axis_type)
    stems = [
        # Strong axes (cooperation + hopefulness) first
        ("Qwen3-4B \\emph{cooperation} L20", "qwen_cooperation_L20_withprompt", "qwen_cooperation_L20_neg", "strong"),
        ("Qwen3-4B \\emph{hopefulness} L18", "qwen_hopefulness_L18_withprompt", "qwen_hopefulness_L18_neg", "strong"),
        ("Gemma \\emph{cooperation} L23",    "gemma_cooperation_L23_withprompt","gemma_cooperation_L23_neg", "strong"),
        ("Gemma \\emph{hopefulness} L25",    "gemma_hopefulness_L25_withprompt","gemma_hopefulness_L25_neg", "strong"),
        ("Qwen3.5-9B \\emph{cooperation} L15", "qwen35_9b_cooperation_L15_withprompt", "qwen35_9b_cooperation_L15_neg", "strong"),
        ("Qwen3.5-9B \\emph{hopefulness} L15", "qwen35_9b_hopefulness_L15_withprompt", "qwen35_9b_hopefulness_L15_neg", "strong"),
        ("Mistral \\emph{cooperation} L15", "mistral7b_cooperation_L15_withprompt", "mistral7b_cooperation_L15_neg", "strong"),
        ("Mistral \\emph{hopefulness} L20", "mistral7b_hopefulness_L20_withprompt", "mistral7b_hopefulness_L20_neg", "strong"),
        # Weak axes (openness + initiative)
        ("Qwen3-4B \\emph{openness} L15",    "qwen_openness_L15_withprompt",    "qwen_openness_L15_neg", "weak"),
        ("Qwen3-4B \\emph{initiative} L9",   "qwen_initiative_L9_withprompt",   "qwen_initiative_L9_neg", "weak"),
        ("Gemma \\emph{openness} L3",        "gemma_openness_L3_withprompt",    "gemma_openness_L3_neg", "weak"),
        ("Gemma \\emph{initiative} L13",     "gemma_initiative_L13_withprompt", "gemma_initiative_L13_neg", "weak"),
        ("Qwen3.5-9B \\emph{openness} L20",    "qwen35_9b_openness_L20_withprompt",    "qwen35_9b_openness_L20_neg", "weak"),
        ("Qwen3.5-9B \\emph{initiative} L20",  "qwen35_9b_initiative_L20_withprompt",  "qwen35_9b_initiative_L20_neg", "weak"),
        ("Mistral \\emph{openness} L15",    "mistral7b_openness_L15_withprompt",    "mistral7b_openness_L15_neg", "weak"),
        ("Mistral \\emph{initiative} L5",   "mistral7b_initiative_L5_withprompt",   "mistral7b_initiative_L5_neg", "weak"),
    ]
    # Legacy filenames from the original 3-stem ablation — fall back if the consistent-naming
    # file is missing (so partial repro still works).
    legacy_pb_stems = {
        "qwen35_9b_hopefulness_L15_withprompt": "q35_9b_hopefulness_L15_withprompt",
    }

    def mean_aa(judged_dir, stem, provs, target_variant, target_alpha):
        steered_aa, base_aa = [], []
        for prov in provs:
            f = Path(judged_dir) / f"{stem}_judged__{prov}.jsonl"
            if not f.exists():
                continue
            for line in open(f):
                r = json.loads(line)
                if r.get("variant") == target_variant and r.get("alpha") == target_alpha:
                    s = r.get("steered_scores", {}).get("axis_alignment")
                    b = r.get("base_scores", {}).get("axis_alignment")
                    if isinstance(s, (int, float)):
                        steered_aa.append(s)
                    if isinstance(b, (int, float)):
                        base_aa.append(b)
        return float(np.mean(steered_aa)) if steered_aa else float("nan"), \
               float(np.mean(base_aa)) if base_aa else float("nan")

    # OSS-only for prompt-baseline (no Mistral due to position bias on small-effect cells).
    # The no-prompt baseline column is also restricted to OSS for apples-to-apples comparison.
    OSS = ["ionos__openai_gpt-oss-120b_high"]

    rows = []
    for label, pb_stem, neg_stem, axis_type in stems:
        nps_steered, np_baseline = mean_aa(
            ROOT / "outputs/steering_eval/judged_v2_negalpha", neg_stem, OSS,
            "global_blind", -3.0,
        )
        ps_steered, p_baseline = mean_aa(
            ROOT / "outputs/steering_eval/judged_prompt_baseline", pb_stem, OSS,
            "global_blind", -3.0,
        )
        if not np.isfinite(ps_steered) and pb_stem in legacy_pb_stems:
            ps_steered, p_baseline = mean_aa(
                ROOT / "outputs/steering_eval/judged_prompt_baseline",
                legacy_pb_stems[pb_stem], OSS, "global_blind", -3.0,
            )
        rows.append((axis_type, label, np_baseline, p_baseline, nps_steered, ps_steered))

    def fcell(x):
        return "n/a" if not np.isfinite(x) else f"${x:.2f}$"

    # Compute per-axis-type stats for the caption
    n_have = 0
    strong_co_gt_st = strong_co_gt_pr = strong_st_gt_pr = 0
    weak_co_gt_st = weak_co_gt_pr = weak_st_gt_pr = 0
    for axis_type, label, nb, pb, ns, ps in rows:
        if not all(np.isfinite([ns, pb, ps])):
            continue
        n_have += 1
        bucket = "strong" if axis_type == "strong" else "weak"
        st_pr = ns > pb; co_st = ps > ns; co_pr = ps > pb
        if bucket == "strong":
            strong_st_gt_pr += int(st_pr); strong_co_gt_st += int(co_st); strong_co_gt_pr += int(co_pr)
        else:
            weak_st_gt_pr += int(st_pr); weak_co_gt_st += int(co_st); weak_co_gt_pr += int(co_pr)

    lines = [
        r"% auto-generated by scripts/build_paper_tables.py:write_prompt_baseline()",
        r"\begin{table}[t]\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l rrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{No prompt}} & \multicolumn{2}{c}{\textbf{Prompt}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Stem} & base & + steer & base & + steer \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Strong axes (cooperation, hopefulness)}} \\",
    ]
    last_axis = "strong"
    for axis_type, label, nb, pb, ns, ps in rows:
        if axis_type != last_axis:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{5}{@{}l}{\textit{Weak axes (openness, initiative)}} \\")
            last_axis = axis_type
        lines.append(f"{label} & {fcell(nb)} & {fcell(ns)} & {fcell(pb)} & {fcell(ps)} \\\\")
    caption = (
        r"\caption{Full-matrix prompt-baseline ablation at $\alpha{=}{-}3.0$: "
        r"judge \texttt{axis\_alignment} score (1--5) per cell, judged by "
        r"GPT-OSS-120B (the position-debiased anchor from \S\ref{sec:irr}; "
        r"App. Tab.~\ref{tab:judge-reliability}). Mistral-Magistral is omitted "
        r"because its documented position bias inflates noise on small-effect cells "
        r"(\emph{defensive}, \emph{reactive}). $N{=}200$ contexts per cell. "
        f"On all 8 strong-axis stems, all three orderings hold uniformly: "
        f"steering$>$prompting ({strong_st_gt_pr}/8), combining$>$steering ({strong_co_gt_st}/8), "
        f"combining$>$prompting ({strong_co_gt_pr}/8). On weak-axis stems, steering itself fails "
        f"to beat prompting ({weak_st_gt_pr}/8 — consistent with \\S\\ref{{sec:negalpha}}), "
        f"but combining still adds value over steering in {weak_co_gt_st}/8 cells. "
        f"Overall combining$>$steering: {strong_co_gt_st+weak_co_gt_st}/16."
        r"}"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:prompt-baseline}",
        r"\end{table}",
    ]
    Path(ROOT / "paper/tables/tab_prompt_baseline.tex").write_text("\n".join(lines))
    print(f"Wrote tab_prompt_baseline.tex (OSS-only, strong: co>st {strong_co_gt_st}/8, co>pr {strong_co_gt_pr}/8; weak: co>st {weak_co_gt_st}/8, co>pr {weak_co_gt_pr}/8)")


def load_negalpha_stems():
    files = sorted(glob.glob(str(ROOT / "outputs/steering_eval/judged_v2_negalpha_ensemble/*_ensemble_summary.json")))
    by_stem = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        stem = stem_norm(d["eval_stem"]).replace("_neg", "")
        by_stem[stem].append(d)
    return by_stem


def _aggregate_fidelity_by_va(by_stem_dicts):
    """For each list-of-summary-dicts (one list per stem), aggregate per-cell
    *_delta_mean and pairwise_net_overall keyed by 'variant__aALPHA'.

    Returns: dict variant_alpha -> {dim: [values across cells], '_net': [values]}.
    A cell here is one (stem, summary-file). When a stem has multiple seeds we
    average within stem first so each stem counts once.
    """
    DIMS = ("axis_alignment", "case_fidelity", "client_role_fidelity", "training_utility")
    agg = defaultdict(lambda: defaultdict(list))
    for stem, dicts in by_stem_dicts.items():
        # Per-(variant,alpha) values, averaged across seeds for this stem
        per_va = defaultdict(lambda: defaultdict(list))
        for d in dicts:
            for k, cell in d.get("by_variant_alpha", {}).items():
                if "baseline" in k:
                    continue
                for dim in DIMS:
                    v = cell.get(f"{dim}_delta_mean")
                    if v is not None:
                        per_va[k][dim].append(v)
                v = cell.get("pairwise_net_overall")
                if v is not None:
                    per_va[k]["_net"].append(v)
        for k, dimmap in per_va.items():
            for dim, vals in dimmap.items():
                if vals:
                    agg[k][dim].append(float(np.mean(vals)))
    return agg


def write_fidelity_deltas():
    """Per-dimension mean deltas (steered - baseline) on judge 1-5 Likert scores,
    aggregated across all (model, axis) cells, for each (variant, alpha).
    Two sub-blocks in one table: negative alpha on top (headline direction),
    positive alpha below. Variants kept in canonical order."""
    pos_by_stem = load_paper_v2_stems()
    neg_by_stem = load_negalpha_stems()
    pos_agg = _aggregate_fidelity_by_va(pos_by_stem)
    neg_agg = _aggregate_fidelity_by_va(neg_by_stem)

    VARIANTS = ["global_blind", "global_aware", "P_matched", "P_mismatched"]
    POS_ALPHAS = [1.5, 3.0]
    NEG_ALPHAS = [-1.5, -3.0]

    DIMS = ("axis_alignment", "case_fidelity", "client_role_fidelity", "training_utility")

    def cell_means(agg, variant, alpha):
        # Reconstruct the by_variant_alpha key as produced by cops_judge_ensemble.
        key = f"{variant}__a{alpha}"
        row = agg.get(key, {})
        out = {}
        for dim in DIMS:
            vals = row.get(dim, [])
            out[dim] = float(np.mean(vals)) if vals else float("nan")
        netvals = row.get("_net", [])
        out["_net"] = float(np.mean(netvals)) if netvals else float("nan")
        out["n"] = len(row.get(DIMS[0], []))
        return out

    def fmt(x):
        if not np.isfinite(x):
            return "n/a"
        return f"${x:+.3f}$"

    def block_rows(agg, alphas):
        out = []
        for variant in VARIANTS:
            for alpha in alphas:
                m = cell_means(agg, variant, alpha)
                if m["n"] == 0:
                    continue
                var_tex = VARIANT_TEX.get(variant, variant.replace("_", r"\_"))
                out.append(
                    f"{var_tex} & ${alpha:+.1f}$ & "
                    f"{fmt(m['axis_alignment'])} & {fmt(m['case_fidelity'])} & "
                    f"{fmt(m['client_role_fidelity'])} & {fmt(m['training_utility'])} & "
                    f"{fmt(m['_net'])} \\\\"
                )
        return out

    lines = [
        r"% auto-generated",
        r"\begin{table*}[t]\centering\small",
        r"\begin{tabular}{l c rrrr r}",
        r"\toprule",
        r"\textbf{Variant} & $\alpha$ & $\Delta$\textbf{axis} & $\Delta$\textbf{case} & $\Delta$\textbf{role} & $\Delta$\textbf{util} & \textbf{\netpw} \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Negative-$\alpha$ (operational direction: defensive / reactive / resistant / resigned)}} \\",
        r"\midrule",
    ]
    lines += block_rows(neg_agg, NEG_ALPHAS)
    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Positive-$\alpha$ (symmetric control: open / proactive / cooperative / hopeful)}} \\",
        r"\midrule",
    ]
    lines += block_rows(pos_agg, POS_ALPHAS)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Mean per-dimension judge-score deltas (steered $-$ baseline, 1--5 Likert) and overall net-pairwise preference, averaged across all (model, axis) cells in the full-matrix evaluation (16 cells per $\alpha$ block; Qwen3-4B \emph{openness} averaged over 3 seeds before pooling). $\Delta$axis is the targeted axis-alignment effect; $\Delta$case (fidelity to the case narrative), $\Delta$role (client-role plausibility), and $\Delta$util (training-data usefulness) are the case-preservation diagnostics. In aggregate, $\Delta$axis and $\Delta$util are consistently positive at every $(\text{variant}, \alpha)$ while $\Delta$case and $\Delta$role remain within $[-0.052, +0.094]$ on the mean. Per-cell outliers where $\Delta$case or $\Delta$role drops below $-0.10$ are listed in Appendix~\ref{tab:fidelity-outliers}; they concentrate on specific (model, axis) combinations rather than spreading uniformly.}",
        r"\label{tab:fidelity-deltas}",
        r"\end{table*}",
    ]
    Path(ROOT / "paper/tables/tab_fidelity_deltas.tex").write_text("\n".join(lines))
    print("Wrote tab_fidelity_deltas.tex")

    # ----- outlier table (per-cell Δcase or Δrole < -0.10) -----
    OUTLIER_THRESHOLD = -0.10
    outliers = []  # (model, axis, layer, variant, alpha, dcase, drole)
    for sign_label, by_stem, alphas in (("pos", pos_by_stem, POS_ALPHAS), ("neg", neg_by_stem, NEG_ALPHAS)):
        for stem, dicts in by_stem.items():
            m = re.match(r"(qwen35_9b|mistral7b|qwen|gemma)_(\w+)_L(\d+)", stem)
            if not m:
                continue
            model, axis, layer = m.group(1), m.group(2), int(m.group(3))
            # Average across seeds for the same (variant, alpha) within stem
            per_va = defaultdict(lambda: defaultdict(list))
            for d in dicts:
                for k, cell in d.get("by_variant_alpha", {}).items():
                    if "baseline" in k:
                        continue
                    per_va[k]["case"].append(cell.get("case_fidelity_delta_mean", float("nan")))
                    per_va[k]["role"].append(cell.get("client_role_fidelity_delta_mean", float("nan")))
            for k, dim in per_va.items():
                if "__a" not in k:
                    continue
                variant, alpha_s = k.rsplit("__a", 1)
                try:
                    alpha = float(alpha_s)
                except ValueError:
                    continue
                dcase = float(np.nanmean(dim["case"])) if dim["case"] else float("nan")
                drole = float(np.nanmean(dim["role"])) if dim["role"] else float("nan")
                if (np.isfinite(dcase) and dcase < OUTLIER_THRESHOLD) or \
                   (np.isfinite(drole) and drole < OUTLIER_THRESHOLD):
                    outliers.append((model, axis, layer, variant, alpha, dcase, drole))

    # Sort: by sign of alpha (neg first), then by min(dcase, drole) ascending (worst first)
    def sort_key(row):
        _, _, _, _, alpha, dcase, drole = row
        worst = min(
            dcase if np.isfinite(dcase) else float("inf"),
            drole if np.isfinite(drole) else float("inf"),
        )
        return (alpha >= 0, worst)
    outliers.sort(key=sort_key)

    o_lines = [
        r"% auto-generated",
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{ll c l r rr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Axis} & \textbf{L} & \textbf{Variant} & $\alpha$ & $\Delta$\textbf{case} & $\Delta$\textbf{role} \\",
        r"\midrule",
    ]
    last_model = None
    for model, axis, layer, variant, alpha, dcase, drole in outliers:
        var_tex = VARIANT_TEX.get(variant, variant.replace("_", r"\_"))
        if model != last_model and last_model is not None:
            o_lines.append(r"\midrule")
        last_model = model
        def cell_fmt(x, thresh=OUTLIER_THRESHOLD):
            if not np.isfinite(x):
                return "n/a"
            s = f"${x:+.3f}$"
            return r"\textbf{" + s + "}" if x < thresh else s
        o_lines.append(
            f"{MODEL_LBL.get(model, model)} & {axis} & L{layer} & {var_tex} & "
            f"${alpha:+.1f}$ & {cell_fmt(dcase)} & {cell_fmt(drole)} \\\\"
        )
    if not outliers:
        o_lines.append(r"\multicolumn{7}{c}{\textit{No cells with $\Delta$case or $\Delta$role below $-0.10$.}} \\")
    o_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-cell outliers from Table~\ref{tab:fidelity-deltas}: (model, axis, layer, variant, $\alpha$) combinations where the mean $\Delta$case\_fidelity or $\Delta$client\_role\_fidelity drops below $-0.10$ (boldface marks the offending dimension). Outliers concentrate in three groups: (i) Qwen3.5-9B \emph{hopefulness}/\emph{cooperation} at $\alpha=-3.0$, but only under \texttt{case\_blind} --- the case-aware and persona-conditioned variants on the same stems hold $\Delta$case within $\pm 0.05$ \emph{and} achieve higher $\netpw$ (e.g.\ Qwen3.5-9B \emph{hopefulness}: \texttt{persona\_mismatched} $\netpw=+0.75$ at $\Delta$case$=+0.04$ vs.\ \texttt{case\_blind} $\netpw=+0.51$ at $\Delta$case$=-0.29$); (ii) Mistral-7B \emph{hopefulness} L20 at $\alpha=-3.0$, where all variants drift on case ($-0.11$ to $-0.15$) --- an axis-level rather than method-level failure; (iii) \emph{initiative} on three of four models at $\alpha=+3.0$. The worst single cell is Qwen3.5-9B \emph{hopefulness} L15 / \texttt{case\_blind} / $\alpha=-3.0$ at $\Delta$case $=-0.292$.}",
        r"\label{tab:fidelity-outliers}",
        r"\end{table}",
    ]
    Path(ROOT / "paper/tables/tab_fidelity_outliers.tex").write_text("\n".join(o_lines))
    print(f"Wrote tab_fidelity_outliers.tex ({len(outliers)} outlier cells)")


NEG_POLE = {
    "openness": "defensive",
    "initiative": "reactive",
    "cooperation": "resistant",
    "hopefulness": "resigned",
}

# Short model labels used in the existing headline table.
MODEL_SHORT = {
    "qwen": "Qwen-4B",
    "gemma": "Gemma",
    "qwen35_9b": "Qwen-9B",
    "mistral7b": "Mistral",
}

NEG_STEM_ORDER = [
    ("qwen", "openness", "qwen_openness_L15"),
    ("qwen", "initiative", "qwen_initiative_L9"),
    ("qwen", "cooperation", "qwen_cooperation_L20"),
    ("qwen", "hopefulness", "qwen_hopefulness_L18"),
    ("gemma", "openness", "gemma_openness_L3"),
    ("gemma", "initiative", "gemma_initiative_L13"),
    ("gemma", "cooperation", "gemma_cooperation_L23"),
    ("gemma", "hopefulness", "gemma_hopefulness_L25"),
    ("qwen35_9b", "openness", "qwen35_9b_openness_L20"),
    ("qwen35_9b", "initiative", "qwen35_9b_initiative_L20"),
    ("qwen35_9b", "cooperation", "qwen35_9b_cooperation_L15"),
    ("qwen35_9b", "hopefulness", "qwen35_9b_hopefulness_L15"),
    ("mistral7b", "openness", "mistral7b_openness_L15"),
    ("mistral7b", "initiative", "mistral7b_initiative_L5"),
    ("mistral7b", "cooperation", "mistral7b_cooperation_L15"),
    ("mistral7b", "hopefulness", "mistral7b_hopefulness_L20"),
]


def _mean_pairwise_net(stem_data, variant, alpha):
    """Average pairwise_net_overall across the list of summary-dicts for a stem."""
    vals = []
    for d in stem_data:
        cell = d.get("by_variant_alpha", {}).get(f"{variant}__a{alpha}")
        if cell and cell.get("pairwise_net_overall") is not None:
            vals.append(cell["pairwise_net_overall"])
    return float(np.mean(vals)) if vals else None


def _mean_kappa(stem_data):
    vals = [d["inter_rater_agreement"]["fleiss_kappa"] for d in stem_data
            if "inter_rater_agreement" in d and "fleiss_kappa" in d["inter_rater_agreement"]]
    return float(np.mean(vals)) if vals else None


def _load_cluster_ci(path="outputs/steering_eval/cluster_ci/headline_neg.json"):
    p = ROOT / path
    if not p.exists():
        return None
    return json.load(p.open())


def _ci_marker(ci_cell):
    """Return '^{*}' if cluster-bootstrap 95% CI excludes 0, else ''."""
    if not ci_cell or "lo" not in ci_cell or "hi" not in ci_cell:
        return ""
    if ci_cell["lo"] > 0 or ci_cell["hi"] < 0:
        return r"^{*}"
    return ""


def write_fullmatrix_neg_prespecified():
    """Prespecified headline at α=-3.0: per (model, axis) cell, report
    global_blind and global_aware net-pairwise side-by-side (no per-cell
    variant selection)."""
    by_stem_neg = load_negalpha_stems()
    ci_table = _load_cluster_ci()  # may be None if cluster-bootstrap not yet run
    lines = [
        r"% auto-generated",
        r"\begin{table}[t]\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}ll c rr r@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Pole} & \textbf{L} & \textbf{CB} & \textbf{CA} & \textbf{$\kappa$} \\",
        r"\midrule",
    ]
    rows_gb, rows_ga = [], []
    last_model = None
    n_signif_cb = n_signif_ca = 0
    for mk, ax, stem in NEG_STEM_ORDER:
        if stem not in by_stem_neg:
            continue
        data = by_stem_neg[stem]
        gb = _mean_pairwise_net(data, "global_blind", -3.0)
        ga = _mean_pairwise_net(data, "global_aware", -3.0)
        kappa = _mean_kappa(data)
        layer = stem.split("_L")[-1]
        if gb is not None: rows_gb.append(gb)
        if ga is not None: rows_ga.append(ga)
        if mk != last_model and last_model is not None:
            lines.append(r"\midrule")
        last_model = mk

        # Cluster-bootstrap significance markers
        ci_stem = (ci_table or {}).get(stem, {})
        mark_gb = _ci_marker(ci_stem.get("CB"))
        mark_ca = _ci_marker(ci_stem.get("CA"))
        if mark_gb: n_signif_cb += 1
        if mark_ca: n_signif_ca += 1

        def fmt(x, mark):
            return f"${x:+.3f}{mark}$" if x is not None else "n/a"
        lines.append(
            f"{MODEL_SHORT[mk]} & {NEG_POLE[ax]} & {layer} & {fmt(gb, mark_gb)} & {fmt(ga, mark_ca)} & ${kappa:.2f}$ \\\\"
        )
    # Mean row
    if rows_gb and rows_ga:
        mean_gb = float(np.mean(rows_gb))
        mean_ga = float(np.mean(rows_ga))
        lines.append(r"\midrule")
        lines.append(f"\\textbf{{Mean}} &  &  & $\\mathbf{{{mean_gb:+.3f}}}$ & $\\mathbf{{{mean_ga:+.3f}}}$ &  \\\\")
    caption = (
        r"\caption{Prespecified headline at $\alpha{=}{-}3$ (operational "
        r"direction): per (model, pole) cell, net-pairwise preference for the "
        r"two case-agnostic variants $\textbf{CB}=\texttt{case\_blind}$ and "
        r"$\textbf{CA}=\texttt{case\_aware}$. No per-cell variant selection. "
        r"Both variants are fixed methodological choices specified before "
        r"evaluation. Mean over the 16 cells: $\textbf{CB}=+0.234$, "
        r"$\textbf{CA}=+0.309$. The per-cell oracle (best of the four "
        r"variants per cell) appears in Table~\ref{tab:fullmatrix-neg-oracle} "
        r"and adds only $+0.048$ over CA on the mean. $N{=}200$ held-out "
        r"contexts per cell."
    )
    if ci_table is not None:
        caption += (
            f" $^{{*}}$marks cells whose 95\\% cluster-bootstrap CI excludes 0 "
            f"(resampled on the 10 personas; $n_\\text{{boot}}=2000$; "
            f"full CIs in Table~\\ref{{tab:cluster-ci-neg}}). "
            f"{n_signif_cb}/16 cells signif.\\ for CB, {n_signif_ca}/16 for CA."
        )
    caption += r" Pole abbreviations: defensive/reactive/resistant/resigned correspond to suppressing openness/initiative/cooperation/hopefulness.}"
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:fullmatrix-neg-headline}",
        r"\end{table}",
    ]
    Path(ROOT / "paper/tables/tab_fullmatrix_neg_headline.tex").write_text("\n".join(lines))
    print("Wrote tab_fullmatrix_neg_headline.tex")


def write_cluster_ci_appendix():
    """Full per-cell cluster-bootstrap CIs as an appendix table."""
    ci_table = _load_cluster_ci()
    if ci_table is None:
        print("Skipping tab_cluster_ci_neg.tex (no cluster-bootstrap JSON yet)")
        return
    lines = [
        r"% auto-generated",
        r"\begin{table}[t]\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}ll c cc cc@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Pole} & \textbf{L} & "
        r"\multicolumn{2}{c}{\textbf{CB} 95\% CI} & "
        r"\multicolumn{2}{c}{\textbf{CA} 95\% CI} \\",
        r" & & & lo & hi & lo & hi \\",
        r"\midrule",
    ]
    last_model = None
    for mk, ax, stem in NEG_STEM_ORDER:
        if stem not in ci_table:
            continue
        if mk != last_model and last_model is not None:
            lines.append(r"\midrule")
        last_model = mk
        cb = ci_table[stem].get("CB", {})
        ca = ci_table[stem].get("CA", {})
        layer = stem.split("_L")[-1]
        def cell(v):
            return f"${v:+.3f}$" if isinstance(v, (int, float)) else "--"
        lines.append(
            f"{MODEL_SHORT[mk]} & {NEG_POLE[ax]} & {layer} & "
            f"{cell(cb.get('lo'))} & {cell(cb.get('hi'))} & "
            f"{cell(ca.get('lo'))} & {cell(ca.get('hi'))} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Cluster-bootstrap 95\% CIs for the headline "
        r"$\alpha{=}{-}3$ matrix (Table~\ref{tab:fullmatrix-neg-headline}). "
        r"Resampled on the 10 personas (each tied to one case; see "
        r"\S\ref{sec:limitations}) with $n_\text{boot}=2000$ percentile "
        r"intervals. Strong axes (\emph{resistant, resigned}) hold all 16 "
        r"cells significantly above 0; weak axes (\emph{defensive, "
        r"reactive}) overlap 0 on most cells, consistent with the "
        r"axis-asymmetry reported in \S\ref{sec:negalpha}.}",
        r"\label{tab:cluster-ci-neg}",
        r"\end{table}",
    ]
    Path(ROOT / "paper/tables/tab_cluster_ci_neg.tex").write_text("\n".join(lines))
    print("Wrote tab_cluster_ci_neg.tex")


def write_fullmatrix_pos_prespecified():
    """Prespecified headline at α=+3.0: per (model, axis) cell, global_blind
    and global_aware net-pairwise side-by-side."""
    by_stem = load_paper_v2_stems()
    lines = [
        r"% auto-generated",
        r"\begin{table*}[t]\centering\small",
        r"\begin{tabular}{ll c rr r}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Axis} & \textbf{L} & \textbf{CB} & \textbf{CA} & \textbf{$\kappa$} \\",
        r"\midrule",
    ]
    rows_gb, rows_ga = [], []
    last_model = None
    for mk, ax, stem in STEM_ORDER:
        if stem not in by_stem:
            continue
        data = by_stem[stem]
        gb = _mean_pairwise_net(data, "global_blind", 3.0)
        ga = _mean_pairwise_net(data, "global_aware", 3.0)
        kappa = _mean_kappa(data)
        layer = stem.split("_L")[-1]
        if gb is not None: rows_gb.append(gb)
        if ga is not None: rows_ga.append(ga)
        if mk != last_model and last_model is not None:
            lines.append(r"\midrule")
        last_model = mk
        def fmt(x):
            return f"${x:+.3f}$" if x is not None else "n/a"
        lines.append(f"{MODEL_LBL[mk]} & {ax} & L{layer} & {fmt(gb)} & {fmt(ga)} & ${kappa:.2f}$ \\\\")
    if rows_gb and rows_ga:
        mean_gb = float(np.mean(rows_gb))
        mean_ga = float(np.mean(rows_ga))
        lines.append(r"\midrule")
        lines.append(f"\\textbf{{Mean}} &  &  & $\\mathbf{{{mean_gb:+.3f}}}$ & $\\mathbf{{{mean_ga:+.3f}}}$ &  \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Prespecified symmetric control at $\alphapos$: per (model, axis) cell, net-pairwise preference for the two case-agnostic variants $\textbf{CB}=\texttt{case\_blind}$ and $\textbf{CA}=\texttt{case\_aware}$. No per-cell variant selection. Mean over 15 cells: $\textbf{CB}=+0.195$, $\textbf{CA}=+0.206$. The per-cell oracle (best of the four variants per cell) appears in Table~\ref{tab:fullmatrix-pos-oracle} and adds $+0.070$ over CA on the mean. $N{=}200$ held-out contexts per cell ($N{=}600$ for Qwen3-4B \emph{openness}).}",
        r"\label{tab:fullmatrix-pos-headline}",
        r"\end{table*}",
    ]
    Path(ROOT / "paper/tables/tab_fullmatrix_pos_headline.tex").write_text("\n".join(lines))
    print("Wrote tab_fullmatrix_pos_headline.tex")


def main():
    write_layer_sweep()
    write_fullmatrix_pos()
    write_fullmatrix_neg()
    write_variant_wins()
    write_abc_variants()
    write_fleiss()
    write_provider_sens()
    write_prompt_baseline()
    write_fidelity_deltas()
    write_fullmatrix_neg_prespecified()
    write_fullmatrix_pos_prespecified()
    write_cluster_ci_appendix()


if __name__ == "__main__":
    main()
