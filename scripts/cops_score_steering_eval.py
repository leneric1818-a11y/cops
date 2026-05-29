#!/usr/bin/env python3
"""Score steering evaluation outputs.

For each (model, axis) eval JSONL file, computes per-row:
  - style_score_positive: classifier probability that response is 'positive-pole' style
  - case_keyword_retention: fraction of case keywords from hauptanliegen present in response
  - case_keywords_n: number of case keywords matched
  - novelty_vs_baseline: Jaccard-based novelty (higher = more rewrite)
  - has_halluzination_flag: rough flag — response contains name/word NOT in context or hauptanliegen
                             (used sparingly, heuristic only)

Then aggregates by (variant, alpha): mean, std, plus delta-vs-baseline.

Outputs:
  - per-row CSV next to the input jsonl
  - aggregate JSON summary
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def extract_case_keywords(ha: str) -> set[str]:
    kws = set()
    for m in re.finditer(r"\b[A-ZÄÖÜ][a-zäöüß]{2,}\b", ha):
        kws.add(m.group())
    for kw in [
        "kifft", "rauchen", "konsumiert", "abhängig", "trennung", "verschwunden",
        "schule", "noten", "freundeskreis", "drogen", "entzug", "kiffen",
    ]:
        if kw in ha.lower():
            kws.add(kw.lower())
    return kws


def jaccard_words(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    inter = wa & wb
    union = wa | wb
    return len(inter) / len(union)


def train_axis_classifier(flat_path: Path, axis: str) -> tuple:
    """Train TF-IDF + LogReg to score positive-pole style.

    flat_path: persona_pairs_<axis>_gpt54mini_1000_flat.jsonl with rows
               {style: ..., response: ...}
    Returns (vectorizer, classifier, positive_label, negative_label)
    """
    with flat_path.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    styles = sorted(set(r["style"] for r in rows))
    assert len(styles) == 2, f"Expected 2 styles for axis {axis}, got {styles}"

    # Positive pole by axis convention
    POS = {
        "openness": "open", "initiative": "explorative",
        "cooperation": "cooperative", "hopefulness": "hopeful",
    }
    pos = POS.get(axis, styles[1])
    neg = [s for s in styles if s != pos][0]

    X_text = [r["response"] for r in rows]
    y = np.array([1 if r["style"] == pos else 0 for r in rows])

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, sublinear_tf=True)
    X = vec.fit_transform(X_text)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    print(f"  axis={axis} classifier test_acc={acc:.3f} pos={pos} neg={neg}")
    return vec, clf, pos, neg


def score_eval_file(eval_jsonl: Path, out_dir: Path, classifiers: dict) -> dict:
    """Score one evaluation JSONL and write per-row + aggregate outputs."""
    with eval_jsonl.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        print(f"  [skip] empty: {eval_jsonl}")
        return {}

    axis = rows[0]["axis"]
    model = rows[0]["model_path"]

    if axis not in classifiers:
        print(f"  [skip] no classifier for axis={axis}")
        return {}
    vec, clf, pos, neg = classifiers[axis]

    # Baseline map: (run_seed, seed_id) -> baseline response (per-seed baselines)
    baseline_by_sid = {
        (r.get("run_seed", 42), r["seed_id"]): r["response"]
        for r in rows if r["variant"] == "baseline"
    }

    scored = []
    for r in rows:
        resp = r["response"]
        ha = r.get("hauptanliegen", "")
        ctx = r.get("context", "")

        X = vec.transform([resp])
        style_score_pos = float(clf.predict_proba(X)[0, 1])

        case_kws = extract_case_keywords(ha)
        if case_kws:
            resp_lower = resp.lower()
            n_matched = sum(1 for k in case_kws if k.lower() in resp_lower)
            kw_retention = n_matched / len(case_kws)
        else:
            n_matched = 0
            kw_retention = float("nan")

        base_key = (r.get("run_seed", 42), r["seed_id"])
        base = baseline_by_sid.get(base_key, "")
        jaccard = jaccard_words(resp, base) if base else float("nan")
        novelty = 1.0 - jaccard if jaccard == jaccard else float("nan")  # 1-jaccard

        scored.append({
            **r,
            "style_score_pos": style_score_pos,
            "case_kw_retention": kw_retention,
            "case_kw_matched": n_matched,
            "case_kw_total": len(case_kws) if case_kws else 0,
            "jaccard_with_baseline": jaccard,
            "novelty_vs_baseline": novelty,
        })

    out_jsonl = out_dir / (eval_jsonl.stem + "_scored.jsonl")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Aggregate by (variant, alpha) — pooled across all seeds
    agg: dict = {}
    by_va: dict = defaultdict(list)
    by_va_seed: dict = defaultdict(list)  # (variant, alpha, seed) groups for per-seed stats
    for s in scored:
        key = f"{s['variant']}__a{s['alpha']}"
        by_va[key].append(s)
        seed_key = (s['variant'], s['alpha'], s.get('run_seed', 42))
        by_va_seed[seed_key].append(s)

    # Per-seed baselines
    baseline_by_seed: dict = defaultdict(list)
    baseline_kw_by_seed: dict = defaultdict(list)
    for s in scored:
        if s["variant"] != "baseline":
            continue
        rs = s.get("run_seed", 42)
        baseline_by_seed[rs].append(s["style_score_pos"])
        if s["case_kw_total"] > 0:
            baseline_kw_by_seed[rs].append(s["case_kw_retention"])

    # Pooled baselines (across all seeds)
    baseline_scores_all = [s["style_score_pos"] for s in scored if s["variant"] == "baseline"]
    baseline_mean = float(np.mean(baseline_scores_all)) if baseline_scores_all else float("nan")
    baseline_kw_all = [s["case_kw_retention"] for s in scored if s["variant"] == "baseline" and s["case_kw_total"] > 0]
    baseline_kw_mean = float(np.mean(baseline_kw_all)) if baseline_kw_all else float("nan")

    # Per-(variant, alpha) aggregation — pooled + per-seed deltas
    for key, group in sorted(by_va.items()):
        variant, alpha_str = key.split("__a")
        alpha_val = float(alpha_str)

        style_scores = [s["style_score_pos"] for s in group]
        kw_rets = [s["case_kw_retention"] for s in group if s["case_kw_total"] > 0]
        novelties = [s["novelty_vs_baseline"] for s in group if s["novelty_vs_baseline"] == s["novelty_vs_baseline"]]

        # Per-seed deltas — one mean delta per seed, then mean+std over seeds
        per_seed_deltas_style = []
        per_seed_deltas_kw = []
        per_seed_counts = []
        for rs, rs_baseline_scores in baseline_by_seed.items():
            seed_group = by_va_seed.get((variant, alpha_val, rs), [])
            if not seed_group:
                continue
            seed_style = [s["style_score_pos"] for s in seed_group]
            seed_base_mean = float(np.mean(rs_baseline_scores))
            per_seed_deltas_style.append(float(np.mean(seed_style) - seed_base_mean))
            per_seed_counts.append(len(seed_group))
            seed_kw = [s["case_kw_retention"] for s in seed_group if s["case_kw_total"] > 0]
            if seed_kw and baseline_kw_by_seed.get(rs):
                per_seed_deltas_kw.append(float(np.mean(seed_kw) - np.mean(baseline_kw_by_seed[rs])))

        n_seeds_used = len(per_seed_deltas_style)
        # Use sample std (ddof=1) — these are seeds treated as samples
        def _safe_std(xs):
            if len(xs) < 2:
                return float("nan")
            return float(np.std(xs, ddof=1))

        def _ci95(xs):
            if len(xs) < 2:
                return float("nan")
            from scipy import stats
            try:
                tcrit = float(stats.t.ppf(0.975, len(xs) - 1))
            except Exception:
                tcrit = 1.96
            return tcrit * _safe_std(xs) / np.sqrt(len(xs))

        agg[key] = {
            "n": len(group),
            "n_seeds": n_seeds_used,
            "style_score_pos_mean": float(np.mean(style_scores)) if style_scores else float("nan"),
            "style_score_pos_std_pooled": float(np.std(style_scores)) if style_scores else float("nan"),
            "directional_delta_vs_baseline": float(np.mean(style_scores) - baseline_mean) if style_scores else float("nan"),
            "directional_delta_mean_over_seeds": float(np.mean(per_seed_deltas_style)) if per_seed_deltas_style else float("nan"),
            "directional_delta_std_over_seeds": _safe_std(per_seed_deltas_style),
            "directional_delta_ci95_halfwidth": _ci95(per_seed_deltas_style),
            "case_kw_retention_mean": float(np.mean(kw_rets)) if kw_rets else float("nan"),
            "case_kw_retention_delta_vs_baseline": (float(np.mean(kw_rets) - baseline_kw_mean) if kw_rets and baseline_kw_all else float("nan")),
            "case_kw_delta_mean_over_seeds": float(np.mean(per_seed_deltas_kw)) if per_seed_deltas_kw else float("nan"),
            "case_kw_delta_std_over_seeds": _safe_std(per_seed_deltas_kw),
            "novelty_vs_baseline_mean": float(np.mean(novelties)) if novelties else float("nan"),
            "per_seed_deltas_style": per_seed_deltas_style,
        }

    summary_path = out_dir / (eval_jsonl.stem + "_summary.json")
    summary = {
        "eval_file": str(eval_jsonl),
        "axis": axis,
        "model_path": model,
        "n_rows": len(rows),
        "n_seeds_detected": len(baseline_by_seed),
        "baseline_style_score_mean": baseline_mean,
        "baseline_case_kw_retention_mean": baseline_kw_mean,
        "baseline_per_seed": {str(rs): float(np.mean(xs)) for rs, xs in baseline_by_seed.items()},
        "by_variant_alpha": agg,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  wrote {out_jsonl.name} + {summary_path.name}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-dir",
        default=str(ROOT / "outputs/steering_eval"),
        help="Directory of eval JSONL files.",
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs/steering_eval/scored"),
    )
    p.add_argument(
        "--flat-pair-dir",
        default=str(ROOT / "outputs/metrics"),
        help="Dir with persona_pairs_<axis>_gpt54mini_1000_flat.jsonl files.",
    )
    p.add_argument(
        "--axes",
        default="openness,initiative,cooperation,hopefulness",
    )
    p.add_argument(
        "--skip-pattern",
        default="SMOKE_",
    )
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train per-axis classifiers
    classifiers = {}
    for axis in args.axes.split(","):
        axis = axis.strip()
        flat_path = Path(args.flat_pair_dir) / f"persona_pairs_{axis}_gpt54mini_1000_flat.jsonl"
        if not flat_path.exists():
            print(f"[WARN] No classifier data for {axis}: {flat_path}")
            continue
        classifiers[axis] = train_axis_classifier(flat_path, axis)

    # Score each eval jsonl
    all_summaries = []
    for jsonl in sorted(eval_dir.glob("*.jsonl")):
        if args.skip_pattern and args.skip_pattern in jsonl.name:
            continue
        if "_scored" in jsonl.name:
            continue
        print(f"\n== {jsonl.name} ==")
        summary = score_eval_file(jsonl, out_dir, classifiers)
        if summary:
            all_summaries.append(summary)

    master = out_dir / "all_summaries.json"
    with master.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\nMaster summary: {master}")


if __name__ == "__main__":
    main()
