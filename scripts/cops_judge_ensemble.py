#!/usr/bin/env python3
"""Ensemble aggregation over multiple judge providers/models.

For each eval jsonl, reads per-provider `*_judged[__provider__model].jsonl`
files and produces an ensembled judgement per (run_seed, seed_id, variant, alpha):

  - Per-dimension ensembled scores: mean (rubric) or vote (winner)
  - Majority-vote pairwise winner (tie if split evenly)
  - Inter-rater agreement (Cohen's κ pairwise, Fleiss's κ for 3+)

Outputs an ensembled JSONL + summary JSON, plus an inter-rater agreement JSON.

Usage:
  scripts/cops_judge_ensemble.py \
      --eval-stems qwen_openness_L1_all,gemma_openness_L3_all \
      --providers openai:gpt-5.4-mini cluster_mistral:mistralai/Magistral-Small-2509 \
      --judged-dir outputs/steering_eval/judged \
      --out-dir outputs/steering_eval/judged_ensemble
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def provider_model_suffix(provider: str, model: str, effort: str | None = None) -> str:
    is_default = (provider == "openai" and model == "gpt-5.4-mini" and not effort)
    if is_default:
        return ""
    model_tag = model.replace("/", "_").replace(":", "_")
    effort_tag = f"_{effort}" if effort else ""
    return f"__{provider}__{model_tag}{effort_tag}"


def parse_provider_spec(entry: str) -> tuple[str, str, str | None]:
    """Accept 'provider:model' or 'provider:model@effort'."""
    if ":" not in entry:
        raise SystemExit(f"Bad provider spec (expected provider:model[@effort]): {entry}")
    provider, rest = entry.split(":", 1)
    effort = None
    if "@" in rest:
        rest, effort = rest.rsplit("@", 1)
    return provider, rest, effort


def load_judged(stem: str, provider: str, model: str, judged_dir: Path, effort: str | None = None) -> list[dict] | None:
    suffix = provider_model_suffix(provider, model, effort)
    path = judged_dir / f"{stem}_judged{suffix}.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def row_key(r: dict) -> tuple:
    """Stable cross-provider row identifier."""
    return (r.get("run_seed", 42), r["seed_id"], r["variant"], float(r["alpha"]))


def ensemble_rubric(rows_by_provider: dict[str, dict]) -> dict:
    """Average rubric scores across providers."""
    dims = ("axis_alignment", "case_fidelity", "client_role_fidelity", "training_utility")
    out_steered: dict = {}
    out_base: dict = {}
    for dim in dims:
        vs_s = [rows_by_provider[p]["steered_scores"].get(dim) for p in rows_by_provider
                if isinstance(rows_by_provider[p]["steered_scores"].get(dim), (int, float))]
        vs_b = [rows_by_provider[p]["base_scores"].get(dim) for p in rows_by_provider
                if isinstance(rows_by_provider[p]["base_scores"].get(dim), (int, float))]
        out_steered[dim] = float(np.mean(vs_s)) if vs_s else float("nan")
        out_base[dim] = float(np.mean(vs_b)) if vs_b else float("nan")
    return {"steered_scores": out_steered, "base_scores": out_base}


def majority_winner(winners: list[str]) -> str:
    c = Counter(winners)
    # Break ties: prefer "tie" when counts equal
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "tie"
    return top[0][0]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's κ for two raters' ordinal labels."""
    labels = sorted(set(a + b))
    if not labels or len(a) != len(b):
        return float("nan")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) * b.count(l)) / (n * n) for l in labels)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """Fleiss's κ for k raters. ratings[i] = list of labels per item from rater i."""
    if len(ratings) < 2:
        return float("nan")
    n_items = len(ratings[0])
    if any(len(r) != n_items for r in ratings):
        return float("nan")
    labels = sorted({lab for r in ratings for lab in r})
    k = len(labels)
    N = len(ratings)
    # n_ij = count of rater choices for category j on item i
    n_ij = np.zeros((n_items, k))
    for rater in ratings:
        for i, lab in enumerate(rater):
            j = labels.index(lab)
            n_ij[i, j] += 1
    # Agreement per item
    P_i = (n_ij * (n_ij - 1)).sum(axis=1) / (N * (N - 1)) if N > 1 else np.zeros(n_items)
    P_bar = P_i.mean()
    # Chance agreement
    p_j = n_ij.sum(axis=0) / (n_items * N)
    P_e = (p_j ** 2).sum()
    if P_e >= 1.0:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def _finite(xs: list[float]) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float)) and np.isfinite(x)]


def _safe_mean(xs: list[float]) -> float:
    vals = _finite(xs)
    return float(np.mean(vals)) if vals else float("nan")


def _safe_std(xs: list[float]) -> float:
    vals = _finite(xs)
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan")


def ensemble_stem(
    stem: str,
    providers: list[tuple[str, str]],
    judged_dir: Path,
    out_dir: Path,
) -> dict | None:
    loaded = {}
    for provider, model, effort in providers:
        rows = load_judged(stem, provider, model, judged_dir, effort)
        if rows is None:
            print(f"  [skip {stem}] missing provider={provider} model={model} effort={effort}")
            return None
        tag = f"{provider}:{model}" + (f"@{effort}" if effort else "")
        loaded[tag] = rows

    # Group rows by row_key per provider
    per_provider_map: dict[str, dict[tuple, dict]] = {}
    for tag, rows in loaded.items():
        per_provider_map[tag] = {row_key(r): r for r in rows}

    # Intersection of keys across all providers
    common = set.intersection(*[set(m.keys()) for m in per_provider_map.values()])
    if not common:
        print(f"  [skip {stem}] no common rows across providers")
        return None

    ensembled = []
    # Winners aligned across providers for κ calculation
    winner_lists: list[list[str]] = [[] for _ in providers]

    for key in sorted(common):
        rows_by_provider = {tag: per_provider_map[tag][key] for tag in per_provider_map}
        # Collect winners
        winners = [rows_by_provider[tag]["pairwise_winner"] for tag in per_provider_map]
        for idx, w in enumerate(winners):
            winner_lists[idx].append(w)
        ensemble_winner = majority_winner(winners)
        ens_scores = ensemble_rubric(rows_by_provider)
        first = next(iter(rows_by_provider.values()))
        ensembled.append({
            "run_seed": first["run_seed"],
            "seed_id": first["seed_id"],
            "variant": first["variant"],
            "alpha": first["alpha"],
            "persona_name": first.get("persona_name"),
            "hauptanliegen": first.get("hauptanliegen"),
            "axis": first.get("axis"),
            "model_path": first.get("model_path"),
            "target_pole": first.get("target_pole"),
            "target_style": first.get("target_style"),
            "pairwise_winner": ensemble_winner,
            "per_provider_winners": dict(zip(per_provider_map.keys(), winners)),
            **ens_scores,
            "n_agree": sum(1 for w in winners if w == ensemble_winner),
            "n_providers": len(providers),
        })

    # Pairwise κ + Fleiss κ
    kappas = {}
    keys = list(per_provider_map.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            kappas[f"{keys[i]}__vs__{keys[j]}"] = cohen_kappa(winner_lists[i], winner_lists[j])
    fleiss = fleiss_kappa(winner_lists)

    # Write ensembled jsonl
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{stem}_ensembled.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in ensembled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    axis = ensembled[0].get("axis")
    model_path = ensembled[0].get("model_path")

    by_va = defaultdict(list)
    by_va_seed = defaultdict(list)
    for r in ensembled:
        k = f"{r['variant']}__a{r['alpha']}"
        by_va[k].append(r)
        by_va_seed[(r["variant"], r["alpha"], r["run_seed"])].append(r)

    agg = {}
    dims = ("axis_alignment", "case_fidelity", "client_role_fidelity", "training_utility")
    for k, group in sorted(by_va.items()):
        variant, alpha_str = k.split("__a")
        alpha_val = float(alpha_str)
        winners = [r["pairwise_winner"] for r in group]
        n = len(group)

        dim_deltas = {}
        dim_pairwise = {}
        for dim in dims:
            valid = [
                (r["steered_scores"].get(dim), r["base_scores"].get(dim))
                for r in group
                if isinstance(r["steered_scores"].get(dim), (int, float))
                and isinstance(r["base_scores"].get(dim), (int, float))
            ]
            deltas = [s - b for s, b in valid]
            dim_deltas[f"{dim}_delta_mean"] = _safe_mean(deltas)
            dim_deltas[f"{dim}_delta_std_pooled"] = float(np.std(deltas)) if deltas else float("nan")

            steered_wins = sum(1 for s, b in valid if s > b)
            base_wins = sum(1 for s, b in valid if s < b)
            ties = sum(1 for s, b in valid if s == b)
            tot = max(1, len(valid))
            dim_pairwise[f"{dim}_pairwise_steered_win_rate"] = steered_wins / tot
            dim_pairwise[f"{dim}_pairwise_base_win_rate"] = base_wins / tot
            dim_pairwise[f"{dim}_pairwise_tie_rate"] = ties / tot
            dim_pairwise[f"{dim}_pairwise_net"] = (steered_wins - base_wins) / tot

        run_seed_values = sorted({r["run_seed"] for r in group})
        per_seed_win_rates = []
        per_seed_case_fidelity_delta = []
        per_seed_dim_net: dict[str, list[float]] = defaultdict(list)
        for rs in run_seed_values:
            seed_group = by_va_seed.get((variant, alpha_val, rs), [])
            if not seed_group:
                continue
            seed_winners = [r["pairwise_winner"] for r in seed_group]
            per_seed_win_rates.append(seed_winners.count("steered") / max(1, len(seed_group)))

            case_del = [
                r["steered_scores"].get("case_fidelity") - r["base_scores"].get("case_fidelity")
                for r in seed_group
                if isinstance(r["steered_scores"].get("case_fidelity"), (int, float))
                and isinstance(r["base_scores"].get("case_fidelity"), (int, float))
            ]
            if case_del:
                per_seed_case_fidelity_delta.append(float(np.mean(case_del)))
            else:
                per_seed_case_fidelity_delta.append(float("nan"))

            for dim in dims:
                valid = [
                    (r["steered_scores"].get(dim), r["base_scores"].get(dim))
                    for r in seed_group
                    if isinstance(r["steered_scores"].get(dim), (int, float))
                    and isinstance(r["base_scores"].get(dim), (int, float))
                ]
                if valid:
                    sw_d = sum(1 for s, b in valid if s > b)
                    bw_d = sum(1 for s, b in valid if s < b)
                    per_seed_dim_net[dim].append((sw_d - bw_d) / len(valid))
                else:
                    per_seed_dim_net[dim].append(float("nan"))

        agg[k] = {
            "n": n,
            "target_pole": group[0].get("target_pole"),
            "target_style": group[0].get("target_style"),
            "pairwise_steered_win_rate": winners.count("steered") / max(1, n),
            "pairwise_tie_rate": winners.count("tie") / max(1, n),
            "pairwise_base_win_rate": winners.count("base") / max(1, n),
            "pairwise_net_overall": (winners.count("steered") - winners.count("base")) / max(1, n),
            **dim_deltas,
            **dim_pairwise,
            "run_seed_values": run_seed_values,
            "win_rate_per_seed": per_seed_win_rates,
            "win_rate_mean_over_seeds": _safe_mean(per_seed_win_rates),
            "win_rate_std_over_seeds": _safe_std(per_seed_win_rates),
            "case_fidelity_delta_per_seed": per_seed_case_fidelity_delta,
            "case_fidelity_delta_mean_over_seeds": _safe_mean(per_seed_case_fidelity_delta),
            "case_fidelity_delta_std_over_seeds": _safe_std(per_seed_case_fidelity_delta),
            "axis_alignment_pairwise_net_per_seed": per_seed_dim_net["axis_alignment"],
            "case_fidelity_pairwise_net_per_seed": per_seed_dim_net["case_fidelity"],
            "client_role_fidelity_pairwise_net_per_seed": per_seed_dim_net["client_role_fidelity"],
            "training_utility_pairwise_net_per_seed": per_seed_dim_net["training_utility"],
            "axis_alignment_pairwise_net_mean_over_seeds": _safe_mean(per_seed_dim_net["axis_alignment"]),
            "axis_alignment_pairwise_net_std_over_seeds": _safe_std(per_seed_dim_net["axis_alignment"]),
            "case_fidelity_pairwise_net_mean_over_seeds": _safe_mean(per_seed_dim_net["case_fidelity"]),
            "case_fidelity_pairwise_net_std_over_seeds": _safe_std(per_seed_dim_net["case_fidelity"]),
            "client_role_fidelity_pairwise_net_mean_over_seeds": _safe_mean(per_seed_dim_net["client_role_fidelity"]),
            "client_role_fidelity_pairwise_net_std_over_seeds": _safe_std(per_seed_dim_net["client_role_fidelity"]),
            "training_utility_pairwise_net_mean_over_seeds": _safe_mean(per_seed_dim_net["training_utility"]),
            "training_utility_pairwise_net_std_over_seeds": _safe_std(per_seed_dim_net["training_utility"]),
        }

    summary = {
        "eval_file": stem,
        "eval_stem": stem,
        "axis": axis,
        "model_path": model_path,
        "providers": [{"provider": p, "model": m, "effort": e} for p, m, e in providers],
        "run_seeds": sorted({row["run_seed"] for row in ensembled}),
        "n_seeds_detected": len({row["run_seed"] for row in ensembled}),
        "n_pairs": len(ensembled),
        "n_ensembled_rows": len(ensembled),
        "inter_rater_agreement": {
            "pairwise_cohen_kappa": kappas,
            "fleiss_kappa": fleiss,
        },
        "by_variant_alpha": agg,
    }
    summary_path = out_dir / f"{stem}_ensemble_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [{stem}] n={len(ensembled)} Fleiss_κ={fleiss:.3f}  wrote {summary_path.name}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-stems", required=True, help="Comma-separated list of eval_jsonl stems (without _judged suffix).")
    p.add_argument(
        "--providers",
        nargs="+",
        required=True,
        help="Space-separated provider:model entries, e.g. openai:gpt-5.4-mini cluster_mistral:mistralai/Magistral-Small-2509",
    )
    p.add_argument("--judged-dir", default=str(ROOT / "outputs/steering_eval/judged"))
    p.add_argument("--out-dir", default=str(ROOT / "outputs/steering_eval/judged_ensemble"))
    return p.parse_args()


def main():
    args = parse_args()
    judged_dir = Path(args.judged_dir)
    out_dir = Path(args.out_dir)
    stems = [s.strip() for s in args.eval_stems.split(",") if s.strip()]
    providers = [parse_provider_spec(entry) for entry in args.providers]
    print(f"Ensembling {len(stems)} stems across {len(providers)} providers")
    all_summaries = []
    for stem in stems:
        s = ensemble_stem(stem, providers, judged_dir, out_dir)
        if s:
            all_summaries.append(s)
    master = out_dir / "all_ensemble_summaries.json"
    master.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Master: {master}")


if __name__ == "__main__":
    main()
