#!/usr/bin/env python3
"""Ablate the content preservation metric weights to test ranking stability.

For each weight configuration from a small grid, recompute
content_preservation_score = w_jaccard * Jaccard + w_novelty * (1 - Novelty) + w_drift * (1 - DriftRate)
and compute Spearman rank correlation of method-family orderings against the reference (0.6/0.2/0.2).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILY_ORDER = ("paired_dense", "prompt_diff", "reft_loreft")
AXIS_ORDER = ("openness", "initiative", "cooperation", "hopefulness")

WEIGHT_GRID = [
    (0.4, 0.3, 0.3),
    (0.5, 0.25, 0.25),
    (0.6, 0.2, 0.2),   # reference
    (0.7, 0.15, 0.15),
    (0.8, 0.1, 0.1),
    (1.0, 0.0, 0.0),   # pure Jaccard baseline
]
REFERENCE_WEIGHTS = (0.6, 0.2, 0.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate content preservation metric weights.")
    parser.add_argument(
        "--benchmarks-root",
        default=str(ROOT / "outputs" / "metrics" / "benchmarks"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "metrics" / "ablation" / "cp_weight_ablation.csv"),
    )
    return parser.parse_args()


def compute_cp(jaccard: float, novelty: float, drift: float, weights: tuple[float, float, float]) -> float:
    w_j, w_n, w_d = weights
    return max(0.0, min(1.0, w_j * jaccard + w_n * (1.0 - novelty) + w_d * (1.0 - drift)))


def spearman(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[order[j]] == vals[order[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((v - mean_rx) ** 2 for v in rx))
    den_y = math.sqrt(sum((v - mean_ry) ** 2 for v in ry))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def load_run_components(benchmarks_root: Path) -> list[dict]:
    """Load per-run component scores from all_runs_summary.csv files."""
    runs = []
    for summary_path in sorted(benchmarks_root.glob("persona_*_v1__*/scoring/content/all_runs_summary.csv")):
        benchmark_name = summary_path.parts[-4]
        axis_model = benchmark_name.replace("persona_", "", 1)
        try:
            axis_name, model_name = axis_model.split("_v1__", maxsplit=1)
        except ValueError:
            continue

        with summary_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                runs.append({
                    "axis_name": axis_name,
                    "model_name": model_name,
                    "config_id": row["config_id"],
                    "jaccard": float(row["mean_token_jaccard_vs_base"]),
                    "novelty": float(row["mean_novelty_ratio"]),
                    "drift": float(row["drift_flag_rate"]),
                })
    return runs


def family_for_config(config_id: str) -> str | None:
    cid = config_id.lower()
    if "prompt_diff" in cid or "prompt-diff" in cid:
        return "prompt_diff"
    if "loreft" in cid or "reft" in cid:
        return "reft_loreft"
    if "layer" in cid or "paired" in cid or "dense" in cid:
        return "paired_dense"
    return None


def family_rankings(runs: list[dict], weights: tuple[float, float, float]) -> list[float]:
    """Return mean CP score per (axis, family) cell as a flat list (axis-major order)."""
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in runs:
        family = family_for_config(run["config_id"])
        if family is None:
            continue
        cp = compute_cp(run["jaccard"], run["novelty"], run["drift"], weights)
        scores[(run["axis_name"], family)].append(cp)

    ranking = []
    for axis in AXIS_ORDER:
        for family in FAMILY_ORDER:
            vals = scores.get((axis, family), [])
            ranking.append(sum(vals) / len(vals) if vals else 0.0)
    return ranking


def main() -> None:
    args = parse_args()
    benchmarks_root = Path(args.benchmarks_root)
    output_path = Path(args.output)

    runs = load_run_components(benchmarks_root)
    if not runs:
        raise SystemExit(f"No content summary CSV files found under {benchmarks_root}")

    reference_ranking = family_rankings(runs, REFERENCE_WEIGHTS)

    results = []
    for weights in WEIGHT_GRID:
        ranking = family_rankings(runs, weights)
        rho = spearman(reference_ranking, ranking)
        w_j, w_n, w_d = weights
        results.append({
            "w_jaccard": f"{w_j:.2f}",
            "w_novelty": f"{w_n:.2f}",
            "w_drift": f"{w_d:.2f}",
            "weight_label": f"{w_j}/{w_n}/{w_d}",
            "is_reference": "yes" if weights == REFERENCE_WEIGHTS else "no",
            "spearman_rho_vs_reference": f"{rho:.4f}" if not math.isnan(rho) else "",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["weight_label", "w_jaccard", "w_novelty", "w_drift", "is_reference", "spearman_rho_vs_reference"]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {output_path}")
    for row in results:
        ref_marker = " (reference)" if row["is_reference"] == "yes" else ""
        print(f"  {row['weight_label']}{ref_marker}: Spearman ρ = {row['spearman_rho_vs_reference']}")


if __name__ == "__main__":
    main()
