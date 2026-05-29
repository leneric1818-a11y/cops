#!/usr/bin/env python3
"""Optimise the content-preservation weighting against judge case-fidelity.

The content-preservation score is currently defined ad-hoc as
``CP = 0.6 * jaccard + 0.2 * (1 - novelty) + 0.2 * (1 - drift_rate)``.
This script finds the weights that maximise predictive alignment between CP
and two judge references (``case_fidelity_steered`` absolute and
``case_fidelity_delta`` = steered - base) on:

  * per-example granularity (~14 k rows across 24 benchmarks), and
  * per-run granularity (aggregates, ~one row per (benchmark, config_key)).

Two weight spaces are explored:

  1. A constrained simplex ``w_J + w_N + w_D = 1, w_* >= 0`` via a grid
     search (step 0.05, 231 points).
  2. An unconstrained linear regression ``y ~ beta_J * J + beta_N * (1-N)
     + beta_D * (1-D) + beta_0``, solved via ``numpy.linalg.lstsq``.

For each (granularity, reference, weight-space) combination we report:

  * best Spearman / Kendall / Pearson correlation and weights;
  * 1000-sample bootstrap 95% CI for the correlation;
  * leave-one-axis-out and leave-one-model-out cross-validation;
  * distance from the current reference weights (0.6, 0.2, 0.2).

Outputs CSVs / JSON summary under
``outputs/metrics/ablation/cp_weights/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Reuse the formula + rank helper from the existing ablation script.
import sys

sys.path.insert(0, str(ROOT / "scripts"))
from ablate_content_preservation_weights import compute_cp, spearman  # noqa: E402

try:
    from scipy.stats import kendalltau, pearsonr
except ImportError:  # pragma: no cover

    def pearsonr(x: Sequence[float], y: Sequence[float]):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.std() == 0 or y.std() == 0:
            return (float("nan"), float("nan"))
        r = float(np.corrcoef(x, y)[0, 1])
        return (r, float("nan"))

    def kendalltau(x, y):
        # Fallback: report NaN to avoid heavy stat code duplication.
        return (float("nan"), float("nan"))


REFERENCE_WEIGHTS = (0.6, 0.2, 0.2)
REFERENCES = ("case_fidelity_steered", "case_fidelity_delta")
GRID_STEP = 0.05


# ---------------------------------------------------------------------------
# Data loading


def _load_scored_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _parse_axis_model(benchmark_name: str) -> tuple[str, str] | None:
    if not benchmark_name.startswith("persona_"):
        return None
    rest = benchmark_name[len("persona_"):]
    if "_v1__" not in rest:
        return None
    axis, model = rest.split("_v1__", 1)
    return axis, model


def load_per_example(benchmarks_root: Path,
                     axes: set[str] | None = None,
                     models: set[str] | None = None) -> list[dict]:
    """Join content and judge scored rows per benchmark and run.

    Returns a list of dicts with fields:
      axis, model, benchmark, run_name, config_key, example_index, seed_id,
      jaccard, novelty, drift_flag (0/1 float),
      case_fidelity_base, case_fidelity_steered, case_fidelity_delta.
    """
    rows: list[dict] = []
    for bench_dir in sorted(benchmarks_root.glob("persona_*_v1__*")):
        parsed = _parse_axis_model(bench_dir.name)
        if parsed is None:
            continue
        axis, model = parsed
        if axes and axis not in axes:
            continue
        if models and model not in models:
            continue
        content_dir = bench_dir / "scoring" / "content"
        judge_dir = bench_dir / "scoring" / "judge"
        if not content_dir.is_dir() or not judge_dir.is_dir():
            continue
        for content_path in sorted(content_dir.glob("*.scored_rows.json")):
            run_name = content_path.name[: -len(".scored_rows.json")]
            judge_path = judge_dir / content_path.name
            if not judge_path.is_file():
                continue
            content_rows = _load_scored_rows(content_path)
            judge_rows = _load_scored_rows(judge_path)
            # Index judge rows for the join.
            judge_by_key: dict[tuple[int, str], dict] = {}
            for r in judge_rows:
                key = (int(r["example_index"]), str(r["config_key"]))
                judge_by_key[key] = r
            for r in content_rows:
                key = (int(r["example_index"]), str(r["config_key"]))
                jr = judge_by_key.get(key)
                if jr is None:
                    continue
                if jr.get("case_fidelity_steered") is None:
                    continue
                try:
                    rows.append({
                        "axis": axis,
                        "model": model,
                        "benchmark": bench_dir.name,
                        "run_name": run_name,
                        "config_key": str(r["config_key"]),
                        "example_index": int(r["example_index"]),
                        "seed_id": r.get("seed_id"),
                        "jaccard": float(r["token_jaccard_vs_base"]),
                        "novelty": float(r["novelty_ratio"]),
                        "drift_flag": 1.0 if r.get("drift_flag") else 0.0,
                        "case_fidelity_base": float(jr["case_fidelity_base"])
                            if jr.get("case_fidelity_base") is not None else None,
                        "case_fidelity_steered": float(jr["case_fidelity_steered"]),
                        "case_fidelity_delta": float(jr["case_fidelity_delta"])
                            if jr.get("case_fidelity_delta") is not None
                            else float(jr["case_fidelity_steered"]) - float(jr.get("case_fidelity_base", 0.0)),
                    })
                except (TypeError, ValueError, KeyError):
                    continue
    return rows


def aggregate_per_run(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["benchmark"], r["run_name"], r["config_key"])].append(r)
    out: list[dict] = []
    for (bench, run, cfg), bucket in buckets.items():
        axis = bucket[0]["axis"]
        model = bucket[0]["model"]
        fid_steered = [b["case_fidelity_steered"] for b in bucket]
        fid_delta = [b["case_fidelity_delta"] for b in bucket
                     if b["case_fidelity_delta"] is not None]
        out.append({
            "axis": axis,
            "model": model,
            "benchmark": bench,
            "run_name": run,
            "config_key": cfg,
            "n": len(bucket),
            "jaccard": sum(b["jaccard"] for b in bucket) / len(bucket),
            "novelty": sum(b["novelty"] for b in bucket) / len(bucket),
            "drift_flag": sum(b["drift_flag"] for b in bucket) / len(bucket),
            "case_fidelity_steered": sum(fid_steered) / len(fid_steered),
            "case_fidelity_delta": (sum(fid_delta) / len(fid_delta)) if fid_delta else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Correlation helpers


def _pearson_safe(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_safe(x: Sequence[float], y: Sequence[float]) -> float:
    # Delegate to the existing helper for consistency.
    return spearman(list(x), list(y))


def _kendall_safe(x: Sequence[float], y: Sequence[float]) -> float:
    try:
        tau, _ = kendalltau(x, y)
        return float(tau) if not math.isnan(tau) else float("nan")
    except Exception:
        return float("nan")


def cp_scores(rows: list[dict], weights: tuple[float, float, float]) -> np.ndarray:
    return np.array([
        compute_cp(r["jaccard"], r["novelty"], r["drift_flag"], weights)
        for r in rows
    ])


def bootstrap_ci(values_x: np.ndarray, values_y: np.ndarray,
                 n_samples: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% CI for Spearman correlation of (x, y)."""
    rng = random.Random(seed)
    n = len(values_x)
    if n < 4:
        return (float("nan"), float("nan"))
    samples = []
    xs = values_x.tolist()
    ys = values_y.tolist()
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        sx = [xs[i] for i in idx]
        sy = [ys[i] for i in idx]
        rho = _spearman_safe(sx, sy)
        if not math.isnan(rho):
            samples.append(rho)
    if not samples:
        return (float("nan"), float("nan"))
    samples.sort()
    lo = samples[int(alpha / 2 * len(samples))]
    hi = samples[min(len(samples) - 1, int((1 - alpha / 2) * len(samples)))]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Simplex grid search


def simplex_grid(step: float = GRID_STEP) -> list[tuple[float, float, float]]:
    """Enumerate all (w_J, w_N, w_D) triples on the simplex with the given step."""
    k = int(round(1.0 / step))
    out = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            kk = k - i - j
            out.append((i / k, j / k, kk / k))
    return out


def grid_search(rows: list[dict], reference: str,
                n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Full grid search. Returns dict with per-point rows + best row."""
    y = np.array([r[reference] for r in rows])
    points = simplex_grid()
    results = []
    for weights in points:
        cp = cp_scores(rows, weights)
        results.append({
            "w_jaccard": weights[0],
            "w_novelty": weights[1],
            "w_drift": weights[2],
            "spearman": _spearman_safe(cp, y),
            "kendall": _kendall_safe(cp, y),
            "pearson": _pearson_safe(cp, y),
        })

    def pick(key: str) -> dict:
        finite = [r for r in results if not math.isnan(r[key])]
        if not finite:
            return {}
        return max(finite, key=lambda r: r[key])

    best = pick("spearman")
    # Bootstrap CI only for the best point.
    if best:
        cp_best = cp_scores(rows, (best["w_jaccard"], best["w_novelty"], best["w_drift"]))
        lo, hi = bootstrap_ci(cp_best, y, n_samples=n_bootstrap, seed=seed)
        best["spearman_ci_low"] = lo
        best["spearman_ci_high"] = hi

    # Reference row and its bootstrap CI.
    ref_cp = cp_scores(rows, REFERENCE_WEIGHTS)
    ref_rho = _spearman_safe(ref_cp, y)
    ref_lo, ref_hi = bootstrap_ci(ref_cp, y, n_samples=n_bootstrap, seed=seed)
    reference_row = {
        "w_jaccard": REFERENCE_WEIGHTS[0],
        "w_novelty": REFERENCE_WEIGHTS[1],
        "w_drift": REFERENCE_WEIGHTS[2],
        "spearman": ref_rho,
        "kendall": _kendall_safe(ref_cp, y),
        "pearson": _pearson_safe(ref_cp, y),
        "spearman_ci_low": ref_lo,
        "spearman_ci_high": ref_hi,
    }

    return {
        "all": results,
        "best_spearman": best,
        "reference": reference_row,
        "n": len(rows),
    }


# ---------------------------------------------------------------------------
# Free linear regression


def linear_fit(rows: list[dict], reference: str,
               n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """OLS fit y = beta_J*J + beta_N*(1-N) + beta_D*(1-D) + beta_0."""
    J = np.array([r["jaccard"] for r in rows])
    N = np.array([1.0 - r["novelty"] for r in rows])
    D = np.array([1.0 - r["drift_flag"] for r in rows])
    y = np.array([r[reference] for r in rows])
    X = np.stack([J, N, D, np.ones_like(J)], axis=1)
    beta, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = len(rows)
    p = 3
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")

    # Normalise raw slopes onto the simplex so they can be compared with the grid
    # optimum. Handle negative slopes by clipping to zero before renormalising.
    raw = beta[:3]
    clipped = np.clip(raw, 0.0, None)
    s = clipped.sum()
    normalised = (clipped / s).tolist() if s > 0 else [float("nan")] * 3

    # Bootstrap CIs for the four raw coefficients.
    rng = np.random.default_rng(seed)
    boots: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b, _, _, _ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boots.append(b)
    boots_arr = np.stack(boots, axis=0)
    ci_low = np.percentile(boots_arr, 2.5, axis=0)
    ci_high = np.percentile(boots_arr, 97.5, axis=0)

    # Spearman of the OLS fit against y, with bootstrap CI on that correlation.
    rho = _spearman_safe(y_hat, y)
    lo, hi = bootstrap_ci(y_hat, y, n_samples=n_bootstrap, seed=seed)

    return {
        "beta_J": float(beta[0]),
        "beta_N": float(beta[1]),
        "beta_D": float(beta[2]),
        "beta_0": float(beta[3]),
        "beta_J_ci": (float(ci_low[0]), float(ci_high[0])),
        "beta_N_ci": (float(ci_low[1]), float(ci_high[1])),
        "beta_D_ci": (float(ci_low[2]), float(ci_high[2])),
        "normalised": {"w_jaccard": normalised[0], "w_novelty": normalised[1],
                        "w_drift": normalised[2]},
        "r2": r2,
        "adj_r2": adj_r2,
        "spearman_of_fit": rho,
        "spearman_ci_low": lo,
        "spearman_ci_high": hi,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Cross-validation


def cross_validate(rows: list[dict], reference: str,
                   fold_key: str, seed: int = 0) -> list[dict]:
    """Leave-one-out CV on a categorical key (``axis`` or ``model``)."""
    fold_values = sorted({r[fold_key] for r in rows})
    cv_rows = []
    for leave_out in fold_values:
        train = [r for r in rows if r[fold_key] != leave_out]
        test = [r for r in rows if r[fold_key] == leave_out]
        if len(train) < 10 or len(test) < 5:
            continue
        train_res = grid_search(train, reference, n_bootstrap=0, seed=seed)
        best = train_res["best_spearman"]
        if not best:
            continue
        test_cp = cp_scores(test, (best["w_jaccard"], best["w_novelty"], best["w_drift"]))
        test_rho = _spearman_safe(test_cp, [r[reference] for r in test])
        cv_rows.append({
            "fold_key": fold_key,
            "leave_out": leave_out,
            "n_train": len(train),
            "n_test": len(test),
            "train_w_jaccard": best["w_jaccard"],
            "train_w_novelty": best["w_novelty"],
            "train_w_drift": best["w_drift"],
            "train_spearman": best["spearman"],
            "test_spearman": test_rho,
        })
    return cv_rows


# ---------------------------------------------------------------------------
# Main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks-root",
                   default=str(ROOT / "outputs" / "metrics" / "benchmarks"))
    p.add_argument("--output-dir",
                   default=str(ROOT / "outputs" / "metrics" / "ablation" / "cp_weights"))
    p.add_argument("--axes", nargs="+", default=None,
                   help="Subset of axis names (for smoke tests).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of model names (for smoke tests).")
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench_root = Path(args.benchmarks_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axes = set(args.axes) if args.axes else None
    models = set(args.models) if args.models else None

    print(f"Loading per-example rows from {bench_root} …")
    per_example = load_per_example(bench_root, axes=axes, models=models)
    print(f"  {len(per_example):,} per-example rows")
    per_run = aggregate_per_run(per_example)
    print(f"  {len(per_run):,} per-run rows")
    if not per_example:
        raise SystemExit("No joined rows. Check benchmarks-root / filters.")

    all_results: dict[str, dict] = {}
    all_grid_points: list[dict] = []
    cv_all: list[dict] = []

    combos = [("per_example", per_example), ("per_run", per_run)]
    for gran_label, gran_rows in combos:
        for ref in REFERENCES:
            key = f"{gran_label}__{ref}"
            print(f"\n== {key}  (n={len(gran_rows):,}) ==")
            grid_res = grid_search(gran_rows, ref, n_bootstrap=args.bootstrap_samples,
                                    seed=args.seed)
            lin_res = linear_fit(gran_rows, ref, n_bootstrap=args.bootstrap_samples,
                                  seed=args.seed)
            best = grid_res["best_spearman"]
            ref_row = grid_res["reference"]
            print(f"  simplex best:    ρ={best['spearman']:.4f} "
                  f"CI[{best.get('spearman_ci_low', float('nan')):.3f},"
                  f"{best.get('spearman_ci_high', float('nan')):.3f}]  "
                  f"w=({best['w_jaccard']:.2f},{best['w_novelty']:.2f},{best['w_drift']:.2f})")
            print(f"  reference 0.6/0.2/0.2: ρ={ref_row['spearman']:.4f} "
                  f"Δ={best['spearman'] - ref_row['spearman']:+.4f}")
            nn = lin_res["normalised"]
            print(f"  OLS normalised:  w=({nn['w_jaccard']:.2f},{nn['w_novelty']:.2f},{nn['w_drift']:.2f})  "
                  f"adj-R²={lin_res['adj_r2']:.3f}  ρ(fit)={lin_res['spearman_of_fit']:.4f}")

            # Cross-validation only on per-example with steered reference (most interesting).
            if gran_label == "per_example" and ref == "case_fidelity_steered":
                for fold_key in ("axis", "model"):
                    fold_rows = cross_validate(gran_rows, ref, fold_key, seed=args.seed)
                    cv_all.extend(fold_rows)
                    if fold_rows:
                        mean_test = sum(r["test_spearman"] for r in fold_rows) / len(fold_rows)
                        std_test = (
                            sum((r["test_spearman"] - mean_test) ** 2 for r in fold_rows)
                            / max(1, len(fold_rows) - 1)
                        ) ** 0.5
                        print(f"  CV leave-one-{fold_key}-out: "
                              f"mean test-ρ={mean_test:.3f} ± {std_test:.3f}  "
                              f"({len(fold_rows)} folds)")

            all_results[key] = {"grid_best": best, "reference_row": ref_row, "ols": lin_res}
            for row in grid_res["all"]:
                all_grid_points.append({"granularity": gran_label, "reference": ref, **row})

    # Write CSVs.
    with (out_dir / "grid_points.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["granularity", "reference", "w_jaccard", "w_novelty",
                      "w_drift", "spearman", "kendall", "pearson"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_grid_points:
            writer.writerow({k: row[k] for k in fieldnames})

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=float)

    with (out_dir / "cv.csv").open("w", newline="", encoding="utf-8") as fh:
        if cv_all:
            writer = csv.DictWriter(fh, fieldnames=list(cv_all[0].keys()))
            writer.writeheader()
            writer.writerows(cv_all)
        else:
            fh.write("fold_key,leave_out,n_train,n_test,train_w_jaccard,train_w_novelty,train_w_drift,train_spearman,test_spearman\n")

    # Save per-run and per-example joined data for plotting / auditing.
    with (out_dir / "joined_per_run.csv").open("w", newline="", encoding="utf-8") as fh:
        if per_run:
            writer = csv.DictWriter(fh, fieldnames=list(per_run[0].keys()))
            writer.writeheader()
            writer.writerows(per_run)

    print(f"\nWrote outputs to {out_dir}")
    print(f"  grid_points.csv   ({len(all_grid_points):,} rows)")
    print(f"  summary.json      (best / reference / OLS per combo)")
    print(f"  cv.csv            ({len(cv_all):,} fold rows)")


if __name__ == "__main__":
    main()
