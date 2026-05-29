#!/usr/bin/env python3
"""Train a binary style-axis classifier and score steering outputs.

This generalizes the earlier defensive/open scorer to arbitrary binary persona
axes such as:

- open vs defensive
- explorative vs reactive
- cooperative vs resistant
- hopeful vs resigned
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary style-axis classifier and score steering outputs."
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="Flat JSONL with fields including response text and style label.",
    )
    parser.add_argument(
        "--score-paths",
        required=True,
        nargs="+",
        help="One or more JSONL files from steering runs to score.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for classifier metrics and scored summaries.",
    )
    parser.add_argument(
        "--positive-label",
        default="open",
        help="Style label treated as the positive direction.",
    )
    parser.add_argument(
        "--negative-label",
        default="defensive",
        help="Style label treated as the negative direction.",
    )
    parser.add_argument(
        "--style-field",
        default="style",
        help="Field name holding style labels in the training data.",
    )
    parser.add_argument(
        "--response-field",
        default="response",
        help="Field name holding response text in the training data.",
    )
    parser.add_argument(
        "--group-field",
        default="seed_id",
        help="Field name used for grouped train/test splitting.",
    )
    parser.add_argument(
        "--axis-name",
        default=None,
        help="Optional human-readable axis name written into the summaries.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Held-out fraction for grouped classifier validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Bootstrap samples for mean-delta confidence intervals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        row
        for row in load_jsonl(Path(args.train_path))
        if row.get(args.style_field) in {args.positive_label, args.negative_label}
    ]
    if not train_rows:
        raise SystemExit("No training rows matched the configured positive/negative labels.")

    train_texts = [str(row[args.response_field]) for row in train_rows]
    y = np.array(
        [1 if row[args.style_field] == args.positive_label else 0 for row in train_rows],
        dtype=np.int64,
    )
    groups = np.array(
        [row.get(args.group_field) or f"row_{idx}" for idx, row in enumerate(train_rows)]
    )

    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed).split(
            train_texts, y, groups
        )
    )
    clf = build_classifier(args.seed)
    clf.fit([train_texts[idx] for idx in train_idx], y[train_idx])
    test_proba = clf.predict_proba([train_texts[idx] for idx in test_idx])[:, 1]
    test_pred = (test_proba >= 0.5).astype(np.int64)

    classifier_metrics = {
        "axis_name": args.axis_name or f"{args.negative_label}_vs_{args.positive_label}",
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "style_field": args.style_field,
        "response_field": args.response_field,
        "group_field": args.group_field,
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "train_group_count": int(len(set(groups[train_idx]))),
        "test_group_count": int(len(set(groups[test_idx]))),
        "test_accuracy": float(accuracy_score(y[test_idx], test_pred)),
        "test_auc": float(roc_auc_score(y[test_idx], test_proba)),
        "test_average_precision": float(average_precision_score(y[test_idx], test_proba)),
        "positive_base_rate_test": float(y[test_idx].mean()),
    }
    (output_dir / "classifier_metrics.json").write_text(
        json.dumps(classifier_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_summary_rows = []
    for score_path_raw in args.score_paths:
        score_path = Path(score_path_raw)
        rows = load_jsonl(score_path)
        summary_rows, scored_rows = score_run(
            rows=rows,
            clf=clf,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            positive_label=args.positive_label,
            axis_name=classifier_metrics["axis_name"],
        )
        stem = score_path.stem
        (output_dir / f"{stem}.summary.json").write_text(
            json.dumps(summary_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output_dir / f"{stem}.summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        (output_dir / f"{stem}.scored_rows.json").write_text(
            json.dumps(scored_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for row in summary_rows:
            enriched = dict(row)
            enriched["source_file"] = str(score_path.resolve())
            all_summary_rows.append(enriched)

    if all_summary_rows:
        with (output_dir / "all_runs_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary_rows)

    print(json.dumps(classifier_metrics, ensure_ascii=False, indent=2))
    print("\nTop configs by mean delta:")
    for row in sorted(all_summary_rows, key=lambda item: item["mean_delta_vs_base"], reverse=True)[:10]:
        print(
            row["source_file"],
            row["config_id"],
            "mean_delta",
            round(row["mean_delta_vs_base"], 4),
            "wins",
            row["wins"],
            "losses",
            row["losses"],
            "wilcoxon_p",
            row["wilcoxon_p"],
        )


def build_classifier(seed: int) -> Pipeline:
    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    clf = LogisticRegression(
        max_iter=2000,
        C=4.0,
        random_state=seed,
        solver="liblinear",
    )
    return Pipeline([("features", vectorizer), ("clf", clf)])


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def score_run(
    rows: list[dict],
    clf: Pipeline,
    bootstrap_samples: int,
    seed: int,
    positive_label: str,
    axis_name: str,
):
    by_example = defaultdict(dict)
    all_texts = []
    row_refs = []

    for row in rows:
        example_index = row["example_index"]
        config_key = "base" if row["mode"] == "base" else build_config_key(row)
        by_example[example_index][config_key] = row
        all_texts.append(row["response"])
        row_refs.append((example_index, config_key))

    probs = clf.predict_proba(all_texts)[:, 1]
    scored_rows = []
    for (example_index, config_key), prob in zip(row_refs, probs):
        row = by_example[example_index][config_key]
        row["positive_probability"] = float(prob)
        scored_rows.append(
            {
                "example_index": example_index,
                "config_key": config_key,
                "mode": row["mode"],
                "positive_label": positive_label,
                "positive_probability": float(prob),
                "response": row["response"],
                "seed_id": row.get("seed_id"),
            }
        )

    config_keys = sorted({key for item in by_example.values() for key in item if key != "base"})
    summary_rows = []
    for config_key in config_keys:
        base_probs = []
        steered_probs = []
        deltas = []
        wins = 0
        losses = 0
        ties = 0
        for example_index, item in by_example.items():
            if "base" not in item or config_key not in item:
                continue
            base_prob = item["base"]["positive_probability"]
            steered_prob = item[config_key]["positive_probability"]
            delta = steered_prob - base_prob
            base_probs.append(base_prob)
            steered_probs.append(steered_prob)
            deltas.append(delta)
            if delta > 1e-12:
                wins += 1
            elif delta < -1e-12:
                losses += 1
            else:
                ties += 1

        deltas_arr = np.array(deltas, dtype=float)
        mean_delta = float(deltas_arr.mean()) if len(deltas_arr) else 0.0
        median_delta = float(np.median(deltas_arr)) if len(deltas_arr) else 0.0
        ci_low, ci_high = bootstrap_mean_ci(
            deltas_arr,
            samples=bootstrap_samples,
            seed=seed,
        )
        sign_p = sign_test_pvalue(wins=wins, losses=losses)
        try:
            wilcoxon_stat, wilcoxon_p = wilcoxon(
                deltas_arr,
                alternative="greater",
                zero_method="wilcox",
            )
            wilcoxon_stat = float(wilcoxon_stat)
            wilcoxon_p = float(wilcoxon_p)
        except ValueError:
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0

        summary_rows.append(
            {
                "axis_name": axis_name,
                "positive_label": positive_label,
                "config_id": config_key,
                "n_examples": len(deltas),
                "mean_base_positive_probability": float(np.mean(base_probs)) if base_probs else 0.0,
                "mean_positive_probability": float(np.mean(steered_probs)) if steered_probs else 0.0,
                "mean_delta_vs_base": mean_delta,
                "median_delta_vs_base": median_delta,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "sign_test_p": sign_p,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
            }
        )

    summary_rows.sort(key=lambda row: row["mean_delta_vs_base"], reverse=True)
    return summary_rows, scored_rows


def build_config_key(row: dict) -> str:
    explicit = row.get("config_id")
    if explicit:
        return str(explicit)
    if row.get("steering_mode") == "weighted":
        layers = row.get("layers", [])
        return f"weighted_{'-'.join(str(layer) for layer in layers)}_a{row['alpha']}"
    if row.get("steering_mode") == "gaussian":
        layers = row.get("layers", [])
        return f"gaussian_{'-'.join(str(layer) for layer in layers)}_a{row['alpha']}"
    layer_idx = row.get("layer_idx")
    alpha = row.get("alpha")
    if layer_idx is None or alpha is None:
        return "unknown"
    return f"layer{layer_idx}_a{alpha}"


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    n = values.size
    for idx in range(samples):
        sample = values[rng.integers(0, n, size=n)]
        means[idx] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return float(binomtest(wins, n=n, p=0.5, alternative="greater").pvalue)


if __name__ == "__main__":
    main()
