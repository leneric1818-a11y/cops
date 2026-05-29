#!/usr/bin/env python3
"""Export the cross-model paper table from final leaderboard outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_MATRIX = ROOT / "configs" / "models" / "persona_model_matrix_v1.json"
DEFAULT_OUTPUT = ROOT / "paper" / "tables" / "table2_model_comparison.csv"
FAMILY_ORDER = ("paired_dense", "prompt_diff", "reft_loreft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the paper's cross-model comparison table as CSV."
    )
    parser.add_argument(
        "--benchmarks-root",
        default=str(ROOT / "outputs" / "metrics" / "benchmarks"),
        help="Root directory containing persona_*__<model>/leaderboard.csv outputs.",
    )
    parser.add_argument(
        "--model-matrix",
        default=str(DEFAULT_MODEL_MATRIX),
        help="Model matrix JSON used to map model ids to paper labels.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Destination CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks_root = Path(args.benchmarks_root)
    model_matrix_path = Path(args.model_matrix)
    output_path = Path(args.output)

    model_matrix = json.loads(model_matrix_path.read_text(encoding="utf-8"))
    models = model_matrix["models"]
    label_by_model = {model["name"]: model["label"] for model in models}
    model_order = [model["name"] for model in models]

    by_model_family: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"rank": [], "content": [], "drift": []}
    )

    for leaderboard_path in sorted(benchmarks_root.glob("persona_*__*/leaderboard.csv")):
        benchmark_name = leaderboard_path.parent.name
        axis_model = benchmark_name.replace("persona_", "", 1)
        try:
            _axis, model_name = axis_model.split("_v1__", maxsplit=1)
        except ValueError as exc:
            raise SystemExit(f"Unexpected benchmark directory name: {benchmark_name}") from exc

        with leaderboard_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        best_by_family: dict[str, dict[str, str]] = {}
        for row in rows:
            family = row["family"]
            if family in FAMILY_ORDER and family not in best_by_family:
                best_by_family[family] = row

        for family, row in best_by_family.items():
            target = by_model_family[(model_name, family)]
            target["rank"].append(float(row["rank_score"]))
            target["content"].append(float(row["content_preservation_score"]))
            target["drift"].append(float(row["drift_flag_rate"]))

    fieldnames = [
        "model_name",
        "model_label",
        "paired_dense_axes",
        "paired_dense_rank",
        "paired_dense_content",
        "paired_dense_content_std",
        "paired_dense_drift",
        "paired_dense_drift_std",
        "prompt_diff_axes",
        "prompt_diff_rank",
        "prompt_diff_content",
        "prompt_diff_content_std",
        "prompt_diff_drift",
        "prompt_diff_drift_std",
        "reft_loreft_axes",
        "reft_loreft_rank",
        "reft_loreft_content",
        "reft_loreft_content_std",
        "reft_loreft_drift",
        "reft_loreft_drift_std",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in model_order:
            row = {
                "model_name": model_name,
                "model_label": label_by_model.get(model_name, model_name),
            }
            for family in FAMILY_ORDER:
                values = by_model_family.get((model_name, family), {"rank": [], "content": [], "drift": []})
                prefix = family
                row[f"{prefix}_axes"] = len(values["rank"])
                row[f"{prefix}_rank"] = format_mean(values["rank"])
                row[f"{prefix}_content"] = format_mean(values["content"])
                row[f"{prefix}_content_std"] = format_std(values["content"])
                row[f"{prefix}_drift"] = format_mean(values["drift"], decimals=2)
                row[f"{prefix}_drift_std"] = format_std(values["drift"], decimals=2)
            writer.writerow(row)

    print(f"Wrote {output_path}")


def format_mean(values: list[float], decimals: int = 3) -> str:
    if not values:
        return ""
    mean_value = sum(values) / len(values)
    return f"{mean_value:.{decimals}f}"


def format_std(values: list[float], decimals: int = 3) -> str:
    if len(values) < 2:
        return ""
    mean_value = sum(values) / len(values)
    variance = sum((v - mean_value) ** 2 for v in values) / (len(values) - 1)
    std_value = variance ** 0.5
    return f"{std_value:.{decimals}f}"


if __name__ == "__main__":
    main()
