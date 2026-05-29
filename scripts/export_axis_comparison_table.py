#!/usr/bin/env python3
"""Export the paper's axis-wise comparison table from multimodel leaderboards."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "tables" / "table1_axis_comparison.csv"
FAMILY_ORDER = ("paired_dense", "prompt_diff", "reft_loreft")
AXIS_ORDER = ("openness", "initiative", "cooperation", "hopefulness")
AXIS_LABELS = {
    "openness": "Openness",
    "initiative": "Initiative",
    "cooperation": "Cooperation",
    "hopefulness": "Hopefulness",
}
FAMILY_LABELS = {
    "paired_dense": "Paired dense",
    "prompt_diff": "Prompt-diff",
    "reft_loreft": "LoReFT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the paper's axis-comparison table as CSV."
    )
    parser.add_argument(
        "--benchmarks-root",
        default=str(ROOT / "outputs" / "metrics" / "benchmarks"),
        help="Root directory containing persona_*__<model>/leaderboard.csv outputs.",
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
    output_path = Path(args.output)

    by_axis_family: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"rank": [], "directional_effect": [], "content": [], "drift": [], "wilcoxon_p": []}
    )

    for leaderboard_path in sorted(benchmarks_root.glob("persona_*_v1__*/leaderboard.csv")):
        benchmark_name = leaderboard_path.parent.name
        axis_name = benchmark_name.replace("persona_", "", 1).split("_v1__", maxsplit=1)[0]

        with leaderboard_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        best_by_family: dict[str, dict[str, str]] = {}
        for row in rows:
            family = row["family"]
            if family in FAMILY_ORDER and family not in best_by_family:
                best_by_family[family] = row

        for family, row in best_by_family.items():
            target = by_axis_family[(axis_name, family)]
            target["rank"].append(float(row["rank_score"]))
            target["directional_effect"].append(float(row["directional_effect"]))
            target["content"].append(float(row["content_preservation_score"]))
            target["drift"].append(float(row["drift_flag_rate"]))
            p_val = row.get("wilcoxon_p", "")
            if p_val:
                target["wilcoxon_p"].append(float(p_val))

    fieldnames = [
        "axis_name",
        "axis_label",
        "family",
        "family_label",
        "models",
        "mean_rank",
        "mean_directional_effect",
        "std_directional_effect",
        "mean_content",
        "std_content",
        "mean_drift",
        "std_drift",
        "median_wilcoxon_p",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for axis_name in AXIS_ORDER:
            for family in FAMILY_ORDER:
                values = by_axis_family.get(
                    (axis_name, family),
                    {"rank": [], "directional_effect": [], "content": [], "drift": [], "wilcoxon_p": []},
                )
                writer.writerow(
                    {
                        "axis_name": axis_name,
                        "axis_label": AXIS_LABELS[axis_name],
                        "family": family,
                        "family_label": FAMILY_LABELS[family],
                        "models": len(values["rank"]),
                        "mean_rank": format_mean(values["rank"]),
                        "mean_directional_effect": format_mean(
                            values["directional_effect"], decimals=4
                        ),
                        "std_directional_effect": format_std(
                            values["directional_effect"], decimals=4
                        ),
                        "mean_content": format_mean(values["content"]),
                        "std_content": format_std(values["content"]),
                        "mean_drift": format_mean(values["drift"], decimals=2),
                        "std_drift": format_std(values["drift"], decimals=2),
                        "median_wilcoxon_p": format_median(values["wilcoxon_p"], decimals=3),
                    }
                )

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


def format_median(values: list[float], decimals: int = 3) -> str:
    if not values:
        return ""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    return f"{median:.{decimals}f}"


if __name__ == "__main__":
    main()
