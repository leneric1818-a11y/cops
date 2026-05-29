#!/usr/bin/env python3
"""Export the paper's axis-wise LLM-judge comparison table from leaderboards."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "tables" / "table3_judge_axis_comparison.csv"
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
        description="Export the paper's judge axis-comparison table as CSV."
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
        lambda: {"win_rate": [], "axis_delta": [], "case_delta": [], "role_delta": []}
    )

    for leaderboard_path in sorted(benchmarks_root.glob("persona_*_v1__*/leaderboard.csv")):
        benchmark_name = leaderboard_path.parent.name
        axis_name = benchmark_name.replace("persona_", "", 1).split("_v1__", maxsplit=1)[0]

        with leaderboard_path.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("judge_pairwise_steered_win_rate")]
        if not rows:
            continue

        best_by_family: dict[str, dict[str, str]] = {}
        for family in FAMILY_ORDER:
            family_rows = [row for row in rows if row["family"] == family]
            if not family_rows:
                continue
            best_by_family[family] = max(
                family_rows,
                key=lambda row: (
                    float(row["judge_pairwise_steered_win_rate"]),
                    float(row["judge_mean_axis_alignment_delta"]),
                    float(row["judge_mean_case_fidelity_delta"]),
                    float(row["judge_mean_client_role_fidelity_delta"]),
                ),
            )

        for family, row in best_by_family.items():
            target = by_axis_family[(axis_name, family)]
            target["win_rate"].append(float(row["judge_pairwise_steered_win_rate"]))
            target["axis_delta"].append(float(row["judge_mean_axis_alignment_delta"]))
            target["case_delta"].append(float(row["judge_mean_case_fidelity_delta"]))
            target["role_delta"].append(float(row["judge_mean_client_role_fidelity_delta"]))

    fieldnames = [
        "axis_name",
        "axis_label",
        "family",
        "family_label",
        "models",
        "mean_win_rate",
        "mean_axis_delta",
        "mean_case_delta",
        "mean_role_delta",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for axis_name in AXIS_ORDER:
            for family in FAMILY_ORDER:
                values = by_axis_family.get(
                    (axis_name, family),
                    {"win_rate": [], "axis_delta": [], "case_delta": [], "role_delta": []},
                )
                writer.writerow(
                    {
                        "axis_name": axis_name,
                        "axis_label": AXIS_LABELS[axis_name],
                        "family": family,
                        "family_label": FAMILY_LABELS[family],
                        "models": len(values["win_rate"]),
                        "mean_win_rate": format_mean(values["win_rate"]),
                        "mean_axis_delta": format_mean(values["axis_delta"]),
                        "mean_case_delta": format_mean(values["case_delta"]),
                        "mean_role_delta": format_mean(values["role_delta"]),
                    }
                )

    print(f"Wrote {output_path}")


def format_mean(values: list[float], decimals: int = 3) -> str:
    if not values:
        return ""
    mean_value = sum(values) / len(values)
    return f"{mean_value:.{decimals}f}"


if __name__ == "__main__":
    main()
