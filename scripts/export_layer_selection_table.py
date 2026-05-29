#!/usr/bin/env python3
"""Export the layer selection table for the paper appendix.

For each (axis, model) pair, reports:
- selected layer index (from actual run JSONL files)
- alpha value used
- test AUC from layerwise separability analysis (where available)
- test balanced accuracy (where available)
- signal-to-noise ratio (where available)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAYERWISE_PLAN = ROOT / "configs" / "benchmarks" / "model_matrix" / "layerwise_plan_v1.json"
BENCHMARKS_ROOT = ROOT / "outputs" / "metrics" / "benchmarks"

AXIS_ORDER = ("openness", "initiative", "cooperation", "hopefulness")
MODEL_ORDER = ("qwen3_4b", "qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b", "gemma_4_e2b", "gemma_4_e4b_it")
MODEL_LABELS = {
    "qwen3_4b": "Qwen3-4B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "qwen3_5_4b": "Qwen3.5-4B",
    "qwen3_5_9b": "Qwen3.5-9B",
    "gemma_4_e2b": "Gemma-4-E2B-it",
    "gemma_4_e4b_it": "Gemma-4-E4B-it",
}
AXIS_LABELS = {
    "openness": "Openness",
    "initiative": "Initiative",
    "cooperation": "Cooperation",
    "hopefulness": "Hopefulness",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export layer selection table for the paper appendix.")
    parser.add_argument("--benchmarks-root", default=str(BENCHMARKS_ROOT))
    parser.add_argument("--output", default=str(ROOT / "paper" / "tables" / "table_layer_selection.csv"))
    return parser.parse_args()


def get_actual_layer(bench_dir: Path, axis: str, model: str) -> tuple[int | None, float | None]:
    """Read layer_idx and alpha from the first paired_dense steered row."""
    runs_dir = bench_dir / f"persona_{axis}_v1__{model}" / "runs"
    if not runs_dir.exists():
        return None, None
    for jsonl_path in sorted(runs_dir.glob("paired_dense_*.jsonl")):
        try:
            rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        except Exception:
            continue
        steered = [r for r in rows if r.get("steering_mode")]
        if steered:
            r = steered[0]
            return r.get("layer_idx"), r.get("alpha")
    return None, None


def get_separability_stats(axis: str, model: str) -> dict[str, str]:
    """Look up AUC and balanced accuracy for the selected layer from separability outputs."""
    sep_dir = ROOT / "outputs" / "metrics" / f"layerwise_separability_{axis}_{model}_gpt54mini_1000"
    if not sep_dir.exists():
        return {}

    best_path = sep_dir / "best_layers.json"
    if not best_path.exists():
        return {}

    best = json.loads(best_path.read_text())
    entry = best.get("best_by_test_auc", {})
    if not entry:
        return {}

    return {
        "sep_layer": str(entry.get("layer", "")),
        "test_auc": f"{entry['test_auc']:.4f}" if "test_auc" in entry else "",
        "test_balanced_accuracy": f"{entry['test_balanced_accuracy']:.4f}" if "test_balanced_accuracy" in entry else "",
        "signal_to_noise": f"{entry['signal_to_noise']:.4f}" if "signal_to_noise" in entry else "",
    }


def main() -> None:
    args = parse_args()
    bench_root = Path(args.benchmarks_root)
    output_path = Path(args.output)

    fieldnames = [
        "axis_name", "axis_label", "model_name", "model_label",
        "selected_layer", "alpha",
        "sep_layer", "test_auc", "test_balanced_accuracy", "signal_to_noise",
    ]

    rows = []
    for axis in AXIS_ORDER:
        for model in MODEL_ORDER:
            layer, alpha = get_actual_layer(bench_root, axis, model)
            sep = get_separability_stats(axis, model)
            rows.append({
                "axis_name": axis,
                "axis_label": AXIS_LABELS[axis],
                "model_name": model,
                "model_label": MODEL_LABELS.get(model, model),
                "selected_layer": str(layer) if layer is not None else "",
                "alpha": str(alpha) if alpha is not None else "",
                **{k: sep.get(k, "") for k in ("sep_layer", "test_auc", "test_balanced_accuracy", "signal_to_noise")},
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path} ({len(rows)} rows)")
    # Print summary
    with_auc = sum(1 for r in rows if r["test_auc"])
    print(f"  Rows with separability AUC: {with_auc}/{len(rows)}")


if __name__ == "__main__":
    main()
