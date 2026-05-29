"""Transition matrix training pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from llm_ft_comparison.data import (
    extract_history,
    extract_prev_category_from_history,
    extract_target_category,
    load_records,
)


class TransitionMatrixPipeline:
    def run(self, config: dict) -> str:
        dataset_cfg = config.get("dataset", {})
        tm_cfg = config.get("transition_matrix", {})

        root = Path(__file__).resolve().parents[3]
        dataset_path = _resolve_path(dataset_cfg.get("path"), root)
        records = load_records(dataset_path)

        from_field = tm_cfg.get("from_field")
        to_field = tm_cfg.get("to_field")
        smoothing = tm_cfg.get("smoothing", 0.0)

        transition_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        row_totals: dict[str, float] = defaultdict(float)

        for item in records:
            if from_field:
                from_cat = item.get(from_field, "")
            else:
                history = extract_history(item)
                from_cat = extract_prev_category_from_history(history)

            if to_field:
                to_cat = item.get(to_field, "")
            else:
                to_cat = extract_target_category(item)

            if not from_cat or not to_cat:
                continue

            transition_counts[from_cat][to_cat] += 1.0
            row_totals[from_cat] += 1.0

        all_categories = sorted(
            set(transition_counts.keys())
            | {cat for counts in transition_counts.values() for cat in counts.keys()}
        )

        matrix = pd.DataFrame(0.0, index=all_categories, columns=all_categories)
        for from_cat, to_counts in transition_counts.items():
            total = row_totals[from_cat]
            if total <= 0:
                continue
            for to_cat, count in to_counts.items():
                matrix.loc[from_cat, to_cat] = count / total

        if smoothing:
            matrix = matrix + smoothing
            matrix = matrix.div(matrix.sum(axis=1), axis=0)

        output_path = _resolve_path(
            tm_cfg.get("output_path", "outputs/metrics/transition_matrix.csv"),
            root,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(output_path)

        if tm_cfg.get("write_priors", True):
            priors = matrix.sum(axis=0) / matrix.values.sum()
            priors_path = output_path.with_name(output_path.stem + "_priors.csv")
            priors.to_csv(priors_path, header=["prior_probability"])

        return str(output_path)


def _resolve_path(path_value: str | None, root: Path) -> Path:
    if not path_value:
        raise ValueError("Missing required path in config")
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root / path).resolve()
