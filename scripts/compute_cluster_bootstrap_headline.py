"""Cluster-bootstrap CIs for the headline negative-alpha table.

Resamples on the persona level (10 clusters) for each of the 16 stems
× 2 case-agnostic variants (case_blind / case_aware) at alpha=-3.0.

Writes ``outputs/steering_eval/cluster_ci/headline_neg.json`` which is
consumed by ``scripts/build_paper_tables.py`` to render CI columns in
``paper/tables/tab_fullmatrix_neg_headline.tex``.

Usage::

    .venv/bin/python scripts/compute_cluster_bootstrap_headline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_ft_comparison.evaluation.cluster_bootstrap import (  # noqa: E402
    add_cluster_field_from_contexts,
    cluster_bootstrap_netpw,
)

ENSEMBLE_DIR = ROOT / "outputs/steering_eval/judged_v2_negalpha_ensemble"
CONTEXTS_PATH = ROOT / "data/processed/cops_contexts.jsonl"
OUTPUT_DIR = ROOT / "outputs/steering_eval/cluster_ci"

# (model_key, axis_key, layer)
STEMS: list[tuple[str, str, int]] = [
    ("qwen", "openness", 15),
    ("qwen", "initiative", 9),
    ("qwen", "cooperation", 20),
    ("qwen", "hopefulness", 18),
    ("gemma", "openness", 3),
    ("gemma", "initiative", 13),
    ("gemma", "cooperation", 23),
    ("gemma", "hopefulness", 25),
    ("qwen35_9b", "openness", 20),
    ("qwen35_9b", "initiative", 20),
    ("qwen35_9b", "cooperation", 15),
    ("qwen35_9b", "hopefulness", 15),
    ("mistral7b", "openness", 15),
    ("mistral7b", "initiative", 5),
    ("mistral7b", "cooperation", 15),
    ("mistral7b", "hopefulness", 20),
]

VARIANTS = {"CB": "global_blind", "CA": "global_aware"}
ALPHA = -3.0
N_BOOT = 2000
SEED = 42


def stem_file(model: str, axis: str, layer: int) -> Path:
    return ENSEMBLE_DIR / f"{model}_{axis}_L{layer}_neg_ensembled.jsonl"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_for_stem(model: str, axis: str, layer: int) -> dict:
    path = stem_file(model, axis, layer)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = load_rows(path)
    rows = add_cluster_field_from_contexts(rows, str(CONTEXTS_PATH), field="persona_name")

    out: dict[str, dict] = {}
    for var_short, var_long in VARIANTS.items():
        sub = [r for r in rows if r["variant"] == var_long and r["alpha"] == ALPHA]
        if not sub:
            out[var_short] = {"error": "no rows"}
            continue
        ci = cluster_bootstrap_netpw(sub, n_boot=N_BOOT, seed=SEED)
        out[var_short] = ci
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for model, axis, layer in STEMS:
        stem_id = f"{model}_{axis}_L{layer}"
        try:
            summary[stem_id] = compute_for_stem(model, axis, layer)
            cb, ca = summary[stem_id].get("CB", {}), summary[stem_id].get("CA", {})
            print(
                f"{stem_id:35s}  "
                f"CB={cb.get('point', float('nan')):+.3f} [{cb.get('lo', float('nan')):+.3f}, {cb.get('hi', float('nan')):+.3f}]  "
                f"CA={ca.get('point', float('nan')):+.3f} [{ca.get('lo', float('nan')):+.3f}, {ca.get('hi', float('nan')):+.3f}]  "
                f"n_clusters={cb.get('n_clusters', '?')}"
            )
        except FileNotFoundError as e:
            print(f"{stem_id:35s}  MISSING: {e}")
            summary[stem_id] = {"error": "file not found"}

    out_path = OUTPUT_DIR / "headline_neg.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
