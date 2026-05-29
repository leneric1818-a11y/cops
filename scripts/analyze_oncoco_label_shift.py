#!/usr/bin/env python3
"""Analyse how vector steering shifts the OnCoCo predicted-label distribution.

Compares the CL-* label frequencies of steered vs. baseline responses, revealing
which utterance-type categories increase or decrease under each steering direction.

Usage:
    python scripts/analyze_oncoco_label_shift.py
    python scripts/analyze_oncoco_label_shift.py --axis openness --pole negative --heatmap
    python scripts/analyze_oncoco_label_shift.py --variant global_blind --heatmap
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS_FILE = ROOT / "outputs/steering_eval/oncoco_labels.jsonl"

# Human-readable short names for CL-* codes (best-effort mapping)
CL_NAMES: dict[str, str] = {
    "CL-FB-*-*-*-*":        "Feedback",
    "CL-E-*-*-PT-*":        "Empathie f. Dritte",
    "CL-E-*-*-ECC-*":       "Mitgefühl m. anderen",
    "CL-E-*-*-ECP-*":       "Sorge um andere",
    "CL-IF-ACP-*-PS-*":     "Problemdarstellung",
    "CL-IF-ACP-*-PD-*":     "Problemdefinition",
    "CL-IF-ACP-*-DPD-*":    "Detaill. Problemdarst.",
    "CL-IF-ACP-*-FPA-*":    "Preisgeben pers. Daten",
    "CL-IF-ACP-*-OE-*":     "Eigene Gefühlsdarst.",
    "CL-IF-ACP-*-Cons-*":   "Zustimmung",
    "CL-IF-ACP-*-Rej-*":    "Ablehnung",
    "CL-IF-ACP-*-Req-*":    "Nachfrage / Bitte",
    "CL-IF-AO-*-Obj-*":     "Einwand",
    "CL-IF-AO-*-Ext-*":     "Erweiterung",
    "CL-IF-Mot-*-FC-*":     "Motivation f. Veränd.",
    "CL-IF-Mot-*-RC-*":     "Widerstand g. Veränd.",
    "CL-IF-RA-*-RF-*":      "Ressourcen: Familie",
    "CL-IF-RA-*-RP-*":      "Ressourcen: profess.",
    "CL-IF-HP-*-PosF-*":    "Allg. pos. Rückmeldung",
    "CL-IF-HP-*-PosFR-*":   "Pos. RM zu Empfehlung",
    "CL-IF-HP-*-NegFR-*":   "Neg. RM zu Empfehlung",
    "CL-IF-HP-*-RepRA-*":   "Bericht Umsetzung",
    "CL-IF-HP-*-Succ-*":    "Erfolg",
    "CL-IF-HP-*-Fail-*":    "Misserfolg",
    "CL-FC-*-*-F-*":        "Formales Abschluss",
    "CL-FC-*-*-UPR-*":      "Unpassende Reaktion",
    "CL-O-*-*-O-*":         "Sonstiges",
    "CL-O-*-*-UCO-*":       "Nicht klassifizierbar",
}

MODEL_DISPLAY = {
    "Qwen/Qwen3-4B": "Qwen3-4B",
    "google/gemma-4-E4B-it": "Gemma-4-E4B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
}


# ---------------------------------------------------------------------------
# Loading & enrichment
# ---------------------------------------------------------------------------

def infer_pole(source_file: str) -> str:
    return "negative" if "negalpha" in source_file else "positive"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            r["pole"] = infer_pole(r.get("_source_file", ""))
            rows.append(r)
    return rows


def apply_filters(
    rows: list[dict],
    axis: str | None,
    pole: str | None,
    variant: str | None,
    model: str | None,
) -> list[dict]:
    out = rows
    if axis:
        out = [r for r in out if r.get("axis") == axis]
    if pole:
        out = [r for r in out if r["pole"] == pole]
    if variant:
        out = [r for r in out if r.get("variant") == variant or r.get("variant") == "baseline"]
    if model:
        out = [r for r in out if model in r.get("model_path", "")]
    return out


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def label_dist(rows: list[dict], cl_only: bool = True) -> dict[str, float]:
    counts: Counter = Counter()
    for r in rows:
        lbl = r["predicted_label"]
        if cl_only and not lbl.startswith("CL-"):
            continue
        counts[lbl] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def dist_delta(base_dist: dict, steer_dist: dict) -> dict[str, float]:
    all_labels = set(base_dist) | set(steer_dist)
    return {lbl: steer_dist.get(lbl, 0.0) - base_dist.get(lbl, 0.0)
            for lbl in all_labels}


def co_rate(rows: list[dict]) -> float:
    n = len(rows)
    if n == 0:
        return float("nan")
    return sum(1 for r in rows if r["predicted_label"].startswith("CO-")) / n


# ---------------------------------------------------------------------------
# Per-axis shift
# ---------------------------------------------------------------------------

def compute_shifts(
    rows: list[dict],
    group_by: str = "axis",
) -> dict[str, dict[str, float]]:
    """Return {group_value: {label: delta_freq}} comparing steered vs baseline."""
    base_rows = [r for r in rows if r["variant"] == "baseline"]
    steer_rows = [r for r in rows if r["variant"] != "baseline"]

    groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for r in base_rows:
        key = r.get(group_by, "?")
        groups[key][0].append(r)
    for r in steer_rows:
        key = r.get(group_by, "?")
        groups[key][1].append(r)

    shifts = {}
    for key, (base, steer) in groups.items():
        bd = label_dist(base)
        sd = label_dist(steer)
        shifts[key] = dist_delta(bd, sd)
    return shifts


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def short_name(code: str) -> str:
    return CL_NAMES.get(code, code.split("-")[-2] if "*" not in code.split("-")[-1] else code.split("*")[0].rstrip("-").split("-")[-1])


def print_shift_table(
    base_rows: list[dict],
    steer_rows: list[dict],
    title: str,
    top_n: int = 20,
) -> None:
    bd = label_dist(base_rows)
    sd = label_dist(steer_rows)
    delta = dist_delta(bd, sd)

    # sort by absolute delta, show top_n
    ranked = sorted(delta.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"  baseline n={len(base_rows)}  steered n={len(steer_rows)}")
    print(f"  CO-* rate  baseline={co_rate(base_rows):.3f}  steered={co_rate(steer_rows):.3f}")
    print(f"{'='*70}")
    hdr = f"  {'Label':<40}  {'base%':>6}  {'steer%':>6}  {'Δ':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lbl, delta_val in ranked:
        if not lbl.startswith("CL-"):
            continue
        name = CL_NAMES.get(lbl, lbl)
        direction = "▲" if delta_val > 0.005 else ("▼" if delta_val < -0.005 else " ")
        print(
            f"  {direction} {name:<38}  "
            f"{bd.get(lbl,0)*100:>5.1f}%  "
            f"{sd.get(lbl,0)*100:>5.1f}%  "
            f"{delta_val:>+8.4f}"
        )


# ---------------------------------------------------------------------------
# Heatmap (label × axis or label × group_by)
# ---------------------------------------------------------------------------

def make_heatmap(
    rows: list[dict],
    group_by: str,
    pole: str | None,
    axis: str | None,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available")
        return

    base_rows = [r for r in rows if r["variant"] == "baseline"]
    steer_rows = [r for r in rows if r["variant"] != "baseline"]

    groups = sorted({r.get(group_by, "?") for r in steer_rows})
    all_cl_labels = sorted(
        {r["predicted_label"] for r in rows if r["predicted_label"].startswith("CL-")}
    )

    # build delta matrix
    data = np.zeros((len(all_cl_labels), len(groups)))
    for j, grp in enumerate(groups):
        b = [r for r in base_rows if r.get(group_by, "?") == grp]
        s = [r for r in steer_rows if r.get(group_by, "?") == grp]
        if not b:
            b = base_rows  # fall back to overall baseline if group has no separate baseline
        bd = label_dist(b)
        sd = label_dist(s)
        delta = dist_delta(bd, sd)
        for i, lbl in enumerate(all_cl_labels):
            data[i, j] = delta.get(lbl, 0.0)

    # sort rows by max absolute delta across groups
    row_order = np.argsort(-np.max(np.abs(data), axis=1))
    data = data[row_order]
    row_labels = [f"{CL_NAMES.get(all_cl_labels[i], all_cl_labels[i])}" for i in row_order]
    col_labels = groups

    vmax = max(0.05, float(np.max(np.abs(data))))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(groups)), max(6, 0.4 * len(row_labels))))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = data[i, j]
            color = "white" if abs(v) > 0.5 * vmax else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax, label="Δ label frequency (steered − baseline)", shrink=0.8)
    suffix = f"{group_by}" + (f"_{pole}" if pole else "") + (f"_{axis}" if axis else "")
    ax.set_title(
        f"OnCoCo label distribution shift\nsteered vs. baseline · grouped by {group_by}"
        + (f"  [{pole}]" if pole else "")
        + (f"  [{axis}]" if axis else ""),
        fontsize=10,
    )
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"oncoco_shift_{suffix}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {out_path}")


# ---------------------------------------------------------------------------
# Full distribution plot (baseline vs steered side-by-side)
# ---------------------------------------------------------------------------

def plot_full_distributions(
    rows: list[dict],
    pole: str | None,
    axis: str | None,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available")
        return

    axes_to_plot = [axis] if axis else ["openness", "initiative", "cooperation", "hopefulness"]
    single = len(axes_to_plot) == 1

    if single:
        fig, axs_grid = plt.subplots(1, 1, figsize=(11, 10))
        axs_list = [axs_grid]
    else:
        fig, axs_grid = plt.subplots(2, 2, figsize=(18, 22))
        axs_list = axs_grid.flatten().tolist()

    for i, ax_name in enumerate(axes_to_plot):
        base = [r for r in rows if r["variant"] == "baseline" and r.get("axis") == ax_name]
        steer = [r for r in rows if r["variant"] != "baseline" and r.get("axis") == ax_name]

        bd = label_dist(base)
        sd = label_dist(steer)

        all_labels = sorted(
            [lbl for lbl in set(bd) | set(sd)
             if bd.get(lbl, 0) >= 0.005 or sd.get(lbl, 0) >= 0.005],
            key=lambda l: bd.get(l, 0), reverse=True,
        )
        label_names = [CL_NAMES.get(l, l) for l in all_labels]
        base_vals = [bd.get(l, 0) * 100 for l in all_labels]
        steer_vals = [sd.get(l, 0) * 100 for l in all_labels]

        y = np.arange(len(all_labels))
        height = 0.35
        ax = axs_list[i]
        ax.barh(y + height / 2, base_vals, height, label="Baseline", color="#4878CF", alpha=0.85)
        ax.barh(y - height / 2, steer_vals, height, label="Steered", color="#D65F5F", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(label_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("% of sentences", fontsize=9)
        ax.set_title(
            f"{ax_name}  (n_base={len(base)}, n_steer={len(steer)})", fontsize=10
        )
        ax.legend(fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    # hide unused subplots in 2×2 grid
    if not single:
        for j in range(len(axes_to_plot), len(axs_list)):
            axs_list[j].set_visible(False)

    pole_str = pole if pole else "both"
    axis_str = axis if axis else "all"
    fig.suptitle(
        f"OnCoCo CL-* label distributions: baseline vs. steered\nPole: {pole_str}",
        fontsize=12,
    )
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"oncoco_full_dist_{pole_str}_{axis_str}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Full distribution plot saved to {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(LABELS_FILE))
    p.add_argument("--axis", default=None,
                   help="Filter to one axis (openness|initiative|cooperation|hopefulness)")
    p.add_argument("--pole", default=None, choices=["negative", "positive"])
    p.add_argument("--variant", default=None,
                   help="Filter steered rows to one variant (still keeps all baseline)")
    p.add_argument("--model", default=None, help="Substring filter on model_path")
    p.add_argument("--group_by", default="axis",
                   help="Dimension to split heatmap columns: axis|pole|model_path|variant (default: axis)")
    p.add_argument("--top_n", type=int, default=20, help="Labels shown in console table")
    p.add_argument("--heatmap", action="store_true")
    p.add_argument("--full_dist", action="store_true",
                   help="Plot complete CL-* label distributions for baseline vs steered side by side")
    p.add_argument("--output_csv", default=None)
    p.add_argument("--plot_dir", default=str(ROOT / "outputs/figures"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    print(f"Loaded {len(rows)} rows.")

    rows = apply_filters(rows, args.axis, args.pole, args.variant, args.model)
    print(f"After filters: {len(rows)} rows.")

    base_rows = [r for r in rows if r["variant"] == "baseline"]
    steer_rows = [r for r in rows if r["variant"] != "baseline"]

    if not base_rows or not steer_rows:
        print("Not enough data after filtering.")
        return

    # overall shift table
    print_shift_table(base_rows, steer_rows, "Overall shift: steered vs. baseline", args.top_n)

    # per-axis breakdown if not already filtered to one axis
    if not args.axis:
        for ax in ["openness", "initiative", "cooperation", "hopefulness"]:
            b = [r for r in base_rows if r.get("axis") == ax]
            s = [r for r in steer_rows if r.get("axis") == ax]
            if b and s:
                print_shift_table(b, s, f"Axis: {ax}", top_n=12)

    if args.heatmap:
        # normalise group_by: model_path → display name
        if args.group_by == "model_path":
            for r in rows:
                r["model_short"] = MODEL_DISPLAY.get(r.get("model_path", ""), r.get("model_path", ""))
            gby = "model_short"
        else:
            gby = args.group_by
        make_heatmap(rows, gby, args.pole, args.axis, Path(args.plot_dir))

    if args.full_dist:
        plot_full_distributions(rows, args.pole, args.axis, Path(args.plot_dir))

    if args.output_csv:
        bd = label_dist(base_rows)
        sd = label_dist(steer_rows)
        delta = dist_delta(bd, sd)
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["label", "label_name", "base_freq", "steer_freq", "delta"])
            for lbl, dv in sorted(delta.items(), key=lambda x: x[1], reverse=True):
                w.writerow([lbl, CL_NAMES.get(lbl, ""), f"{bd.get(lbl,0):.5f}",
                             f"{sd.get(lbl,0):.5f}", f"{dv:.5f}"])
        print(f"CSV written to {path}")


if __name__ == "__main__":
    main()
