#!/usr/bin/env python3
"""Build LaTeX figures for the COPS v2 paper from canonical ensemble JSON files.

Currently:
- ``asymmetry_scatter.pdf`` — Difficult-pole vs easy-pole net-pairwise per
  (model, axis) cell, ``case_aware`` variant (JSON key ``global_aware``)
  at ``|alpha|=3``. Color by
  axis, marker by model, y=x reference. Companion to Figure 1 (qualitative)
  and the Three-findings roadmap callout in section 5.

Data sources are reused from ``build_paper_tables.py``: ``load_paper_v2_stems``
(positive pole) and ``load_negalpha_stems`` (difficult pole). No hardcoded
values; consistent with the existing canonical-pipeline convention.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Reuse the data loaders and stem ordering from the table builder so figure
# data cannot drift from table data.
from build_paper_tables import (  # type: ignore
    NEG_STEM_ORDER,
    MODEL_SHORT,
    load_negalpha_stems,
    load_paper_v2_stems,
    _mean_pairwise_net,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"

AXIS_COLOR = {
    "cooperation": "#1f77b4",
    "hopefulness": "#d62728",
    "initiative":  "#2ca02c",
    "openness":    "#ff7f0e",
}
AXIS_ORDER = ["cooperation", "hopefulness", "initiative", "openness"]

MODEL_MARKER = {
    "Qwen-4B":  "o",
    "Gemma":    "s",
    "Qwen-9B":  "^",
    "Mistral":  "D",
}
MODEL_ORDER = ["Qwen-4B", "Gemma", "Qwen-9B", "Mistral"]


def collect_paired_cells():
    """Return list of (model_short, axis, diff_net, easy_net) tuples for cells
    where both the difficult-pole (alpha=-3) and easy-pole (alpha=+3)
    case_aware (JSON key: ``global_aware``) net-pairwise are available."""
    neg = load_negalpha_stems()
    pos = load_paper_v2_stems()
    cells = []
    for mk, ax, stem in NEG_STEM_ORDER:
        if stem not in neg or stem not in pos:
            continue
        diff_net = _mean_pairwise_net(neg[stem], "global_aware", -3.0)
        easy_net = _mean_pairwise_net(pos[stem], "global_aware", 3.0)
        if diff_net is None or easy_net is None:
            continue
        cells.append((MODEL_SHORT[mk], ax, diff_net, easy_net))
    return cells


def build_asymmetry_scatter() -> None:
    cells = collect_paired_cells()
    n_paired = len(cells)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))

    lim_lo, lim_hi = -0.12, 0.80

    # Faint y=x reference and quadrant shading
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            linestyle="--", color="#999", linewidth=0.9, zorder=1)
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], lim_hi,
                    color="#1f77b4", alpha=0.035, zorder=0)
    ax.fill_between([lim_lo, lim_hi], lim_lo, [lim_lo, lim_hi],
                    color="#d62728", alpha=0.035, zorder=0)
    ax.axhline(0, color="#ccc", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#ccc", linewidth=0.5, zorder=0)

    # Diagonal-aligned region labels (rotated to follow the y=x line)
    ax.text(0.18, 0.30, "difficult pole stronger",
            fontsize=8, color="#1f77b4", ha="center", va="bottom",
            rotation=45, rotation_mode="anchor", alpha=0.95)
    ax.text(0.46, 0.34, "easier pole stronger",
            fontsize=8, color="#d62728", ha="center", va="top",
            rotation=45, rotation_mode="anchor", alpha=0.95)

    # Data points
    for model, axis_lbl, diff_net, easy_net in cells:
        ax.scatter(easy_net, diff_net,
                   c=AXIS_COLOR[axis_lbl],
                   marker=MODEL_MARKER[model],
                   s=68, edgecolor="black", linewidth=0.6,
                   zorder=3)

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"Easy-pole net-pairwise ($\alpha{=}{+}3$, CA)",
                  fontsize=10)
    ax.set_ylabel(r"Difficult-pole net-pairwise ($\alpha{=}{-}3$, CA)",
                  fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.3)

    # Legends placed outside the chart on the right to avoid data overlap.
    axis_handles = [
        mpatches.Patch(facecolor=AXIS_COLOR[a], edgecolor="black",
                       linewidth=0.5, label=a)
        for a in AXIS_ORDER
    ]
    model_handles = [
        Line2D([0], [0], marker=MODEL_MARKER[m], color="w",
               markerfacecolor="#888", markeredgecolor="black",
               markeredgewidth=0.5, markersize=8, label=m)
        for m in MODEL_ORDER
    ]
    leg1 = ax.legend(handles=axis_handles, title="Axis",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, title_fontsize=8, frameon=False,
                     handlelength=1.2, borderaxespad=0)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model",
              loc="upper left", bbox_to_anchor=(1.02, 0.55),
              fontsize=8, title_fontsize=8, frameon=False,
              handlelength=1.0, borderaxespad=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "asymmetry_scatter.pdf"
    png_path = OUT_DIR / "asymmetry_scatter.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"wrote {pdf_path}  ({n_paired} paired cells)")
    print(f"wrote {png_path}")


def main() -> None:
    build_asymmetry_scatter()


if __name__ == "__main__":
    main()
