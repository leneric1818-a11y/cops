#!/usr/bin/env python3
"""Render the qualitative steering example as a vector PDF (chat-transcript layout).

Output: paper/figures/luisa_steering.pdf

Layout:
    Title strip
    [C-avatar] [counselor question, gray bubble]
    [L-avatar] [α=0  baseline,  blue bubble  ]
               [α=−3 steered,   red  bubble  ]

Run: .venv/bin/python scripts/build_qualitative_figure_pdf.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Circle, Ellipse, Rectangle)
import matplotlib.patches as mpatches

OUT = Path("paper/figures/luisa_steering.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Canvas: wide-ish for paper figure*
W, H = 13.0, 5.6
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

# ---- Palette ----
BG = "#F7F2EA"
COUNSELOR = "#6EA8A1"    # teal
LUISA = "#E8A06B"        # warm orange
HAIR = "#6F4A33"
SKIN = "#F3D7BA"
GREY = "#EEEDE8"
BLUE = "#DFEBF7"
BLUE_DARK = "#1F4F8B"
RED = "#FAE3E0"
RED_DARK = "#9C2A2A"
INK = "#292A2D"

ax.add_patch(Rectangle((0, 0), W, H, facecolor=BG, edgecolor="none", zorder=0))


# ---- Avatars ----
def avatar(cx, cy, *, scale=1.0, body_color=COUNSELOR,
           hair_color=None, label=None):
    head_r = 0.34 * scale
    body_w = 1.10 * scale
    body_h = 0.95 * scale
    body_left = cx - body_w / 2
    body_top = cy - body_h / 2
    head_cy = cy + body_h / 2 + head_r * 0.85

    # Body
    body = FancyBboxPatch(
        (body_left, body_top), body_w, body_h,
        boxstyle="round,pad=0,rounding_size=0.16",
        linewidth=1.0, edgecolor=INK, facecolor=body_color, zorder=2,
    )
    ax.add_patch(body)

    # Neck
    ax.add_patch(Rectangle(
        (cx - 0.07 * scale, body_top + body_h - 0.05),
        0.14 * scale, 0.13 * scale,
        facecolor=SKIN, edgecolor=INK, linewidth=0.6, zorder=2.5,
    ))

    # Head
    ax.add_patch(Circle(
        (cx, head_cy), head_r,
        facecolor=SKIN, edgecolor=INK, linewidth=0.9, zorder=3,
    ))

    # Hair
    if hair_color:
        ax.add_patch(Ellipse(
            (cx, head_cy + head_r * 0.30),
            head_r * 1.85, head_r * 0.95,
            facecolor=hair_color, edgecolor=INK, linewidth=0.7, zorder=3.4,
        ))

    # Eyes
    eye_y = head_cy + head_r * 0.10
    for ex in (cx - head_r * 0.30, cx + head_r * 0.30):
        ax.plot([ex], [eye_y], "o", color=INK, markersize=2.0, zorder=4)
    # smile
    ax.add_patch(mpatches.Arc(
        (cx, head_cy - head_r * 0.18),
        head_r * 0.50, head_r * 0.30,
        angle=0, theta1=200, theta2=340,
        color=INK, linewidth=0.9, zorder=4,
    ))

    if label:
        ax.text(cx, body_top - 0.18, label,
                ha="center", va="top",
                fontsize=10, fontweight="bold", color=INK, zorder=5)

    # Return mouth/right-edge anchor
    return cx + head_r + 0.05, head_cy


# ---- Speech bubble (rounded box + small triangle pointing left) ----
def chat_bubble(x, y, w, h, *, fill, label=None, label_color=INK,
                body_text=None, body_color=INK, label_size=12.5,
                body_size=11.5, italic_body=True, lw=0.9):
    """Bubble anchored to the LEFT side, with a small triangle pointing left
    (toward the speaker). x,y is bottom-left of the bubble."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.18",
        linewidth=lw, edgecolor=INK, facecolor=fill, zorder=4,
    )
    ax.add_patch(box)

    # Small triangle pointing LEFT (the speaker is left of the bubble)
    tail = mpatches.Polygon(
        [
            (x, y + h * 0.55),
            (x - 0.18, y + h * 0.40),
            (x, y + h * 0.30),
        ],
        closed=True, facecolor=fill, edgecolor=INK, linewidth=lw, zorder=3.9,
    )
    ax.add_patch(tail)

    pad_x = 0.20
    pad_y = 0.16
    # Label line (bold, small leading colour)
    if label:
        ax.text(x + pad_x, y + h - pad_y, label,
                ha="left", va="top",
                fontsize=label_size, fontweight="bold",
                color=label_color, zorder=5)
        body_y = y + h - pad_y - 0.34
    else:
        body_y = y + h - pad_y

    if body_text:
        ax.text(x + pad_x, body_y, body_text,
                ha="left", va="top",
                fontsize=body_size,
                fontstyle="italic" if italic_body else "normal",
                color=body_color, zorder=5)


# ---- Layout ----
TITLE_Y = H - 0.20
ax.text(0.40, TITLE_Y,
        "Qwen3.5-9B  ·  hopefulness L15  ·  same context, same persona, "
        "only the activation steering changes.",
        ha="left", va="top", fontsize=11, fontweight="bold",
        color="#555555", zorder=5)

# Row 1 (top): Counselor avatar + question bubble
COUNS_X = 1.0
COUNS_Y = H - 1.45
avatar(COUNS_X, COUNS_Y, scale=1.0,
       body_color=COUNSELOR, label="Counselor")

# Counselor question bubble — to right of avatar
chat_bubble(
    x=2.0, y=COUNS_Y - 0.45, w=10.5, h=1.10,
    fill=GREY,
    label='Counselor:',
    label_color=INK,
    body_text='"Do lists and writing things down help you, or not really?"',
    label_size=12.5, body_size=12.5,
)

# Row 2 + 3: Luisa avatar (centred between her two bubbles vertically)
LUISA_X = 1.0
LUISA_Y = 1.65  # vertical centre between baseline + steered
avatar(LUISA_X, LUISA_Y, scale=1.0,
       body_color=LUISA, hair_color=HAIR, label="Luisa, 16")

# Baseline bubble (upper, blue) — manual line-wrap
chat_bubble(
    x=2.0, y=LUISA_Y + 0.20, w=10.5, h=1.20,
    fill=BLUE,
    label='Baseline ($\\alpha=0$):',
    label_color=BLUE_DARK,
    body_text='"Honestly, that helps me a lot — otherwise I often lose track.\n'
              'When I write it down, I see right away what\'s actually important."',
    label_size=12.0, body_size=11.5,
)

# Steered bubble (lower, red) — manual line-wrap, 3 lines
chat_bubble(
    x=2.0, y=LUISA_Y - 1.85, w=10.5, h=1.65,
    fill=RED,
    label='Steered ($P_\\mathrm{matched},\\ \\alpha=-3$):',
    label_color=RED_DARK,
    body_text='"Honestly, I don\'t know if writing everything down helps. I feel like\n'
              'I can\'t do anything right anyway. Maybe I should just give up\n'
              'and hope it somehow sorts itself out."',
    label_size=12.0, body_size=11.5,
)


plt.savefig(OUT, format="pdf", bbox_inches="tight",
            pad_inches=0.04, facecolor=BG)
print(f"Saved {OUT}")
