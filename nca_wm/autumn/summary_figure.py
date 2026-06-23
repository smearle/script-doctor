#!/usr/bin/env python3
"""Capstone figure: the hidden-state blindspot and its fix across AutumnBench envs.

Each environment has hidden state that determines dynamics/an action's effect but
isn't visible in the grid. A single-frame NCA defaults to the majority outcome;
the right memory architecture (1-frame history, or a recurrent hidden grid)
recovers it. Bars show the key per-environment metric, single-frame vs fixed.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (env, hidden state, metric name, single-frame, fixed, fix-type)
ROWS = [
    ("gameOfLife", "none (Markovian)", "rule exact", 0.99, 0.99, "n/a"),
    ("mario\n(enemy)", "patrol direction", "enemy recall", 0.59, 1.00, "history"),
    ("wind", "wind dir (±1)", "changed-cell acc", 0.67, 0.85, "history"),
    ("snake", "move direction", "mean-exact", 0.73, 0.81, "history"),
    ("mario\n(bullets)", "bullet counter", "fire recall", 0.09, 0.65, "recurrent"),
    ("sand", "clickType brush", "water-brush (AR)", 0.00, 0.50, "recurrent"),
    ("paint", "currColor (5-cycle)", "paint color", 0.00, 1.00, "recurrent"),
]
colors = {"history": "#3b82f6", "recurrent": "#ef4444", "n/a": "#9ca3af"}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(ROWS))
w = 0.38
sf = [r[3] for r in ROWS]
fx = [r[4] for r in ROWS]
ax.bar(x - w / 2, sf, w, label="single-frame", color="#d1d5db", edgecolor="k")
ax.bar(x + w / 2, fx, w, label="with memory fix",
       color=[colors[r[5]] for r in ROWS], edgecolor="k")
for i, r in enumerate(ROWS):
    ax.text(i - w / 2, r[3] + 0.02, f"{r[3]:.2f}", ha="center", fontsize=11)
    ax.text(i + w / 2, r[4] + 0.02, f"{r[4]:.2f}", ha="center", fontsize=11)
    ax.text(i, -0.13, r[2], ha="center", fontsize=10, style="italic", color="#444")
    if r[5] != "n/a":
        ax.text(i + w / 2, r[4] / 2, r[5], ha="center", va="center", rotation=90,
                fontsize=10, color="white", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in ROWS], fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("key metric (higher = better)", fontsize=13)
ax.set_title("AutumnBench hidden-state blindspots and the memory fix\n"
             "(single-frame NCA vs. history / recurrent-hidden-grid)", fontsize=14)
ax.legend(loc="upper left", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "nca_wm/autumn/figures/blindspot_taxonomy"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=140, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print(f"saved -> {out}.png / .pdf")
