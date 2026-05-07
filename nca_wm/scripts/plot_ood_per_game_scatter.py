#!/usr/bin/env python3
"""Per-game OOD 1-step (TF) cell-error scatter: cond vs uncond on Heldout-26.

Reads per-game cell-error from
  nca_wm/paper/figures/cond_vs_uncond_match/per_game_step1.csv

Emits
  nca_wm/paper/figures/cond_vs_uncond_match/ood_per_game_scatter.{pdf,png}

Layout: scatter of (cond_err, uncond_err) across 26 heldout games at the
broadest training scale (Train-199), with the y=x diagonal and identity
threshold lines for each game. Points are coloured by which model
beats identity on that game (cond only / uncond only / both / neither).

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_ood_per_game_scatter.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH  = REPO_ROOT / "nca_wm" / "paper" / "figures" / "cond_vs_uncond_match" / "per_game_step1.csv"
OUT_DIR   = REPO_ROOT / "nca_wm" / "paper" / "figures" / "cond_vs_uncond_match"

SCALE = "Train-199"

RC_PARAMS = {
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
}


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    cond_col   = f"Rule-conditional, {SCALE}"
    uncond_col = f"Unconditional, {SCALE}"

    games  = []
    cond_e = []
    unc_e  = []
    ident  = []
    for r in rows:
        try:
            c = float(r[cond_col]); u = float(r[uncond_col]); i = float(r["identity"])
        except (KeyError, ValueError):
            continue
        if i <= 0:
            continue
        games.append(r["game"])
        cond_e.append(c); unc_e.append(u); ident.append(i)
    cond_e = np.array(cond_e); unc_e = np.array(unc_e); ident = np.array(ident)

    # Ratios to per-game identity baseline. <1 beats identity; >1 worse.
    cond_r = cond_e / ident
    unc_r  = unc_e  / ident

    cond_wins = cond_r < 1
    unc_wins  = unc_r  < 1
    both    = cond_wins &  unc_wins
    only_c  = cond_wins & ~unc_wins
    only_u  = ~cond_wins &  unc_wins
    neither = ~cond_wins & ~unc_wins

    plt.rcParams.update(RC_PARAMS)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    spec = [
        (both,    "#2ca02c", "both beat identity"),
        (only_c,  "#d62728", "cond only"),
        (only_u,  "#1f77b4", "uncond only"),
        (neither, "#7f7f7f", "neither"),
    ]
    for mask, color, label in spec:
        if mask.any():
            ax.scatter(cond_r[mask], unc_r[mask],
                       s=46, c=color, alpha=0.9,
                       edgecolors="white", linewidths=0.6,
                       label=f"{label} ({mask.sum()})", zorder=4)

    # Tight log-log range around the points; pad ~25% on each end.
    lo = float(min(cond_r.min(), unc_r.min())) / 1.25
    hi = float(max(cond_r.max(), unc_r.max())) * 1.25
    lo = max(lo, 1e-3); hi = min(hi, 1e2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # x=1 and y=1: per-game identity threshold for each model.
    ax.axvline(1.0, color="black", linewidth=1.0, alpha=0.6, zorder=2)
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.6, zorder=2)
    # y=x: cond and uncond equally far from identity on this game.
    diag = np.array([lo, hi])
    ax.plot(diag, diag, color="black", linewidth=0.9, linestyle="--",
            alpha=0.5, zorder=2, label="$y = x$")

    ax.set_xlabel("Cond error / identity error")
    ax.set_ylabel("Uncond error / identity error")
    ax.set_title(f"Per-game OOD error vs. identity ({SCALE})")
    ax.grid(False)
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    pdf = OUT_DIR / "ood_per_game_scatter.pdf"
    png = OUT_DIR / "ood_per_game_scatter.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    plt.close(fig)
    for p in (pdf, png):
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
