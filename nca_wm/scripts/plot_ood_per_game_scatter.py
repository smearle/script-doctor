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
        games.append(r["game"])
        cond_e.append(c); unc_e.append(u); ident.append(i)
    cond_e = np.array(cond_e); unc_e = np.array(unc_e); ident = np.array(ident)

    cond_wins = cond_e <  ident
    unc_wins  = unc_e  <  ident
    both = cond_wins &  unc_wins
    only_c = cond_wins & ~unc_wins
    only_u = ~cond_wins &  unc_wins
    neither = ~cond_wins & ~unc_wins

    plt.rcParams.update(RC_PARAMS)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))

    spec = [
        (both,    "#2ca02c", "both beat identity"),
        (only_c,  "#d62728", "cond only beats identity"),
        (only_u,  "#1f77b4", "uncond only beats identity"),
        (neither, "#7f7f7f", "neither beats identity"),
    ]
    for mask, color, label in spec:
        if mask.any():
            ax.scatter(100*cond_e[mask], 100*unc_e[mask],
                       s=42, c=color, alpha=0.85,
                       edgecolors="white", linewidths=0.6,
                       label=f"{label} ({mask.sum()})", zorder=3)

    lo = max(1e-2, 100*float(min(cond_e.min(), unc_e.min(), 1e-4)))
    hi = 100*float(max(cond_e.max(), unc_e.max(), 0.4)) * 1.2
    diag = np.array([lo, hi])
    ax.plot(diag, diag, color="black", linewidth=1.0, linestyle="--",
            alpha=0.5, label="$y = x$", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Rule-conditional 1-step (TF) cell-error (%) [{SCALE}]")
    ax.set_ylabel(f"Unconditional 1-step (TF) cell-error (%) [{SCALE}]")
    ax.set_title(f"Per-game OOD error on Heldout-26 ({SCALE})")
    ax.grid(True, which="both", alpha=0.25)
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
