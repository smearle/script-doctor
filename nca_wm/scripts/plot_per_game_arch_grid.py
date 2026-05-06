"""Heatmap figure for the per-game × per-architecture grid.

Reads nca_wm/figures/per_game_arch/summary.csv (produced by
summarize_per_game_arch_grid.py) and renders a 2-panel heatmap:
  - left: BFS cell-error on training level (L0)
  - right: BFS cell-error on held-out levels (mean of L1+; blank if none)
Cells annotate the percentage; rows are games, cols are arch buckets.

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_per_game_arch_grid.py
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))

_BUCKET_LABELS = {
    "A": "A:\npool ON\nshared",
    "B": "B:\npool ON\nper-step",
    "C": "C:\nno-pool+skip\nshared",
    "D": "D:\nno-pool+skip\nper-step",
}


def _load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("bfs_err_l0", "bfs_err_heldout", "best_loss",
                      "random_tf_err_mean", "random_err_mean"):
                v = r.get(k, "")
                try:
                    r[k] = float(v) if v not in ("", "None") else float("nan")
                except ValueError:
                    r[k] = float("nan")
            rows.append(r)
    return rows


def _matrix(rows, games, buckets, key):
    M = np.full((len(games), len(buckets)), np.nan)
    by = {(r["game"], r["bucket"]): r for r in rows}
    for i, g in enumerate(games):
        for j, b in enumerate(buckets):
            r = by.get((g, b))
            if r is not None:
                M[i, j] = r.get(key, np.nan)
    return M


def _annotate(ax, M, fmt="{:.1f}%", scale=100):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="black")
            else:
                ax.text(j, i, fmt.format(v * scale), ha="center", va="center",
                        fontsize=9, color="black")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nca_wm/figures/per_game_arch/summary.csv")
    ap.add_argument("--outdir", default="nca_wm/figures/per_game_arch")
    ap.add_argument(
        "--exclude",
        default="sokoban_basic",
        help="Comma-separated game names to drop from the figure. "
        "sokoban_basic is excluded by default because Microban covers the same "
        "dynamics with more authored levels for held-out signal.",
    )
    args = ap.parse_args()

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_REPO, args.csv)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(_REPO, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    rows = _load(csv_path)
    if not rows:
        print(f"No rows in {csv_path}")
        return

    excludes = {x.strip() for x in args.exclude.split(",") if x.strip()}
    rows = [r for r in rows if r["game"] not in excludes]

    games = sorted({r["game"] for r in rows})
    buckets = ["A", "B", "C", "D"]

    M_l0 = _matrix(rows, games, buckets, "bfs_err_l0")
    M_held = _matrix(rows, games, buckets, "bfs_err_heldout")

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.constrained_layout.use": True,
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, max(3.0, 0.55 * len(games) + 1.5)))

    # Shared color scale for fairness; clip top at 30% so small-difference cells
    # remain readable. NaNs render as the "bad" color (light grey).
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#dddddd")
    vmax = float(np.nanmax(np.concatenate([M_l0[~np.isnan(M_l0)], M_held[~np.isnan(M_held)]])) ) if (np.isfinite(M_l0).any() or np.isfinite(M_held).any()) else 0.3
    vmax = max(min(vmax, 0.30), 0.05)

    im0 = axL.imshow(M_l0, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    _annotate(axL, M_l0)
    axL.set_xticks(range(len(buckets)))
    axL.set_xticklabels([_BUCKET_LABELS[b] for b in buckets])
    axL.set_yticks(range(len(games)))
    axL.set_yticklabels(games)
    axL.set_title("BFS cell-error — training level (L0)")

    im1 = axR.imshow(M_held, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    _annotate(axR, M_held)
    axR.set_xticks(range(len(buckets)))
    axR.set_xticklabels([_BUCKET_LABELS[b] for b in buckets])
    axR.set_yticks(range(len(games)))
    axR.set_yticklabels(games)
    axR.set_title("BFS cell-error — held-out levels (L1+ mean)")

    cbar = fig.colorbar(im1, ax=[axL, axR], shrink=0.85, pad=0.02)
    cbar.set_label("cell-error (capped at {:.0f}%)".format(vmax * 100))

    fig.suptitle("Per-game × per-architecture grid (n_nca_steps=8, rule_attn h=256)")
    fig.savefig(os.path.join(outdir, "heatmap_bfs.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "heatmap_bfs.png"), bbox_inches="tight", dpi=150)
    print(f"  wrote {outdir}/heatmap_bfs.pdf|.png")
    plt.close(fig)

    # Bonus: best-loss panel (log-scaled). One axes only.
    M_loss = _matrix(rows, games, buckets, "best_loss")
    if np.isfinite(M_loss).any():
        fig2, ax2 = plt.subplots(figsize=(6.5, max(3.0, 0.55 * len(games) + 1.5)))
        with np.errstate(invalid="ignore"):
            logM = np.log10(M_loss)
        im2 = ax2.imshow(logM, aspect="auto", cmap=cmap)
        for i in range(M_loss.shape[0]):
            for j in range(M_loss.shape[1]):
                v = M_loss[i, j]
                txt = "—" if np.isnan(v) else f"{v:.1e}"
                ax2.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
        ax2.set_xticks(range(len(buckets)))
        ax2.set_xticklabels([_BUCKET_LABELS[b] for b in buckets])
        ax2.set_yticks(range(len(games)))
        ax2.set_yticklabels(games)
        ax2.set_title("Best train loss (BCE; log10 colormap)")
        fig2.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02, label="log10(loss)")
        fig2.savefig(os.path.join(outdir, "heatmap_loss.pdf"), bbox_inches="tight")
        fig2.savefig(os.path.join(outdir, "heatmap_loss.png"), bbox_inches="tight", dpi=150)
        print(f"  wrote {outdir}/heatmap_loss.pdf|.png")
        plt.close(fig2)


if __name__ == "__main__":
    main()
