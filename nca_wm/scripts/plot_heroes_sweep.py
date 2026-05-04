"""Paper-style figure for the Heroes_of_Sokoban depth × sharing × pool sweep.

Reads nca_wm/figures/heroes_sweep/summary.csv and writes
  - heroes_bfs_by_depth.{pdf,png}: BFS rollout error vs depth, line per bucket.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))


def _load(p):
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            for k in ("best_loss","final_loss","final_change_acc",
                      "bfs_err_mean","astar_err_mean","random_err_mean",
                      "random_tf_err_mean"):
                v = r.get(k, "")
                try:
                    r[k] = float(v) if v else float("nan")
                except ValueError:
                    r[k] = float("nan")
            for k in ("n_steps", "n_repeats", "n_layers"):
                r[k] = int(r[k]) if r.get(k) else 0
            rows.append(r)
    return rows


def _bucket(name):
    if name.startswith("heroes_A_"): return "A: pool ON, shared"
    if name.startswith("heroes_B_"): return "B: pool ON, per-step"
    if name.startswith("heroes_C_"): return "C: pool OFF + skip, shared"
    if name.startswith("heroes_D_"): return "D: pool OFF + skip, per-step"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nca_wm/figures/heroes_sweep/summary.csv")
    ap.add_argument("--outdir", default="nca_wm/figures/heroes_sweep")
    args = ap.parse_args()

    p = args.csv if os.path.isabs(args.csv) else os.path.join(_REPO, args.csv)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(_REPO, args.outdir)
    rows = _load(p)
    print(f"Loaded {len(rows)} rows.")

    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
        "figure.constrained_layout.use": True,
    })

    bys = {}
    for r in rows:
        b = _bucket(r["run"])
        if b is None: continue
        bys.setdefault(b, []).append(r)

    # Plot 1: bfs_err vs depth (semi-log y for d=4 outliers)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"A: pool ON, shared":"C0", "B: pool ON, per-step":"C1",
              "C: pool OFF + skip, shared":"C2", "D: pool OFF + skip, per-step":"C3"}
    markers = {"A: pool ON, shared":"o", "B: pool ON, per-step":"s",
               "C: pool OFF + skip, shared":"^", "D: pool OFF + skip, per-step":"D"}
    for b in ["A: pool ON, shared", "B: pool ON, per-step",
              "C: pool OFF + skip, shared", "D: pool OFF + skip, per-step"]:
        rs = sorted(bys.get(b, []), key=lambda r: r["n_steps"])
        if not rs: continue
        x = [r["n_steps"] for r in rs]
        y = [r["bfs_err_mean"] * 100 for r in rs]
        ax.plot(x, y, marker=markers[b], color=colors[b], label=b,
                linewidth=2, markersize=10)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels([4, 8, 16, 32])
    ax.set_xlabel("NCA depth (n_steps)")
    ax.set_ylabel("BFS rollout cell-error (%)")
    ax.set_title("Heroes_of_Sokoban L0: depth × sharing × pool")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "heroes_bfs_by_depth.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "heroes_bfs_by_depth.png"), bbox_inches="tight", dpi=150)
    print(f"  wrote {outdir}/heroes_bfs_by_depth.pdf|.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
