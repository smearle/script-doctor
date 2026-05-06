"""Loss-curve overlays per game, one panel per game, four lines per panel
(buckets A/B/C/D). Lets us check whether high-loss cells were actually
converged at training-budget cutoff.

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_per_game_arch_curves.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))

_RUN_RE = re.compile(r"^(?P<game>.+)__(?P<bucket>[A-D])_d(?P<depth>\d+)$")
_BUCKET_COLORS = {"A": "C0", "B": "C1", "C": "C2", "D": "C3"}
_BUCKET_LABELS = {
    "A": "A: pool ON, shared",
    "B": "B: pool ON, per-step",
    "C": "C: no-pool+skip, shared",
    "D": "D: no-pool+skip, per-step",
}


def _load_curves(run_dir):
    cs = sorted(
        glob.glob(os.path.join(run_dir, "curves_step*.npz")),
        key=lambda p: int(os.path.basename(p)[len("curves_step"):-len(".npz")]),
    )
    if not cs:
        return None
    z = np.load(cs[-1])
    return np.asarray(z["losses"]).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="nca_wm/logs_per_game_arch")
    ap.add_argument("--outdir", default="nca_wm/figures/per_game_arch")
    args = ap.parse_args()

    logs = args.logs if os.path.isabs(args.logs) else os.path.join(_REPO, args.logs)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(_REPO, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    by_game = {}
    for d in sorted(glob.glob(os.path.join(logs, "*"))):
        name = os.path.basename(d)
        m = _RUN_RE.match(name)
        if not m:
            continue
        losses = _load_curves(d)
        if losses is None or losses.size == 0:
            continue
        by_game.setdefault(m.group("game"), {})[m.group("bucket")] = losses

    games = sorted(by_game.keys())
    n = len(games)
    if n == 0:
        print("no curves found")
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, g in zip(axes, games):
        for b in ["A", "B", "C", "D"]:
            ys = by_game[g].get(b)
            if ys is None:
                continue
            xs = np.arange(len(ys))
            ax.plot(xs, ys, color=_BUCKET_COLORS[b], label=_BUCKET_LABELS[b], linewidth=1.5, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(g, fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[: n]:
        ax.set_xlabel("training step")
    axes[0].set_ylabel("BCE loss (log scale)")
    axes[0].legend(loc="upper right", fontsize=8)

    fig.suptitle("Per-game training curves (buckets A/B/C/D, n_nca_steps=8)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss_curves.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "loss_curves.png"), bbox_inches="tight", dpi=150)
    print(f"  wrote {outdir}/loss_curves.pdf|.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
