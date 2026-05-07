"""Render two minimal latent-space scatter PDFs for the teaser.

Both come from a multi-game checkpoint's saved heldout_overlay_flat.npz
(produced by latent_overlay_heldout.py). One shows only the training
games (gray dots); the other adds the held-out games as red stars.

Outputs:
    latent_train_only.pdf  / .png
    latent_with_ood.pdf    / .png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[4]
RUN = _REPO / "nca_wm" / "logs" / "multi_scaling_gallery_v3" / "interp" \
    / "heldout_overlay_flat.npz"
OUT = Path(__file__).resolve().parent

d = np.load(RUN, allow_pickle=True)
pt = d["proj_train"]
ph = d["proj_heldout"]

xmin = min(pt[:, 0].min(), ph[:, 0].min())
xmax = max(pt[:, 0].max(), ph[:, 0].max())
ymin = min(pt[:, 1].min(), ph[:, 1].min())
ymax = max(pt[:, 1].max(), ph[:, 1].max())
pad_x = (xmax - xmin) * 0.08
pad_y = (ymax - ymin) * 0.08


def _setup(ax):
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("black")
        s.set_linewidth(0.4)


def _save(name, fn):
    fig, ax = plt.subplots(figsize=(1.4, 1.4), dpi=200)
    _setup(ax)
    fn(ax)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf + .png")


def _train_only(ax):
    ax.scatter(pt[:, 0], pt[:, 1],
               s=6, c="#666666", alpha=0.75, linewidths=0)


def _with_ood(ax):
    ax.scatter(pt[:, 0], pt[:, 1],
               s=6, c="#666666", alpha=0.65, linewidths=0)
    ax.scatter(ph[:, 0], ph[:, 1],
               s=22, c="#C0463E", marker="*",
               edgecolors="black", linewidths=0.3)


_save("latent_train_only", _train_only)
_save("latent_with_ood",   _with_ood)
