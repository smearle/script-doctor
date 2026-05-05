"""Per-game heatmap of BFS final cell-error on scaling_14.

Reads each scaling_14_<arch>_s<seed>/eval_multigame.npz, averages
final-step BFS cell-error across that game's levels, and renders a
heatmap of (game × architecture) with cell colour ∝ error and value
written in each cell. When multiple seeds are present, reports the
mean across seeds. Saves to:

    nca_wm/figures/baseline_comparison/scaling_14_per_game_heatmap.{pdf,png}
"""
from __future__ import annotations

import os
import re
from glob import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGDIR = os.path.join(REPO, "nca_wm", "logs_baselines")
FIGDIR = os.path.join(REPO, "nca_wm", "figures", "baseline_comparison")


ARCH_ORDER = ["nca_shared", "nca_perstep", "cnn_d4", "unet_l2", "vit_l4"]
ARCH_DISPLAY = {
    "nca_shared":  "NCA shared",
    "nca_perstep": "NCA perstep",
    "cnn_d4":      "CNN",
    "unet_l2":     "U-Net",
    "vit_l4":      "ViT",
}
RUN_RE = re.compile(r"scaling_14_(.+)_s(\d+)$")
LEVEL_RE = re.compile(r"^(.+?)_L(\d+)_bfs_cell_error_rate$")


def collect():
    """Returns dict {arch: {game: list[per_level_final_err]}} averaged across seeds."""
    by_arch = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob(os.path.join(LOGDIR, "scaling_14_*_s*"))):
        m = RUN_RE.search(os.path.basename(d))
        if m is None:
            continue
        arch = m.group(1)
        ev_path = os.path.join(d, "eval_multigame.npz")
        if not os.path.isfile(ev_path):
            continue
        ev = dict(np.load(ev_path, allow_pickle=True))
        per_game_levels = defaultdict(list)
        for k, v in ev.items():
            mm = LEVEL_RE.match(k)
            if mm is None:
                continue
            game = mm.group(1)
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.size > 0:
                per_game_levels[game].append(float(arr[-1]))
        for game, vs in per_game_levels.items():
            by_arch[arch][game].append(float(np.mean(vs)))
    return by_arch


def main():
    by_arch = collect()
    if not by_arch:
        raise SystemExit(f"no scaling_14 eval data under {LOGDIR}")

    archs = [a for a in ARCH_ORDER if a in by_arch]
    games = sorted({g for a in archs for g in by_arch[a].keys()})

    # mat[i, j] = mean across seeds of game i's mean-across-levels final err
    M = np.full((len(games), len(archs)), np.nan)
    for j, a in enumerate(archs):
        for i, g in enumerate(games):
            vs = by_arch[a].get(g, [])
            if vs:
                M[i, j] = float(np.mean(vs))

    # Order games by NCA-shared (or first available arch) error, ascending —
    # easiest games at top.
    sort_key = M[:, 0] if "nca_shared" in archs else M[:, 0]
    order = np.argsort(np.where(np.isnan(sort_key), 1.0, sort_key))
    M = M[order]
    games = [games[i] for i in order]

    fig, ax = plt.subplots(figsize=(1.5 + 1.0 * len(archs),
                                      0.4 * max(len(games), 6) + 1.0))
    im = ax.imshow(M, aspect="auto", cmap="magma_r",
                    vmin=0.0, vmax=min(1.0, max(0.5, np.nanmax(M))))
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([ARCH_DISPLAY.get(a, a) for a in archs], rotation=30,
                        ha="right")
    ax.set_yticks(range(len(games)))
    ax.set_yticklabels(games, fontsize=8)
    ax.set_title("BFS final-step cell-error rate (mean over levels, then seeds)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("cell-error rate")
    # annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.2f}" if v >= 0.01 else "0"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                     color="white" if v > 0.4 else "black")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGDIR, f"scaling_14_per_game_heatmap.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
