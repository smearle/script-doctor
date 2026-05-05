"""Paper-style figure: Heroes L0-only vs L0-L7 training transfer comparison.

Per-level cell-error breakdown across all 22 authored levels of
Heroes_of_Sokoban, comparing four configs:
  - L0-only training, depth d=4 (recipe C; from heroes_C_nopool_shared_d4)
  - L0-L7 training, depth d=4 (heroes_l07_C_d4)
  - L0-only training, depth d=8 (heroes_C_nopool_shared_d8)
  - L0-L7 training, depth d=8 (heroes_l07_C_d8)

Writes nca_wm/figures/heroes_l07_transfer/{per_level_bfs, summary_bars}.{pdf,png}.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
_OUT = os.path.join(_REPO, "nca_wm/figures/heroes_l07_transfer")

PAT = re.compile(r"^(?P<game>.+)_L(?P<lvl>\d+)_(?P<kind>bfs|astar|random|random_tf)_cell_error_rate$")

RUNS = {
    "L0-only, d=4": "nca_wm/logs_heroes/heroes_C_nopool_shared_d4",
    "L0–L7, d=4":   "nca_wm/logs_heroes_authored/heroes_l07_C_d4",
    "L0-only, d=8": "nca_wm/logs_heroes/heroes_C_nopool_shared_d8",
    "L0–L7, d=8":   "nca_wm/logs_heroes_authored/heroes_l07_C_d8",
}


def _load(d):
    full = os.path.join(_REPO, d) if not os.path.isabs(d) else d
    for fn in ("eval_multigame_tlfix.npz", "eval_multigame.npz"):
        p = os.path.join(full, fn)
        if os.path.exists(p):
            break
    z = np.load(p, allow_pickle=True)
    by_lvl = defaultdict(dict)
    for k in z.files:
        m = PAT.match(k)
        if not m:
            continue
        by_lvl[int(m.group("lvl"))][m.group("kind")] = float(np.nanmean(z[k]))
    return by_lvl


def main():
    os.makedirs(_OUT, exist_ok=True)
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 13, "legend.fontsize": 12,
        "figure.constrained_layout.use": True,
    })
    data = {n: _load(p) for n, p in RUNS.items()}
    all_levels = sorted(set().union(*(d.keys() for d in data.values())))

    # Plot 1: per-level BFS cell-error, grouped bars by depth
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    palette = {"L0-only, d=4": "C0", "L0–L7, d=4": "C2",
               "L0-only, d=8": "C1", "L0–L7, d=8": "C3"}
    for ax, dpair in zip(axes, [("L0-only, d=4", "L0–L7, d=4"),
                                  ("L0-only, d=8", "L0–L7, d=8")]):
        x = np.arange(len(all_levels))
        w = 0.4
        for i, name in enumerate(dpair):
            vals = [data[name].get(l, {}).get("bfs", np.nan) * 100 for l in all_levels]
            ax.bar(x + (i - 0.5) * w, vals, w, label=name, color=palette[name],
                   edgecolor="black", linewidth=0.3)
        ax.axvline(7.5, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(3.5, ax.get_ylim()[1] * 0.92, "L0–L7 (training set\nfor 'L0–L7' run)",
                ha="center", va="top", fontsize=10)
        ax.text(14.5, ax.get_ylim()[1] * 0.92, "L8–L21 (held-out for 'L0–L7' run)",
                ha="center", va="top", fontsize=10)
        ax.set_ylabel("BFS cell-error (%)")
        ax.legend(loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"L{l}" for l in all_levels], rotation=0)
    axes[1].set_xlabel("Authored level")
    fig.suptitle("Heroes_of_Sokoban: per-level transfer, L0-only vs L0–L7 training")
    fig.savefig(os.path.join(_OUT, "per_level_bfs.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(_OUT, "per_level_bfs.png"), bbox_inches="tight", dpi=150)
    print(f"  wrote {_OUT}/per_level_bfs.pdf|.png")
    plt.close(fig)

    # Plot 2: heldout-mean summary bars (4 configs, 4 metrics)
    train_l0 = {0}
    train_l07 = set(range(8))

    def held_mean(d, kind, train_set):
        vs = [d[lvl][kind] for lvl in d if lvl not in train_set and kind in d[lvl]]
        return float(np.mean(vs)) if vs else np.nan

    fig2, ax = plt.subplots(figsize=(10, 5.5))
    metrics = [("bfs", "BFS"), ("astar", "A*"),
               ("random_tf", "random TF"), ("random", "random AR")]
    n_metrics = len(metrics)
    n_runs = len(RUNS)
    bar_w = 0.18
    x = np.arange(n_metrics)
    for i, (name, _) in enumerate(RUNS.items()):
        ts = train_l0 if name.startswith("L0-only") else train_l07
        vals = [held_mean(data[name], k, ts) * 100 for k, _ in metrics]
        ax.bar(x + (i - 1.5) * bar_w, vals, bar_w, label=name,
               color=palette[name], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_xlabel("Rollout type (held-out levels)")
    ax.set_ylabel("Held-out cell-error (%)")
    ax.set_title("Heroes_of_Sokoban: held-out transfer, L0-only vs L0–L7")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig2.savefig(os.path.join(_OUT, "summary_bars.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(_OUT, "summary_bars.png"), bbox_inches="tight", dpi=150)
    print(f"  wrote {_OUT}/summary_bars.pdf|.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
