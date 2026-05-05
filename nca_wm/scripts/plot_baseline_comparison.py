"""Bar charts comparing the architectures on the headline metrics.

Reads logs_baselines/summary.csv (produced by
aggregate_baseline_comparison.py) and emits per-games_tag bar plots:
    figures/baseline_comparison_{games_tag}_{metric}.{pdf,png}
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGDIR = os.path.join(REPO, "nca_wm", "logs_baselines")
FIGDIR = os.path.join(REPO, "nca_wm", "figures", "baseline_comparison")
os.makedirs(FIGDIR, exist_ok=True)


METRICS = [
    ("best_loss",                 "Best train loss (lower is better)"),
    ("bfs_final_cellerr",         "Cell-error rate at end of BFS rollout"),
    ("random_rollout_cellerr",    "Mean cell-error rate over random rollout"),
    ("astar_first_div",           "First-divergence step under A* (higher is better)"),
]
ARCH_ORDER = ["nca_shared", "nca_perstep", "cnn_d4", "unet_l2", "vit_l4"]
ARCH_COLOURS = {
    "nca_shared":  "#1f77b4",
    "nca_perstep": "#7fbfff",
    "cnn_d4":      "#ff7f0e",
    "unet_l2":     "#2ca02c",
    "vit_l4":      "#d62728",
}


def load():
    path = os.path.join(LOGDIR, "summary.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}; run aggregate_baseline_comparison.py first")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def main():
    rows = load()
    by_games = defaultdict(list)
    for r in rows:
        by_games[r["games"]].append(r)

    for games, group in by_games.items():
        seeds_per_arch = defaultdict(list)
        for r in group:
            seeds_per_arch[r["arch"]].append(r)

        for metric, ylabel in METRICS:
            archs_present = [a for a in ARCH_ORDER if a in seeds_per_arch]
            if not archs_present:
                continue
            xs = []
            means = []
            stds = []
            params = []
            for a in archs_present:
                vals = [to_float(r.get(metric, "")) for r in seeds_per_arch[a]]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                xs.append(a)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                pvals = [to_float(r.get("n_params", "")) for r in seeds_per_arch[a]]
                pvals = [v for v in pvals if v is not None]
                params.append(np.mean(pvals) if pvals else None)
            if not xs:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            colours = [ARCH_COLOURS.get(a, "#888") for a in xs]
            ax.bar(xs, means, yerr=stds, color=colours, capsize=3)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{games} | {metric}")
            for i, (m, p) in enumerate(zip(means, params)):
                annot = f"{m:.3g}"
                if p is not None:
                    annot += f"\n({p / 1e6:.1f}M)"
                ax.text(i, m + 0.02 * max(means), annot,
                        ha="center", va="bottom", fontsize=9)
            ax.margins(y=0.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for ext in ("pdf", "png"):
                path = os.path.join(FIGDIR,
                                     f"{games}_{metric}.{ext}")
                fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"wrote {games}/{metric}")


if __name__ == "__main__":
    main()
