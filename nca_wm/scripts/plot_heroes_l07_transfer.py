"""Paper-style figure: Heroes training-data ablation, per-level transfer.

Compares up to eight training-data sources, replicated at depths d=4 and d=8:
  - L0-only authored                (heroes_C_nopool_shared_d{4,8})
  - L0-L7 authored                  (heroes_l07_C_d{4,8})
  - Pure-evolve synth, fitness sel. (heroes_synth_pureEvolve_d{4,8})
  - Evolve seeded from L0, fitness  (heroes_synth_seedL0_d{4,8})
  - Evolve seeded from L0-L7, fit.  (heroes_synth_seedL07_d{4,8})
  - Pure-evolve synth, NSLC         (heroes_synth_nslc_pureEvolve_d{4,8})
  - Evolve seeded from L0, NSLC     (heroes_synth_nslc_seedL0_d{4,8})
  - Evolve seeded from L0-L7, NSLC  (heroes_synth_nslc_seedL07_d{4,8})

Runs are skipped if the npz isn't present, so the script is safe to re-run
while sweeps are still going.

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


_COLOR = {
    ("L0", 4):           ("#1f77b4", "L0 authored"),
    ("L0", 8):           ("#ff7f0e", "L0 authored"),
    ("L07", 4):          ("#2ca02c", "L0–L7 authored"),
    ("L07", 8):          ("#d62728", "L0–L7 authored"),
    ("pure", 4):         ("#9467bd", "Pure-evolve synth"),
    ("pure", 8):         ("#8c564b", "Pure-evolve synth"),
    ("seedL0", 4):       ("#e377c2", "Synth seeded from L0"),
    ("seedL0", 8):       ("#7f7f7f", "Synth seeded from L0"),
    ("seedL07", 4):      ("#bcbd22", "Synth seeded from L0–L7"),
    ("seedL07", 8):      ("#17becf", "Synth seeded from L0–L7"),
    ("nslc_pure", 4):    ("#aec7e8", "Pure-evolve synth + NSLC"),
    ("nslc_pure", 8):    ("#ffbb78", "Pure-evolve synth + NSLC"),
    ("nslc_seedL0", 4):  ("#98df8a", "Synth seeded from L0 + NSLC"),
    ("nslc_seedL0", 8):  ("#ff9896", "Synth seeded from L0 + NSLC"),
    ("nslc_seedL07", 4): ("#c5b0d5", "Synth seeded from L0–L7 + NSLC"),
    ("nslc_seedL07", 8): ("#c49c94", "Synth seeded from L0–L7 + NSLC"),
}


def _r(tag, depth):
    """Return (label, run_dir, train_levels_set, palette_color)."""
    color, base_label = _COLOR[(tag, depth)]
    label = f"{base_label}, d={depth}"
    if tag == "L0":
        run_dir = f"nca_wm/logs_heroes/heroes_C_nopool_shared_d{depth}"
        train_set = {0}
    elif tag == "L07":
        run_dir = f"nca_wm/logs_heroes_authored/heroes_l07_C_d{depth}"
        train_set = set(range(8))
    elif tag == "pure":
        run_dir = f"nca_wm/logs_heroes_synth/heroes_synth_pureEvolve_d{depth}"
        train_set = set()
    elif tag == "seedL0":
        run_dir = f"nca_wm/logs_heroes_synth/heroes_synth_seedL0_d{depth}"
        train_set = {0}
    elif tag == "seedL07":
        run_dir = f"nca_wm/logs_heroes_synth/heroes_synth_seedL07_d{depth}"
        train_set = set(range(8))
    elif tag == "nslc_pure":
        run_dir = f"nca_wm/logs_heroes_synth/heroes_synth_nslc_pureEvolve_d{depth}"
        train_set = set()
    elif tag == "nslc_seedL0":
        run_dir = f"nca_wm/logs_heroes_synth/heroes_synth_nslc_seedL0_d{depth}"
        train_set = {0}
    elif tag == "nslc_seedL07":
        run_dir = f"nca_wm/logs_heroes_synth/heroes_synth_nslc_seedL07_d{depth}"
        train_set = set(range(8))
    else:
        raise ValueError(tag)
    return label, run_dir, train_set, color


TAGS = ("L0", "L07", "pure", "seedL0", "seedL07",
        "nslc_pure", "nslc_seedL0", "nslc_seedL07")
DEPTHS = (4, 8)


def _load(d):
    full = os.path.join(_REPO, d) if not os.path.isabs(d) else d
    p = os.path.join(full, "eval_multigame.npz")
    if not os.path.isfile(p):
        return None
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
        "xtick.labelsize": 12, "ytick.labelsize": 13, "legend.fontsize": 11,
        "figure.constrained_layout.use": True,
    })

    runs = {}  # label → (run_dir, train_set, color, tag, depth)
    for depth in DEPTHS:
        for tag in TAGS:
            label, run_dir, train_set, color = _r(tag, depth)
            runs[label] = (run_dir, train_set, color, tag, depth)

    data = {}
    missing = []
    for label, (run_dir, *_) in runs.items():
        d = _load(run_dir)
        if d is None:
            missing.append(label)
        else:
            data[label] = d
    if missing:
        print("  missing (will be skipped):")
        for m in missing:
            print(f"    {m}  → {runs[m][0]}/eval_multigame.npz")

    all_levels = sorted(set().union(*(d.keys() for d in data.values())))

    # Plot 1: per-level BFS cell-error, grouped bars per depth (one panel each)
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, depth in zip(axes, DEPTHS):
        labels_here = [
            lbl for lbl, (_, _, _, _, d) in runs.items()
            if d == depth and lbl in data
        ]
        n = len(labels_here)
        if n == 0:
            ax.set_visible(False)
            continue
        x = np.arange(len(all_levels))
        w = 0.8 / max(n, 1)
        for i, lbl in enumerate(labels_here):
            color = runs[lbl][2]
            vals = [data[lbl].get(l, {}).get("bfs", np.nan) * 100 for l in all_levels]
            offset = (i - (n - 1) / 2.0) * w
            ax.bar(x + offset, vals, w, label=lbl, color=color,
                   edgecolor="black", linewidth=0.3)
        ax.axvline(0.5, color="k", linestyle=":", alpha=0.4, linewidth=1)
        ax.axvline(7.5, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_ylabel("BFS cell-error (%)")
        ax.set_title(f"d = {depth}")
        ax.legend(loc="upper left", ncol=2, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"L{l}" for l in all_levels], rotation=0)
    axes[-1].set_xlabel("Authored level")
    fig.suptitle("Heroes_of_Sokoban: per-level transfer across training-data sources")
    fig.savefig(os.path.join(_OUT, "per_level_bfs.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(_OUT, "per_level_bfs.png"), bbox_inches="tight", dpi=150)
    print(f"  wrote {_OUT}/per_level_bfs.pdf|.png")
    plt.close(fig)

    # Plot 2: heldout-mean summary bars. For each run, "held-out" = authored
    # levels not in its own train_set (synth runs have train_set covering
    # whichever authored levels they were seeded from; pure-evolve has none).
    def held_mean(d, kind, train_set):
        vs = [d[lvl][kind] for lvl in d if lvl not in train_set and kind in d[lvl]]
        return float(np.mean(vs)) if vs else np.nan

    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    metrics = [("bfs", "BFS"), ("astar", "A*"),
               ("random_tf", "random TF"), ("random", "random AR")]
    for ax, depth in zip(axes2, DEPTHS):
        labels_here = [
            lbl for lbl, (_, _, _, _, d) in runs.items()
            if d == depth and lbl in data
        ]
        n = len(labels_here)
        if n == 0:
            ax.set_visible(False)
            continue
        x = np.arange(len(metrics))
        bar_w = 0.8 / max(n, 1)
        for i, lbl in enumerate(labels_here):
            _, train_set, color, _, _ = runs[lbl]
            vals = [held_mean(data[lbl], k, train_set) * 100 for k, _ in metrics]
            offset = (i - (n - 1) / 2.0) * bar_w
            ax.bar(x + offset, vals, bar_w, label=lbl, color=color,
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in metrics])
        ax.set_xlabel("Rollout type")
        ax.set_title(f"d = {depth}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    axes2[0].set_ylabel("Held-out cell-error (%)")
    fig2.suptitle("Heroes_of_Sokoban: held-out transfer (mean over each run's held-out levels)")
    fig2.savefig(os.path.join(_OUT, "summary_bars.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(_OUT, "summary_bars.png"), bbox_inches="tight", dpi=150)
    print(f"  wrote {_OUT}/summary_bars.pdf|.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
