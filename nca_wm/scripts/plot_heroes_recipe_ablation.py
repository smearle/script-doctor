"""Recipe ablation on NSLC + seedL07 backbone (Heroes_of_Sokoban).

Two knobs varied on top of the NSLC + seedL07 baseline at d=4 and d=8:
  - solvable:    + --synthetic_require_solvable + --synthetic_fallback_dynamics
  - rulecov:     + --synthetic_track_rules_fired + --synthetic_rule_coverage_weight=50

Writes nca_wm/figures/heroes_l07_transfer/recipe_ablation.{pdf,png} —
4-condition × 2-depth grouped bars on held-out (L8-L21) BFS cell-error.
"""
from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
_OUT = os.path.join(_REPO, "nca_wm/figures/heroes_l07_transfer")
PAT = re.compile(r"^.+_L(\d+)_bfs_cell_error_rate$")

CONDITIONS = [
    ("ref",          "NSLC + seedL07",            "#9467bd"),
    ("solv",         "+ require_solvable",        "#d62728"),
    ("rulecov",      "+ rule_coverage_w=50",      "#2ca02c"),
    ("solv_rulecov", "+ both",                    "#ff7f0e"),
]


def _heldout_bfs(run_dir):
    p = os.path.join(_REPO, run_dir, "eval_multigame.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p, allow_pickle=True)
    vals = []
    for k in z.files:
        m = PAT.match(k)
        if m and int(m.group(1)) >= 8:
            vals.append(float(np.nanmean(z[k])) * 100)
    return float(np.mean(vals)) if vals else None


def _dir(tag, depth):
    if tag == "ref":
        return f"nca_wm/logs_heroes_synth/heroes_synth_nslc_seedL07_d{depth}"
    return f"nca_wm/logs_heroes_synth/heroes_synth_nslc_seedL07_{tag}_d{depth}"


def main():
    os.makedirs(_OUT, exist_ok=True)
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
        "figure.constrained_layout.use": True,
    })

    depths = (4, 8)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(depths))
    bar_w = 0.8 / len(CONDITIONS)
    for i, (tag, label, color) in enumerate(CONDITIONS):
        vals = [_heldout_bfs(_dir(tag, d)) for d in depths]
        offset = (i - (len(CONDITIONS) - 1) / 2.0) * bar_w
        bars = ax.bar(x + offset, [v if v is not None else 0 for v in vals],
                      bar_w, label=label, color=color,
                      edgecolor="black", linewidth=0.3)
        for b, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.15,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d = {d}" for d in depths])
    ax.set_ylabel("Held-out BFS cell-error (L8–L21, %)")
    ax.set_title("Recipe ablation on NSLC + seedL07 backbone")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(_OUT, "recipe_ablation.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(_OUT, "recipe_ablation.png"),
                bbox_inches="tight", dpi=150)
    print(f"  wrote {_OUT}/recipe_ablation.pdf|.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
