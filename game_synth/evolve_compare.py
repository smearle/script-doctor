"""Compare GP-evolve runs by their activated-mechanics trajectories.

Plots mean activated mechanics and dead-game fraction per generation for any
set of (label, run_dir) pairs. Used for the WM-loss-only vs activation-weighted
selection comparison.

    .venv/bin/python -m game_synth.evolve_compare \
        w0:game_synth/evolve_w0 w1:game_synth/evolve_w1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[1]


def load(run_dir):
    return json.loads((Path(run_dir) / "metrics.json").read_text())


def main():
    pairs = []
    for arg in sys.argv[1:]:
        label, _, d = arg.partition(":")
        pairs.append((label, d))
    if not pairs:
        pairs = [("WM-loss only", "game_synth/evolve_w0"),
                 ("+ activation weight", "game_synth/evolve_w1")]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]
    for (label, d), c in zip(pairs, colors):
        m = load(_REPO / d)
        g = [r["gen"] for r in m]
        ax[0].plot(g, [r["mean_activated"] for r in m], "-o", color=c, label=label)
        ax[1].plot(g, [r["frac_dead"] * 100 for r in m], "-o", color=c, label=label)
    ax[0].set_xlabel("generation", fontsize=13)
    ax[0].set_ylabel("mean activated mechanics / game", fontsize=13)
    ax[0].set_title("Mechanics that actually fire in play", fontsize=14)
    ax[0].legend(fontsize=12); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("generation", fontsize=13)
    ax[1].set_ylabel("% games firing ZERO rules", fontsize=13)
    ax[1].set_title("Dead-game fraction under selection", fontsize=14)
    ax[1].legend(fontsize=12); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = _REPO / "nca_wm" / "figures"
    for e in ("png", "pdf"):
        fig.savefig(out / f"activation_selection_compare.{e}", dpi=140, bbox_inches="tight")
    print(f"saved {out}/activation_selection_compare.png", flush=True)
    for label, d in pairs:
        m = load(_REPO / d)
        last = m[-1]
        print(f"  {label:24s} final gen {last['gen']}: act_mean {last['mean_activated']:.2f} "
              f"max {last['max_activated']} dead {last['frac_dead']:.0%}", flush=True)


if __name__ == "__main__":
    main()
