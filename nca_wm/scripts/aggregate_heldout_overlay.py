"""Aggregate `latent_overlay_heldout` JSON outputs across checkpoints into a
table + scaling-curve plot.

Walks `nca_wm/logs/<run>/interp/heldout_overlay_*.json` for each provided run
and reports a summary table of (n_games, recipe_flags, median_1nn,
mean_1nn). Optionally plots median 1-NN cosine distance vs n_games.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_run_meta(run_dir):
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    gi_path = os.path.join(run_dir, "game_infos.pkl")
    if os.path.exists(gi_path):
        with open(gi_path, "rb") as f:
            gi = pickle.load(f)
        n_games = len(gi)
    else:
        n_games = None
    return {
        "preset": cfg.get("games", "?"),
        "arch": cfg.get("architecture", "?"),
        "n_games": n_games,
        "n_updates": cfg.get("n_updates", "?"),
        "n_nca_steps": cfg.get("n_nca_steps", "?"),
        "n_nca_repeats": cfg.get("n_nca_repeats", 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run dirs (relative to repo or absolute).")
    ap.add_argument("--reduce", default="flat",
                    choices=["flat", "mean", "dyn_mean"])
    ap.add_argument("--out_dir", default="nca_wm/figures/heldout_scaling")
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        run_abs = (run if os.path.isabs(run)
                   else os.path.join(REPO, run))
        meta = load_run_meta(run_abs)
        if meta is None:
            print(f"  skip (no config): {run}", file=sys.stderr)
            continue
        overlay_path = os.path.join(run_abs, "interp",
                                    f"heldout_overlay_{args.reduce}.json")
        if not os.path.exists(overlay_path):
            print(f"  skip (no overlay): {run}", file=sys.stderr)
            continue
        with open(overlay_path) as f:
            ov = json.load(f)
        rows.append({
            "run": os.path.basename(run_abs),
            **meta,
            "median_1nn": ov["nn1_cos_dist_median"],
            "mean_1nn": ov["nn1_cos_dist_mean"],
            "min_1nn": ov["nn1_cos_dist_min"],
            "max_1nn": ov["nn1_cos_dist_max"],
            "n_heldout": ov["n_heldout"],
        })

    if not rows:
        print("No rows aggregated.", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: (r["n_games"] or 0))

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Table (markdown)
    md_path = os.path.join(out_dir, f"summary_{args.reduce}.md")
    with open(md_path, "w") as f:
        f.write(f"# Held-out overlay scaling table (reduce={args.reduce})\n\n")
        f.write(f"All runs evaluated on the same {rows[0]['n_heldout']}-game "
                f"held-out set.\n\n")
        f.write("| run | n_games | n_updates | nca_steps | "
                "median 1-NN | mean 1-NN | min | max |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| `{r['run']}` | {r['n_games']} | "
                    f"{r['n_updates']} | {r['n_nca_steps']} | "
                    f"{r['median_1nn']:.3f} | {r['mean_1nn']:.3f} | "
                    f"{r['min_1nn']:.3f} | {r['max_1nn']:.3f} |\n")
    print(f"Wrote {md_path}")

    # Plot: median 1-NN vs n_games
    fig, ax = plt.subplots(figsize=(8, 5))
    pts = [(r["n_games"], r["median_1nn"], r["run"])
           for r in rows if r["n_games"] is not None]
    pts.sort()
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        names = [p[2] for p in pts]
        ax.plot(xs, ys, marker="o", markersize=10, linewidth=1.5,
                color="#1f77b4")
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), fontsize=7, xytext=(5, 5),
                        textcoords="offset points")

    ax.set_xlabel("n_games (training)")
    ax.set_ylabel("median held-out 1-NN cosine distance")
    ax.set_title(f"Held-out latent compression vs scale "
                 f"(N_held={rows[0]['n_heldout']}, reduce={args.reduce})")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    base = os.path.join(out_dir, f"median_1nn_vs_n_games_{args.reduce}")
    fig.savefig(base + ".png", dpi=150)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"Wrote {base}.png + .pdf")

    # Print table to stdout
    print()
    with open(md_path) as f:
        sys.stdout.write(f.read())


if __name__ == "__main__":
    main()
