#!/usr/bin/env python3
"""Two-panel slope plot for the in-distribution dilution / intersection finding.

Reads nca_wm/paper/figures/indist_intersection/summary.json and emits
a two-panel figure (cond | uncond) showing per-game 1-step (TF)
cell-error trajectories across Train-14 -> Train-59 -> Train-199 plus
the aggregate mean / median across the 14 intersection games.

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_indist_intersection.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_intersection" / "summary.json"
OUT_DIR = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_intersection"

# Match the column ordering used in collate_indist_intersection.py.
SCALES = ["Train-14", "Train-59", "Train-199"]
RUN_BY = {
    ("Train-14",  "cond"):   "multi_scaling_14_cond_match_s0",
    ("Train-14",  "uncond"): "multi_scaling_14_uncond_match_s0",
    ("Train-59",  "cond"):   "multi_scaling_gallery_v2_cond_match_s0",
    ("Train-59",  "uncond"): "multi_scaling_gallery_v2_uncond_match_s0",
    ("Train-199", "cond"):   "multi_scaling_gallery_v4_cond_match_s0",
    ("Train-199", "uncond"): "multi_scaling_gallery_v4_uncond_match_s0",
}
HEADLINE = "random_tf"  # 1-step (TF) cell-error


def _load_eval_per_game(run_dir: str, games: list[str]) -> dict[str, float]:
    """Per-game headline-regime mean (over levels) for the listed games."""
    import re
    npz_path = REPO_ROOT / "nca_wm" / "logs" / run_dir / "eval_multigame.npz"
    if not npz_path.exists():
        return {}
    d = np.load(npz_path, allow_pickle=True)
    pat = re.compile(rf"^(?P<game>.+?)_L\d+_{HEADLINE}_cell_error_rate$")
    accum: dict[str, list[float]] = {}
    for k in d.files:
        m = pat.match(k)
        if not m: continue
        if m.group("game") not in games: continue
        arr = np.asarray(d[k], dtype=np.float64).ravel()
        if arr.size == 0: continue
        accum.setdefault(m.group("game"), []).append(float(arr.mean()))
    return {g: float(np.mean(v)) for g, v in accum.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    summary = json.loads(SUMMARY.read_text())
    games = summary["intersection_games"]

    # per_game[(scale, model)] = {game: error}
    per_game = {}
    aggregate_avail = {}
    for (scale, model), run_dir in RUN_BY.items():
        per_game[(scale, model)] = _load_eval_per_game(run_dir, games)
        head = summary["runs"][run_dir]["regimes"].get(HEADLINE)
        aggregate_avail[(scale, model)] = head is not None

    plt.rcParams.update({
        "font.size":        13,
        "axes.titlesize":   14,
        "axes.labelsize":   13,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  11,
    })

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))
    x = np.arange(len(SCALES))

    series = [
        ("cond",   "Rule-conditioned", "#d62728"),
        ("uncond", "Unconditional",    "#1f77b4"),
    ]
    fig_data = {}
    for model, label, color in series:
        # Thin per-game lines, color-coded by model.
        for g in games:
            ys = [per_game[(scale, model)].get(g) for scale in SCALES]
            xs_p = [xi for xi, yi in zip(x, ys) if yi is not None and yi > 0]
            ys_p = [100*yi for yi in ys if yi is not None and yi > 0]
            if len(xs_p) >= 2:
                ax.plot(xs_p, ys_p, color=color, linewidth=0.7, alpha=0.25,
                        marker="o", markersize=2.5, zorder=1)

        # Aggregate mean / median across all 14 games at scales where every
        # game has data (skip otherwise so partial points don't pull the line).
        means, medians, xs_present = [], [], []
        for xi, scale in enumerate(SCALES):
            vals = [per_game[(scale, model)].get(g) for g in games]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(games):
                means.append(100*float(np.mean(vals)))
                medians.append(100*float(np.median(vals)))
                xs_present.append(xi)
        if len(xs_present) >= 1:
            ax.plot(xs_present, means, color=color, linewidth=2.6,
                    marker="o", markersize=8, zorder=3,
                    label=f"{label} (mean)")
            ax.plot(xs_present, medians, color=color, linewidth=2.0,
                    marker="s", markersize=7, zorder=3, linestyle="--",
                    label=f"{label} (median)")
            fig_data[model] = {"means": means, "medians": medians, "xs": xs_present}

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(SCALES)
    ax.set_xlabel("Training corpus")
    ax.set_ylabel("1-step (TF) cell-error (%)")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(-0.25, len(SCALES) - 0.75)

    # Annotate T14 -> T59 multiplicative growth for each condition; this
    # segment is currently the one where both cond and uncond have data.
    annotations = []
    for model, label, color in series:
        d = fig_data.get(model)
        if d is None or len(d["means"]) < 2:
            continue
        m0, m1 = d["means"][0], d["means"][1]
        if m0 > 0 and m1 > 0:
            annotations.append((label, m1 / m0, color))
    if annotations:
        text_lines = [
            rf"{lab}: $\approx{f:.0f}\times$ T14$\to$T59"
            for (lab, f, _c) in annotations
        ]
        ax.text(0.02, 0.97, "\n".join(text_lines),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#666666", alpha=0.95))

    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()

    out_pdf = OUT_DIR / "intersection_slope.pdf"
    out_png = OUT_DIR / "intersection_slope.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    print(f"Wrote: {out_pdf}")
    print(f"       {out_png}")


if __name__ == "__main__":
    main()
