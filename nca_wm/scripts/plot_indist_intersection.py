#!/usr/bin/env python3
"""Slope + heatmap figures for the in-distribution intersection finding.

Emits two PDFs:

  intersection_slope.pdf
    Two-panel slope plot. Left: aggregate mean / median over the 14
    games shared by every training preset. Right: aggregate mean /
    median over each preset's full training set (the same data as
    Table tab:indist-scaling). Both panels share axes and styling so
    the two views can be compared at a glance.

  intersection_heatmap.pdf
    Per-game heatmap on the 14-game intersection (rows: games sorted
    by max cell value, columns: preset x cond/uncond). Standalone
    appendix figure.

Reads:
  - nca_wm/paper/figures/indist_intersection/summary.json   (intersection)
  - nca_wm/paper/figures/indist_scaling/summary.json        (full corpus)
  - per-run nca_wm/logs/<run>/eval_multigame.npz            (per-game)

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_indist_intersection.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT      = Path(__file__).resolve().parents[2]
INTER_SUMMARY  = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_intersection" / "summary.json"
FULL_SUMMARY   = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_scaling"     / "summary.json"
OUT_DIR        = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_intersection"

SCALES = ["Train-14", "Train-59", "Train-199"]
RUN_BY = {
    ("Train-14",  "cond"):   "multi_scaling_14_cond_match_s0",
    ("Train-14",  "uncond"): "multi_scaling_14_uncond_match_s0",
    ("Train-59",  "cond"):   "multi_scaling_gallery_v2_cond_match_s0",
    ("Train-59",  "uncond"): "multi_scaling_gallery_v2_uncond_match_s0",
    ("Train-199", "cond"):   "multi_scaling_gallery_v4_cond_match_s0",
    ("Train-199", "uncond"): "multi_scaling_gallery_v4_uncond_match_s0",
}
HEADLINE = "random_tf"

SERIES = [
    ("cond",   "Rule-conditioned", "#d62728"),
    ("uncond", "Unconditional",    "#1f77b4"),
]

RC_PARAMS = {
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
}


def _load_eval_per_game(run_dir: str, games: list[str]) -> dict[str, float]:
    """Per-game headline-regime mean (over levels) for the listed games."""
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


def _slope_panel(ax, scale_to_mean: dict[tuple[str, str], float],
                 scale_to_median: dict[tuple[str, str], float],
                 title: str) -> dict[str, dict]:
    """Plot a cond/uncond mean+median slope in one axes."""
    x = np.arange(len(SCALES))
    fig_data = {}
    for model, label, color in SERIES:
        means, medians, xs_present = [], [], []
        for xi, scale in enumerate(SCALES):
            mu = scale_to_mean.get((scale, model))
            md = scale_to_median.get((scale, model))
            if mu is not None and md is not None:
                means.append(100*mu)
                medians.append(100*md)
                xs_present.append(xi)
        if xs_present:
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
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(-0.25, len(SCALES) - 0.75)
    return fig_data


def _annotate_t14_to_t59(ax, fig_data: dict[str, dict]) -> None:
    annotations = []
    for model, label, _color in SERIES:
        d = fig_data.get(model)
        if d is None or len(d["means"]) < 2:
            continue
        m0, m1 = d["means"][0], d["means"][1]
        if m0 > 0 and m1 > 0:
            annotations.append((label, m1 / m0))
    if not annotations:
        return
    text_lines = [
        rf"{lab}: $\approx{f:.0f}\times$ T14$\to$T59" for (lab, f) in annotations
    ]
    ax.text(0.03, 0.97, "\n".join(text_lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#666666", alpha=0.95))


def _figure_slope(per_game: dict[tuple[str, str], dict[str, float]],
                  games_intersection: list[str],
                  full_summary: dict) -> plt.Figure:
    """Two-panel slope: left = intersection, right = full-corpus aggregate."""
    plt.rcParams.update(RC_PARAMS)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    # Left panel: 14-game intersection.
    inter_mean, inter_median = {}, {}
    for scale in SCALES:
        for model in ("cond", "uncond"):
            vals = [per_game[(scale, model)].get(g) for g in games_intersection]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(games_intersection):
                inter_mean[(scale, model)]   = float(np.mean(vals))
                inter_median[(scale, model)] = float(np.median(vals))
    fd_left = _slope_panel(
        ax_left, inter_mean, inter_median,
        title="14-game intersection (fixed game set)",
    )
    ax_left.set_ylabel("1-step (TF) cell-error (%)")
    _annotate_t14_to_t59(ax_left, fd_left)
    ax_left.legend(loc="lower right", framealpha=0.95)

    # Right panel: full-corpus per-preset aggregate (the data behind
    # tab:indist-scaling -- changing game set per scale).
    full_mean, full_median = {}, {}
    for run_dir, blob in full_summary.items():
        scale = blob["preset"]
        model = blob["model"]
        head = blob["results"].get(HEADLINE) if blob["results"] else None
        if head is not None:
            full_mean[(scale, model)]   = float(head["mean"])
            full_median[(scale, model)] = float(head["median"])
    fd_right = _slope_panel(
        ax_right, full_mean, full_median,
        title="Full training set (changing game set)",
    )
    _annotate_t14_to_t59(ax_right, fd_right)

    fig.tight_layout()
    return fig


def _figure_heatmap(per_game: dict[tuple[str, str], dict[str, float]],
                    games: list[str]) -> plt.Figure:
    """Standalone per-game heatmap on the 14-game intersection."""
    plt.rcParams.update(RC_PARAMS)

    def _row_key(g):
        cells = []
        for scale in SCALES:
            for model in ("cond", "uncond"):
                v = per_game[(scale, model)].get(g)
                if v is not None:
                    cells.append(v)
        return -max(cells) if cells else 0.0
    sorted_games = sorted(games, key=_row_key)

    col_labels = []
    matrix = np.full((len(sorted_games), 2*len(SCALES)), np.nan, dtype=float)
    for ci, scale in enumerate(SCALES):
        for cj, model in enumerate(("cond", "uncond")):
            col = 2*ci + cj
            col_labels.append(f"{scale}\n{model}")
            for ri, g in enumerate(sorted_games):
                v = per_game[(scale, model)].get(g)
                if v is not None:
                    matrix[ri, col] = 100*v

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    floor = 1e-3
    matrix_log = np.log10(np.clip(matrix, floor, None))
    im = ax.imshow(matrix_log, aspect="auto", cmap="magma_r",
                   vmin=np.log10(floor), vmax=np.log10(30.0))

    ax.set_yticks(range(len(sorted_games)))
    ax.set_yticklabels([g.replace("_", " ") for g in sorted_games])
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)

    for sep in (1.5, 3.5):
        ax.axvline(sep, color="white", linewidth=1.5)

    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            v = matrix[ri, ci]
            if not np.isfinite(v):
                continue
            text_color = "white" if matrix_log[ri, ci] > np.log10(0.3) else "black"
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("cell-error (%, log)", fontsize=10)
    log_ticks = [-3, -2, -1, 0, 1]
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"{10**t:g}" for t in log_ticks])

    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    inter_summary = json.loads(INTER_SUMMARY.read_text())
    games_intersection = inter_summary["intersection_games"]
    full_summary = json.loads(FULL_SUMMARY.read_text())

    per_game = {
        (scale, model): _load_eval_per_game(run_dir, games_intersection)
        for (scale, model), run_dir in RUN_BY.items()
    }

    fig_slope = _figure_slope(per_game, games_intersection, full_summary)
    slope_pdf = OUT_DIR / "intersection_slope.pdf"
    slope_png = OUT_DIR / "intersection_slope.png"
    fig_slope.savefig(slope_pdf, bbox_inches="tight")
    fig_slope.savefig(slope_png, bbox_inches="tight", dpi=160)
    plt.close(fig_slope)

    fig_heat = _figure_heatmap(per_game, games_intersection)
    heat_pdf = OUT_DIR / "intersection_heatmap.pdf"
    heat_png = OUT_DIR / "intersection_heatmap.png"
    fig_heat.savefig(heat_pdf, bbox_inches="tight")
    fig_heat.savefig(heat_png, bbox_inches="tight", dpi=160)
    plt.close(fig_heat)

    for p in (slope_pdf, slope_png, heat_pdf, heat_png):
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
