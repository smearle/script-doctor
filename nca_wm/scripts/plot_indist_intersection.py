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
        "font.size":        12,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
        "xtick.labelsize":  11,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
    })

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12.0, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.45]},
    )

    # ---- Left: aggregate slope (mean + median per condition). ----
    x = np.arange(len(SCALES))
    series = [
        ("cond",   "Rule-conditioned", "#d62728"),
        ("uncond", "Unconditional",    "#1f77b4"),
    ]
    fig_data = {}
    for model, label, color in series:
        means, medians, xs_present = [], [], []
        for xi, scale in enumerate(SCALES):
            vals = [per_game[(scale, model)].get(g) for g in games]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(games):
                means.append(100*float(np.mean(vals)))
                medians.append(100*float(np.median(vals)))
                xs_present.append(xi)
        if len(xs_present) >= 1:
            ax_left.plot(xs_present, means, color=color, linewidth=2.6,
                         marker="o", markersize=8, zorder=3,
                         label=f"{label} (mean)")
            ax_left.plot(xs_present, medians, color=color, linewidth=2.0,
                         marker="s", markersize=7, zorder=3, linestyle="--",
                         label=f"{label} (median)")
            fig_data[model] = {"means": means, "medians": medians, "xs": xs_present}

    ax_left.set_yscale("log")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(SCALES)
    ax_left.set_xlabel("Training corpus")
    ax_left.set_ylabel("1-step (TF) cell-error (%)")
    ax_left.set_title("Aggregate over 14 shared games")
    ax_left.grid(True, which="both", alpha=0.25)
    ax_left.set_xlim(-0.25, len(SCALES) - 0.75)

    # T14 -> T59 multiplicative growth annotation for each condition.
    annotations = []
    for model, label, _color in series:
        d = fig_data.get(model)
        if d is None or len(d["means"]) < 2:
            continue
        m0, m1 = d["means"][0], d["means"][1]
        if m0 > 0 and m1 > 0:
            annotations.append((label, m1 / m0))
    if annotations:
        text_lines = [
            rf"{lab}: $\approx{f:.0f}\times$ T14$\to$T59"
            for (lab, f) in annotations
        ]
        ax_left.text(0.03, 0.97, "\n".join(text_lines),
                     transform=ax_left.transAxes, ha="left", va="top",
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               ec="#666666", alpha=0.95))
    ax_left.legend(loc="lower right", framealpha=0.95)

    # ---- Right: per-game heatmap. ----
    # Sort games by their max error across all six cells, descending,
    # so the visually striking outliers (nekopuzzle, notsnake) appear
    # at the top.
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

    # Log-color scale, clipped at floor for log domain.
    floor = 1e-3  # 0.001 %
    matrix_log = np.log10(np.clip(matrix, floor, None))
    im = ax_right.imshow(matrix_log, aspect="auto", cmap="magma_r",
                         vmin=np.log10(floor), vmax=np.log10(30.0))

    ax_right.set_yticks(range(len(sorted_games)))
    ax_right.set_yticklabels([g.replace("_", " ") for g in sorted_games])
    ax_right.set_xticks(range(len(col_labels)))
    ax_right.set_xticklabels(col_labels, fontsize=9)
    ax_right.set_title("Per-game 1-step (TF) cell-error (%)")

    # Vertical separators between scale groups (after every 2 cols).
    for sep in (1.5, 3.5):
        ax_right.axvline(sep, color="white", linewidth=1.5)

    # Cell-value annotations (small font, contrast-aware).
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            v = matrix[ri, ci]
            if not np.isfinite(v):
                continue
            text_color = "white" if matrix_log[ri, ci] > np.log10(0.3) else "black"
            ax_right.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                          fontsize=7.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax_right, fraction=0.045, pad=0.02)
    cbar.set_label("cell-error (%, log)", fontsize=10)
    log_ticks = [-3, -2, -1, 0, 1]
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"{10**t:g}" for t in log_ticks])

    fig.tight_layout()

    out_pdf = OUT_DIR / "intersection_slope.pdf"
    out_png = OUT_DIR / "intersection_slope.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    print(f"Wrote: {out_pdf}")
    print(f"       {out_png}")


if __name__ == "__main__":
    main()
