#!/usr/bin/env python3
"""Slope + heatmap figures for the in-distribution intersection finding.

Emits four PDFs:

  intersection_slope.pdf
    Two-panel slope plot. Left: in-distribution aggregate mean / median
    on the 14 games shared by every training preset (rule-conditioning
    slows per-game dilution). Right: out-of-distribution aggregate
    mean / median on Heldout-26 (the non-truncated subset of
    Table tab:cond-vs-uncond-match), with the identity baseline as a
    reference. Both panels share log axes (numeric x at 14/59/199) and
    line styling so the in-dist and OOD trends can be compared at a
    glance.

  intersection_slope_id.pdf / intersection_slope_id_ar.pdf
    Single-panel versions of the in-distribution panel for 1-step (TF)
    and random-action AR, for use as subfigures in the main paper.

  intersection_slope_ood.pdf / intersection_slope_ood_ar.pdf
    Single-panel versions of the out-of-distribution panel for 1-step
    (TF) and random-action AR, for use as subfigures in the main paper.

  intersection_heatmap.pdf
    Per-game heatmap on the 14-game intersection (rows: games sorted
    by max cell value, columns: preset x cond/uncond). Standalone
    appendix figure.

Reads:
  - nca_wm/paper/figures/indist_intersection/summary.json   (intersection)
  - nca_wm/paper/figures/cond_vs_uncond_match/summary.csv   (OOD aggregate)
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
INTER_SUMMARY  = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_intersection"  / "summary.json"
INTER_IDENTITY = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_intersection"  / "identity.json"
OOD_SUMMARY    = REPO_ROOT / "nca_wm" / "paper" / "figures" / "cond_vs_uncond_match" / "summary.csv"
GAMES_META     = REPO_ROOT / "data" / "games_metadata.json"
OUT_DIR        = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_intersection"

SCALES   = ["Train-14", "Train-59", "Train-199"]
SCALE_N  = {"Train-14": 14, "Train-59": 59, "Train-199": 199}  # numeric x positions
RUN_BY   = {
    ("Train-14",  "cond"):   [
        "multi_scaling_14_cond_match_s0",
        "multi_scaling_14_cond_match_s2",
    ],
    ("Train-14",  "uncond"): [
        "multi_scaling_14_uncond_match_s0",
        "multi_scaling_14_uncond_match_s1",
        "multi_scaling_14_uncond_match_s2",
        "multi_scaling_14_uncond_match_s3",
    ],
    ("Train-59",  "cond"):   [
        "multi_scaling_gallery_v2_cond_match_s0",
        "multi_scaling_gallery_v2_cond_match_s1",
    ],
    ("Train-59",  "uncond"): [
        "multi_scaling_gallery_v2_uncond_match_s0",
        "multi_scaling_gallery_v2_uncond_match_s1",
    ],
    ("Train-199", "cond"):   [
        "multi_scaling_gallery_v4_cond_match_s0",
        "multi_scaling_gallery_v4_cond_match_s1",
    ],
    ("Train-199", "uncond"): [
        "multi_scaling_gallery_v4_uncond_match_s0",
        "multi_scaling_gallery_v4_uncond_match_s1",
    ],
}
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


def _load_eval_per_game(run_dirs: list[str] | str, games: list[str],
                        regime: str = "random_tf") -> dict[str, float]:
    """Per-game headline-regime mean (over levels) for the listed games,
    averaged across seeds whose eval_multigame.npz exists. Accepts a
    single run dir (legacy) or a list of seed dirs."""
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    pat = re.compile(rf"^(?P<game>.+?)_L\d+_{regime}_cell_error_rate$")
    per_seed: list[dict[str, float]] = []
    for rd in run_dirs:
        npz_path = REPO_ROOT / "nca_wm" / "logs" / rd / "eval_multigame.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        accum: dict[str, list[float]] = {}
        for k in d.files:
            m = pat.match(k)
            if not m: continue
            if m.group("game") not in games: continue
            arr = np.asarray(d[k], dtype=np.float64).ravel()
            if arr.size == 0: continue
            accum.setdefault(m.group("game"), []).append(float(arr.mean()))
        per_seed.append({g: float(np.mean(v)) for g, v in accum.items()})
    if not per_seed:
        return {}
    games_seen = sorted(set().union(*[set(d.keys()) for d in per_seed]))
    out: dict[str, float] = {}
    for g in games_seen:
        vs = [d[g] for d in per_seed if g in d]
        if vs:
            out[g] = float(np.mean(vs))
    return out


def _slope_panel(ax, scale_to_mean: dict[tuple[str, str], float],
                 scale_to_median: dict[tuple[str, str], float],
                 title: str) -> dict[str, dict]:
    """Plot a cond/uncond mean+median slope on a numeric (log) x-axis."""
    fig_data = {}
    for model, label, color in SERIES:
        xs, means, medians = [], [], []
        for scale in SCALES:
            mu = scale_to_mean.get((scale, model))
            md = scale_to_median.get((scale, model))
            if mu is not None and md is not None:
                xs.append(SCALE_N[scale])
                means.append(100*mu)
                medians.append(100*md)
        if xs:
            ax.plot(xs, means, color=color, linewidth=2.6,
                    marker="o", markersize=8, zorder=3,
                    label=f"{label} (mean)")
            ax.plot(xs, medians, color=color, linewidth=2.0,
                    marker="s", markersize=7, zorder=3, linestyle="--",
                    label=f"{label} (median)")
            fig_data[model] = {"xs": xs, "means": means, "medians": medians}
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([SCALE_N[s] for s in SCALES])
    ax.set_xticklabels(SCALES)
    ax.minorticks_off()  # don't decorate between 14/59/199 with auto minor ticks
    ax.set_xlabel("Training corpus (games, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(11, 250)
    return fig_data


def _annotate_t14_to_t59(ax, fig_data: dict[str, dict]) -> None:
    """Multiplicative growth from the smallest to the next-smallest scale."""
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
        rf"{lab}: $\approx{f:.0f}\times$ 14$\to$59" for (lab, f) in annotations
    ]
    ax.text(0.03, 0.97, "\n".join(text_lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#666666", alpha=0.95))


def _annotate_t14_to_t199(ax, fig_data: dict[str, dict]) -> None:
    """Multiplicative shrinkage from the smallest to the largest scale (OOD)."""
    annotations = []
    for model, label, _color in SERIES:
        d = fig_data.get(model)
        if d is None or len(d["means"]) < 2:
            continue
        # Only annotate if we have both endpoints (xs should bracket 14 and 199).
        if d["xs"][0] != SCALE_N["Train-14"] or d["xs"][-1] != SCALE_N["Train-199"]:
            continue
        m0, m_end = d["means"][0], d["means"][-1]
        if m0 > 0 and m_end > 0:
            annotations.append((label, m0 / m_end))
    if not annotations:
        return
    text_lines = [
        rf"{lab}: $\approx{f:.0f}\times$ better 14$\to$199"
        for (lab, f) in annotations
    ]
    ax.text(0.03, 0.07, "\n".join(text_lines),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#666666", alpha=0.95))


def _load_ood_summary(metric: str = "tf") -> tuple[dict[tuple[str, str], float],
                                                   dict[tuple[str, str], float],
                                                   float, float]:
    """Read OOD aggregates from cond_vs_uncond_match/summary.csv.

    Returns (mean_by_(scale,model), median_by_(scale,model), id_mean, id_median)
    in [0,1] units (matching the indist summary)."""
    import csv as _csv
    if metric == "tf":
        prefix = "ood_tf"
    elif metric == "ar30":
        prefix = "ood_ar30"
    else:
        raise ValueError(f"unknown OOD metric: {metric}")
    means, medians = {}, {}
    id_mean = id_median = float("nan")
    with open(OOD_SUMMARY) as f:
        for row in _csv.DictReader(f):
            scale = row["preset"]
            model = "cond" if row["kind"] == "cond" else "uncond"
            try:
                m  = float(row[f"{prefix}_model_mean"])
                md = float(row[f"{prefix}_model_median"])
                means[(scale, model)]   = m
                medians[(scale, model)] = md
            except ValueError:
                continue
            try:
                id_mean   = float(row[f"{prefix}_identity_mean"])
                id_median = float(row[f"{prefix}_identity_median"])
            except (ValueError, KeyError):
                pass
    return means, medians, id_mean, id_median


def _intersection_aggregates(per_game: dict[tuple[str, str], dict[str, float]],
                             games_intersection: list[str]
                             ) -> tuple[dict[tuple[str, str], float],
                                         dict[tuple[str, str], float]]:
    """Aggregate per-game ID errors to mean/median by (scale, model)."""
    inter_mean, inter_median = {}, {}
    for scale in SCALES:
        for model in ("cond", "uncond"):
            vals = [per_game[(scale, model)].get(g) for g in games_intersection]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(games_intersection):
                inter_mean[(scale, model)]   = float(np.mean(vals))
                inter_median[(scale, model)] = float(np.median(vals))
    return inter_mean, inter_median


def _load_id_identity() -> tuple[float, float]:
    """Read the in-distribution identity baseline (mean, median) from
    identity.json, or (nan, nan) if the file isn't present."""
    if not INTER_IDENTITY.exists():
        return float("nan"), float("nan")
    j = json.loads(INTER_IDENTITY.read_text())
    agg = j.get("aggregate") or {}
    return (
        float(agg.get("mean",   float("nan"))),
        float(agg.get("median", float("nan"))),
    )


def _draw_id_panel(ax, inter_mean, inter_median,
                   *, title: str, ylabel: str,
                   id_mean: float = float("nan"),
                   id_median: float = float("nan")) -> None:
    fd = _slope_panel(
        ax, inter_mean, inter_median,
        title=title,
    )
    ax.set_ylabel(ylabel)
    if np.isfinite(id_mean):
        ax.axhline(100*id_mean, color="black", linewidth=1.4,
                   linestyle=":", alpha=0.7,
                   label="identity (mean)", zorder=2)
    if np.isfinite(id_median):
        ax.axhline(100*id_median, color="black", linewidth=1.0,
                   linestyle=(0, (1, 2)), alpha=0.55,
                   label="identity (median)", zorder=2)
    _annotate_t14_to_t59(ax, fd)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)


def _draw_ood_panel(ax, ood_mean, ood_median, id_mean, id_median,
                    *, title: str, ylabel: str) -> None:
    fd = _slope_panel(
        ax, ood_mean, ood_median,
        title=title,
    )
    ax.set_ylabel(ylabel)
    if np.isfinite(id_mean):
        ax.axhline(100*id_mean, color="black", linewidth=1.4,
                   linestyle=":", alpha=0.7,
                   label="identity (mean)", zorder=2)
    if np.isfinite(id_median):
        ax.axhline(100*id_median, color="black", linewidth=1.0,
                   linestyle=(0, (1, 2)), alpha=0.55,
                   label="identity (median)", zorder=2)
    _annotate_t14_to_t199(ax, fd)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)


def _figure_slope(per_game: dict[tuple[str, str], dict[str, float]],
                  games_intersection: list[str]) -> plt.Figure:
    """Two-panel slope: left = in-dist intersection, right = OOD heldout."""
    plt.rcParams.update(RC_PARAMS)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11.0, 4.6),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    inter_mean, inter_median = _intersection_aggregates(per_game, games_intersection)
    id_mean_id, id_median_id = _load_id_identity()
    _draw_id_panel(
        ax_left, inter_mean, inter_median,
        title="In-distribution (14-game intersection)",
        ylabel="1-step (TF) cell-error (%)",
        id_mean=id_mean_id, id_median=id_median_id,
    )
    ood_mean, ood_median, id_mean, id_median = _load_ood_summary("tf")
    _draw_ood_panel(
        ax_right, ood_mean, ood_median, id_mean, id_median,
        title="Out-of-distribution (Heldout-26)",
        ylabel="1-step (TF) cell-error (%)",
    )
    fig.tight_layout()
    return fig


def _figure_id(per_game: dict[tuple[str, str], dict[str, float]],
               games_intersection: list[str],
               *, title: str, ylabel: str) -> plt.Figure:
    """Single-panel ID slope, for use as a subfigure."""
    plt.rcParams.update(RC_PARAMS)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    inter_mean, inter_median = _intersection_aggregates(per_game, games_intersection)
    id_mean, id_median = _load_id_identity()
    _draw_id_panel(ax, inter_mean, inter_median, title=title, ylabel=ylabel,
                   id_mean=id_mean, id_median=id_median)
    fig.tight_layout()
    return fig


def _figure_ood(*, metric: str, title: str, ylabel: str) -> plt.Figure:
    """Single-panel OOD slope, for use as a subfigure."""
    plt.rcParams.update(RC_PARAMS)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ood_mean, ood_median, id_mean, id_median = _load_ood_summary(metric)
    _draw_ood_panel(ax, ood_mean, ood_median, id_mean, id_median,
                    title=title, ylabel=ylabel)
    fig.tight_layout()
    return fig


def _load_rule_counts(games: list[str]) -> dict[str, int]:
    """Look up per-game n_rules from games_metadata.json (best-effort)."""
    if not GAMES_META.exists():
        return {}
    meta = json.loads(GAMES_META.read_text())
    rules = {}
    for g in games:
        for cand in (g + ".txt", g.replace(" ", "_") + ".txt",
                     g.lower() + ".txt"):
            if cand in meta:
                rules[g] = int(meta[cand].get("n_rules", -1))
                break
    return rules


def _load_astar_difficulty(games: list[str]) -> dict[str, float]:
    """Per-game mean A* iterations to solve (across levels with a win).

    Reads per-level JSON from data/js_sols/<game>/astar_<budget>-steps_level-*.json,
    preferring the largest available step budget per game. Games with zero
    solved levels return NaN; the caller decides how to slot them.
    """
    JS_SOLS = REPO_ROOT / "data" / "js_sols"
    out: dict[str, float] = {}
    for g in games:
        gdir = JS_SOLS / g
        if not gdir.is_dir():
            out[g] = float("nan")
            continue
        # Prefer largest budget present.
        budgets = set()
        for p in gdir.glob("astar_*-steps_level-*.json"):
            stem = p.name
            try:
                budgets.add(int(stem.split("_")[1].split("-")[0]))
            except (IndexError, ValueError):
                continue
        if not budgets:
            out[g] = float("nan")
            continue
        budget = max(budgets)
        iters = []
        for p in sorted(gdir.glob(f"astar_{budget}-steps_level-*.json")):
            try:
                rec = json.loads(p.read_text())
            except Exception:
                continue
            if rec.get("won"):
                v = rec.get("iterations")
                if isinstance(v, (int, float)):
                    iters.append(float(v))
        out[g] = float(np.mean(iters)) if iters else float("nan")
    return out


def _figure_heatmap(per_game: dict[tuple[str, str], dict[str, float]],
                    games: list[str]) -> plt.Figure:
    """Per-game heatmap on the 14-game intersection.

    Rows are ordered by rule count (ascending), with the count
    appended to each y-label. Cell values from per_game stay
    unchanged.
    """
    plt.rcParams.update(RC_PARAMS)

    rule_counts = _load_rule_counts(games)
    def _row_key(g):
        return (rule_counts.get(g, 99), g.lower())
    sorted_games = sorted(games, key=_row_key)

    col_labels = []
    matrix = np.full((len(sorted_games), 2*len(SCALES)), np.nan, dtype=float)
    for ci, scale in enumerate(SCALES):
        for cj, model in enumerate(("cond", "uncond")):
            col = 2*ci + cj
            col_labels.append(f"{scale} {model}")
            for ri, g in enumerate(sorted_games):
                v = per_game[(scale, model)].get(g)
                if v is not None:
                    matrix[ri, col] = 100*v

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    floor = 1e-3
    matrix_log = np.log10(np.clip(matrix, floor, None))
    im = ax.imshow(matrix_log, aspect="auto", cmap="magma_r",
                   vmin=np.log10(floor), vmax=np.log10(30.0))

    ax.set_yticks(range(len(sorted_games)))
    ylabels = []
    for g in sorted_games:
        rc = rule_counts.get(g)
        suffix = f" (n={rc})" if rc is not None else ""
        ylabels.append(g.replace("_", " ") + suffix)
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, rotation=45,
                       ha="right", rotation_mode="anchor")

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

    per_game_tf = {
        (scale, model): _load_eval_per_game(run_dir, games_intersection, "random_tf")
        for (scale, model), run_dir in RUN_BY.items()
    }
    per_game_ar = {
        (scale, model): _load_eval_per_game(run_dir, games_intersection, "random")
        for (scale, model), run_dir in RUN_BY.items()
    }

    fig_slope = _figure_slope(per_game_tf, games_intersection)
    slope_pdf = OUT_DIR / "intersection_slope.pdf"
    slope_png = OUT_DIR / "intersection_slope.png"
    fig_slope.savefig(slope_pdf, bbox_inches="tight")
    fig_slope.savefig(slope_png, bbox_inches="tight", dpi=160)
    plt.close(fig_slope)

    fig_id = _figure_id(
        per_game_tf, games_intersection,
        title="In-distribution (14-game intersection)",
        ylabel="1-step (TF) cell-error (%)",
    )
    id_pdf = OUT_DIR / "intersection_slope_id.pdf"
    id_png = OUT_DIR / "intersection_slope_id.png"
    fig_id.savefig(id_pdf, bbox_inches="tight")
    fig_id.savefig(id_png, bbox_inches="tight", dpi=160)
    plt.close(fig_id)

    fig_ood = _figure_ood(
        metric="tf",
        title="Out-of-distribution (Heldout-26)",
        ylabel="1-step (TF) cell-error (%)",
    )
    ood_pdf = OUT_DIR / "intersection_slope_ood.pdf"
    ood_png = OUT_DIR / "intersection_slope_ood.png"
    fig_ood.savefig(ood_pdf, bbox_inches="tight")
    fig_ood.savefig(ood_png, bbox_inches="tight", dpi=160)
    plt.close(fig_ood)

    fig_id_ar = _figure_id(
        per_game_ar, games_intersection,
        title="In-distribution AR (14-game intersection)",
        ylabel="random-action AR cell-error (%)",
    )
    id_ar_pdf = OUT_DIR / "intersection_slope_id_ar.pdf"
    id_ar_png = OUT_DIR / "intersection_slope_id_ar.png"
    fig_id_ar.savefig(id_ar_pdf, bbox_inches="tight")
    fig_id_ar.savefig(id_ar_png, bbox_inches="tight", dpi=160)
    plt.close(fig_id_ar)

    fig_ood_ar = _figure_ood(
        metric="ar30",
        title="Out-of-distribution AR (Heldout-26)",
        ylabel="random-action AR cell-error (%)",
    )
    ood_ar_pdf = OUT_DIR / "intersection_slope_ood_ar.pdf"
    ood_ar_png = OUT_DIR / "intersection_slope_ood_ar.png"
    fig_ood_ar.savefig(ood_ar_pdf, bbox_inches="tight")
    fig_ood_ar.savefig(ood_ar_png, bbox_inches="tight", dpi=160)
    plt.close(fig_ood_ar)

    fig_heat = _figure_heatmap(per_game_tf, games_intersection)
    heat_pdf = OUT_DIR / "intersection_heatmap.pdf"
    heat_png = OUT_DIR / "intersection_heatmap.png"
    fig_heat.savefig(heat_pdf, bbox_inches="tight")
    fig_heat.savefig(heat_png, bbox_inches="tight", dpi=160)
    plt.close(fig_heat)

    for p in (slope_pdf, slope_png, id_pdf, id_png,
              ood_pdf, ood_png, id_ar_pdf, id_ar_png,
              ood_ar_pdf, ood_ar_png, heat_pdf, heat_png):
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
