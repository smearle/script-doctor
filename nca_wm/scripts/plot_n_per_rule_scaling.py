#!/usr/bin/env python3
"""Two-panel data-scaling slope figure for the structured n_per_rule sweep.

This is the n_per_rule analogue of plot_indist_intersection.py's two-panel
slope figure. Instead of three coarse presets (Train-14/59/199), the x-axis
is the structured corpus-size sweep n_per_rule in
{1,2,3,5,10,20,50,100,200,(400)} -- the number of mechanically distinct
games (stratified across rule counts) the model is trained on.

  Left  (in-distribution): mean/median TF cell-error over each scale's own
    training games (from eval_multigame.npz). As the corpus widens at fixed
    capacity, in-distribution fit dilutes; rule-conditioning slows the
    dilution relative to a parameter-matched unconditional model.

  Right (out-of-distribution): mean/median TF cell-error on Heldout-26
    (the common (game, level) pairs scored by every run in
    heldout_v4_n30/results.json), with the identity baseline as a reference.
    Both models improve toward identity-level error as the corpus widens.

Outputs (paper/figures/n_per_rule_scaling/):
  n_per_rule_slope.{pdf,png}      -- combined two-panel figure
  n_per_rule_slope_id.{pdf,png}   -- in-distribution panel (subfigure)
  n_per_rule_slope_ood.{pdf,png}  -- out-of-distribution panel (subfigure)

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_n_per_rule_scaling.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR   = REPO_ROOT / "nca_wm" / "logs"
OUT_DIR   = REPO_ROOT / "nca_wm" / "paper" / "figures" / "n_per_rule_scaling"

# Corpus-size sweep (games per rule count). cond runs to 400, uncond to 200.
NS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 400]

# (model_key, label, color); colors match plot_indist_intersection.py:
# cond red, uncond blue.
SERIES = [
    ("cond",   "Rule-conditioned", "#d62728"),
    ("uncond", "Unconditional",    "#1f77b4"),
]

# Run-dir base per (model, n); we glob "<base>_s*" so every available seed of
# the plain cond/uncond dp variants is picked up automatically. The base
# stops before "_s" so the nosprites/objperm variants (which have a different
# token after "_dp") are never matched.
RUN_BASE = {
    "cond":   "n_per_rule_{n}_cond_dp_val0.10",
    "uncond": "n_per_rule_{n}_uncond_h288_dp_val0.10",
}


def _seed_run_dirs(model: str, n: int) -> list[str]:
    """All completed seed run-dir names for (model, n), sorted by seed."""
    base = RUN_BASE[model].format(n=n)
    dirs = []
    for d in sorted(LOG_DIR.glob(base + "_s*")):
        if (d / "eval_multigame.npz").exists():
            dirs.append(d.name)
    return dirs

# Floor (in %) for the log axes; exact-zero perfect-fit cells clip here.
FLOOR_PCT = 1e-3

# Bootstrap resamples for the across-game / across-pair CI on the mean.
N_BOOT = 2000
BOOT_SEED = 0

# In-distribution identity baseline (per-cell churn), cached per game by
# compute_n_per_rule_id_identity.py.
ID_IDENTITY_JSON = OUT_DIR / "id_identity.json"

# No-rule baseline (force-driven player movement with collision, empty RULES
# section), cached by compute_n_per_rule_id_norule.py (per training game) and
# compute_n_per_rule_ood_norule.py (per Heldout-26 (game, level) pair).
ID_NORULE_JSON  = OUT_DIR / "id_norule.json"
OOD_NORULE_JSON = OUT_DIR / "ood_norule.json"

# Baseline reference styling (identity: black dotted; no-rule: brown dash-dot).
IDENTITY_COLOR = "black"
NORULE_COLOR   = "#8c564b"
IDENTITY_LABEL = "identity (copy last state)"
NORULE_LABEL   = "no-rule (force-only movement)"

# Shared y-axis label: cell_error_rate = wrong_cells / (H*W) per step, i.e.
# the fraction of grid cells whose predicted object-stack differs from ground
# truth (a cell counts as wrong if ANY object channel is wrong), averaged over
# rollout steps and over games/levels. Per-cell, not per-state Hamming.
YLABEL = "Per-cell error rate (%)\n(grid cells mispredicted)"

RC_PARAMS = {
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
}

_TF_PAT = re.compile(r"^(?P<game>.+?)_L\d+_random_tf_cell_error_rate$")


# --------------------------------------------------------------------------- #
# In-distribution: mean/median TF cell-error over each scale's training games.
# --------------------------------------------------------------------------- #
def _id_per_game(run: str) -> dict[str, float]:
    """Per-game TF cell-error (mean over levels) from eval_multigame.npz."""
    npz = LOG_DIR / run / "eval_multigame.npz"
    if not npz.exists():
        return {}
    d = np.load(npz, allow_pickle=True)
    accum: dict[str, list[float]] = {}
    for k in d.files:
        m = _TF_PAT.match(k)
        if not m:
            continue
        arr = np.asarray(d[k], dtype=np.float64).ravel()
        if arr.size == 0:
            continue
        accum.setdefault(m.group("game"), []).append(float(arr.mean()))
    return {g: float(np.mean(v)) for g, v in accum.items()}


def _id_aggregates() -> tuple[dict[str, dict[int, np.ndarray]],
                              dict[str, dict[int, list[str]]],
                              dict[str, dict[int, np.ndarray]]]:
    """({model: {n: seed_avg_per_game_pct}},
        {model: {n: game_names}},
        {model: {n: per_seed_mean_pct}}).

    Per-game values are averaged across available seeds (for the median line
    and across-game band); per_seed_mean is the mean-over-games for each seed
    (for the across-seed error bar).
    """
    vals: dict[str, dict[int, np.ndarray]] = {"cond": {}, "uncond": {}}
    names: dict[str, dict[int, list[str]]] = {"cond": {}, "uncond": {}}
    seedm: dict[str, dict[int, np.ndarray]] = {"cond": {}, "uncond": {}}
    for model in ("cond", "uncond"):
        for n in NS:
            per_seed = [_id_per_game(rd) for rd in _seed_run_dirs(model, n)]
            per_seed = [p for p in per_seed if p]
            if not per_seed:
                continue
            gnames = sorted(set().union(*[set(p) for p in per_seed]))
            pg_avg = [100.0 * np.mean([p[g] for p in per_seed if g in p])
                      for g in gnames]
            seed_means = [100.0 * np.mean(list(p.values())) for p in per_seed]
            vals[model][n] = np.asarray(pg_avg, dtype=np.float64)
            names[model][n] = gnames
            seedm[model][n] = np.asarray(seed_means, dtype=np.float64)
    return vals, names, seedm


def _id_identity_by_n(id_names: dict[str, dict[int, list[str]]]
                      ) -> dict[int, float]:
    """Per-cell identity error (%) averaged over each scale's own training
    games, using the cond game sets (widest coverage). {} if the cache is
    absent."""
    if not ID_IDENTITY_JSON.exists():
        return {}
    per_game = json.loads(ID_IDENTITY_JSON.read_text())
    out: dict[int, float] = {}
    for n in NS:
        gnames = id_names.get("cond", {}).get(n) or id_names.get("uncond", {}).get(n)
        if not gnames:
            continue
        vs = [per_game[g] for g in gnames if g in per_game]
        if vs:
            out[n] = float(np.mean(vs)) * 100.0
    return out


def _id_norule_by_n(id_names: dict[str, dict[int, list[str]]]
                    ) -> dict[int, float]:
    """Per-cell no-rule error (%) averaged over each scale's own training
    games, using the cond game sets (widest coverage). {} if the cache is
    absent."""
    if not ID_NORULE_JSON.exists():
        return {}
    per_game = json.loads(ID_NORULE_JSON.read_text())
    out: dict[int, float] = {}
    for n in NS:
        gnames = id_names.get("cond", {}).get(n) or id_names.get("uncond", {}).get(n)
        if not gnames:
            continue
        vs = [per_game[g] for g in gnames if g in per_game]
        if vs:
            out[n] = float(np.mean(vs)) * 100.0
    return out


# --------------------------------------------------------------------------- #
# Out-of-distribution: mean/median TF cell-error over common Heldout-26 pairs.
# --------------------------------------------------------------------------- #
def _ood_load_pairs(run: str) -> dict[tuple[str, str], tuple[float, float]]:
    """{(game, lvl): (model_tf_mean, identity_tf_mean)} from results.json."""
    p = LOG_DIR / run / "heldout_v4_n30" / "results.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for game, lvls in r["heldout"].items():
        for lvl, rollouts in lvls.items():
            tf = rollouts.get("random_tf")
            if tf is None:
                continue
            out[(game, str(lvl))] = (
                float(tf["model_cell_err_mean"]),
                float(tf["identity_cell_err_mean"]),
            )
    return out


def _ood_norule_flat(common: list[tuple[str, str]]) -> float:
    """Mean per-cell no-rule error (%) over the common Heldout-26 pairs, from
    the ood_norule.json cache. NaN if the cache is absent or covers no common
    pair."""
    if not OOD_NORULE_JSON.exists():
        return float("nan")
    cache = json.loads(OOD_NORULE_JSON.read_text())
    vs = [cache[f"{g}|{lvl}"]["norule"] * 100.0
          for (g, lvl) in common if f"{g}|{lvl}" in cache]
    return float(np.mean(vs)) if vs else float("nan")


def _ood_aggregates() -> tuple[dict[str, dict[int, np.ndarray]], float, float,
                               dict[str, dict[int, np.ndarray]]]:
    """({model: {n: seed_avg_per_pair_pct}}, identity_mean_pct,
        norule_mean_pct, {model: {n: per_seed_mean_pct}}).

    Restricts to the (game, level) pairs scored by *every* loaded run (all
    seeds, all cells), so the aggregate is over a fixed Heldout-26 set. Per
    pair we seed-average; per_seed_mean is the mean over the common pairs for
    each seed (for the across-seed error bar). The no-rule baseline is averaged
    over the same common pairs from the ood_norule.json cache.
    """
    loaded: dict[tuple[str, int], list[dict]] = {}
    for model in ("cond", "uncond"):
        for n in NS:
            seeds = [_ood_load_pairs(rd) for rd in _seed_run_dirs(model, n)]
            seeds = [s for s in seeds if s]
            if seeds:
                loaded[(model, n)] = seeds
    if not loaded:
        return ({"cond": {}, "uncond": {}}, float("nan"), float("nan"),
                {"cond": {}, "uncond": {}})
    all_pair_sets = [set(s.keys()) for seeds in loaded.values() for s in seeds]
    common = sorted(set.intersection(*all_pair_sets))

    ident_vals: list[float] = []
    out: dict[str, dict[int, np.ndarray]] = {"cond": {}, "uncond": {}}
    seedm: dict[str, dict[int, np.ndarray]] = {"cond": {}, "uncond": {}}
    for (model, n), seeds in loaded.items():
        per_pair_avg = [100.0 * np.mean([s[k][0] for s in seeds]) for k in common]
        seed_means = [100.0 * np.mean([s[k][0] for k in common]) for s in seeds]
        out[model][n] = np.asarray(per_pair_avg, dtype=np.float64)
        seedm[model][n] = np.asarray(seed_means, dtype=np.float64)
        ident_vals.extend([seeds[0][k][1] * 100.0 for k in common])
    identity = float(np.mean(ident_vals)) if ident_vals else float("nan")
    norule = _ood_norule_flat(common)
    return out, identity, norule, seedm


def _boot_ci(vals: np.ndarray) -> tuple[float, float, float]:
    """(mean, lo, hi) with a percentile bootstrap 95% CI of the mean across
    the units (games / pairs). Single-unit cells get a degenerate CI at the
    point itself."""
    mean = float(vals.mean())
    if vals.size < 2:
        return mean, mean, mean
    rng = np.random.RandomState(BOOT_SEED)
    idx = rng.randint(0, vals.size, size=(N_BOOT, vals.size))
    boot_means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #
def _draw_panel(ax, agg: dict[str, dict[int, np.ndarray]], *, title: str,
                seedm: dict[str, dict[int, np.ndarray]] | None = None,
                identity_flat: float | None = None,
                identity_by_n: dict[int, float] | None = None,
                norule_flat: float | None = None,
                norule_by_n: dict[int, float] | None = None) -> None:
    """Plot cond/uncond mean (solid) + median (dashed) per corpus size, plus
    an identity reference. The shaded band is the bootstrap 95% CI across the
    aggregation units (games for ID, heldout pairs for OOD). Solid capped
    error bars show the across-SEED spread (min..max of the per-seed
    mean-over-games), drawn only where >=2 seeds exist; the legend reports the
    max seed count so it's explicit how many cells are multi-seed."""
    floor = FLOOR_PCT
    seedm = seedm or {}
    for model, label, color in SERIES:
        series = agg.get(model, {})
        xs = sorted(series.keys())
        if not xs:
            continue
        means, medians, los, his = [], [], [], []
        bar_x, bar_y, bar_lo, bar_hi = [], [], [], []
        max_seeds = 1
        for n in xs:
            vals = series[n]
            mean, lo, hi = _boot_ci(vals)
            means.append(max(mean, floor))
            medians.append(max(float(np.median(vals)), floor))
            los.append(max(lo, floor))
            his.append(max(hi, floor))
            sm = seedm.get(model, {}).get(n)
            if sm is not None and sm.size >= 2:
                max_seeds = max(max_seeds, sm.size)
                m = float(sm.mean())
                bar_x.append(n)
                bar_y.append(max(m, floor))
                bar_lo.append(max(m - sm.min(), 0.0))
                bar_hi.append(sm.max() - m)
        ax.fill_between(xs, los, his, color=color, alpha=0.15, zorder=1)
        ax.plot(xs, means, color=color, linewidth=2.6, marker="o",
                markersize=7, zorder=4, label=f"{label} (mean)")
        ax.plot(xs, medians, color=color, linewidth=1.8, linestyle="--",
                marker="s", markersize=6, zorder=3, label=f"{label} (median)")
        if bar_x:
            # Across-seed range bars; intentionally not in the legend (the
            # standalone caption explains them).
            ax.errorbar(bar_x, bar_y, yerr=[bar_lo, bar_hi], fmt="none",
                        ecolor=color, elinewidth=1.6, capsize=4, capthick=1.6,
                        zorder=5, label="_nolegend_")
    if norule_by_n:
        xs = sorted(norule_by_n.keys())
        ys = [max(norule_by_n[n], floor) for n in xs]
        ax.plot(xs, ys, color=NORULE_COLOR, linewidth=1.6, linestyle="-.",
                marker="^", markersize=5, alpha=0.85, zorder=2,
                label=NORULE_LABEL)
    elif norule_flat is not None and np.isfinite(norule_flat):
        ax.axhline(max(norule_flat, floor), color=NORULE_COLOR, linewidth=1.6,
                   linestyle="-.", alpha=0.85, zorder=2, label=NORULE_LABEL)
    if identity_by_n:
        xs = sorted(identity_by_n.keys())
        ys = [identity_by_n[n] for n in xs]
        ax.plot(xs, ys, color=IDENTITY_COLOR, linewidth=1.6, linestyle=":",
                alpha=0.8, zorder=2,
                label=IDENTITY_LABEL)
    elif identity_flat is not None and np.isfinite(identity_flat):
        ax.axhline(identity_flat, color=IDENTITY_COLOR, linewidth=1.6,
                   linestyle=":", alpha=0.8, zorder=2, label=IDENTITY_LABEL)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlim(0.85, 470)
    ax.set_xlabel("Training corpus (distinct games, log scale)")
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)


def _make_id_panel(ax, id_agg, id_identity_by_n, id_norule_by_n,
                   id_seedm=None, draw_legend=True):
    _draw_panel(ax, id_agg, seedm=id_seedm,
                title="In-distribution (training games)",
                identity_by_n=id_identity_by_n,
                norule_by_n=id_norule_by_n)
    if draw_legend:
        ax.legend(loc="lower left", framealpha=0.95, fontsize=8)


def _make_ood_panel(ax, ood_agg, identity, norule, ood_seedm=None,
                    draw_legend=True):
    _draw_panel(ax, ood_agg, seedm=ood_seedm,
                title="Out-of-distribution (Heldout-26)",
                identity_flat=identity, norule_flat=norule)
    if draw_legend:
        ax.legend(loc="upper right", framealpha=0.95, fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(RC_PARAMS)

    id_agg, id_names, id_seedm = _id_aggregates()
    id_identity_by_n = _id_identity_by_n(id_names)
    id_norule_by_n = _id_norule_by_n(id_names)
    ood_agg, identity, norule, ood_seedm = _ood_aggregates()
    if not id_identity_by_n:
        print("WARNING: id_identity.json missing; run "
              "compute_n_per_rule_id_identity.py for the ID identity baseline.")
    if not id_norule_by_n:
        print("WARNING: id_norule.json missing; run "
              "compute_n_per_rule_id_norule.py for the ID no-rule baseline.")
    if not np.isfinite(norule):
        print("WARNING: ood_norule.json missing; run "
              "compute_n_per_rule_ood_norule.py for the OOD no-rule baseline.")

    # Report seed counts per cell so multi-seed coverage is explicit.
    print("seeds per (model, n):")
    for model in ("cond", "uncond"):
        cnts = {n: len(_seed_run_dirs(model, n)) for n in NS
                if _seed_run_dirs(model, n)}
        print(f"  {model:7s}: " + "  ".join(f"n{n}={c}" for n, c in cnts.items()))

    # Combined two-panel figure with a SINGLE shared legend (the panels have
    # identical series) placed below, centered.
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    _make_id_panel(axl, id_agg, id_identity_by_n, id_norule_by_n, id_seedm,
                   draw_legend=False)
    _make_ood_panel(axr, ood_agg, identity, norule, ood_seedm,
                    draw_legend=False)
    handles, labels = axl.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               framealpha=0.95, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 160})):
        fig.savefig(OUT_DIR / f"n_per_rule_slope{ext}", bbox_inches="tight", **kw)
    plt.close(fig)

    # Single-panel subfigures.
    fig_id, ax_id = plt.subplots(figsize=(5.8, 4.8))
    _make_id_panel(ax_id, id_agg, id_identity_by_n, id_norule_by_n, id_seedm)
    fig_id.tight_layout()
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 160})):
        fig_id.savefig(OUT_DIR / f"n_per_rule_slope_id{ext}", bbox_inches="tight", **kw)
    plt.close(fig_id)

    fig_ood, ax_ood = plt.subplots(figsize=(5.8, 4.8))
    _make_ood_panel(ax_ood, ood_agg, identity, norule, ood_seedm)
    fig_ood.tight_layout()
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 160})):
        fig_ood.savefig(OUT_DIR / f"n_per_rule_slope_ood{ext}", bbox_inches="tight", **kw)
    plt.close(fig_ood)

    # Console summary (mean [95% CI], median).
    print(f"OOD identity (Heldout-26 TF mean): {identity:.3f}%   "
          f"OOD no-rule (Heldout-26 TF mean): {norule:.3f}%")
    print(f"\n{'n':>5} | {'cond mean (CI)':>22} {'cond med':>8} {'ID ident':>8} "
          f"{'ID norule':>9} "
          f"| {'unc mean (CI)':>22} {'unc med':>8} "
          f"|| {'cond OOD (CI)':>22} {'unc OOD (CI)':>22}")
    def fci(d, n):
        if n not in d:
            return f"{'--':>22}"
        m, lo, hi = _boot_ci(d[n])
        return f"{m:7.3f} [{lo:6.3f},{hi:6.3f}]"
    def fmed(d, n):
        return f"{np.median(d[n]):8.3f}" if n in d else f"{'--':>8}"
    for n in NS:
        idi = f"{id_identity_by_n[n]:8.2f}" if n in id_identity_by_n else f"{'--':>8}"
        nri = f"{id_norule_by_n[n]:9.2f}" if n in id_norule_by_n else f"{'--':>9}"
        print(f"{n:>5} | {fci(id_agg['cond'], n)} {fmed(id_agg['cond'], n)} {idi} "
              f"{nri} "
              f"| {fci(id_agg['uncond'], n)} {fmed(id_agg['uncond'], n)} "
              f"|| {fci(ood_agg['cond'], n)} {fci(ood_agg['uncond'], n)}")

    for name in ("n_per_rule_slope", "n_per_rule_slope_id", "n_per_rule_slope_ood"):
        print(f"Wrote: {OUT_DIR / (name + '.pdf')}")


if __name__ == "__main__":
    main()
