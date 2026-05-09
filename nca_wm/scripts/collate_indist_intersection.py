#!/usr/bin/env python3
r"""In-distribution dilution table on the 14-game intersection.

Holds the 14 Train-14 games fixed and reports how cell-error on those
same 14 games shifts as the training corpus grows from 14 -> 59 -> 199.
The Train-14 game set is a subset of Train-59 and Train-199, so the
intersection is exactly the Train-14 games (asserted at runtime).

Reads each run's eval_multigame.npz and emits:
  nca_wm/paper/figures/indist_intersection/{table.tex, summary.json}

Run:
    .venv/bin/python3 nca_wm/scripts/collate_indist_intersection.py

Re-run once a missing eval_multigame.npz lands (e.g. the in-flight
multi_scaling_gallery_v4_uncond_match_s0). Missing runs render as "--".
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / "nca_wm" / "logs"
OUT_DIR = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_intersection"

# Column layout: ascending corpus size, with cond/uncond paired at each scale.
# Each cell may contain multiple seed dirs; per-game numbers are averaged
# across whichever seeds have eval_multigame.npz present. The first dir is
# the canonical run referenced elsewhere (e.g. as the table's run_dir).
RUNS = [
    ("Train-14",  "cond",   [
        "multi_scaling_14_cond_match_s0",
        "multi_scaling_14_cond_match_s2",
    ]),
    ("Train-14",  "uncond", [
        # Parameter-matched uncond: n_hid=288 ≈ 16.6M total params,
        # slightly above cond@n_hid=256's 16.03M. Honest comparison
        # against cond. The non-param-matched runs (n_hid=256, ~13M)
        # remain on disk but are no longer plotted/tabled.
        "multi_scaling_14_uncond_match_s0_h288",
    ]),
    ("Train-59",  "cond",   [
        "multi_scaling_gallery_v2_cond_match_s0",
        "multi_scaling_gallery_v2_cond_match_s1",
    ]),
    ("Train-59",  "uncond", [
        "multi_scaling_gallery_v2_uncond_match_s0_h288",
    ]),
    ("Train-199", "cond",   [
        "multi_scaling_gallery_v4_cond_match_s0",
        "multi_scaling_gallery_v4_cond_match_s1",
        "multi_scaling_gallery_v4_cond_match_s2",
    ]),
    ("Train-199", "uncond", [
        "multi_scaling_gallery_v4_uncond_match_s0_h288",
    ]),
]

# Reported regimes (matching collate_indist_scaling_table.py).
REGIMES = [
    ("random_tf", "1-step (TF)"),
    ("bfs",       "BFS (AR)"),
    ("random",    "random (AR)"),
]
PRIMARY_REGIME = "random_tf"  # the per-game-rows table reports this regime

KEY_RE = re.compile(
    r"^(?P<game>.+?)_L(?P<level>\d+)_(?P<algo>random_tf|random|bfs|astar)_cell_error_rate$"
)


def _per_game(eval_npz: Path) -> dict[str, dict[str, float]]:
    """{game: {algo: mean over levels}} from a single eval_multigame.npz."""
    if not eval_npz.exists():
        return {}
    d = np.load(eval_npz, allow_pickle=True)
    accum: dict[str, dict[str, list[float]]] = {}
    for k in d.files:
        m = KEY_RE.match(k)
        if m is None:
            continue
        algo = m.group("algo")
        arr = np.asarray(d[k], dtype=np.float64).ravel()
        if arr.size == 0:
            continue
        accum.setdefault(m.group("game"), {}).setdefault(algo, []).append(float(arr.mean()))
    return {g: {a: float(np.mean(v)) for a, v in algos.items()} for g, algos in accum.items()}


def _fmt_pct(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{100*x:.3f}"


def _fmt_pct_pm(mean: float | None, std: float | None) -> str:
    """`mean` (no decoration) or `mean$\\pm$std` when std is known.
    Both rendered in the same percent units used by `_fmt_pct`."""
    if mean is None or not np.isfinite(mean):
        return "--"
    base = f"{100*mean:.3f}"
    if std is None or not np.isfinite(std):
        return base
    return rf"{base}$\pm${100*std:.3f}"


def _per_run_mean_across_seeds(
    seed_dirs: list[str],
) -> tuple[
    dict[str, dict[str, float]],
    list[str],
    list[dict[str, dict[str, float]]],
]:
    """Average per-game per-algo numbers across seeds whose
    eval_multigame.npz is present. Returns (merged_per_game, used_seeds,
    per_seed_per_game). The third return preserves each seed's per-game
    table so the caller can compute std-across-seeds of any aggregate
    they construct (mean over 14 games, median over 14 games, etc.)."""
    used: list[str] = []
    per_seed: list[dict[str, dict[str, float]]] = []
    for sd in seed_dirs:
        eval_npz = LOGS_ROOT / sd / "eval_multigame.npz"
        if eval_npz.exists():
            per_seed.append(_per_game(eval_npz))
            used.append(sd)
    if not per_seed:
        return {}, used, []
    games = sorted(set().union(*[set(d.keys()) for d in per_seed]))
    merged: dict[str, dict[str, float]] = {}
    for g in games:
        algos = sorted(set().union(*[set(d.get(g, {}).keys()) for d in per_seed]))
        merged[g] = {}
        for a in algos:
            vs = [d[g][a] for d in per_seed if a in d.get(g, {})]
            if vs:
                merged[g][a] = float(np.mean(vs))
    return merged, used, per_seed


def _per_seed_aggregate(
    per_seed: list[dict[str, dict[str, float]]],
    games: list[str],
    algo: str,
    stat: str,
) -> list[float]:
    """For each seed compute `stat` (mean / median) over `games` of `algo`.
    Drops seeds that don't cover every requested game (so std is computed
    only on a like-for-like basis)."""
    out: list[float] = []
    for d in per_seed:
        vals = [d.get(g, {}).get(algo) for g in games]
        if any(v is None for v in vals):
            continue
        if stat == "mean":
            out.append(float(np.mean(vals)))
        elif stat == "median":
            out.append(float(np.median(vals)))
        else:
            raise ValueError(f"unknown stat {stat}")
    return out


def _std_or_none(vals: list[float]) -> float | None:
    """Sample std (ddof=1); returns None when fewer than two seeds."""
    if len(vals) < 2:
        return None
    return float(np.std(vals, ddof=1))


def write_table(rows_by_game: list[tuple[str, dict]], aggregates: dict, out_path: Path) -> None:
    """Emit per-game-rows table for the primary regime + aggregate footer.

    Columns: game | T14 cond | T59 cond | T199 cond | T14 uncond | T59 uncond | T199 uncond.
    Aggregate footer rows: mean, median (per-game then averaged across the 14 games).
    """
    # Each cell-key in rows_by_game/aggregates is the FIRST seed dir,
    # used as the canonical identifier for the cell.
    col_spec = "l " + " ".join(["c"] * len(RUNS))

    def cell(d: dict, run_dir: str) -> str:
        return _fmt_pct(d.get(run_dir))

    # The std `±` decorations on the mean / median rows already signal
    # which cells aggregated across multiple seeds; we keep the column
    # headers compact and let the figure caption mention seed counts.
    headers = ["Game"] + [f"{p} {m}" for p, m, _seeds in RUNS]

    lines = [
        "% AUTOGENERATED by nca_wm/scripts/collate_indist_intersection.py -- do not edit.",
        "% Reported regime: 1-step (TF) cell-error, %% per cell, mean across each game's authored levels.",
        "\\begin{adjustbox}{max width=\\linewidth}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "  \\toprule",
        "  " + " & ".join(headers) + r" \\",
        "  \\midrule",
    ]
    for game, row in rows_by_game:
        cells = [cell(row, seeds[0]) for _, _, seeds in RUNS]
        safe_game = game.replace("_", r"\_")
        lines.append("  " + " & ".join([safe_game] + cells) + r" \\")
    lines.append("  \\midrule")
    for stat in ("mean", "median"):
        std_key = f"{stat}_std"
        cells = [
            _fmt_pct_pm(
                aggregates[stat][m].get(seeds[0]),
                aggregates[std_key][m].get(seeds[0]),
            )
            for _, m, seeds in RUNS
        ]
        lines.append("  " + " & ".join([f"\\textbf{{{stat}}}"] + cells) + r" \\")
    lines += ["  \\bottomrule", "\\end{tabular}", "\\end{adjustbox}", ""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all runs (averaged across seeds where multiple are present).
    per_run: dict[str, dict[str, dict[str, float]]] = {}
    per_run_per_seed: dict[str, list[dict[str, dict[str, float]]]] = {}
    used_seeds: dict[str, list[str]] = {}
    available_cells: list[str] = []
    missing_cells: list[str] = []
    for _, _, seed_dirs in RUNS:
        canonical = seed_dirs[0]
        merged, used, per_seed = _per_run_mean_across_seeds(seed_dirs)
        per_run[canonical] = merged
        per_run_per_seed[canonical] = per_seed
        used_seeds[canonical] = used
        if merged:
            available_cells.append(canonical)
        else:
            missing_cells.append(canonical)

    # Train-14 cond is the anchor: its 14 games define the intersection.
    anchor = "multi_scaling_14_cond_match_s0"
    if anchor not in available_cells:
        raise SystemExit(f"anchor run {anchor} has no eval_multigame.npz; cannot define intersection")
    intersection = sorted(per_run[anchor].keys())

    # Cross-check that every Train-59 / Train-199 cond run that exists
    # contains the Train-14 games as a subset.
    for run in ("multi_scaling_gallery_v2_cond_match_s0",
                "multi_scaling_gallery_v4_cond_match_s0"):
        if run not in available_cells:
            continue
        missing_in_super = [g for g in intersection if g not in per_run[run]]
        assert not missing_in_super, f"{run} missing games from Train-14: {missing_in_super}"

    # Build per-game rows for the primary regime.
    rows_by_game: list[tuple[str, dict]] = []
    for game in intersection:
        row = {}
        for _, _, seeds in RUNS:
            canonical = seeds[0]
            v = per_run[canonical].get(game, {}).get(PRIMARY_REGIME)
            if v is not None:
                row[canonical] = v
        rows_by_game.append((game, row))

    # Per-run aggregates (mean / median across the 14 intersection games)
    # for every regime, plus std-across-seeds of those same aggregates.
    aggregates: dict[str, dict[str, dict[str, float]]] = {
        "mean":     {"cond": {}, "uncond": {}},
        "median":   {"cond": {}, "uncond": {}},
        "mean_std":   {"cond": {}, "uncond": {}},
        "median_std": {"cond": {}, "uncond": {}},
    }
    summary: dict[str, dict] = {}
    for preset, model, seeds in RUNS:
        canonical = seeds[0]
        run_summary = {
            "preset": preset, "model": model,
            "available": canonical in available_cells,
            "seed_dirs": seeds,
            "used_seeds": used_seeds[canonical],
            "n_seeds": len(used_seeds[canonical]),
        }
        run_summary["regimes"] = {}
        for algo, _label in REGIMES:
            vals = []
            for g in intersection:
                v = per_run[canonical].get(g, {}).get(algo)
                if v is not None:
                    vals.append(v)
            if vals and len(vals) == len(intersection):
                # Cross-seed stds: for each seed compute the same aggregate
                # over the 14 intersection games, then take ddof=1 std.
                seed_means = _per_seed_aggregate(
                    per_run_per_seed[canonical], intersection, algo, "mean")
                seed_medians = _per_seed_aggregate(
                    per_run_per_seed[canonical], intersection, algo, "median")
                run_summary["regimes"][algo] = {
                    "n_games":    len(vals),
                    "n_seeds":    len(seed_means),
                    "mean":       float(np.mean(vals)),
                    "median":     float(np.median(vals)),
                    "mean_std":   _std_or_none(seed_means),
                    "median_std": _std_or_none(seed_medians),
                    "per_seed_mean":   seed_means,
                    "per_seed_median": seed_medians,
                }
            else:
                run_summary["regimes"][algo] = None
        # Primary regime aggregates feed the table footer.
        head = run_summary["regimes"].get(PRIMARY_REGIME)
        if head is not None:
            aggregates["mean"][model][canonical]       = head["mean"]
            aggregates["median"][model][canonical]     = head["median"]
            aggregates["mean_std"][model][canonical]   = head["mean_std"]
            aggregates["median_std"][model][canonical] = head["median_std"]
        summary[canonical] = run_summary

    write_table(rows_by_game, aggregates, OUT_DIR / "table.tex")
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "intersection_games": intersection,
        "primary_regime":     PRIMARY_REGIME,
        "runs":               summary,
    }, indent=2))

    # Console report.
    print(f"Intersection: {len(intersection)} games (anchor = {anchor}).")
    print(f"Available cells: {len(available_cells)} / {len(RUNS)}")
    if missing_cells:
        print(f"Cells with no seed available (rendered as '--'):")
        for r in missing_cells:
            print(f"  {r}")
    print()
    print(f"Primary regime ({PRIMARY_REGIME}) aggregates on the {len(intersection)}-game intersection:")
    print(f"  {'preset/model':<24s}  {'n_seeds':>7s}  {'mean':>9s}  {'median':>9s}")
    for preset, model, seeds in RUNS:
        canonical = seeds[0]
        head = summary[canonical]["regimes"].get(PRIMARY_REGIME)
        n = summary[canonical]["n_seeds"]
        if head is None:
            print(f"  {preset+' '+model:<24s}  {n:>7d}  {'--':>9s}  {'--':>9s}")
        else:
            print(f"  {preset+' '+model:<24s}  {n:>7d}  {100*head['mean']:>8.3f}%  {100*head['median']:>8.3f}%")
    print()
    print(f"Wrote: {OUT_DIR/'table.tex'}")
    print(f"       {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
