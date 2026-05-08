#!/usr/bin/env python3
r"""Generate the main-body in-distribution scaling LaTeX table.

Reads eval_multigame.npz from the three rule-conditional cond_match
runs (Train-14, Train-59, Train-199) — the same checkpoints whose OOD
numbers populate the cond-vs-uncond match table — and emits a row per
preset showing per-cell rollout error under each of the four regimes
(random AR / random TF / BFS oracle / A* oracle).

Output: nca_wm/paper/figures/indist_scaling/{table.tex, summary.json}

Usage:
    .venv/bin/python3 nca_wm/scripts/collate_indist_scaling_table.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / "nca_wm" / "logs"
OUT_DIR = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_scaling"

# (seed_dirs, preset_label, n_games, model_label) — rows follow this
# order. Each cell may carry multiple seed dirs; per-cell aggregates
# (mean / median over each game's authored levels) are averaged across
# seeds whose eval_multigame.npz exists, and ddof=1 stds are reported
# in cells with N>=2 seeds. Conditional and unconditional runs at each
# preset are paired so the rule-encoder contribution reads down each
# column. Missing eval data renders as "--" (the in-flight uncond runs
# land here once they finish).
RUNS = [
    (["multi_scaling_14_uncond_match_s0",
      "multi_scaling_14_uncond_match_s1",
      "multi_scaling_14_uncond_match_s2",
      "multi_scaling_14_uncond_match_s3"],         "Train-14",  14,  "uncond"),
    (["multi_scaling_14_cond_match_s0",
      "multi_scaling_14_cond_match_s2"],           "Train-14",  14,  "cond"),
    (["multi_scaling_gallery_v2_uncond_match_s0",
      "multi_scaling_gallery_v2_uncond_match_s1"], "Train-59",  59,  "uncond"),
    (["multi_scaling_gallery_v2_cond_match_s0",
      "multi_scaling_gallery_v2_cond_match_s1"],   "Train-59",  59,  "cond"),
    (["multi_scaling_gallery_v4_uncond_match_s0",
      "multi_scaling_gallery_v4_uncond_match_s1"], "Train-199", 199, "uncond"),
    (["multi_scaling_gallery_v4_cond_match_s0",
      "multi_scaling_gallery_v4_cond_match_s1"],   "Train-199", 199, "cond"),
    # Parameter-matched uncond legs (n_hid=288 ≈ 16.6M total, slightly
    # above cond at n_hid=256 which is 16.03M). Render as "--" until
    # each row's eval_multigame.npz lands.
    (["multi_scaling_14_uncond_match_s0_h288"],         "Train-14",  14,  r"uncond ($n_h{=}288$)"),
    (["multi_scaling_gallery_v2_uncond_match_s0_h288"], "Train-59",  59,  r"uncond ($n_h{=}288$)"),
    (["multi_scaling_gallery_v4_uncond_match_s0_h288"], "Train-199", 199, r"uncond ($n_h{=}288$)"),
]

REGIMES = [
    ("random",    "random (AR)"),
    ("random_tf", "1-step (TF)"),
    ("bfs",       "BFS (AR)"),
    ("astar",     "A* (AR)"),
]


_KEY_RE = re.compile(
    r"^(?P<game>.+?)_L(?P<level>\d+)_(?P<algo>random_tf|random|bfs|astar)_cell_error_rate$"
)


def _aggregate(eval_npz: Path) -> dict:
    """Per-algo {n_games, mean, median} on the games this single seed
    evaluated."""
    if not eval_npz.exists():
        raise FileNotFoundError(eval_npz)
    d = np.load(eval_npz, allow_pickle=True)
    per_algo: dict[str, dict[str, list[float]]] = {a: {} for a, _ in REGIMES}
    for key in d.files:
        m = _KEY_RE.match(key)
        if m is None:
            continue
        algo = m.group("algo")
        if algo not in per_algo:
            continue
        arr = np.asarray(d[key], dtype=np.float64).ravel()
        if arr.size == 0:
            continue
        per_algo[algo].setdefault(m.group("game"), []).append(float(arr.mean()))
    out: dict[str, dict | None] = {}
    for algo, _ in REGIMES:
        per_game_means = [float(np.mean(v)) for v in per_algo[algo].values()]
        if not per_game_means:
            out[algo] = None
        else:
            out[algo] = {
                "n_games": len(per_game_means),
                "mean": float(np.mean(per_game_means)),
                "median": float(np.median(per_game_means)),
            }
    return out


def _aggregate_cell(seed_dirs: list[str]) -> tuple[dict, dict, list[str]]:
    """Aggregate `_aggregate` across seeds. Returns:
        (means_per_algo, stds_per_algo, used_seed_dirs)
    where each *_per_algo[algo] is the dict {n_games, mean, median}
    averaged across seeds (mean and median fields are seed-averages),
    and stds_per_algo[algo] = {mean_std, median_std} with None when
    fewer than two seeds have the metric."""
    per_seed_results: list[dict] = []
    used: list[str] = []
    for sd in seed_dirs:
        npz = LOGS_ROOT / sd / "eval_multigame.npz"
        if not npz.exists():
            continue
        per_seed_results.append(_aggregate(npz))
        used.append(sd)
    means_per_algo: dict[str, dict | None] = {}
    stds_per_algo: dict[str, dict] = {}
    for algo, _ in REGIMES:
        per_seed_mean = [r[algo]["mean"] for r in per_seed_results
                         if r.get(algo) is not None]
        per_seed_median = [r[algo]["median"] for r in per_seed_results
                           if r.get(algo) is not None]
        if not per_seed_mean:
            means_per_algo[algo] = None
            stds_per_algo[algo] = {"mean_std": None, "median_std": None}
            continue
        # Use the first seed's n_games as the canonical count (all seeds
        # train on the same preset, so this should match across seeds).
        n_games = next(
            r[algo]["n_games"] for r in per_seed_results
            if r.get(algo) is not None
        )
        means_per_algo[algo] = {
            "n_games": n_games,
            "n_seeds": len(per_seed_mean),
            "mean":    float(np.mean(per_seed_mean)),
            "median":  float(np.mean(per_seed_median)),
        }
        stds_per_algo[algo] = {
            "mean_std":   float(np.std(per_seed_mean, ddof=1))   if len(per_seed_mean) >= 2 else None,
            "median_std": float(np.std(per_seed_median, ddof=1)) if len(per_seed_median) >= 2 else None,
        }
    return means_per_algo, stds_per_algo, used


def _fmt(stat: dict | None, std: dict | None = None) -> str:
    """Render `mean / median` (no std) or `mean$\\pm$std / median$\\pm$std`
    when stds are present."""
    if stat is None:
        return "--"
    m = f"{100*stat['mean']:.2f}"
    md = f"{100*stat['median']:.2f}"
    if std is not None:
        if std.get("mean_std") is not None:
            m = m + rf"$\pm${100*std['mean_std']:.2f}"
        if std.get("median_std") is not None:
            md = md + rf"$\pm${100*std['median_std']:.2f}"
    return f"{m} / {md}"


def write_table(rows: list[tuple[str, int, str, dict, dict, int]],
                 out_path: Path) -> None:
    """Emit a tabular wrapped in adjustbox so it shrinks to \\linewidth.

    Two leading columns: training preset (\\multirow-grouped, with
    n_games annotation) and model variant (cond / uncond, suffixed
    with (N=k) when k>=2). Remaining columns are the four rollout
    regimes, mean / median (%) decorated with $\\pm$std where std is
    known (N>=2 seeds).
    """
    col_spec = "l l " + " ".join(["c"] * len(REGIMES))
    lines = [
        "% AUTOGENERATED by nca_wm/scripts/collate_indist_scaling_table.py — do not edit.",
        "\\begin{adjustbox}{max width=\\linewidth}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "  \\toprule",
        "  Preset & Model & " + " & ".join(label for _, label in REGIMES) + r" \\",
        "  & & " + " & ".join([r"\small mean / median (\%)"] * len(REGIMES)) + r" \\",
        "  \\midrule",
    ]
    # Group consecutive rows with the same preset; emit \multirow for the
    # first row of each group and a midrule between groups.
    groups: list[list[tuple[str, int, str, dict, dict, int]]] = []
    for r in rows:
        preset = r[0]
        if groups and groups[-1][0][0] == preset:
            groups[-1].append(r)
        else:
            groups.append([r])
    for gi, group in enumerate(groups):
        if gi > 0:
            lines.append("  \\midrule")
        size = len(group)
        preset = group[0][0]
        preset_cell = (
            f"\\multirow{{{size}}}{{*}}{{\\textsc{{{preset}}}}}"
            if size > 1
            else f"\\textsc{{{preset}}}"
        )
        for ri, (_, _, model, results, stds, n_seeds) in enumerate(group):
            cells = [_fmt(results.get(algo), stds.get(algo))
                     for algo, _ in REGIMES]
            first = preset_cell if ri == 0 else ""
            model_cell = (model + (rf" ($N{{=}}{n_seeds}$)" if n_seeds >= 2 else ""))
            lines.append("  " + " & ".join([first, model_cell] + cells) + r" \\")
    lines += ["  \\bottomrule", "\\end{tabular}", "\\end{adjustbox}", ""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, str, dict, dict, int]] = []
    summary: dict[str, dict] = {}
    for seed_dirs, preset, n_games, model in RUNS:
        canonical = seed_dirs[0]
        results, stds, used = _aggregate_cell(seed_dirs)
        n_seeds = len(used)
        label = f"{preset} {model}"
        if n_seeds == 0:
            print(f"[indist-scaling] no seed for {seed_dirs[0]}; rendering as '--'")
            rows.append((preset, n_games, model,
                         {a: None for a, _ in REGIMES},
                         {a: {"mean_std": None, "median_std": None} for a, _ in REGIMES},
                         0))
            summary[canonical] = {"preset": preset, "model": model,
                                   "results": None, "n_seeds": 0,
                                   "used_seeds": []}
            continue
        rows.append((preset, n_games, model, results, stds, n_seeds))
        summary[canonical] = {
            "preset": preset, "model": model,
            "results": results, "stds": stds,
            "n_seeds": n_seeds, "used_seeds": used,
            "seed_dirs": seed_dirs,
        }
        for algo, alabel in REGIMES:
            r = results.get(algo)
            s = stds.get(algo) or {}
            if r is None:
                print(f"  {label:18s} {alabel:24s} no data")
            else:
                std_m = s.get("mean_std")
                std_md = s.get("median_std")
                std_str = ""
                if std_m is not None:
                    std_str = f"  mean_std={100*std_m:.2f}% median_std={100*std_md:.2f}%"
                print(f"  {label:18s} {alabel:24s} n={r['n_games']:3d}  "
                      f"seeds={r['n_seeds']}  "
                      f"mean={100*r['mean']:.2f}%  median={100*r['median']:.2f}%{std_str}")
    if not rows:
        print("[indist-scaling] no runs found; nothing written")
        return
    write_table(rows, OUT_DIR / "table.tex")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote: {OUT_DIR / 'table.tex'}")
    print(f"       {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
