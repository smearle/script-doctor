#!/usr/bin/env python3
r"""Generate the architecture-baseline LaTeX tables (paper Tables 1-4).

Reads nca_wm/paper/figures/baselines/summary.csv (a snapshot of the
aggregator output from logs_baselines/, currently produced on box 210
by nca_wm/scripts/aggregate_baseline_comparison.py) and emits one
LaTeX fragment per table:

  * sokoban_authored.tex   - Table 1 (single-game Sokoban / Microban)
  * sokoban_synth.tex      - Table 2 (multi-grid synth Sokoban)
  * ablation_pool.tex      - Table 3 (per-feature pool / input-skip)
  * scaling_14.tex         - Table 4 (multi-game Train-14)

To refresh the CSV from box 210:

  ssh 210 'cd /home/jupyter-earle/script-doctor &&
           .venv/bin/python3 nca_wm/scripts/aggregate_baseline_comparison.py'
  scp 210:/home/jupyter-earle/script-doctor/nca_wm/logs_baselines/summary.csv \\
      nca_wm/paper/figures/baselines/summary.csv

then re-run this script.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_CSV = REPO_ROOT / "nca_wm" / "paper" / "figures" / "baselines" / "summary.csv"
OUT_DIR = REPO_ROOT / "nca_wm" / "paper" / "figures" / "baselines"

# Maps the CSV's `arch` column to a paper-side display name. Used by all
# four tables; the order in each table is given separately by `arch_order`
# below (which the table sorts further by BFS final-err).
ARCH_DISPLAY = {
    "nca_shared":  r"\textbf{NCA (shared, $T{=}4, n_r{=}T$)}",
    "nca_perstep": "NCA (per-step, $T{=}4, n_r{=}1$)",
    "cnn_d4":      "CNN (4-block ResNet)",
    "unet_l2":     "U-Net (2 levels)",
    "vit_l4":      "ViT (4 layers)",
}

# Per-feature ablation rows. Order follows the paper Table 3.
ABLATION_DISPLAY = {
    "nca_shared":         (r"\textbf{Full (axis + cummax + global + input-skip)}", True),
    "nca_shared_noac":    ("no axis-cummax",                                       False),
    "nca_shared_nois":    ("no input-skip",                                        False),
    "nca_shared_noap":    ("no axis-pool",                                         False),
    "nca_shared_nogp":    ("no global-pool",                                       False),
    "nca_shared_nopool":  ("no pool at all",                                       False),
}


def _load_rows() -> list[dict]:
    if not SRC_CSV.exists():
        raise FileNotFoundError(
            f"missing {SRC_CSV}; refresh from box 210 (see docstring)"
        )
    with SRC_CSV.open() as f:
        return list(csv.DictReader(f))


def _fnum(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _aggregate(rows: list[dict], games: str, arch: str) -> dict:
    """Mean / std over seeds for a (games, arch) cell."""
    seeds = [r for r in rows if r["games"] == games and r["arch"] == arch]
    if not seeds:
        return {}
    n_params = next((int(r["n_params"]) for r in seeds if r.get("n_params")), None)
    out = {"n_params": n_params, "n_seeds": len(seeds)}
    for k in (
        "random_tf_rollout_cellerr",
        "bfs_rollout_cellerr",
    ):
        vals = [_fnum(r, k) for r in seeds]
        vals = [v for v in vals if v is not None]
        if vals:
            # ddof=0 (population std) — np.std default; matches the
            # numbers originally generated for the paper.
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _fmt_pct(stat: tuple[float, float] | None, bold: bool = False) -> str:
    if stat is None:
        return "--"
    m, s = stat
    # Use an extra decimal place when the mean is sub-1%, so values like
    # 0.002% don't round to "0.00%" alongside 0.005% etc.
    nd = 3 if m < 0.01 else 2
    inner = f"{100*m:.{nd}f}\\% \\pm {100*s:.{nd}f}\\%"
    if bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$"


def _fmt_params(n_params: int | None) -> str:
    if n_params is None:
        return "--"
    return f"{n_params/1e6:.2f}"


def _table_baseline(
    rows: list[dict],
    games: str,
    arch_order: list[str],
    out_path: Path,
) -> None:
    """Emit a 5-architecture table sorted by BFS-AR cell-error ascending.

    Columns: arch | params (M) | one-step (TF) cell-err | BFS (AR) cell-err.
    Both error columns are per-cell, averaged across rollout steps; matches
    the metric definitions in \\S 4.4 and the indist / cond-vs-uncond tables.
    """
    aggs = {a: _aggregate(rows, games, a) for a in arch_order}
    aggs = {a: v for a, v in aggs.items() if v}
    if not aggs:
        print(f"[baselines] no data for games={games}; skipped {out_path.name}")
        return
    # Sort by BFS rollout cell-err ascending; missing → infinity.
    def _bfs(a: str) -> float:
        s = aggs[a].get("bfs_rollout_cellerr")
        return s[0] if s else float("inf")

    sorted_archs = sorted(aggs.keys(), key=_bfs)
    best_arch = sorted_archs[0]
    least_params = min(
        (aggs[a]["n_params"] for a in sorted_archs if aggs[a].get("n_params")),
        default=None,
    )

    body = []
    for a in sorted_archs:
        agg = aggs[a]
        is_best_bfs = a == best_arch
        is_least_params = (
            least_params is not None
            and agg.get("n_params") == least_params
        )
        params_str = _fmt_params(agg.get("n_params"))
        if is_least_params:
            params_str = f"\\textbf{{{params_str}}}"
        body.append(
            "  " + " & ".join([
                ARCH_DISPLAY.get(a, a),
                params_str,
                _fmt_pct(agg.get("random_tf_rollout_cellerr")),
                _fmt_pct(agg.get("bfs_rollout_cellerr"), bold=is_best_bfs),
            ]) + r" \\"
        )
    table = (
        "% AUTOGENERATED by nca_wm/scripts/collate_baseline_tables.py — do not edit.\n"
        "\\begin{tabular}{l c c c}\n"
        "  \\toprule\n"
        "  Architecture & params (M) & one-step (TF) & BFS (AR) \\\\\n"
        "  \\midrule\n"
        + "\n".join(body)
        + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )
    out_path.write_text(table)
    print(f"[baselines] wrote {out_path}")


def _table_ablation(rows: list[dict], out_path: Path) -> None:
    """Emit Table 3: per-feature ablation on sokoban_basic_synth (multi-grid).

    Baseline row first (full recipe), then ablations sorted by Δ BFS ascending.
    """
    games = "sokoban_basic_synth"
    aggs: dict[str, dict] = {}
    for arch in ABLATION_DISPLAY:
        a = _aggregate(rows, games, arch)
        if a:
            aggs[arch] = a
    if "nca_shared" not in aggs:
        print(f"[ablation] missing baseline (full nca_shared) row; skipped")
        return
    full = aggs["nca_shared"]
    full_bfs = full["bfs_rollout_cellerr"][0]
    ablations = [a for a in aggs if a != "nca_shared"]
    ablations.sort(
        key=lambda a: aggs[a]["bfs_rollout_cellerr"][0] - full_bfs
    )

    def _row(arch: str) -> str:
        agg = aggs[arch]
        label, is_baseline = ABLATION_DISPLAY[arch]
        bfs = agg["bfs_rollout_cellerr"][0]
        delta_abs = (bfs - full_bfs) * 100
        rel = (bfs - full_bfs) / full_bfs if full_bfs > 0 else float("nan")
        if is_baseline:
            delta_str = "---"
        else:
            sign = "+" if delta_abs >= 0 else ""
            delta_str = f"${sign}{delta_abs:.2f}$ ({sign}{100*rel:.0f}\\%)"
        bfs_pct = _fmt_pct(agg["bfs_rollout_cellerr"], bold=is_baseline)
        return "  " + " & ".join([
            label,
            _fmt_params(agg.get("n_params")),
            _fmt_pct(agg.get("random_tf_rollout_cellerr")),
            bfs_pct,
            delta_str,
        ]) + r" \\"

    body_lines = [_row("nca_shared"), "  \\midrule"]
    for a in ablations:
        body_lines.append(_row(a))
    table = (
        "% AUTOGENERATED by nca_wm/scripts/collate_baseline_tables.py — do not edit.\n"
        "\\begin{tabular}{l c c c c}\n"
        "  \\toprule\n"
        "  Recipe & params (M) & one-step (TF) & BFS (AR) & $\\Delta$BFS vs.\\ full \\\\\n"
        "  \\midrule\n"
        + "\n".join(body_lines)
        + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )
    out_path.write_text(table)
    print(f"[ablation] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()

    arch_order = ["nca_shared", "nca_perstep", "cnn_d4", "unet_l2", "vit_l4"]

    _table_baseline(
        rows,
        games="microban_authored",
        arch_order=arch_order,
        out_path=OUT_DIR / "sokoban_authored.tex",
    )
    _table_baseline(
        rows,
        games="sokoban_basic_synth",
        arch_order=arch_order,
        out_path=OUT_DIR / "sokoban_synth.tex",
    )
    _table_baseline(
        rows,
        games="scaling_14",
        arch_order=arch_order,
        out_path=OUT_DIR / "scaling_14.tex",
    )
    _table_ablation(rows, OUT_DIR / "ablation_pool.tex")


if __name__ == "__main__":
    main()
