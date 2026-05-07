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
        "best_loss",
        "bfs_final_cellerr",
        "astar_final_cellerr",
        "astar_first_div",
        "bfs_first_div",
    ):
        vals = [_fnum(r, k) for r in seeds]
        vals = [v for v in vals if v is not None]
        if vals:
            # ddof=0 (population std) — np.std default; matches the
            # numbers originally generated for the paper.
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _fmt_loss(stat: tuple[float, float] | None) -> str:
    """Format loss: scientific notation for small values, fixed for the rest.

    Train loss on Train-14 is in the 1e-5 - 1e-3 range; Sokoban / Microban
    runs sit at $\\approx 0.1$. Pick a representation per-row.
    """
    if stat is None:
        return "--"
    m, _ = stat
    if m < 1e-4:
        return r"$\approx 0$"
    if m < 1e-2:
        # Scientific: 7.7 \times 10^{-4}
        exp = int(np.floor(np.log10(m)))
        mant = m / (10 ** exp)
        return f"${mant:.1f} \\times 10^{{{exp}}}$"
    return f"${m:.3f} \\pm {stat[1]:.3f}$"


def _fmt_pct(stat: tuple[float, float] | None, bold: bool = False) -> str:
    if stat is None:
        return "--"
    m, s = stat
    inner = f"{100*m:.2f}\\% \\pm {100*s:.2f}\\%"
    if bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$"


def _fmt_first_div(stat: tuple[float, float] | None) -> str:
    if stat is None:
        return "--"
    m, _ = stat
    # Show one decimal unless the mean lands exactly on an integer.
    if m == int(m):
        return f"${int(m)}$"
    return f"${m:.1f}$"


def _fmt_params(n_params: int | None) -> str:
    if n_params is None:
        return "--"
    return f"{n_params/1e6:.2f}"


def _table_baseline(
    rows: list[dict],
    games: str,
    arch_order: list[str],
    out_path: Path,
    *,
    a_metric: str = "first_div",
) -> None:
    """Emit a 5-architecture table sorted by BFS final-err ascending.

    Columns: arch | params (M) | best loss | BFS final-err | A* metric
    where `a_metric` is "first_div" (Tables 1, 2) or "final_err" (Table 4).
    """
    aggs = {a: _aggregate(rows, games, a) for a in arch_order}
    aggs = {a: v for a, v in aggs.items() if v}
    if not aggs:
        print(f"[baselines] no data for games={games}; skipped {out_path.name}")
        return
    # Sort by BFS final-err ascending; missing → infinity.
    def _bfs(a: str) -> float:
        s = aggs[a].get("bfs_final_cellerr")
        return s[0] if s else float("inf")

    sorted_archs = sorted(aggs.keys(), key=_bfs)
    best_arch = sorted_archs[0]
    least_params = min(
        (aggs[a]["n_params"] for a in sorted_archs if aggs[a].get("n_params")),
        default=None,
    )

    a_label = "A* first-div" if a_metric == "first_div" else "A* final-err"
    if a_metric == "first_div":
        a_fmt = lambda agg: _fmt_first_div(agg.get("astar_first_div"))
    else:
        a_fmt = lambda agg: _fmt_pct(agg.get("astar_final_cellerr"))

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
                _fmt_loss(agg.get("best_loss")),
                _fmt_pct(agg.get("bfs_final_cellerr"), bold=is_best_bfs),
                a_fmt(agg),
            ]) + r" \\"
        )
    table = (
        "% AUTOGENERATED by nca_wm/scripts/collate_baseline_tables.py — do not edit.\n"
        "\\begin{tabular}{l c c c c}\n"
        "  \\toprule\n"
        f"  Architecture & params (M) & best loss & BFS final-err & {a_label} \\\\\n"
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
    full_bfs = full["bfs_final_cellerr"][0]
    ablations = [a for a in aggs if a != "nca_shared"]
    ablations.sort(
        key=lambda a: aggs[a]["bfs_final_cellerr"][0] - full_bfs
    )

    def _row(arch: str) -> str:
        agg = aggs[arch]
        label, is_baseline = ABLATION_DISPLAY[arch]
        bfs = agg["bfs_final_cellerr"][0]
        delta_abs = (bfs - full_bfs) * 100
        rel = (bfs - full_bfs) / full_bfs if full_bfs > 0 else float("nan")
        if is_baseline:
            delta_str = "---"
        else:
            sign = "+" if delta_abs >= 0 else ""
            delta_str = f"${sign}{delta_abs:.2f}$ ({sign}{100*rel:.0f}\\%)"
        bfs_pct = _fmt_pct(agg["bfs_final_cellerr"], bold=is_baseline)
        return "  " + " & ".join([
            label,
            _fmt_params(agg.get("n_params")),
            _fmt_loss(agg.get("best_loss")),
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
        "  Recipe & params (M) & best loss & BFS final-err & $\\Delta$BFS vs.\\ full \\\\\n"
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
        a_metric="first_div",
    )
    _table_baseline(
        rows,
        games="sokoban_basic_synth",
        arch_order=arch_order,
        out_path=OUT_DIR / "sokoban_synth.tex",
        a_metric="first_div",
    )
    _table_baseline(
        rows,
        games="scaling_14",
        arch_order=arch_order,
        out_path=OUT_DIR / "scaling_14.tex",
        a_metric="final_err",
    )
    _table_ablation(rows, OUT_DIR / "ablation_pool.tex")


if __name__ == "__main__":
    main()
