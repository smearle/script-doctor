#!/usr/bin/env python3
"""Quick summary for a single NCA WM run directory (per-game AND per-level).

Usage::
    python nca_wm/scripts/summarize_run.py <run_dir>
    python nca_wm/scripts/summarize_run.py nca_wm/logs/take_heart_lass/pool_on_skip_on_cap300k

Reports:
    - final train loss / change_err (per the latest curves_step*.npz)
    - per-game change_err trajectory (from per_game_* arrays if present)
    - per-game eval error (random / bfs / astar) from eval_multigame.npz
    - per-(game, level) scorecard: AR / TF / BFS / A* wrong-cells + first-div
Writes ``eval_summary.txt`` and ``eval_summary.png/.pdf`` (per-level error bars)
into the run directory so every finished run carries its own scorecard.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np

# eval_multigame.npz keys are "<game>_L<level>_<rtype>_<metric>".
_EVAL_RE = re.compile(
    r"^(.+)_L(\d+)_(random_tf|random|bfs|astar)_"
    r"(error_rate|cell_error_rate|wrong_tiles|wrong_cells|mean_wrong_cells|mean_first_div|first_div_step)$"
)

_OUT_LINES: list[str] = []


def emit(s: str = "") -> None:
    """Print and capture for eval_summary.txt."""
    print(s)
    _OUT_LINES.append(s)


def _latest_curves(run_dir: str):
    curve_files = sorted(glob.glob(os.path.join(run_dir, "curves_step*.npz")))
    if not curve_files:
        return None, 0
    def step_of(p):
        m = re.search(r"curves_step(\d+)", p)
        return int(m.group(1)) if m else 0
    path = max(curve_files, key=step_of)
    return dict(np.load(path)), step_of(path)


def _eval_per_game(eval_data, rollout="random", metric="error_rate"):
    per_game: dict[str, list[float]] = {}
    for k, arr in eval_data.items():
        m = _EVAL_RE.match(k)
        if not m:
            continue
        g, lvl, rt, mt = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if rt != rollout or mt != metric:
            continue
        per_game.setdefault(g, []).append(float(np.asarray(arr).mean()))
    return {g: float(np.mean(v)) for g, v in per_game.items()}


def _eval_nested(eval_data):
    """results[game][level][rtype][metric] -> np.ndarray."""
    out: dict = {}
    for k, arr in eval_data.items():
        m = _EVAL_RE.match(k)
        if not m:
            continue
        g, lvl, rt, mt = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        out.setdefault(g, {}).setdefault(lvl, {}).setdefault(rt, {})[mt] = np.asarray(arr)
    return out


def _per_level_table(nested):
    """Print per-(game, level) scorecard; return rows for plotting."""
    rows = []  # (game, level, ar_wc, tf_wc, bfs_wc, astar_wc, perfect)
    hdr = (f"    {'lvl':>4} | {'AR_wc':>8} {'AR_fdiv':>8} | {'TF_wc':>8} "
           f"{'TF_err%':>8} | {'bfs_wc':>7} {'bfs_fd':>7} | "
           f"{'as_wc':>7} {'as_fd':>7} | status")
    for game in sorted(nested):
        emit(f"\n  per-level scorecard [{game}]:")
        emit(hdr)
        emit("    " + "-" * (len(hdr) - 4))
        n_perfect = 0
        for lvl in sorted(nested[game]):
            r = nested[game][lvl]
            def g(rt, met, agg):
                a = r.get(rt, {}).get(met)
                if a is None:
                    return float("nan")
                a = np.asarray(a, dtype=np.float64)
                return float(agg(a)) if a.size else float("nan")
            ar_wc = g("random", "mean_wrong_cells", np.mean)
            ar_fd = g("random", "mean_first_div", lambda x: x.item() if x.size == 1 else x.mean())
            tf_wc = g("random_tf", "mean_wrong_cells", np.mean)
            tf_err = g("random_tf", "error_rate", np.mean) * 100.0
            bfs_wc = g("bfs", "wrong_cells", np.max)
            bfs_fd = g("bfs", "first_div_step", lambda x: x.item() if x.size == 1 else x.max())
            as_wc = g("astar", "wrong_cells", np.max)
            as_fd = g("astar", "first_div_step", lambda x: x.item() if x.size == 1 else x.max())
            perfect = all(np.nan_to_num(v) < 1e-6 for v in (ar_wc, tf_wc, bfs_wc, as_wc))
            n_perfect += perfect
            emit(f"    {lvl:>4} | {ar_wc:>8.3f} {ar_fd:>8.2f} | {tf_wc:>8.3f} "
                 f"{tf_err:>8.3f} | {bfs_wc:>7.0f} {bfs_fd:>7.0f} | "
                 f"{as_wc:>7.0f} {as_fd:>7.0f} | "
                 f"{'perfect' if perfect else 'RESIDUAL'}")
            rows.append((game, lvl, ar_wc, tf_wc, bfs_wc, as_wc))
        emit(f"    -> {n_perfect}/{len(nested[game])} levels perfect")
    return rows


def _plot(rows, run_dir):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    multi = len({g for g, *_ in rows}) > 1
    labels = [(f"{g[:6]}.L{l}" if multi else f"L{l}") for (g, l, *_ ) in rows]
    ar = np.nan_to_num([r[2] for r in rows]); tf = np.nan_to_num([r[3] for r in rows])
    bfs = np.nan_to_num([r[4] for r in rows]); ast = np.nan_to_num([r[5] for r in rows])
    x = np.arange(len(rows)); ww = 0.2
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(rows)), 4.6))
    ax.bar(x - 1.5 * ww, ar, ww, label="AR mean wrong cells")
    ax.bar(x - 0.5 * ww, tf, ww, label="TF mean wrong cells")
    ax.bar(x + 0.5 * ww, bfs, ww, label="BFS max wrong cells")
    ax.bar(x + 1.5 * ww, ast, ww, label="A* max wrong cells")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("wrong cells"); ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title(f"Per-level eval error — {os.path.basename(run_dir.rstrip('/'))}")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_pdf = os.path.join(run_dir, "eval_summary.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    emit(f"\nWrote: {out_pdf}")
    emit(f"       {out_pdf.replace('.pdf', '.png')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--no_fig", action="store_true")
    args = p.parse_args()
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    cfg_path = os.path.join(run_dir, "config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    meta_path = os.path.join(run_dir, "train_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    emit(f"== {os.path.basename(run_dir)} ==")
    emit(f"  games={cfg.get('games')!r}  n_hid={cfg.get('n_hid')}  "
         f"n_nca_steps={cfg.get('n_nca_steps')}")
    pool_bits = [k for k in ("axis_pool", "axis_cummax", "global_pool") if cfg.get(k)]
    emit(f"  pool_flags={pool_bits or 'none'}  "
         f"balanced={cfg.get('balanced_sampling', False)}  "
         f"max_transitions_per_game={cfg.get('max_transitions_per_game')}  "
         f"val_frac={cfg.get('val_frac')}")
    emit(f"  total_steps={meta.get('total_steps', 'N/A')}  "
         f"best_step={meta.get('best_step', 'N/A')}  "
         f"best_loss={meta.get('best_loss', 'N/A')}")

    curves, step = _latest_curves(run_dir)
    if curves is not None:
        last_n = min(200, len(curves["losses"]))
        avg_loss = float(curves["losses"][-last_n:].mean())
        avg_cerr = float(1.0 - curves["change_accs"][-last_n:].mean())
        emit(f"\n  train @ step {step:,}: loss={avg_loss:.4e}  change_err={avg_cerr:.4e}")
        if "val_change_acc" in curves and len(curves["val_change_acc"]):
            v_cerr = float(1.0 - np.asarray(curves["val_change_acc"])[-1])
            v_step = int(np.asarray(curves["val_step"])[-1])
            emit(f"  val   @ step {v_step:,}: change_err={v_cerr:.4e}")

        games = set()
        for k in curves:
            m = re.match(r"per_game_(.+)_change_acc", k)
            if m: games.add(m.group(1))
        if games:
            emit("\n  per-game final change_err (train eval subset):")
            entries = []
            for g in sorted(games):
                cacc = curves.get(f"per_game_{g}_change_acc")
                if cacc is None or len(cacc) == 0: continue
                entries.append((g, float(1.0 - cacc[-1])))
            entries.sort(key=lambda x: -x[1])
            for g, cerr in entries:
                flag = "  <-- bottleneck" if cerr > 1e-2 else ""
                emit(f"    {g:<32}  change_err={cerr:.3e}{flag}")

    ev_path = os.path.join(run_dir, "eval_multigame.npz")
    if os.path.isfile(ev_path):
        ev = dict(np.load(ev_path))
        emit("\n  eval per-game error (mean over levels, random rollout):")
        by_rtype = {
            "random": _eval_per_game(ev, "random", "error_rate"),
            "random_tf": _eval_per_game(ev, "random_tf", "error_rate"),
            "astar": _eval_per_game(ev, "astar", "error_rate"),
        }
        games = sorted(by_rtype["random"].keys() | by_rtype.get("random_tf", {}).keys())
        emit(f"    {'game':<32} {'random':>10} {'random_tf':>10} {'astar':>10}")
        rows = [(g, by_rtype["random"].get(g, float("nan")),
                 by_rtype["random_tf"].get(g, float("nan")),
                 by_rtype["astar"].get(g, float("nan"))) for g in games]
        rows.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else -1)
        for g, r, rtf, a in rows:
            def fmt(v):
                return f"{v:>10.4f}" if not np.isnan(v) else f"{'-':>10}"
            flag = "  <-- bottleneck" if r > 0.02 else ""
            emit(f"    {g:<32} {fmt(r)} {fmt(rtf)} {fmt(a)}{flag}")

        nested = _eval_nested(ev)
        plot_rows = _per_level_table(nested)
        if not args.no_fig:
            _plot(plot_rows, run_dir)
    else:
        emit("\n  (no eval_multigame.npz yet — training likely still running)")

    out_txt = os.path.join(run_dir, "eval_summary.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(_OUT_LINES) + "\n")
    print(f"\nWrote: {out_txt}")


if __name__ == "__main__":
    main()
