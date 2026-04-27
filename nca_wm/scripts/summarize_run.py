#!/usr/bin/env python3
"""Quick per-game summary for a single NCA WM run directory.

Usage::
    python nca_wm/scripts/summarize_run.py <run_dir>
    python nca_wm/scripts/summarize_run.py nca_wm/logs/multi_small_cond_bal_ap_ac_gp_...

Reports:
    - final train loss / change_err (per the latest curves_step*.npz)
    - per-game change_err trajectory (from per_game_* arrays if present)
    - per-game eval error (random / bfs / astar) from eval_multigame.npz
Sorted by per-game worst-fit so bottleneck games float to the top.
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
    pat = re.compile(
        r"^(.+)_L(\d+)_(random_tf|random|bfs|astar)_"
        r"(error_rate|cell_error_rate|wrong_tiles|wrong_cells|mean_wrong_cells|mean_first_div|first_div_step)$"
    )
    per_game: dict[str, list[float]] = {}
    for k, arr in eval_data.items():
        m = pat.match(k)
        if not m:
            continue
        g, lvl, rt, mt = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if rt != rollout or mt != metric:
            continue
        per_game.setdefault(g, []).append(float(np.asarray(arr).mean()))
    return {g: float(np.mean(v)) for g, v in per_game.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
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

    print(f"== {os.path.basename(run_dir)} ==")
    print(f"  games={cfg.get('games')!r}  n_hid={cfg.get('n_hid')}  "
          f"n_nca_steps={cfg.get('n_nca_steps')}")
    pool_bits = [k for k in ("axis_pool", "axis_cummax", "global_pool") if cfg.get(k)]
    print(f"  pool_flags={pool_bits or 'none'}  "
          f"balanced={cfg.get('balanced_sampling', False)}  "
          f"sprite_loss_weight={cfg.get('sprite_loss_weight', 0.0)}")
    print(f"  total_steps={meta.get('total_steps', 'N/A')}")

    curves, step = _latest_curves(run_dir)
    if curves is not None:
        last_n = min(200, len(curves["losses"]))
        avg_loss = float(curves["losses"][-last_n:].mean())
        avg_cerr = float(1.0 - curves["change_accs"][-last_n:].mean())
        print(f"\n  train @ step {step:,}: loss={avg_loss:.4e}  change_err={avg_cerr:.4e}")

        # Per-game trajectories
        games = set()
        for k in curves:
            m = re.match(r"per_game_(.+)_change_acc", k)
            if m: games.add(m.group(1))
        if games:
            print("\n  per-game final change_err (train eval subset):")
            entries = []
            for g in sorted(games):
                cacc = curves.get(f"per_game_{g}_change_acc")
                if cacc is None or len(cacc) == 0: continue
                entries.append((g, float(1.0 - cacc[-1])))
            entries.sort(key=lambda x: -x[1])  # worst first
            for g, cerr in entries:
                flag = "  <-- bottleneck" if cerr > 1e-2 else ""
                print(f"    {g:<32}  change_err={cerr:.3e}{flag}")

    # eval_multigame.npz (only present after eval pass completes)
    ev_path = os.path.join(run_dir, "eval_multigame.npz")
    if os.path.isfile(ev_path):
        ev = dict(np.load(ev_path))
        print("\n  eval per-game error (mean over levels, random rollout):")
        by_rtype = {
            "random": _eval_per_game(ev, "random", "error_rate"),
            "random_tf": _eval_per_game(ev, "random_tf", "error_rate"),
            "astar": _eval_per_game(ev, "astar", "error_rate"),
        }
        games = sorted(by_rtype["random"].keys() | by_rtype.get("random_tf", {}).keys())
        header = f"    {'game':<32} {'random':>10} {'random_tf':>10} {'astar':>10}"
        print(header)
        rows = [(g, by_rtype["random"].get(g, float("nan")),
                 by_rtype["random_tf"].get(g, float("nan")),
                 by_rtype["astar"].get(g, float("nan"))) for g in games]
        rows.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else -1)
        for g, r, rtf, a in rows:
            def fmt(v):
                return f"{v:>10.4f}" if not np.isnan(v) else f"{'-':>10}"
            flag = "  <-- bottleneck" if r > 0.02 else ""
            print(f"    {g:<32} {fmt(r)} {fmt(rtf)} {fmt(a)}{flag}")
    else:
        print("\n  (no eval_multigame.npz yet — training likely still running)")


if __name__ == "__main__":
    main()
