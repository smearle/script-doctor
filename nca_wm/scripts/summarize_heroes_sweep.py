"""Summarize the Heroes_of_Sokoban depth × sharing × pool sweep.

For each completed run in nca_wm/logs_heroes/heroes_*:
  - parse config (depth, repeats, pool flags, input_skip)
  - read final train metrics from latest curves_step*.npz
  - read eval_multigame.npz to extract authored-level rollout BFS error
    (mean over all (level, search_algo) cells; we use bfs as the canonical)

Outputs:
  - nca_wm/figures/heroes_sweep/summary.csv
  - nca_wm/figures/heroes_sweep/summary.md (4 sub-tables, one per bucket)
  - nca_wm/figures/heroes_sweep/argmax_by_depth_pool_shared.{pdf,png}

Usage:
    .venv/bin/python3 nca_wm/scripts/summarize_heroes_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))


def _final_train_metrics(run_dir):
    curves = sorted(
        glob.glob(os.path.join(run_dir, "curves_step*.npz")),
        key=lambda p: int(os.path.basename(p)[len("curves_step"):-len(".npz")])
    )
    if not curves:
        return {}
    z = np.load(curves[-1])
    out = {}
    for k in z.files:
        v = z[k]
        if v.ndim == 1 and v.size > 0:
            out[f"final_{k}"] = float(v[-1])
            out[f"best_{k}"] = float(v.min()) if "loss" in k else float(v.max())
    return out


def _bfs_rollout_err(run_dir):
    """Read eval_multigame.npz and extract per-action rollout cell error.

    Returns mean over all levels for {bfs, astar, random} sub-keys, and
    per-action min over levels (best level), max over levels (worst level).
    """
    p = os.path.join(run_dir, "eval_multigame.npz")
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    out = {}
    for kind in ("bfs", "astar", "random", "random_tf"):
        # Records stored as dict-like arrays; need to introspect shape.
        # eval_multigame stores per-game per-level per-kind. Iterate.
        try:
            data = z["eval"].item() if "eval" in z.files else None
        except Exception:
            data = None
        if not data:
            return out
        # data: {game_name: {level_i: {kind: array_or_dict}}}
        all_means = []
        all_total = []
        for game, levels in data.items():
            if not isinstance(levels, dict):
                continue
            for li, kinds in levels.items():
                if not isinstance(kinds, dict) or kind not in kinds:
                    continue
                cell_err = kinds[kind].get("model_wrong_cells")
                tiles = kinds[kind].get("total_cells")
                if cell_err is None or tiles is None or tiles == 0:
                    continue
                cell_err = np.asarray(cell_err)
                m = float(np.nanmean(cell_err) / tiles)
                all_means.append(m)
                all_total.append(tiles)
        if all_means:
            out[f"{kind}_cell_err_mean"] = float(np.mean(all_means))
            out[f"{kind}_cell_err_max"]  = float(np.max(all_means))
    return out


def _eval_from_log(run_dir):
    """Parse eval lines from `<run>.out` log file:
       `  <game> L<lvl> <algo>      wrong: mean=N  max=M  (S steps, T tiles) ...`
    Returns (algo: list of (mean, total)).
    """
    log = run_dir + ".out"
    if not os.path.exists(log):
        return {}
    import re
    pat = re.compile(r"\s+(\S+) L(\d+) (\S+)\s+wrong: mean=(\d+)\s+max=\d+\s+\((\d+) steps,\s+(\d+) tiles\)")
    by_algo = defaultdict(list)
    with open(log) as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            algo = m.group(3)
            mean_w = int(m.group(4))
            steps  = int(m.group(5))
            tiles  = int(m.group(6))
            # mean_w is the mean #wrong cells over the rollout — already
            # per-step; tiles is per-step cell count. So per-step error = mean_w/tiles.
            by_algo[algo].append((mean_w, tiles))
    out = {}
    for algo, rows in by_algo.items():
        per_step = [m / t for m, t in rows if t > 0]
        if per_step:
            out[f"{algo}_cell_err_mean"] = float(np.mean(per_step))
            out[f"{algo}_cell_err_max"]  = float(np.max(per_step))
    return out


def _summarize_run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    name = os.path.basename(run_dir)
    train = _final_train_metrics(run_dir)
    ev = _eval_from_log(run_dir)
    n_steps = cfg.get("n_nca_steps")
    n_reps = cfg.get("n_nca_repeats", 1)
    return {
        "run": name,
        "n_steps": n_steps,
        "n_repeats": n_reps,
        "n_layers": (n_steps // max(n_reps, 1)) if n_steps else None,
        "axis_pool": cfg.get("axis_pool"),
        "global_pool": cfg.get("global_pool"),
        "input_skip": cfg.get("input_skip", False),
        "best_loss": train.get("best_losses"),
        "final_loss": train.get("final_losses"),
        "final_change_acc": train.get("final_change_accs"),
        "bfs_err_mean": ev.get("bfs_cell_err_mean"),
        "astar_err_mean": ev.get("astar_cell_err_mean"),
        "random_err_mean": ev.get("random_cell_err_mean"),
        "random_tf_err_mean": ev.get("random_tf_cell_err_mean"),
    }


def _bucket(name):
    """Map run name to bucket. Returns None if unmatched."""
    if name.startswith("heroes_A_pool_shared_d"): return "A: pool ON, shared"
    if name.startswith("heroes_B_pool_perstep_d"): return "B: pool ON, per-step"
    if name.startswith("heroes_C_nopool_shared_d"): return "C: pool OFF + skip, shared"
    if name.startswith("heroes_D_nopool_perstep_d"): return "D: pool OFF + skip, per-step"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="nca_wm/logs_heroes")
    p.add_argument("--out_csv", default="nca_wm/figures/heroes_sweep/summary.csv")
    p.add_argument("--out_md",  default="nca_wm/figures/heroes_sweep/summary.md")
    args = p.parse_args()

    runs = sorted(glob.glob(os.path.join(_REPO, args.logs, "heroes_*")))
    runs = [r for r in runs if os.path.isdir(r)
            and os.path.exists(os.path.join(r, "config.json"))
            and os.path.exists(os.path.join(r, "params.pkl"))
            and os.path.exists(os.path.join(r, "train_meta.json"))]
    print(f"Found {len(runs)} completed runs.")

    rows = [_summarize_run(r) for r in runs]
    rows.sort(key=lambda r: (r["run"]))

    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(_REPO, args.out_csv)
    out_md  = args.out_md  if os.path.isabs(args.out_md)  else os.path.join(_REPO, args.out_md)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {out_csv}")

    def fmt_loss(x): return "—" if x is None else f"{x:.2e}"
    def fmt_err(x):  return "—" if x is None else f"{x*100:.2f}%"

    lines = ["# Heroes_of_Sokoban L0 depth × sharing × pool sweep\n"]
    lines.append(
        "Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, lr=3e-4, "
        "15k updates, --balanced_sampling, change_loss_weight=5.0. "
        "Single-level (level 0).\n"
    )
    lines.append(
        "`bfs_err`/`astar_err` = mean per-step cell error over BFS/A* oracle rollouts. "
        "`random_err`/`random_tf_err` = same for random-action rollouts (autoregressive / teacher-forced).\n"
    )

    by_bucket = defaultdict(list)
    for r in rows:
        b = _bucket(r["run"])
        if b is None:
            continue
        by_bucket[b].append(r)

    for bucket in ["A: pool ON, shared", "B: pool ON, per-step",
                   "C: pool OFF + skip, shared", "D: pool OFF + skip, per-step"]:
        rs = sorted(by_bucket.get(bucket, []), key=lambda r: r["n_steps"])
        if not rs:
            continue
        lines.append(f"\n## {bucket}\n")
        lines.append("| depth | best loss | final loss | final change_acc | bfs err | astar err | random_tf err | random err |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rs:
            lines.append(
                f"| {r['n_steps']} | {fmt_loss(r['best_loss'])} | "
                f"{fmt_loss(r['final_loss'])} | "
                f"{(str(round(r['final_change_acc']*100,2)) + '%') if r['final_change_acc'] else '—'} | "
                f"{fmt_err(r['bfs_err_mean'])} | "
                f"{fmt_err(r['astar_err_mean'])} | "
                f"{fmt_err(r['random_tf_err_mean'])} | "
                f"{fmt_err(r['random_err_mean'])} |"
            )

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
