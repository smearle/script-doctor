"""Summarize the per-game × per-architecture grid.

Reads each completed run under nca_wm/logs_per_game_arch/<game>__<bucket>_d<D>/
and aggregates per-cell:
  - best/final train loss (from latest curves_step*.npz)
  - rollout cell-error from eval_multigame[_tlfix].npz, split by oracle
    policy (bfs / astar / random / random_tf) and by L0 vs heldout

Outputs:
  - nca_wm/figures/per_game_arch/summary.csv  (one row per cell)
  - nca_wm/figures/per_game_arch/summary.md   (game-major table per bucket)

Usage:
    .venv/bin/python3 nca_wm/scripts/summarize_per_game_arch_grid.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))


def _final_train_metrics(run_dir):
    curves = sorted(
        glob.glob(os.path.join(run_dir, "curves_step*.npz")),
        key=lambda p: int(os.path.basename(p)[len("curves_step"):-len(".npz")]),
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
    # Convergence diagnostic: relative descent over the last 10% of training.
    # If first-tenth-mean / last-tenth-mean > 1.10, the loss is still meaningfully
    # falling at cutoff — i.e. the run was almost certainly under-trained.
    losses = z.get("losses")
    if losses is not None and losses.ndim == 1 and losses.size >= 200:
        n = losses.size
        # Compare loss in the last 25% to loss in the preceding 25% (windows
        # 50–75% vs 75–100%). On a converged run this ratio sits at ~1.0; on a
        # still-descending one it's >1.0 by however much the loss has fallen.
        prev = float(np.mean(losses[int(n * 0.50): int(n * 0.75)]))
        tail = float(np.mean(losses[int(n * 0.75):]))
        out["loss_q3"] = prev
        out["loss_q4"] = tail
        out["loss_ratio_q3_to_q4"] = (prev / tail) if tail > 1e-30 else float("nan")
        out["n_steps"] = int(n)
    return out


def _rollout_err(run_dir):
    """Read per-cell rollout cell-error npz and aggregate by kind/level-split."""
    p = None
    for cand in ("eval_multigame_tlfix.npz", "eval_multigame.npz"):
        cp = os.path.join(run_dir, cand)
        if os.path.exists(cp):
            p = cp
            break
    if p is None:
        return {}
    z = np.load(p, allow_pickle=True)
    pat = re.compile(r"^(?P<game>.+)_L(?P<lvl>\d+)_(?P<kind>bfs|astar|random|random_tf)_cell_error_rate$")
    by_kind_all = defaultdict(list)
    by_kind_l0 = defaultdict(list)
    by_kind_held = defaultdict(list)
    for k in z.files:
        m = pat.match(k)
        if not m:
            continue
        kind = m.group("kind")
        lvl = int(m.group("lvl"))
        v = np.asarray(z[k]).astype(float)
        if v.size == 0 or not np.isfinite(v).any():
            continue
        mean_v = float(np.nanmean(v))
        by_kind_all[kind].append(mean_v)
        if lvl == 0:
            by_kind_l0[kind].append(mean_v)
        else:
            by_kind_held[kind].append(mean_v)
    out = {}
    for kind in ("bfs", "astar", "random", "random_tf"):
        if by_kind_all[kind]:
            out[f"{kind}_err_mean"] = float(np.mean(by_kind_all[kind]))
        if by_kind_l0[kind]:
            out[f"{kind}_err_l0"] = float(np.mean(by_kind_l0[kind]))
        if by_kind_held[kind]:
            out[f"{kind}_err_heldout"] = float(np.mean(by_kind_held[kind]))
    return out


_RUN_RE = re.compile(r"^(?P<game>.+)__(?P<bucket>[A-D])_d(?P<depth>\d+)$")


def _parse_run_name(name):
    m = _RUN_RE.match(name)
    if not m:
        return None
    return {
        "run": name,
        "game": m.group("game"),
        "bucket": m.group("bucket"),
        "depth": int(m.group("depth")),
    }


def _summarize_run(run_dir):
    name = os.path.basename(run_dir)
    parsed = _parse_run_name(name)
    if parsed is None:
        return None
    cfg = {}
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            pass
    train = _final_train_metrics(run_dir)
    ev = _rollout_err(run_dir)
    row = dict(parsed)
    row.update({
        "n_steps": cfg.get("n_nca_steps"),
        "n_repeats": cfg.get("n_nca_repeats"),
        "axis_pool": cfg.get("axis_pool"),
        "global_pool": cfg.get("global_pool"),
        "axis_cummax": cfg.get("axis_cummax"),
        "input_skip": cfg.get("input_skip"),
        "best_loss": train.get("best_losses"),
        "final_loss": train.get("final_losses"),
        "final_change_acc": train.get("final_change_accs"),
        "loss_ratio_q3_to_q4": train.get("loss_ratio_q3_to_q4"),
        "n_train_steps": train.get("n_steps"),
        "bfs_err_mean": ev.get("bfs_err_mean"),
        "bfs_err_l0": ev.get("bfs_err_l0"),
        "bfs_err_heldout": ev.get("bfs_err_heldout"),
        "astar_err_mean": ev.get("astar_err_mean"),
        "astar_err_l0": ev.get("astar_err_l0"),
        "astar_err_heldout": ev.get("astar_err_heldout"),
        "random_err_mean": ev.get("random_err_mean"),
        "random_tf_err_mean": ev.get("random_tf_err_mean"),
    })
    return row


_BUCKET_LABELS = {
    "A": "A: pool ON, shared",
    "B": "B: pool ON, per-step",
    "C": "C: no-pool + skip, shared",
    "D": "D: no-pool + skip, per-step",
}


def _fmt_loss(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2e}"


def _fmt_err(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="nca_wm/logs_per_game_arch")
    ap.add_argument("--out_csv", default="nca_wm/figures/per_game_arch/summary.csv")
    ap.add_argument("--out_md", default="nca_wm/figures/per_game_arch/summary.md")
    args = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(_REPO, args.logs, "*")))
    runs = [
        r for r in runs
        if os.path.isdir(r)
        and os.path.exists(os.path.join(r, "params.pkl"))
        and os.path.exists(os.path.join(r, "train_meta.json"))
    ]
    print(f"Found {len(runs)} completed runs.")

    rows = []
    for r in runs:
        s = _summarize_run(r)
        if s is not None:
            rows.append(s)
    rows.sort(key=lambda r: (r["game"], r["bucket"]))

    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(_REPO, args.out_csv)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(_REPO, args.out_md)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {out_csv}")

    games = sorted({r["game"] for r in rows})
    buckets = ["A", "B", "C", "D"]
    by_cell = {(r["game"], r["bucket"]): r for r in rows}

    lines = ["# Per-game × per-architecture grid\n"]
    lines.append(
        "Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, lr=3e-4, "
        "n_nca_steps=8, --balanced_sampling, change_loss_weight=5.0. Trained on level 0 of each game.\n"
    )
    lines.append("Buckets:\n")
    for k in buckets:
        lines.append(f"- **{k}** — {_BUCKET_LABELS[k]}")
    lines.append("")

    def _table(metric_key, header):
        out = [f"\n## {header}\n"]
        out.append("| game | " + " | ".join(_BUCKET_LABELS[b] for b in buckets) + " |")
        out.append("|---" * (1 + len(buckets)) + "|")
        for g in games:
            cells = []
            for b in buckets:
                r = by_cell.get((g, b))
                v = (r or {}).get(metric_key)
                cells.append(_fmt_err(v))
            out.append(f"| {g} | " + " | ".join(cells) + " |")
        return out

    lines += _table("bfs_err_l0", "BFS rollout cell-error — training level (L0)")
    lines += _table("bfs_err_heldout", "BFS rollout cell-error — held-out levels (L1+; mean)")
    lines += _table("random_tf_err_mean", "Teacher-forced 1-step cell-error (random actions)")

    # Best-loss table for completeness.
    out = ["\n## Best train loss (cross-entropy)\n"]
    out.append("| game | " + " | ".join(_BUCKET_LABELS[b] for b in buckets) + " |")
    out.append("|---" * (1 + len(buckets)) + "|")
    for g in games:
        cells = []
        for b in buckets:
            r = by_cell.get((g, b))
            v = (r or {}).get("best_loss")
            cells.append(_fmt_loss(v))
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += out

    # Convergence diagnostic — ratio of loss in the 50–75% window vs the
    # 75–100% window. ~1.0 means the loss has flattened; >1.20 means the model
    # is still meaningfully descending at cutoff (almost certainly under-trained).
    out = [
        "\n## Convergence diagnostic\n",
        "Ratio of mean loss in training window 50–75% to mean loss in window 75–100%. "
        "~1.0 means flat; **⚠ flagged** when ratio>1.05 *and* absolute final loss > 1e-4 "
        "(numerical-floor cells, like fully-fit sokoban_basic, get noisy ratios but no flag). "
        "Compare same-row cells: if all four are still descending, extend the budget; if "
        "one bucket asymptotes much higher than its neighbors, that's a real architectural "
        "fit ceiling.\n",
    ]
    out.append("| game | " + " | ".join(_BUCKET_LABELS[b] for b in buckets) + " |")
    out.append("|---" * (1 + len(buckets)) + "|")
    for g in games:
        cells = []
        for b in buckets:
            r = by_cell.get((g, b))
            ratio = (r or {}).get("loss_ratio_q3_to_q4")
            tail = (r or {}).get("best_loss")  # tail loss is ~best at end
            if ratio is None or (isinstance(ratio, float) and np.isnan(ratio)):
                cells.append("—")
            else:
                # Flag only when (a) loss is still meaningfully descending AND
                # (b) absolute loss is still well above the numerical floor.
                still_falling = ratio > 1.05
                meaningful = (tail is not None) and (tail > 1e-4)
                tag = " ⚠" if (still_falling and meaningful) else ""
                cells.append(f"{ratio:.2f}{tag}")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += out

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
