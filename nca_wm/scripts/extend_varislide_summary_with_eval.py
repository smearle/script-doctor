"""Extend the varislide post-fix sweep summary with held-out map-size eval.

For each completed run in nca_wm/logs_canary/varislide_postfix*:
  - Extract per-authored-level (L0..L7) TF + AR mean cell-error rate from
    `eval_multigame.npz` (auto-eval that runs at end of training).
  - Tag each level by its width (training widths {6,8,10,12,16} vs OOD).

Authored varislide level widths (from the canonical varislide game):
  L0=6 (in), L1=7 (interp), L2=8 (in), L3=9 (interp), L4=10 (in),
  L5=12 (in), L6=15 (interp), L7=19 (OOD wider).

Training synth widths: 6, 8, 10, 12, 16. So OOD-wider = L7 only;
in-distribution / interpolation = L0..L6.

Outputs:
  - nca_wm/figures/varislide_postfix/eval_summary.csv (per-run, all levels)
  - nca_wm/figures/varislide_postfix/eval_summary.md (4 sub-tables: A/B/C/Cp)
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

# Authored varislide level widths (verified empirically 2026-05-04 — L0=L1=L2=6).
LEVEL_W = {0: 6, 1: 6, 2: 6, 3: 7, 4: 9, 5: 11, 6: 15, 7: 19}
TRAIN_W = {6, 8, 10, 12, 16}


def _load_eval(run_dir):
    # Prefer the post-fix re-eval (top-left slicing) when present, fall back
    # to the original npz for backwards compat.
    for fname in ("eval_multigame_tlfix.npz", "eval_multigame.npz"):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            break
    else:
        return None
    z = np.load(p, allow_pickle=True)
    out = {"per_level": {}}
    for li in range(8):
        rec = {}
        for kind in ("random", "random_tf", "bfs"):
            key = f"varislide_L{li}_{kind}_cell_error_rate"
            if key in z.files:
                arr = np.asarray(z[key])
                if arr.size > 0:
                    rec[f"{kind}_mean"] = float(np.mean(arr))
                    rec[f"{kind}_max"]  = float(np.max(arr))
                    rec[f"{kind}_step1"]= float(arr[0])
        if rec:
            rec["W"] = LEVEL_W.get(li)
            rec["regime"] = "train" if rec["W"] in TRAIN_W else (
                "OOD" if rec["W"] > max(TRAIN_W) else "interp")
            out["per_level"][li] = rec
    return out


def _summarize_run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    name = os.path.basename(run_dir)
    ev = _load_eval(run_dir)
    base = {
        "run": name,
        "n_steps": cfg.get("n_nca_steps"),
        "n_repeats": cfg.get("n_nca_repeats"),
        "n_layers": (cfg.get("n_nca_steps", 0) // max(cfg.get("n_nca_repeats", 1), 1)),
        "axis_pool": cfg.get("axis_pool", True),
        "global_pool": cfg.get("global_pool", True),
        "input_skip": cfg.get("input_skip", False),
        "seed": cfg.get("seed", 0),
    }
    if ev is None:
        return base, None
    return base, ev["per_level"]


def _bucket(name):
    if name.startswith("varislide_postfixA_"): return "A"
    if name.startswith("varislide_postfixB_"): return "B"
    if name.startswith("varislide_postfixC_nopool_"): return "C"
    if name.startswith("varislide_postfixCp_"): return "Cp"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="nca_wm/logs_canary")
    p.add_argument("--out_csv", default="nca_wm/figures/varislide_postfix/eval_summary.csv")
    p.add_argument("--out_md",  default="nca_wm/figures/varislide_postfix/eval_summary.md")
    args = p.parse_args()

    runs = sorted(glob.glob(os.path.join(_REPO, args.logs, "varislide_postfix[ABC]*")))
    runs = [r for r in runs if os.path.isdir(r)
            and os.path.exists(os.path.join(r, "config.json"))
            and os.path.exists(os.path.join(r, "params.pkl"))
            and os.path.exists(os.path.join(r, "train_meta.json"))]
    print(f"Found {len(runs)} runs.")

    all_rows = []
    summary_per_run = {}
    for r in runs:
        base, per_level = _summarize_run(r)
        if per_level:
            for li, rec in per_level.items():
                row = {**base, "level": li, **rec}
                all_rows.append(row)
            summary_per_run[base["run"]] = (base, per_level)

    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(_REPO, args.out_csv)
    out_md  = args.out_md  if os.path.isabs(args.out_md)  else os.path.join(_REPO, args.out_md)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if all_rows:
        cols = sorted({k for r in all_rows for k in r.keys()})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"wrote {out_csv}")

    # Markdown summary: aggregate per-run train-regime mean and OOD-width stats.
    def fmt(x): return "—" if x is None else f"{x*100:.2f}%"

    lines = ["# Varislide post-bitpack-fix sweep — held-out map-size eval\n"]
    lines.append(
        "Numbers from each run's `eval_multigame.npz` (auto-eval at end of "
        "training). Aggregates per-step cell-error mean over the 30-step "
        "rollout, averaged across levels in each width regime.\n"
    )
    lines.append(
        "Training synth widths: {6, 8, 10, 12, 16}. Authored widths: "
        f"{[LEVEL_W[i] for i in range(8)]}. "
        "Regime: 'train' = W ∈ training set, 'interp' = W in [min, max] but not in training, "
        "'OOD' = W > 16 (only L7 = W19).\n"
    )
    lines.append(
        "Columns: random_tf = teacher-forced (each step gets real prev state); "
        "random = autoregressive (model's own prediction fed back). bfs = "
        "BFS-oracle action sequence (autoregressive). All metrics are mean cell-error / step.\n"
    )

    # Per-bucket per-regime aggregate
    buckets = defaultdict(list)
    for run, (base, levels) in summary_per_run.items():
        b = _bucket(run)
        if b:
            buckets[b].append((base, levels))

    bucket_titles = {
        "A": "## A: depth × seed (fully-shared, n_repeats = n_steps)",
        "B": "## B: L × R factor at total=16",
        "C": "## C: pool OFF + input_skip",
        "Cp": "## C': pool OFF without input_skip control",
    }

    def regime_mean(levels, regime, kind):
        vals = [rec.get(f"{kind}_mean") for rec in levels.values()
                if rec.get("regime") == regime and f"{kind}_mean" in rec]
        return float(np.mean(vals)) if vals else None

    def per_level_one(levels, li, kind):
        return levels.get(li, {}).get(f"{kind}_mean")

    for bkey in ["A", "B", "C", "Cp"]:
        rows = sorted(buckets.get(bkey, []),
                      key=lambda x: (x[0]["n_layers"], x[0]["n_steps"],
                                     x[0]["n_repeats"], x[0]["seed"]))
        if not rows:
            continue
        lines.append(f"\n{bucket_titles[bkey]}\n")
        if bkey == "A":
            cols = "| depth | seed | TF train | TF interp | TF OOD(L7) | AR train | AR interp | AR OOD(L7) | BFS train | BFS interp | BFS OOD(L7) |"
        elif bkey == "B":
            cols = "| L | R | seed | TF train | TF interp | TF OOD(L7) | AR train | AR interp | AR OOD(L7) | BFS train | BFS interp | BFS OOD(L7) |"
        else:
            cols = "| depth | seed | TF train | TF interp | TF OOD(L7) | AR train | AR interp | AR OOD(L7) | BFS train | BFS interp | BFS OOD(L7) |"
        lines.append(cols)
        lines.append("|" + "---|" * (cols.count("|") - 1))
        for base, levels in rows:
            tf_t = regime_mean(levels, "train", "random_tf")
            tf_i = regime_mean(levels, "interp", "random_tf")
            tf_o = per_level_one(levels, 7, "random_tf")
            ar_t = regime_mean(levels, "train", "random")
            ar_i = regime_mean(levels, "interp", "random")
            ar_o = per_level_one(levels, 7, "random")
            bf_t = regime_mean(levels, "train", "bfs")
            bf_i = regime_mean(levels, "interp", "bfs")
            bf_o = per_level_one(levels, 7, "bfs")
            if bkey == "B":
                head = f"| {base['n_layers']} | {base['n_repeats']} | {base['seed']} |"
            else:
                head = f"| {base['n_steps']} | {base['seed']} |"
            row = (head +
                   f" {fmt(tf_t)} | {fmt(tf_i)} | {fmt(tf_o)} |" +
                   f" {fmt(ar_t)} | {fmt(ar_i)} | {fmt(ar_o)} |" +
                   f" {fmt(bf_t)} | {fmt(bf_i)} | {fmt(bf_o)} |")
            lines.append(row)

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
