"""Plots for the post-bitpack-fix varislide sweep.

Reads nca_wm/figures/varislide_postfix/summary.csv produced by
summarize_varislide_postfix_sweep.py and writes paper-ready PNG+PDF figures
to nca_wm/figures/varislide_postfix/:
  - argmax_by_depth.{png,pdf}: bucket A — per-distance argmax % vs depth
  - argmax_by_LR.{png,pdf}: bucket B — argmax % across L×R configs at total=16
  - argmax_pool_onoff.{png,pdf}: A vs C — pool ON / pool OFF + input_skip
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))


def _load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in ("argmax_overall", "argmax_d1", "argmax_d2", "argmax_d3p",
                     "final_loss", "final_acc", "final_change_acc"):
                v = r.get(k, "")
                try:
                    r[k] = float(v) if v else float("nan")
                except ValueError:
                    r[k] = float("nan")
            for k in ("n_steps", "n_repeats", "n_layers", "seed", "n_total"):
                v = r.get(k, "")
                try:
                    r[k] = int(v) if v else 0
                except ValueError:
                    r[k] = 0
            rows.append(r)
    return rows


def _setup_paper_style():
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.constrained_layout.use": True,
    })


def _save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    fig.savefig(path_no_ext + ".png", bbox_inches="tight", dpi=150)
    print(f"  wrote {path_no_ext}.pdf / .png")


def plot_depth_seed(rows, outdir):
    """Bucket A: per-depth argmax with seed dots and median bars."""
    A = [r for r in rows if r["run"].startswith("varislide_postfixA_")]
    if not A:
        print("[skip] no postfixA rows")
        return
    depths = sorted({r["n_steps"] for r in A})
    metrics = [("argmax_overall", "overall"),
               ("argmax_d1", "d=1"),
               ("argmax_d2", "d=2"),
               ("argmax_d3p", "d≥3")]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.18
    xs = np.arange(len(depths))
    for i, (key, lbl) in enumerate(metrics):
        ys, errs = [], []
        for d in depths:
            vals = [r[key] for r in A if r["n_steps"] == d
                    and not np.isnan(r[key])]
            ys.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals) if len(vals) > 1 else 0)
        ax.bar(xs + (i - 1.5) * width, np.array(ys) * 100,
               width, yerr=np.array(errs) * 100, label=lbl, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("NCA depth (n_steps, fully shared)")
    ax.set_ylabel("Argmax-correct (%)")
    ax.set_title("Varislide post-bitpack-fix: depth × per-distance argmax")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", title="slide distance")
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(outdir, exist_ok=True)
    _save(fig, os.path.join(outdir, "argmax_by_depth"))
    plt.close(fig)


def plot_LR(rows, outdir):
    """Bucket B: L×R factor at total=16."""
    B = [r for r in rows if r["run"].startswith("varislide_postfixB_")]
    if not B:
        print("[skip] no postfixB rows")
        return
    LRs = sorted({(r["n_layers"], r["n_repeats"]) for r in B})
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = [("argmax_overall", "overall"),
               ("argmax_d1", "d=1"),
               ("argmax_d2", "d=2"),
               ("argmax_d3p", "d≥3")]
    width = 0.18
    xs = np.arange(len(LRs))
    for i, (key, lbl) in enumerate(metrics):
        ys, errs = [], []
        for L, R in LRs:
            vals = [r[key] for r in B
                    if r["n_layers"] == L and r["n_repeats"] == R
                    and not np.isnan(r[key])]
            ys.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals) if len(vals) > 1 else 0)
        ax.bar(xs + (i - 1.5) * width, np.array(ys) * 100,
               width, yerr=np.array(errs) * 100, label=lbl, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L={L},R={R}" for L, R in LRs])
    ax.set_xlabel("(n_layers, n_repeats) — total = 16")
    ax.set_ylabel("Argmax-correct (%)")
    ax.set_title("Varislide post-bitpack-fix: L × R factor")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", title="slide distance")
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(outdir, exist_ok=True)
    _save(fig, os.path.join(outdir, "argmax_by_LR"))
    plt.close(fig)


def plot_pool_onoff(rows, outdir):
    """A vs C: same depth, pool ON / pool OFF + input_skip."""
    A = [r for r in rows if r["run"].startswith("varislide_postfixA_")]
    C = [r for r in rows if r["run"].startswith("varislide_postfixC_nopool_d")]
    if not (A and C):
        print("[skip] missing A or C rows")
        return
    depths = sorted({r["n_steps"] for r in A + C})
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(depths))
    width = 0.35
    for offset, group, lbl in [(-width / 2, A, "pool ON"),
                                (width / 2, C, "pool OFF + input_skip")]:
        ys, errs = [], []
        for d in depths:
            vals = [r["argmax_overall"] for r in group
                    if r["n_steps"] == d and not np.isnan(r["argmax_overall"])]
            ys.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals) if len(vals) > 1 else 0)
        ax.bar(xs + offset, np.array(ys) * 100, width,
               yerr=np.array(errs) * 100, label=lbl, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("NCA depth (n_steps)")
    ax.set_ylabel("Overall argmax-correct (%)")
    ax.set_title("Varislide post-bitpack-fix: pool ablation")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(outdir, exist_ok=True)
    _save(fig, os.path.join(outdir, "argmax_pool_onoff"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="nca_wm/figures/varislide_postfix/summary.csv")
    p.add_argument("--outdir", default="nca_wm/figures/varislide_postfix")
    args = p.parse_args()
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_REPO, args.csv)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(_REPO, args.outdir)
    _setup_paper_style()
    rows = _load(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    plot_depth_seed(rows, outdir)
    plot_LR(rows, outdir)
    plot_pool_onoff(rows, outdir)


if __name__ == "__main__":
    main()
