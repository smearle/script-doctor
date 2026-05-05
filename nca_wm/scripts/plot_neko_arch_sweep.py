"""Plot the nekopuzzle synth architecture sweep results.

Reads `nca_wm/figures/neko_arch/summary.csv` produced by
`reeval_neko_arch_sweep.py` and emits paper-ready figures to the same dir.
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))


def _read_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("tf_step1", "tf_mean", "ar_step1", "ar_mean",
                  "ho_mean", "ood_mean"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
        r["depth"] = int(r["depth"])
    return rows


def _setup():
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 12, "figure.dpi": 130,
    })


def _save(fig, base):
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_depth(rows, out_dir, metric, title, ylabel):
    """4 lines: pool×share. X=depth. Y=metric."""
    _setup()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cfg_styles = {
        ("shared", "ON"):  ("o-",  "C0", "shared / pool ON"),
        ("shared", "OFF"): ("s--", "C1", "shared / pool OFF"),
        ("per-step", "ON"):  ("^-",  "C2", "per-step / pool ON"),
        ("per-step", "OFF"): ("v--", "C3", "per-step / pool OFF"),
    }
    for (share, pool), (mk, c, label) in cfg_styles.items():
        sel = sorted([r for r in rows if r["share"] == share
                       and r["pool"] == pool], key=lambda r: r["depth"])
        if not sel: continue
        xs = [r["depth"] for r in sel]
        ys = [100 * r[metric] for r in sel]
        ax.plot(xs, ys, mk, color=c, label=label, lw=2, ms=9)
    ax.set_xlabel("depth (n_nca_steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted({r["depth"] for r in rows}))
    ax.grid(True, alpha=0.3)
    ax.legend()
    base = os.path.join(out_dir, f"by_depth_{metric}")
    _save(fig, base)
    print(f"wrote {base}.png")


def plot_grouped_bars(rows, out_dir):
    """Bar chart: each config as a group, panels for {authored AR mean, holdout, ood}."""
    _setup()
    cfg_keys = sorted({(r["depth"], r["share"], r["pool"]) for r in rows})
    cfg_labels = [f"d{d}\n{s[:5]}\n{p}" for (d, s, p) in cfg_keys]
    metrics = [
        ("ar_mean", "Authored AR (30-step mean)"),
        ("ho_mean", "Held-out synth (trained sizes)"),
        ("ood_mean", "OOD synth (9x9)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    for ax, (metric, title) in zip(axes, metrics):
        ys = []
        for k in cfg_keys:
            sel = [r for r in rows if (r["depth"], r["share"], r["pool"]) == k]
            ys.append(100 * sel[0][metric] if sel else 0.0)
        bars = ax.bar(range(len(cfg_keys)), ys,
                       color=["C0" if k[2] == "ON" else "C3" for k in cfg_keys])
        ax.set_xticks(range(len(cfg_keys)))
        ax.set_xticklabels(cfg_labels, fontsize=10)
        ax.set_ylabel("cell-error %")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width()/2, y,
                     f"{y:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Nekopuzzle synth-trained architecture sweep — held-out generalization", fontsize=18)
    base = os.path.join(out_dir, "all_metrics_grouped")
    _save(fig, base)
    print(f"wrote {base}.png")


def _read_combined(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("bfs_authored", "rnd_authored", "tf_authored",
                  "holdout_synth", "ood_synth"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
        r["depth"] = int(r["depth"])
    return rows


def plot_metric_by_depth_combined(rows, out_dir, metric, title, ylabel):
    """Plots from summary_combined.csv (different keys: 'share' uses 'shared'/'perstep')."""
    _setup()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cfg_styles = {
        ("shared", "ON"):  ("o-",  "C0", "shared / pool ON"),
        ("shared", "OFF"): ("s--", "C1", "shared / pool OFF"),
        ("perstep", "ON"):  ("^-",  "C2", "per-step / pool ON"),
        ("perstep", "OFF"): ("v--", "C3", "per-step / pool OFF"),
    }
    for (share, pool), (mk, c, label) in cfg_styles.items():
        sel = sorted([r for r in rows if r["share"] == share
                       and r["pool"] == pool], key=lambda r: r["depth"])
        if not sel: continue
        xs = [r["depth"] for r in sel]
        ys = [100 * r[metric] for r in sel]
        ax.plot(xs, ys, mk, color=c, label=label, lw=2, ms=9)
    ax.set_xlabel("depth (n_nca_steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted({r["depth"] for r in rows}))
    ax.grid(True, alpha=0.3)
    ax.legend()
    base = os.path.join(out_dir, f"by_depth_{metric}")
    _save(fig, base)
    print(f"wrote {base}.png")


def main():
    csv_path = os.path.join(_REPO, "nca_wm/figures/neko_arch/summary.csv")
    out_dir = os.path.dirname(csv_path)
    rows = _read_rows(csv_path)
    if not rows:
        print(f"no rows in {csv_path}")
        return
    print(f"loaded {len(rows)} rows from {csv_path}")
    plot_metric_by_depth(rows, out_dir, "ar_mean",
        "Authored AR cell-error vs depth", "AR mean cell-error %")
    plot_metric_by_depth(rows, out_dir, "tf_step1",
        "Authored TF step-1 cell-error vs depth", "TF step-1 cell-error %")
    plot_metric_by_depth(rows, out_dir, "ho_mean",
        "Held-out synth cell-error vs depth", "Cell-error %")
    plot_metric_by_depth(rows, out_dir, "ood_mean",
        "OOD synth (9x9) cell-error vs depth", "Cell-error %")
    plot_grouped_bars(rows, out_dir)

    # BFS-from-combined plot
    combined_path = os.path.join(out_dir, "summary_combined.csv")
    if os.path.exists(combined_path):
        crows = _read_combined(combined_path)
        plot_metric_by_depth_combined(crows, out_dir, "bfs_authored",
            "Authored BFS-rollout cell-error vs depth (in-train eval)",
            "BFS rollout cell-error %")


if __name__ == "__main__":
    main()
