"""Training curves overlay for multi_scaling_gallery_* runs.

Loads the last curves_step*.npz from each run, smooths the raw per-step
top-level metrics (loss, acc, change_acc), and writes a 1x3 paper-ready
figure (PDF + PNG) to nca_wm/figures/multi_scaling_gallery_curves/.

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_multi_scaling_gallery_curves.py
    .venv/bin/python3 nca_wm/scripts/plot_multi_scaling_gallery_curves.py \
        --pattern 'multi_scaling_gallery_v[34]_*' --window 1000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))


def _last_curves_path(run_dir):
    cs = sorted(
        glob.glob(os.path.join(run_dir, "curves_step*.npz")),
        key=lambda p: int(os.path.basename(p)[len("curves_step"):-len(".npz")]),
    )
    return cs[-1] if cs else None


def _smooth(x, window):
    if window <= 1 or x.size <= window:
        return x
    w = np.ones(window) / window
    return np.convolve(x, w, mode="valid")


def _label_for(name):
    # Drop the common prefix for readability.
    return re.sub(r"^multi_scaling_gallery_", "", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="nca_wm/logs")
    ap.add_argument("--pattern", default="multi_scaling_gallery_*")
    ap.add_argument("--exclude", default="aborted",
                    help="substring filter to drop runs (comma-separated)")
    ap.add_argument("--outdir", default="nca_wm/figures/multi_scaling_gallery_curves")
    ap.add_argument("--window", type=int, default=2000,
                    help="moving-average smoothing window (steps)")
    args = ap.parse_args()

    logs = args.logs if os.path.isabs(args.logs) else os.path.join(_REPO, args.logs)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(_REPO, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    excludes = [s.strip() for s in args.exclude.split(",") if s.strip()]

    runs = []
    for d in sorted(glob.glob(os.path.join(logs, args.pattern))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if any(e in name for e in excludes):
            continue
        cp = _last_curves_path(d)
        if cp is None:
            continue
        runs.append((name, d, cp))

    if not runs:
        print("no runs matched")
        return

    # Read configs for the legend annotation.
    cond_by_run = {}
    for name, d, _ in runs:
        cfg_path = os.path.join(d, "config.json")
        try:
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            cond_by_run[name] = bool(cfg.get("conditional", False))
        except FileNotFoundError:
            cond_by_run[name] = None

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
    })

    # (key, ylabel, log_y, transform)
    metrics = [
        ("losses", "loss", True, lambda y: y),
        ("accs", "1 - acc", True, lambda y: np.clip(1.0 - y, 1e-6, None)),
        ("change_accs", "1 - change_acc", True,
         lambda y: np.clip(1.0 - y, 1e-4, None)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.get_cmap("tab20")

    for ax, (key, ylabel, log_y, fn) in zip(axes, metrics):
        for i, (name, _, cp) in enumerate(runs):
            z = np.load(cp)
            if key not in z.files:
                continue
            y = fn(np.asarray(z[key], dtype=float))
            ys = _smooth(y, args.window)
            xs = np.arange(ys.size) + (y.size - ys.size) // 2
            color = cmap(i % cmap.N)
            ax.plot(xs, ys, color=color, linewidth=1.3, alpha=0.9,
                    label=_label_for(name))
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

    # One legend below the figure for all panels.
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(3, max(1, (len(labels) + 4) // 5))
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle(f"multi_scaling_gallery training curves "
                 f"(smoothing window = {args.window} steps)")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))

    pdf = os.path.join(outdir, "training_curves.pdf")
    png = os.path.join(outdir, "training_curves.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)
    print("wrote", png)

    # Also dump a small summary CSV.
    csv_path = os.path.join(outdir, "summary.csv")
    with open(csv_path, "w") as fh:
        fh.write("run,steps,final_loss,final_acc,final_change_acc,conditional\n")
        for name, _, cp in runs:
            z = np.load(cp)
            n = int(z["losses"].size) if "losses" in z.files else 0
            fl = float(z["losses"][-1]) if "losses" in z.files else float("nan")
            fa = float(z["accs"][-1]) if "accs" in z.files else float("nan")
            fc = float(z["change_accs"][-1]) if "change_accs" in z.files else float("nan")
            fh.write(f"{name},{n},{fl:.6g},{fa:.6f},{fc:.6f},{cond_by_run.get(name)}\n")
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
