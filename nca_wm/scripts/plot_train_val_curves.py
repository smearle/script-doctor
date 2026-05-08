#!/usr/bin/env python3
"""Plot train vs held-out (val) loss/error curves over training steps.

Reads the curves_step*.npz file from each given run directory and produces
one figure per run with two panels: loss (BCE) and change-error (1 - change_acc).
The val curves are plotted from the val_step / val_loss / val_acc /
val_change_acc arrays added by --val_frac > 0 training runs.

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_train_val_curves.py \\
        nca_wm/logs/n_per_rule_1_cond_val0.10_s0 \\
        nca_wm/logs/n_per_rule_2_cond_val0.10_s0 \\
        nca_wm/logs/n_per_rule_3_cond_val0.10_s0
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _latest_curves_npz(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("curves_step*.npz"),
                        key=lambda p: int(p.stem.split("step")[-1]))
    return candidates[-1] if candidates else None


def _smoothed(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # Same-length output so x-axis aligns with original.
    return np.convolve(arr, kernel, mode="same")


def _plot_one(run_dir: Path, ax_loss, ax_err, *, color: str, label: str,
              smooth: int) -> bool:
    """Draw train (lines) + val (markers) on the two axes. Returns True
    if any data was plotted."""
    npz_path = _latest_curves_npz(run_dir)
    if npz_path is None:
        print(f"  [skip] {run_dir.name}: no curves_step*.npz", file=sys.stderr)
        return False
    d = np.load(npz_path, allow_pickle=True)
    step_x = np.arange(1, len(d["losses"]) + 1)

    # Train curves (one point per training step).
    train_loss = np.asarray(d["losses"], dtype=np.float64)
    train_change_err = 1.0 - np.asarray(d["change_accs"], dtype=np.float64)
    if smooth > 1:
        train_loss = _smoothed(train_loss, smooth)
        train_change_err = _smoothed(train_change_err, smooth)
    ax_loss.plot(step_x, train_loss, color=color, linewidth=1.4,
                 alpha=0.85, label=f"{label} train")
    ax_err.plot(step_x, train_change_err, color=color, linewidth=1.4,
                alpha=0.85, label=f"{label} train")

    # Val curves (sparser; only present when run was launched with --val_frac > 0).
    if "val_step" in d.files and len(d["val_step"]) > 0:
        v_step = np.asarray(d["val_step"], dtype=np.int64)
        v_loss = np.asarray(d["val_loss"], dtype=np.float64)
        v_change_err = 1.0 - np.asarray(d["val_change_acc"], dtype=np.float64)
        ax_loss.plot(v_step, v_loss, color=color, linestyle="--",
                     linewidth=2.0, marker="o", markersize=4,
                     label=f"{label} val")
        ax_err.plot(v_step, v_change_err, color=color, linestyle="--",
                    linewidth=2.0, marker="o", markersize=4,
                    label=f"{label} val")
        n_val = int(d["val_n_transitions"][-1]) if "val_n_transitions" in d.files and len(d["val_n_transitions"]) > 0 else 0
        print(f"  {run_dir.name}: train n={len(train_loss):,}, "
              f"val n={len(v_step)} pts (last batch n={n_val:,})")
    else:
        print(f"  {run_dir.name}: train-only (no val_step in curves)",
              file=sys.stderr)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="Run directories. Each must contain a curves_step*.npz.")
    ap.add_argument("--out", type=Path,
                    default=Path("nca_wm/figures/train_val_curves.pdf"),
                    help="Output PDF path. PNG companion written next to it.")
    ap.add_argument("--smooth", type=int, default=50,
                    help="Train-loss smoothing window (per-step). 1 disables.")
    ap.add_argument("--title", type=str, default="Train vs val (held-out 10%)",
                    help="Figure title.")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12,
        "legend.fontsize": 8, "axes.labelsize": 11,
    })
    fig, (ax_loss, ax_err) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    cmap = plt.get_cmap("tab10")
    plotted = 0
    for i, run_dir in enumerate(args.run_dirs):
        run_dir = Path(run_dir)
        label = run_dir.name
        if _plot_one(run_dir, ax_loss, ax_err,
                     color=cmap(i % cmap.N), label=label,
                     smooth=args.smooth):
            plotted += 1

    if plotted == 0:
        sys.exit("no runs had usable data")

    for ax, ylabel in [(ax_loss, "loss (BCE)"), (ax_err, "1 - change_acc")]:
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", framealpha=0.9)
    fig.suptitle(args.title)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print(f"\nWrote: {args.out}")
    print(f"       {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
