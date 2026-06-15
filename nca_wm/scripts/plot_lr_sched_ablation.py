#!/usr/bin/env python3
"""Plot the LR-schedule ablation (cosine vs constant) val change_err curves.

Parses the per-run training logs in nca_wm/logs/lr_sched_ablation/<game>__<cond>.out,
extracts the held-out val change_err trajectory, and produces a paper-ready
cosine-vs-constant comparison (one panel per game) plus a text summary.

The question: does constant LR reach AND HOLD the same val change_err as the
default cosine schedule? If so the schedule is not load-bearing and can be
dropped, which makes resuming/extending a run natural (cosine ties decay to a
fixed n_updates).

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_lr_sched_ablation.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LOGDIR = REPO / "nca_wm" / "logs" / "lr_sched_ablation"
FIGDIR = REPO / "nca_wm" / "figures" / "lr_sched_ablation"

STEP_RE = re.compile(r"^\s*step ([\d,]+)/([\d,]+).*?change_err=([\d.eE+-]+)")
VAL_RE = re.compile(r"val \(held-out.*?change_err=([\d.eE+-]+)")


def parse(out_path: Path):
    """Return dict with arrays: step, train_cerr, val_step, val_cerr."""
    steps, train_cerr = [], []
    val_steps, val_cerr = [], []
    last_step = None
    for line in out_path.read_text(errors="replace").splitlines():
        m = STEP_RE.match(line)
        if m:
            last_step = int(m.group(1).replace(",", ""))
            steps.append(last_step)
            train_cerr.append(float(m.group(3)))
            continue
        v = VAL_RE.search(line)
        if v and last_step is not None:
            val_steps.append(last_step)
            val_cerr.append(float(v.group(1)))
    return dict(
        step=np.array(steps), train_cerr=np.array(train_cerr),
        val_step=np.array(val_steps), val_cerr=np.array(val_cerr),
    )


def first_zero_step(val_step, val_cerr):
    """First step at which val change_err hits exactly 0 and stays 0 after."""
    z = np.where(val_cerr == 0.0)[0]
    if len(z) == 0:
        return None
    # first index from which all subsequent are zero
    for i in z:
        if np.all(val_cerr[i:] == 0.0):
            return int(val_step[i])
    return int(val_step[z[0]])  # hit zero but not permanently


def main():
    plt.rcParams.update({
        "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 19,
        "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })
    runs = sorted(LOGDIR.glob("*__*.out"))
    games = sorted({p.stem.rsplit("__", 1)[0] for p in runs})
    if not games:
        sys.exit(f"No runs found in {LOGDIR}")

    data = {}
    for p in runs:
        game, cond = p.stem.rsplit("__", 1)
        data[(game, cond)] = parse(p)

    color = {"cosine": "tab:blue", "constant": "tab:red"}
    EPS = 1e-7  # floor for log plot (0 -> EPS)

    fig, axes = plt.subplots(1, len(games), figsize=(6.2 * len(games), 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    summary = []
    summary.append(f"{'game':<22} {'cond':<9} {'final_val_cerr':>15} "
                   f"{'first_perm_zero':>16} {'max_cerr_last20%':>17} {'steps':>7}")
    for ax, game in zip(axes, games):
        for cond in ("cosine", "constant"):
            d = data.get((game, cond))
            if d is None or len(d["val_step"]) == 0:
                continue
            vs, vc = d["val_step"], d["val_cerr"]
            ax.plot(vs, np.maximum(vc, EPS), color=color[cond], lw=2,
                    label=cond, marker="o", ms=3, alpha=0.85)
            fz = first_zero_step(vs, vc)
            tail = vc[vs >= 0.8 * vs.max()] if len(vs) else np.array([np.nan])
            summary.append(
                f"{game:<22} {cond:<9} {vc[-1]:>15.3e} "
                f"{str(fz):>16} {np.nanmax(tail):>17.3e} {int(vs.max()):>7}")
        ax.set_yscale("log")
        ax.set_title(game)
        ax.set_xlabel("training step")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="LR schedule")
    axes[0].set_ylabel("held-out val change_err  (0 plotted at 1e-7)")
    fig.suptitle("LR-schedule ablation: cosine vs constant — val change_err",
                 fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"val_change_err.{ext}")
    print("\n".join(summary))
    (FIGDIR / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nWrote {FIGDIR}/val_change_err.pdf (+.png) and summary.txt")


if __name__ == "__main__":
    main()
