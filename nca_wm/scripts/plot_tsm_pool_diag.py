"""Plot the TSM global-pooling diagnostic: train vs held-out-val changed-cell
error over training, for each (pooling x input_skip) cell.

Parses the dense per-500-step stdout logs in logs/tsm_pool_diag/<cfg>.out
(`step N ... change_err=X` and `val (held-out ...) change_err=Y`). Writes
paper-ready PDF+PNG to figures/tsm_pool_diag/ plus a summary table.

Usage: .venv/bin/python3 nca_wm/scripts/plot_tsm_pool_diag.py
"""
from __future__ import annotations
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGDIR = os.path.join(REPO, "nca_wm", "logs", "tsm_pool_diag")
OUTDIR = os.path.join(REPO, "nca_wm", "figures", "tsm_pool_diag")

STEP_RE = re.compile(r"^\s*step\s+([\d,]+)/[\d,]+.*?change_err=([\d.eE+-]+)")
VAL_RE = re.compile(r"val \(held-out[^)]*\):.*?change_err=([\d.eE+-]+)")

CFGS = ["pool_off_skip_on", "pool_on_skip_on", "pool_off_skip_off", "pool_on_skip_off"]
STYLE = {
    "pool_off_skip_on":  {"label": "pool OFF + skip",  "color": "#1f77b4"},
    "pool_on_skip_on":   {"label": "pool ON + skip",   "color": "#d62728"},
    "pool_off_skip_off": {"label": "pool OFF, no skip", "color": "#2ca02c"},
    "pool_on_skip_off":  {"label": "pool ON, no skip",  "color": "#ff7f0e"},
}


def parse(cfg):
    path = os.path.join(LOGDIR, f"{cfg}.out")
    if not os.path.isfile(path):
        return None
    steps, tr, vsteps, va = [], [], [], []
    cur_step = None
    with open(path) as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                cur_step = int(m.group(1).replace(",", ""))
                steps.append(cur_step); tr.append(float(m.group(2)))
                continue
            mv = VAL_RE.search(line)
            if mv and cur_step is not None:
                vsteps.append(cur_step); va.append(float(mv.group(1)))
    return dict(steps=np.array(steps), tr=np.array(tr),
               vsteps=np.array(vsteps), va=np.array(va))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = {c: parse(c) for c in CFGS}
    data = {c: d for c, d in data.items() if d is not None and len(d["steps"])}

    # floor for log plot
    floor = 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, key, ttl in [(axes[0], "tr", "Train changed-cell error"),
                          (axes[1], "va", "Held-out (val) changed-cell error")]:
        for c, d in data.items():
            st = d["steps"] if key == "tr" else d["vsteps"]
            y = d[key].copy()
            if len(st) == 0:
                continue
            y = np.clip(y, floor, None)
            ax.plot(st, y, color=STYLE[c]["color"], label=STYLE[c]["label"], lw=1.8)
        ax.set_yscale("log")
        ax.set_xlabel("training step", fontsize=13)
        ax.set_title(ttl, fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel("changed-cell error (log)", fontsize=13)
    axes[0].legend(fontsize=11, loc="upper right")
    fig.suptitle("Travelling_salesman single-game: pooling × input_skip", fontsize=15)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"curves.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # summary table (final values)
    print(f"{'config':>20s} {'last_step':>10s} {'train_cerr':>12s} {'val_cerr':>12s} {'min_val_cerr':>13s}")
    rows = []
    for c in CFGS:
        d = data.get(c)
        if d is None or not len(d["steps"]):
            print(f"{c:>20s} {'<none>':>10s}"); continue
        last_step = int(d["steps"][-1])
        tr = d["tr"][-1]
        va = d["va"][-1] if len(d["va"]) else float("nan")
        minva = float(np.min(d["va"])) if len(d["va"]) else float("nan")
        print(f"{c:>20s} {last_step:>10,d} {tr:>12.3e} {va:>12.3e} {minva:>13.3e}")
        rows.append((c, last_step, tr, va, minva))
    # write md summary
    with open(os.path.join(OUTDIR, "summary.md"), "w") as f:
        f.write("# TSM pooling × input_skip — single-game, all levels, val_frac=0.1\n\n")
        f.write("| config | last step | train change_err | val change_err | min val change_err |\n")
        f.write("|---|---|---|---|---|\n")
        for c, ls, tr, va, mv in rows:
            f.write(f"| {STYLE[c]['label']} | {ls:,} | {tr:.3e} | {va:.3e} | {mv:.3e} |\n")
    print(f"\nfigures + summary -> {OUTDIR}/")


if __name__ == "__main__":
    main()
