"""Per-game change_acc diagnostics for multi_scaling_gallery_* runs.

Three artifacts per invocation:

1. ``laggards.csv`` — every (run, game) final change_acc, sorted ascending.
2. ``laggards_<run>.pdf/.png`` — grid of change_acc curves for the bottom
   ``--top_k`` games of each run, sorted worst-first. Lets us see whether
   the lagging games are flat or still descending at the end of training.
3. ``heatmap.pdf/.png`` — runs (rows) by games (cols, sorted by mean final
   change_acc ascending), colored by final change_acc. Quickly shows which
   games are stuck across all runs.

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_multi_scaling_gallery_per_game.py
    .venv/bin/python3 nca_wm/scripts/plot_multi_scaling_gallery_per_game.py \
        --pattern 'multi_scaling_gallery_v4_*' --top_k 30
"""
from __future__ import annotations

import argparse
import glob
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


def _per_game(z):
    out = {}
    for k in z.files:
        m = re.match(r"^per_game_(.+)_change_acc$", k)
        if not m:
            continue
        g = m.group(1)
        step_key = f"per_game_{g}_step"
        if step_key not in z.files:
            continue
        step = np.asarray(z[step_key])
        val = np.asarray(z[k], dtype=float)
        n = min(step.size, val.size)
        if n == 0:
            continue
        out[g] = (step[:n], val[:n])
    return out


def _short(name):
    return re.sub(r"^multi_scaling_gallery_", "", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="nca_wm/logs")
    ap.add_argument("--pattern", default="multi_scaling_gallery_*")
    ap.add_argument("--exclude", default="aborted")
    ap.add_argument("--outdir", default="nca_wm/figures/multi_scaling_gallery_per_game")
    ap.add_argument("--top_k", type=int, default=30,
                    help="number of worst games to plot per run")
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
        runs.append((name, cp))

    if not runs:
        print("no runs matched")
        return

    # Load per-run per-game series.
    pg_by_run = {}
    for name, cp in runs:
        z = np.load(cp)
        pg_by_run[name] = _per_game(z)
        print(f"{name}: {len(pg_by_run[name])} games")

    # Build the laggards CSV (every run × game final change_acc, sorted).
    rows = []
    for name, pg in pg_by_run.items():
        for g, (step, val) in pg.items():
            rows.append((name, g, int(step[-1]), float(val[-1])))
    rows.sort(key=lambda r: r[3])
    csv_path = os.path.join(outdir, "laggards.csv")
    with open(csv_path, "w") as fh:
        fh.write("run,game,final_step,final_change_acc\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f}\n")
    print("wrote", csv_path)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.titlesize": 14,
    })

    # Per-run grid of bottom-K games.
    for name, pg in pg_by_run.items():
        if not pg:
            continue
        ranked = sorted(pg.items(), key=lambda kv: kv[1][1][-1])
        sel = ranked[:args.top_k]
        n = len(sel)
        cols = 5
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols,
                                 figsize=(3.0 * cols, 2.0 * rows_n),
                                 sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, (g, (step, val)) in zip(axes, sel):
            ax.plot(step, val, color="C0", linewidth=1.4)
            ax.axhline(1.0, color="0.7", linestyle=":", linewidth=0.8)
            ax.set_ylim(0.0, 1.02)
            ax.set_title(f"{g}\nfinal={val[-1]:.3f}")
            ax.grid(True, alpha=0.3)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"{_short(name)} — bottom {n} games by final change_acc")
        fig.supxlabel("step")
        fig.supylabel("change_acc")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf = os.path.join(outdir, f"laggards_{_short(name)}.pdf")
        png = os.path.join(outdir, f"laggards_{_short(name)}.png")
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", pdf)

    # Heatmap across runs.
    all_games = sorted({g for pg in pg_by_run.values() for g in pg.keys()})
    run_names = [name for name, _ in runs]
    M = np.full((len(run_names), len(all_games)), np.nan)
    for i, name in enumerate(run_names):
        pg = pg_by_run[name]
        for j, g in enumerate(all_games):
            if g in pg:
                M[i, j] = pg[g][1][-1]
    # Sort columns by mean final change_acc ascending (worst first).
    col_mean = np.nanmean(M, axis=0)
    order = np.argsort(col_mean)
    M_s = M[:, order]
    games_s = [all_games[j] for j in order]

    fig_w = max(14, 0.08 * len(games_s) + 4)
    fig_h = max(4, 0.45 * len(run_names) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(M_s, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(run_names)))
    ax.set_yticklabels([_short(n) for n in run_names])
    # Tick every Nth game to keep labels legible.
    stride = max(1, len(games_s) // 60)
    xt = list(range(0, len(games_s), stride))
    ax.set_xticks(xt)
    ax.set_xticklabels([games_s[j] for j in xt], rotation=90, fontsize=6)
    ax.set_xlabel("game (sorted by mean final change_acc, worst first)")
    fig.colorbar(im, ax=ax, label="final change_acc")
    ax.set_title("multi_scaling_gallery — per-game final change_acc")
    fig.tight_layout()
    pdf = os.path.join(outdir, "heatmap.pdf")
    png = os.path.join(outdir, "heatmap.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
