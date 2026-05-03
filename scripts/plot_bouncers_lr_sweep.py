"""Paper-ready figures for the Bouncers n_layers × n_repeats sweep.

Auto-discovers runs in `nca_wm/logs/single_Bouncers_L*_R*/`. Re-running this
script after each new config finishes adds a line to the figures without
code edits.

Outputs (both PDF and PNG, in nca_wm/figures/bouncers_lr_sweep/):
  1. train_loss_curves.{pdf,png}       — change_acc / loss vs step, one line per (L,R)
  2. final_rollout_bars.{pdf,png}      — final autoregressive rollout cell-error per (L,R)
  3. params_vs_rollout.{pdf,png}       — efficiency frontier: params vs rollout error
  4. summary_table.csv                  — all numbers in a tidy CSV for the paper

Usage:
    .venv/bin/python3 scripts/plot_bouncers_lr_sweep.py
"""
from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "nca_wm" / "logs"
OUT = ROOT / "nca_wm" / "figures" / "bouncers_lr_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# Paper-ready styling.
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def discover_runs():
    """Return list of (variant, L, R, run_dir).

    Variants: 'pool' (default), 'nopool' (no global-context flags),
    'nopool_stab' (LN+input_skip bundled), 'nopool_lnonly', 'nopool_skiponly'.
    """
    pat_full = re.compile(r"single_Bouncers_(?:(\w+?)_)?L(\d+)_R(\d+)$")
    runs = []
    for d in sorted(LOGS.glob("single_Bouncers_*L*_R*")):
        m = pat_full.search(d.name)
        if not m:
            continue
        variant = m.group(1) or "pool"
        L, R = int(m.group(2)), int(m.group(3))
        runs.append((variant, L, R, d))
    return runs


def load_run(variant, L, R, run_dir):
    """Return dict with keys: variant, L, R, total, curves, eval_npz."""
    info = {"variant": variant, "L": L, "R": R, "total": L * R, "run_dir": run_dir}
    # Final config to extract param count
    cfg_path = run_dir / "config.json"
    info["cfg"] = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    # Latest curves file
    curves_files = sorted(run_dir.glob("curves_step*.npz"),
                          key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
    info["curves_path"] = curves_files[-1] if curves_files else None
    # Eval file
    info["eval_path"] = (run_dir / "eval_multigame.npz") if (run_dir / "eval_multigame.npz").exists() else None
    # train_meta has best_loss and total_steps
    meta_path = run_dir / "train_meta.json"
    info["meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # params count: load params.pkl if present, count leaves
    return info


def load_curves(info):
    if info["curves_path"] is None:
        return None
    d = np.load(info["curves_path"], allow_pickle=True)
    losses = d["losses"] if "losses" in d.files else None
    change_accs = d["change_accs"] if "change_accs" in d.files else None
    return {"losses": losses, "change_accs": change_accs}


def per_level_rollout(eval_path, algo: str):
    """Return list of (level_idx, final_step_err) for `algo` ∈ {astar, bfs, random, random_tf}.
    Bouncers has no winning transitions so astar isn't recorded — caller falls
    back to bfs in that case."""
    if eval_path is None:
        return []
    d = np.load(eval_path, allow_pickle=True)
    out = []
    pattern = re.compile(rf"^(.+)_L(\d+)_{algo}_cell_error_rate$")
    for k in d.files:
        m = pattern.match(k)
        if not m:
            continue
        level = int(m.group(2))
        v = d[k]
        if v.size == 0:
            continue
        out.append((level, float(v[-1])))
    return sorted(out)


def headline_metric(eval_path):
    """Pick the algo that covers the most levels (most representative summary),
    breaking ties astar > bfs. Returns (algo_used, list of (level, final_err))."""
    a = per_level_rollout(eval_path, "astar")
    b = per_level_rollout(eval_path, "bfs")
    if not a and not b:
        return None, []
    if len(b) > len(a):
        return "bfs", b
    if len(a) > len(b):
        return "astar", a
    # tie: prefer astar
    return ("astar", a) if a else ("bfs", b)


def count_params(run_dir):
    """Best-effort: count params from saved pickle."""
    pkl = run_dir / "params.pkl"
    if not pkl.exists():
        return None
    try:
        import pickle
        with pkl.open("rb") as f:
            params = pickle.load(f)
        import jax
        return int(sum(np.size(p) for p in jax.tree_util.tree_leaves(params)))
    except Exception:
        return None


def palette(runs):
    """Color by (L, R); pool=solid, nopool=hatched/dashed handled at plot time."""
    pool_runs = sorted({(r["L"], r["R"]) for r in runs}, key=lambda x: (x[0]*x[1], x[1]))
    cmap = plt.get_cmap("viridis")
    n = max(len(pool_runs) - 1, 1)
    return {lr: cmap(i / n) for i, lr in enumerate(pool_runs)}


def label(L, R, variant=None):
    base = f"L={L}, R={R} (steps={L*R})"
    if variant == "nopool":
        return base + "  [no pool]"
    return base


def fig_train_loss(runs, colors):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in runs:
        cv = load_curves(r)
        if cv is None or cv["change_accs"] is None:
            continue
        ca = np.asarray(cv["change_accs"])
        x = np.arange(len(ca))
        if len(x) > 200:
            stride = len(x) // 200
            x = x[::stride]
            ca = ca[::stride]
        linestyle = "--" if r["variant"] == "nopool" else "-"
        ax.plot(x, 1.0 - ca,
                color=colors[(r["L"], r["R"])],
                linestyle=linestyle,
                label=label(r["L"], r["R"], r["variant"]),
                linewidth=1.6,
                alpha=0.85)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Train change-cell error  (1 − change\\_acc)")
    ax.set_yscale("log")
    ax.set_title("Bouncers: convergence by (n\\_layers, n\\_repeats)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "train_loss_curves.pdf")
    fig.savefig(OUT / "train_loss_curves.png", dpi=180)
    plt.close(fig)


def fig_final_rollout_bars(runs, colors):
    """Bar chart: per-(L,R), mean astar cell-error over levels (final-step)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for r in runs:
        algo, levels = headline_metric(r["eval_path"])
        if not levels:
            continue
        r["headline_algo"] = algo
        mean_err = float(np.mean([v for _, v in levels]))
        rows.append((r, mean_err, len(levels)))
    # Group by (L,R); plot pool and nopool side by side per (L,R) bucket.
    cfgs = sorted({(r["L"], r["R"]) for r, _, _ in rows}, key=lambda lr: (lr[0]*lr[1], lr[1]))
    by_variant = {(r["variant"], r["L"], r["R"]): m for r, m, _ in rows}
    if not rows:
        ax.text(0.5, 0.5, "No eval results yet — rerun after sweep completes.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        x = np.arange(len(cfgs))
        w = 0.4
        for i, var in enumerate(["pool", "nopool"]):
            ys = [by_variant.get((var, L, R), np.nan) for (L, R) in cfgs]
            cs = [colors[(L, R)] for (L, R) in cfgs]
            offsets = -w/2 if var == "pool" else +w/2
            hatch = None if var == "pool" else "//"
            ax.bar(x + offsets, ys, w, color=cs, edgecolor="black",
                   linewidth=0.5, hatch=hatch,
                   label=("with pool" if var == "pool" else "no pool"))
            for xi, yi in zip(x + offsets, ys):
                if not np.isnan(yi):
                    ax.text(xi, yi + 0.0005, f"{yi:.3f}", ha="center", va="bottom",
                            fontsize=8, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([label(L, R) for (L, R) in cfgs], rotation=20, ha="right")
        ax.set_ylabel("Mean rollout cell-error  (BFS or A*, final-step, avg over levels)")
        ax.set_title("Bouncers: rollout error — pool vs no pool")
        ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "final_rollout_bars.pdf")
    fig.savefig(OUT / "final_rollout_bars.png", dpi=180)
    plt.close(fig)


def fig_params_vs_rollout(runs, colors):
    """Efficiency frontier: x=params, y=rollout error."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pts = []
    for r in runs:
        algo, levels = headline_metric(r["eval_path"])
        if not levels:
            continue
        n_params = count_params(r["run_dir"])
        if n_params is None:
            continue
        mean_err = float(np.mean([v for _, v in levels]))
        pts.append((r, n_params, mean_err))
    if not pts:
        ax.text(0.5, 0.5, "Awaiting eval + params.", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for r, n, e in pts:
            marker = "o" if r["variant"] == "pool" else "s"
            ax.scatter(n, e, color=colors[(r["L"], r["R"])], marker=marker, s=140,
                       edgecolor="black", linewidth=0.7, zorder=3,
                       label=label(r["L"], r["R"], r["variant"]))
        # Dedup-legend: keep one per (L,R,variant) — already unique above.
        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles, labels_, loc="upper right", framealpha=0.9, fontsize=9)
        ax.set_xlabel("Total parameter count")
        ax.set_ylabel("Mean A* rollout cell-error")
        ax.set_xscale("log")
        ax.set_title("Bouncers: parameter efficiency")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "params_vs_rollout.pdf")
    fig.savefig(OUT / "params_vs_rollout.png", dpi=180)
    plt.close(fig)


def fig_stab_disagg(runs):
    """Paper figure: 4-cell ablation (bare, LN-only, skip-only, bundled) on
    the two regimes (L=1,R=16 max-shared) and (L=16,R=1 per-step deep).
    Grouped bar chart with cell colors highlighting the active component."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    regimes = [
        ("(L=1, R=16)\nmax-shared", 1, 16),
        ("(L=16, R=1)\nper-step deep", 16, 1),
    ]
    cell_order = [
        ("nopool",          "bare"),
        ("nopool_lnonly",   "LN only"),
        ("nopool_skiponly", "input_skip only"),
        ("nopool_stab",     "bundled (LN + skip)"),
    ]
    cell_colors = {
        "bare": "#888",
        "LN only": "#7777cc",
        "input_skip only": "#33aa33",
        "bundled (LN + skip)": "#cc7733",
    }
    for ax, (title, L, R) in zip(axes, regimes):
        ys = []
        labels = []
        cs = []
        for variant, label in cell_order:
            err = None
            for r in runs:
                if (r["variant"] == variant and r["L"] == L and r["R"] == R
                        and r["eval_path"] is not None):
                    levels = per_level_rollout(r["eval_path"], "bfs")
                    if levels:
                        err = float(np.mean([v for _, v in levels]))
                    break
            if err is not None:
                ys.append(err)
                labels.append(label)
                cs.append(cell_colors[label])
        x = np.arange(len(ys))
        bars = ax.bar(x, ys, color=cs, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        for xi, yi in zip(x, ys):
            ax.text(xi, yi + 0.0008, f"{yi*100:.2f}%", ha="center", va="bottom",
                    fontsize=10)
        ax.set_title(title)
        ax.set_ylim(0, max(ys) * 1.25 if ys else 0.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("BFS rollout cell-error (mean over levels)")
    fig.suptitle("Bouncers (no pool): LN vs input_skip ablation",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "stab_disagg.pdf")
    fig.savefig(OUT / "stab_disagg.png", dpi=180)
    plt.close(fig)


def write_summary_csv(runs):
    rows = ["variant,L,R,total_steps,n_params,best_train_loss,headline_algo,mean_rollout_err,bfs_err,random_err,n_levels_eval"]
    for r in sorted(runs, key=lambda x: (x["variant"], x["total"], x["R"])):
        algo, levels = headline_metric(r["eval_path"])
        mean_err = float(np.mean([v for _, v in levels])) if levels else float("nan")
        bfs_lvls = per_level_rollout(r["eval_path"], "bfs")
        rnd_lvls = per_level_rollout(r["eval_path"], "random")
        bfs_err = float(np.mean([v for _, v in bfs_lvls])) if bfs_lvls else float("nan")
        rnd_err = float(np.mean([v for _, v in rnd_lvls])) if rnd_lvls else float("nan")
        n_levels = len(levels)
        n_params = count_params(r["run_dir"])
        best_loss = r["meta"].get("best_loss", float("nan"))
        rows.append(f"{r['variant']},{r['L']},{r['R']},{r['total']},{n_params},{best_loss:.6e},{algo or 'NA'},{mean_err:.4f},{bfs_err:.4f},{rnd_err:.4f},{n_levels}")
    (OUT / "summary_table.csv").write_text("\n".join(rows) + "\n")


def main():
    discovered = discover_runs()
    if not discovered:
        print(f"No runs found under {LOGS}/single_Bouncers_L*_R*. Nothing to plot.")
        return
    runs = [load_run(variant, L, R, d) for variant, L, R, d in discovered]
    print(f"Discovered {len(runs)} run(s):")
    for r in runs:
        has_curves = r["curves_path"] is not None
        has_eval = r["eval_path"] is not None
        print(f"  [{r['variant']:6s}] L={r['L']} R={r['R']} (steps={r['total']})  "
              f"curves={'Y' if has_curves else 'N'}  eval={'Y' if has_eval else 'N'}")
    colors = palette(runs)
    fig_train_loss(runs, colors)
    fig_final_rollout_bars(runs, colors)
    fig_params_vs_rollout(runs, colors)
    fig_stab_disagg(runs)
    write_summary_csv(runs)
    print(f"Wrote figures + CSV to {OUT}")


if __name__ == "__main__":
    main()
