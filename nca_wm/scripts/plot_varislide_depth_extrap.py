"""Plot varislide depth-extrapolation results.

Inputs: depth_extrap_eval.json files written by eval_varislide_depth_extrap.py.

Generates paper-ready figures (PDF + PNG, large fonts, tight layout) into
nca_wm/figures/varislide_depth_extrap/:

  fig_argmax_vs_Deval.{pdf,png}
      Per train-depth panel; x = D_eval, y = argmax accuracy averaged over
      slide-distance buckets {short (d≤2), medium (3-7), long (≥8)}.
      One line per width regime (in-train widths {6,8} vs OOD {10,12,16}).

  fig_argmax_by_distance.{pdf,png}
      Heatmap per train-depth: rows = slide distance d, cols = D_eval, cell
      = argmax-correct (averaged over seeds and OOD widths).

  summary.csv / summary.md
      Per (D_train, D_eval, slide_d_bucket) seed-mean argmax accuracy.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.constrained_layout.use": True,
})

DIST_BUCKETS = [
    ("short (d≤2)", lambda d: d <= 2),
    ("med (3-7)",   lambda d: 3 <= d <= 7),
    ("long (d≥8)",  lambda d: d >= 8),
]
TRAIN_WIDTHS = {6, 8}


def _bucketize(per_d_entries, bucket_fn):
    n_total = 0
    n_correct = 0
    for e in per_d_entries:
        if bucket_fn(e["d"]):
            n_total += e["n"]
            n_correct += int(round(e["argmax_correct"] * e["n"]))
    return n_correct, n_total


def _load_runs(run_dirs):
    """Returns dict[(D_train, pool_on, seed)] = parsed json."""
    out = {}
    for d in run_dirs:
        path = os.path.join(d, "depth_extrap_eval.json")
        if not os.path.exists(path):
            print(f"  [warn] no eval json at {path}; skip")
            continue
        j = json.load(open(path))
        cfg = j["config"]
        d_train = int(cfg["n_nca_steps"])
        pool_on = bool(cfg.get("axis_pool") or cfg.get("axis_cummax")
                       or cfg.get("global_pool"))
        # Recover seed from save_dir tag of the form .../<tag>_s<seed>
        tag = os.path.basename(d.rstrip("/"))
        seed = None
        if "_s" in tag:
            try:
                seed = int(tag.rsplit("_s", 1)[-1])
            except ValueError:
                pass
        if seed is None:
            seed = 0
        out[(d_train, pool_on, seed)] = j
    return out


def _aggregate(runs, *, pool_on: bool):
    """Returns nested dict: D_train -> D_eval -> bucket_label -> list of acc per (seed, width)."""
    nested = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))
    for (d_train, p_on, seed), j in runs.items():
        if p_on != pool_on:
            continue
        for d_entry in j["depths"]:
            d_eval = d_entry["D_eval"]
            for w_entry in d_entry["per_width"]:
                w = w_entry["width"]
                width_kind = "train" if w in TRAIN_WIDTHS else "ood"
                for blabel, bfn in DIST_BUCKETS:
                    n_correct, n_total = _bucketize(w_entry["per_d"], bfn)
                    if n_total == 0:
                        continue
                    nested[d_train][d_eval][(blabel, width_kind)].append(
                        n_correct / n_total)
    return nested


def _save_summary(nested_nopool, out_dir):
    rows = []
    for d_train in sorted(nested_nopool.keys()):
        for d_eval in sorted(nested_nopool[d_train].keys()):
            for key, accs in nested_nopool[d_train][d_eval].items():
                blabel, width_kind = key
                rows.append({
                    "D_train": d_train,
                    "D_eval": d_eval,
                    "slide_bucket": blabel,
                    "width_kind": width_kind,
                    "n_runs": len(accs),
                    "argmax_mean": float(np.mean(accs)),
                    "argmax_std": float(np.std(accs)),
                })
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
        if w is not None:
            w.writeheader()
            w.writerows(rows)
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Varislide depth-extrapolation summary (pool OFF)\n\n")
        f.write("Per (D_train, D_eval, slide-distance bucket, width regime), "
                "seed-mean argmax-correct.\n\n")
        f.write("| D_train | D_eval | bucket | width | n | mean | std |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['D_train']} | {r['D_eval']} | {r['slide_bucket']} | "
                    f"{r['width_kind']} | {r['n_runs']} | "
                    f"{r['argmax_mean']:.3f} | {r['argmax_std']:.3f} |\n")
    print(f"  wrote {csv_path}, {md_path}")


def _plot_argmax_vs_Deval(nested_nopool, nested_pool, out_dir):
    train_depths = sorted(nested_nopool.keys())
    n_panels = len(train_depths)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.6),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    bucket_colors = {"short (d≤2)": "tab:green",
                     "med (3-7)":   "tab:orange",
                     "long (d≥8)":  "tab:red"}
    width_styles = {"train": "-", "ood": "--"}

    for ax, d_train in zip(axes, train_depths):
        per_d_eval = nested_nopool[d_train]
        d_evals = sorted(per_d_eval.keys())
        for blabel, _ in DIST_BUCKETS:
            for w_kind in ["train", "ood"]:
                ys, xs, errs = [], [], []
                for d_eval in d_evals:
                    accs = per_d_eval[d_eval].get((blabel, w_kind), [])
                    if not accs:
                        continue
                    xs.append(d_eval)
                    ys.append(np.mean(accs))
                    errs.append(np.std(accs))
                if xs:
                    ax.errorbar(xs, ys, yerr=errs,
                                color=bucket_colors[blabel],
                                linestyle=width_styles[w_kind],
                                marker="o" if w_kind == "train" else "s",
                                markersize=5, capsize=2, elinewidth=0.8,
                                label=f"{blabel} ({w_kind})")
        ax.axvline(d_train, color="black", linestyle=":", alpha=0.4,
                   label=f"D_train={d_train}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("D_eval (NCA steps at inference)")
        ax.set_title(f"D_train = {d_train} (pool OFF)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("argmax-correct (right-action slides)")

    # Single legend below the panels
    handles, labels = axes[-1].get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend([h for h, _ in uniq], [l for _, l in uniq],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Iteration extrapolation in varislide NCA WM (pool OFF)")
    out_pdf = os.path.join(out_dir, "fig_argmax_vs_Deval.pdf")
    out_png = os.path.join(out_dir, "fig_argmax_vs_Deval.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {out_pdf}\n  wrote {out_png}")


def _plot_heatmap_by_distance(runs, out_dir, *, pool_on: bool, suffix: str):
    """Heatmap of argmax-correct vs (slide distance, D_eval) per D_train.

    Aggregates over seeds and all eval widths.
    """
    # nested[d_train][d_eval][d_slide] -> list of accs
    nested = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))
    counts = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(int)))
    for (d_train, p_on, seed), j in runs.items():
        if p_on != pool_on:
            continue
        for d_entry in j["depths"]:
            d_eval = d_entry["D_eval"]
            for w_entry in d_entry["per_width"]:
                for x in w_entry["per_d"]:
                    d = x["d"]
                    nested[d_train][d_eval][d].append(x["argmax_correct"] * x["n"])
                    counts[d_train][d_eval][d] += x["n"]

    train_depths = sorted(nested.keys())
    if not train_depths:
        return
    fig, axes = plt.subplots(1, len(train_depths),
                             figsize=(4.4 * len(train_depths), 4.0),
                             sharey=True)
    if len(train_depths) == 1:
        axes = [axes]

    # Determine axes
    all_d_eval = sorted({de for d_train in train_depths
                         for de in nested[d_train]})
    all_d_slide = sorted({d for d_train in train_depths
                          for de in nested[d_train]
                          for d in nested[d_train][de]})
    for ax, d_train in zip(axes, train_depths):
        H = np.full((len(all_d_slide), len(all_d_eval)), np.nan)
        for j_eval, d_eval in enumerate(all_d_eval):
            for i_d, d_slide in enumerate(all_d_slide):
                tot = counts[d_train][d_eval].get(d_slide, 0)
                if tot > 0:
                    H[i_d, j_eval] = sum(nested[d_train][d_eval][d_slide]) / tot
        im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=1,
                       extent=[-0.5, len(all_d_eval) - 0.5,
                               -0.5, len(all_d_slide) - 0.5])
        ax.set_xticks(range(len(all_d_eval)))
        ax.set_xticklabels([str(x) for x in all_d_eval])
        ax.set_yticks(range(len(all_d_slide)))
        ax.set_yticklabels([str(x) for x in all_d_slide])
        ax.set_xlabel("D_eval")
        ax.set_title(f"D_train = {d_train}{suffix}")
    axes[0].set_ylabel("slide distance d")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("argmax-correct")
    pool_tag = "pool ON" if pool_on else "pool OFF"
    fig.suptitle(f"Per-distance argmax accuracy vs inference depth ({pool_tag})")
    out_pdf = os.path.join(out_dir,
                           f"fig_argmax_by_distance{'_pool' if pool_on else ''}.pdf")
    out_png = os.path.join(out_dir,
                           f"fig_argmax_by_distance{'_pool' if pool_on else ''}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {out_pdf}\n  wrote {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out_dir",
                   default="/home/jupyter-smearle/script-doctor/nca_wm/figures/varislide_depth_extrap")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = _load_runs(args.runs)
    if not runs:
        print("no runs loaded")
        sys.exit(1)
    print(f"loaded {len(runs)} run-eval files")
    nested_nopool = _aggregate(runs, pool_on=False)
    nested_pool = _aggregate(runs, pool_on=True)
    _save_summary(nested_nopool, args.out_dir)
    _plot_argmax_vs_Deval(nested_nopool, nested_pool, args.out_dir)
    _plot_heatmap_by_distance(runs, args.out_dir, pool_on=False, suffix=" (pool OFF)")
    if nested_pool:
        _plot_heatmap_by_distance(runs, args.out_dir, pool_on=True, suffix=" (pool ON)")


if __name__ == "__main__":
    main()
