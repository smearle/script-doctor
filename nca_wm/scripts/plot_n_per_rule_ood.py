"""Aggregate heldout_v4_n30 results across the n_per_rule_*_dp_* sweep and
plot OOD error vs n_per_rule for each variant (cond, uncond_h288, nosprites
cond, objperm cond), with the identity baseline for reference.

Restricts to (game, level) pairs present in every run to avoid bias from
skipped pairs varying by run.

Outputs:
  nca_wm/figures/n_per_rule_ood/tf_mean.{pdf,png}    -- main: TF rollout cell-err mean
  nca_wm/figures/n_per_rule_ood/ar_mean.{pdf,png}    -- AR rollout cell-err mean
  nca_wm/figures/n_per_rule_ood/step1.{pdf,png}      -- step-1 cell-err (TF)
  nca_wm/figures/n_per_rule_ood/by_rollout.{pdf,png} -- 2x2 per-rollout-mode breakdown
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(REPO, "nca_wm", "logs")
OUT_DIR = os.path.join(REPO, "nca_wm", "figures", "n_per_rule_ood")

# Variants we plot. Each entry is (label, color, marker, list of (n, [run_dirs]))
# Multiple run_dirs per n means seed-average.
def discover_runs():
    """Discover all available n_per_rule_*_dp_* runs with heldout results."""
    pattern = os.path.join(LOG_DIR, "n_per_rule_*_dp*", "heldout_v4_n30", "results.json")
    runs = {}  # (variant_key, n) -> list of results.json paths
    for path in sorted(glob.glob(pattern)):
        run = path.split(os.sep)[-3]
        # Parse: n_per_rule_<n>_<cond|uncond>_h288?_dp[_<tag>]?_val0.10_s<seed>
        # Strip prefix/suffix
        parts = run.split("_")
        # parts: n_per_rule_<n>_<cond|uncond>...
        n = int(parts[3])
        rest = "_".join(parts[4:])  # e.g. "cond_dp_val0.10_s0" or "uncond_h288_dp_val0.10_s0"
        # Identify variant key
        if rest.startswith("cond_dp_val"):
            key = "cond"
        elif rest.startswith("uncond_h288_dp_val"):
            key = "uncond_h288"
        elif rest.startswith("cond_dp_nosprites_val"):
            key = "cond_nosprites"
        elif rest.startswith("cond_dp_objperm_val"):
            key = "cond_objperm"
        else:
            print(f"  unknown variant: {run}", file=sys.stderr)
            continue
        runs.setdefault((key, n), []).append(path)
    return runs


def load_pairs(results_path):
    """Return dict[(game, lvl)] -> dict[rollout] -> {model_step1, model_mean, identity_step1, identity_mean}."""
    with open(results_path) as f:
        r = json.load(f)
    out = {}
    for game, lvls in r["heldout"].items():
        for lvl, rollouts in lvls.items():
            metrics = {}
            for rname, m in rollouts.items():
                metrics[rname] = {
                    "model_step1": m["model_cell_err_step1"],
                    "model_mean": m["model_cell_err_mean"],
                    "identity_step1": m["identity_cell_err_step1"],
                    "identity_mean": m["identity_cell_err_mean"],
                }
            out[(game, str(lvl))] = metrics
    return out


def aggregate(runs):
    """Return: dict variant -> sorted list of (n, n_seeds, dict[rollout][metric] -> mean over common pairs)."""
    # Find the common pair set across all runs in `runs`.
    pair_sets = []
    parsed = {}  # path -> pairs dict
    for paths in runs.values():
        for p in paths:
            pairs = load_pairs(p)
            parsed[p] = pairs
            pair_sets.append(set(pairs.keys()))
    common = set.intersection(*pair_sets) if pair_sets else set()
    print(f"common (game, level) pairs across {len(parsed)} runs: {len(common)}")

    # Per-variant per-n aggregate
    out = {}
    identity_per_rollout_step1 = defaultdict(list)
    identity_per_rollout_mean = defaultdict(list)
    for (variant, n), paths in runs.items():
        # mean across seeds, mean across common pairs
        per_rollout = defaultdict(lambda: defaultdict(list))  # rollout -> metric -> list[float]
        for p in paths:
            pairs = parsed[p]
            for k in common:
                for rname, m in pairs[k].items():
                    per_rollout[rname]["model_step1"].append(m["model_step1"])
                    per_rollout[rname]["model_mean"].append(m["model_mean"])
                    per_rollout[rname]["identity_step1"].append(m["identity_step1"])
                    per_rollout[rname]["identity_mean"].append(m["identity_mean"])
        agg = {}
        for rname, metrics in per_rollout.items():
            agg[rname] = {k: float(np.mean(v)) for k, v in metrics.items()}
            # collect identity (same regardless of variant)
            identity_per_rollout_step1[rname].extend(metrics["identity_step1"])
            identity_per_rollout_mean[rname].extend(metrics["identity_mean"])
        out.setdefault(variant, []).append((n, len(paths), agg))
    for v in out:
        out[v].sort(key=lambda x: x[0])
    identity_baseline = {
        r: {"step1": float(np.mean(identity_per_rollout_step1[r])),
            "mean": float(np.mean(identity_per_rollout_mean[r]))}
        for r in identity_per_rollout_step1
    }
    return out, identity_baseline


VARIANT_STYLE = {
    "cond":           {"label": "Conditional",                 "color": "#1f77b4", "marker": "o"},
    "uncond_h288":    {"label": "Unconditional (h=288)",       "color": "#d62728", "marker": "s"},
    "cond_nosprites": {"label": "Cond, no sprite encoder",     "color": "#2ca02c", "marker": "^"},
    "cond_objperm":   {"label": "Cond + obj-perm aug",         "color": "#9467bd", "marker": "D"},
}


def plot_metric(agg, identity_baseline, rollout, metric_key, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    plotted_any = False
    for variant, series in agg.items():
        style = VARIANT_STYLE.get(variant, {"label": variant, "color": "k", "marker": "o"})
        xs, ys, n_seeds_list = [], [], []
        for n, n_seeds, per_rollout in series:
            if rollout not in per_rollout or metric_key not in per_rollout[rollout]:
                continue
            xs.append(n); ys.append(per_rollout[rollout][metric_key] * 100); n_seeds_list.append(n_seeds)
        if not xs:
            continue
        plotted_any = True
        ax.plot(xs, ys, marker=style["marker"], color=style["color"], label=style["label"],
                linewidth=1.6, markersize=6)
        for x, y, s in zip(xs, ys, n_seeds_list):
            if s > 1:
                ax.annotate(f"s={s}", (x, y), fontsize=7, color=style["color"],
                            textcoords="offset points", xytext=(4, 4))
    # identity baseline
    if rollout in identity_baseline:
        ident = identity_baseline[rollout][metric_key.replace("model_", "")] * 100
        ax.axhline(ident, color="0.4", linestyle="--", linewidth=1.2,
                   label=f"Identity baseline ({ident:.2f}%)")
    ax.set_xscale("log")
    ax.set_xlabel("Games per rule count (n_per_rule)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=160)
    plt.close(fig)
    return plotted_any


def plot_by_rollout(agg, identity_baseline, metric_key, out_path):
    rollouts = ["random_tf", "random", "bfs", "astar"]
    rollout_titles = {"random_tf": "Random actions, teacher-forced",
                      "random": "Random actions, autoregressive",
                      "bfs": "BFS solver trajectory",
                      "astar": "A* solver trajectory"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, rollout in zip(axes.flat, rollouts):
        for variant, series in agg.items():
            style = VARIANT_STYLE.get(variant, {"label": variant, "color": "k", "marker": "o"})
            xs, ys = [], []
            for n, _, per_rollout in series:
                if rollout in per_rollout and metric_key in per_rollout[rollout]:
                    xs.append(n); ys.append(per_rollout[rollout][metric_key] * 100)
            if xs:
                ax.plot(xs, ys, marker=style["marker"], color=style["color"], label=style["label"],
                        linewidth=1.4, markersize=5)
        if rollout in identity_baseline:
            ident = identity_baseline[rollout][metric_key.replace("model_", "")] * 100
            ax.axhline(ident, color="0.4", linestyle="--", linewidth=1.0,
                       label=f"Identity ({ident:.2f}%)")
        ax.set_xscale("log")
        ax.set_title(rollout_titles[rollout], fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.tick_params(labelsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Games per rule count (n_per_rule)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cell err mean (%)", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("OOD heldout cell-err mean by rollout mode (Heldout-26)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = discover_runs()
    print(f"discovered {sum(len(v) for v in runs.values())} run dirs across {len(runs)} (variant, n) cells")
    for (variant, n), paths in sorted(runs.items()):
        print(f"  {variant:>16s}  n={n:<4d}  seeds={len(paths)}")
    agg, identity = aggregate(runs)
    print("\nidentity baseline (heldout-26):")
    for r, m in identity.items():
        print(f"  {r:>10s}: step1={m['step1']*100:.2f}%  mean={m['mean']*100:.2f}%")

    plot_metric(agg, identity, "random_tf", "model_mean",
                "Cell err mean (%) -- teacher-forced rollout",
                "OOD Heldout-26: TF rollout mean error vs n_per_rule",
                os.path.join(OUT_DIR, "tf_mean"))
    plot_metric(agg, identity, "random", "model_mean",
                "Cell err mean (%) -- autoregressive rollout",
                "OOD Heldout-26: AR rollout mean error vs n_per_rule",
                os.path.join(OUT_DIR, "ar_mean"))
    plot_metric(agg, identity, "random_tf", "model_step1",
                "Cell err step 1 (%)",
                "OOD Heldout-26: 1-step error vs n_per_rule",
                os.path.join(OUT_DIR, "step1"))
    plot_by_rollout(agg, identity, "model_mean",
                    os.path.join(OUT_DIR, "by_rollout"))

    # print a compact table to stdout
    print("\n=== TF rollout mean (cell err %) ===")
    print(f"{'variant':>16s} {'n=1':>7s} {'n=2':>7s} {'n=3':>7s} {'n=5':>7s} {'n=10':>7s} {'n=20':>7s} {'n=50':>7s} {'n=100':>7s} {'n=200':>7s} {'n=400':>7s} {'n=800':>7s}")
    ns_show = [1, 2, 3, 5, 10, 20, 50, 100, 200, 400, 800]
    for variant, series in agg.items():
        per_n = {n: per_rollout for n, _, per_rollout in series}
        row = [f"{variant:>16s}"]
        for n in ns_show:
            if n in per_n and "random_tf" in per_n[n]:
                row.append(f"{per_n[n]['random_tf']['model_mean']*100:6.2f}")
            else:
                row.append("    --")
        print(" ".join(row))
    if "random_tf" in identity:
        print(f"{'identity':>16s} " + " ".join(f"{identity['random_tf']['mean']*100:6.2f}" for _ in ns_show))

    print(f"\nfigures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
