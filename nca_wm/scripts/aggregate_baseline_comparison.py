"""Aggregate baseline-vs-NCA results into a CSV + markdown table.

Reads all save_dir/curves_step*.npz and save_dir/eval_multigame*.npz under
nca_wm/logs_baselines/ and emits:
    - logs_baselines/summary.csv
    - logs_baselines/summary.md

For each (games_tag, arch_tag, seed):
    - best train change_acc and final loss (from curves)
    - parameter count (parsed from the run's stdout if available, else None)
    - mean autoregressive cell-error-rate across all (game, level) pairs
      under each policy ∈ {random, random_tf, bfs, astar} — averaged over
      the rollout time axis ("mean") and at the final step ("final")
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from glob import glob
import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGDIR = os.path.join(REPO, "nca_wm", "logs_baselines")


RUN_RE = re.compile(r"^(?P<tag>.*?)_s(?P<seed>\d+)$")
EVAL_KEY_RE = re.compile(
    r"^(?P<game>.+?)_L(?P<level>\d+)_(?P<policy>random_tf|random|bfs|astar)_"
    r"(?P<metric>cell_error_rate|error_rate|mean_first_div|first_div_step)$"
)
PARAMS_RE = re.compile(r"Model params:\s*([\d,]+)")


def parse_run(path):
    """e.g. 'microban_authored_nca_shared_s0' → (games_tag, arch_tag, seed).

    Tries each known prefix (so the games_tag is unambiguous regardless of
    underscores in the architecture tag).
    """
    base = os.path.basename(path.rstrip("/"))
    m = RUN_RE.match(base)
    if m is None:
        return None
    seed = int(m.group("seed"))
    rest = m.group("tag")
    for prefix in ("microban_authored", "sokoban_basic_synth", "scaling_14"):
        if rest.startswith(prefix + "_"):
            return prefix, rest[len(prefix) + 1:], seed
    return None


def load_curves(save_dir):
    npzs = sorted(
        glob(os.path.join(save_dir, "curves_step*.npz")),
        key=lambda p: int(re.search(r"step(\d+)", p).group(1)),
    )
    if not npzs:
        return None
    return dict(np.load(npzs[-1], allow_pickle=True))


def load_eval(save_dir):
    candidates = [
        os.path.join(save_dir, "eval_multigame_tlfix.npz"),
        os.path.join(save_dir, "eval_multigame.npz"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return dict(np.load(p, allow_pickle=True))
    return None


def parse_param_count(save_dir):
    """Scrape 'Model params: <N>' from the run's stdout file (one dir up)."""
    base = os.path.basename(save_dir.rstrip("/"))
    out_path = os.path.join(os.path.dirname(save_dir), base + ".out")
    if not os.path.isfile(out_path):
        return None
    try:
        with open(out_path) as f:
            for line in f:
                m = PARAMS_RE.search(line)
                if m:
                    return int(m.group(1).replace(",", ""))
    except OSError:
        pass
    return None


def summarize_eval(ev):
    """Average each policy's cell_error_rate over all (game, level) pairs.

    Returns dict with keys:
        {policy}_mean    : final-step cell error averaged across (game, level)
        {policy}_50step  : same metric averaged over the rollout time axis
                          (so a model that diverges late still scores poorly)
        {policy}_first_div: mean first-divergence step (random/random_tf
                          report mean_first_div; bfs/astar report
                          first_div_step which can be -1 = no divergence)
    """
    policies = ("random", "random_tf", "bfs", "astar")
    bucket = {p: {"final": [], "rollout": [], "first_div": []} for p in policies}
    for k, v in ev.items():
        m = EVAL_KEY_RE.match(k)
        if m is None:
            continue
        policy = m.group("policy")
        metric = m.group("metric")
        arr = np.asarray(v)
        if metric == "cell_error_rate" and arr.ndim == 1 and arr.size > 0:
            bucket[policy]["final"].append(float(arr[-1]))
            bucket[policy]["rollout"].append(float(arr.mean()))
        elif metric == "first_div_step" and arr.shape == ():
            # bfs/astar: -1 means no divergence over the entire rollout.
            val = int(arr)
            if val >= 0:
                bucket[policy]["first_div"].append(val)
        elif metric == "mean_first_div" and arr.shape == ():
            # random/random_tf: average step at which random rollout first
            # diverged from the engine. Higher is better.
            bucket[policy]["first_div"].append(float(arr))
    out = {}
    for p, d in bucket.items():
        if d["final"]:
            out[f"{p}_final_cellerr"] = float(np.mean(d["final"]))
            out[f"{p}_rollout_cellerr"] = float(np.mean(d["rollout"]))
        if d["first_div"]:
            out[f"{p}_first_div"] = float(np.mean(d["first_div"]))
    return out


def main():
    rows = []
    for save_dir in sorted(glob(os.path.join(LOGDIR, "*"))):
        if not os.path.isdir(save_dir):
            continue
        parsed = parse_run(save_dir)
        if parsed is None:
            continue
        games_tag, arch_tag, seed = parsed
        curves = load_curves(save_dir)
        ev = load_eval(save_dir)

        row = {
            "games": games_tag,
            "arch": arch_tag,
            "seed": seed,
            "n_params": parse_param_count(save_dir),
        }
        if curves is not None:
            losses = np.asarray(curves.get("losses", []))
            change_accs = np.asarray(curves.get("change_accs", []))
            if losses.size:
                row["final_loss"] = float(losses[-1])
                row["best_loss"] = float(losses.min())
            if change_accs.size:
                row["final_change_acc"] = float(change_accs[-1])
                row["best_change_acc"] = float(change_accs.max())
            row["n_steps_logged"] = int(losses.size)
        if ev is not None:
            row.update(summarize_eval(ev))
        rows.append(row)

    if not rows:
        print(f"No runs found under {LOGDIR}", file=sys.stderr)
        return

    keys = sorted({k for r in rows for k in r.keys()})
    csv_path = os.path.join(LOGDIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # Aggregate (mean ± std) over seeds within each (games, arch) group.
    groups = defaultdict(list)
    for r in rows:
        groups[(r["games"], r["arch"])].append(r)

    md_path = os.path.join(LOGDIR, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Baseline-vs-NCA aggregate\n\n")
        for games, arch in sorted(groups):
            seeds = groups[(games, arch)]
            n_params = next((s["n_params"] for s in seeds
                             if s.get("n_params") is not None), None)
            f.write(f"## {games} / {arch} ({len(seeds)} seeds")
            if n_params is not None:
                f.write(f", {n_params/1e6:.2f}M params")
            f.write(")\n\n")
            f.write("| metric | mean | std |\n|---|---|---|\n")
            metrics = [
                "best_loss", "final_loss", "best_change_acc", "final_change_acc",
                "random_tf_final_cellerr", "random_tf_rollout_cellerr",
                "random_final_cellerr", "random_rollout_cellerr",
                "random_first_div", "random_tf_first_div",
                "bfs_final_cellerr", "bfs_rollout_cellerr",
                "astar_final_cellerr", "astar_rollout_cellerr",
                "bfs_first_div", "astar_first_div",
            ]
            for m in metrics:
                vals = [r[m] for r in seeds if m in r and r[m] is not None]
                if not vals:
                    continue
                f.write(f"| {m} | {np.mean(vals):.4g} | {np.std(vals):.4g} |\n")
            f.write("\n")

        # Cross-arch headline table for each games_tag.
        games_tags = sorted({r["games"] for r in rows})
        for gt in games_tags:
            arch_groups = sorted({r["arch"] for r in rows if r["games"] == gt})
            f.write(f"## Headline: {gt}\n\n")
            f.write("| arch | params | best_loss | bfs_final_err | astar_final_err | random_final_err |\n")
            f.write("|---|---|---|---|---|---|\n")
            for arch in arch_groups:
                seeds = [r for r in rows if r["games"] == gt and r["arch"] == arch]
                np_count = next((s["n_params"] for s in seeds
                                  if s.get("n_params") is not None), None)
                np_str = f"{np_count/1e6:.2f}M" if np_count else "?"
                def stat(metric):
                    vals = [s[metric] for s in seeds
                            if metric in s and s[metric] is not None]
                    if not vals:
                        return ""
                    return f"{np.mean(vals):.4g}"
                f.write(
                    f"| {arch} | {np_str} | {stat('best_loss')} | "
                    f"{stat('bfs_final_cellerr')} | "
                    f"{stat('astar_final_cellerr')} | "
                    f"{stat('random_final_cellerr')} |\n"
                )
            f.write("\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
