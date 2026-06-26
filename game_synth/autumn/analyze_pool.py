"""Analyze an Autumn-ELM run: yield curve, reject reasons, mechanic diversity.

Reads <run>/manifest.jsonl and writes paper-ready figures (PDF+PNG, big fonts).

Usage: PYTHONPATH=game_synth/autumn .venv/bin/python3 \
         game_synth/autumn/analyze_pool.py game_synth/autumn/runs/mario_pool200
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18, "axes.labelsize": 20, "axes.titlesize": 20,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    "figure.autolayout": True,
})

RESULTS = ["accept", "dup", "invalid", "dead", "parse_fail"]
COLORS = {"accept": "#2ca02c", "dup": "#7f7f7f", "invalid": "#d62728",
          "dead": "#9467bd", "parse_fail": "#ff7f0e"}


def main():
    run = sys.argv[1]
    figdir = os.path.join(run, "figures")
    os.makedirs(figdir, exist_ok=True)
    recs = [json.loads(l) for l in open(os.path.join(run, "manifest.jsonl")) if l.strip()]

    n = len(recs)
    cum_accept = []
    counts = {r: 0 for r in RESULTS}
    c = 0
    for r in recs:
        if r["result"] == "accept":
            c += 1
        cum_accept.append(c)
        counts[r["result"]] = counts.get(r["result"], 0) + 1

    # diversity among accepted
    accepted = [r for r in recs if r["result"] == "accept"]
    covered_sets = {tuple(r.get("covered", [])) for r in accepted}
    kinds = {r.get("kind", "?").lower().strip() for r in accepted}

    # --- figure 1: cumulative yield ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n + 1), cum_accept, lw=3, color=COLORS["accept"])
    ax.set_xlabel("ELM iteration")
    ax.set_ylabel("Accepted variants (cumulative)")
    ax.set_title(f"Mario ELM yield: {c}/{n} accepted")
    ax.grid(alpha=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"yield_curve.{ext}"))
    plt.close(fig)

    # --- figure 2: reject-reason breakdown ---
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r for r in RESULTS if counts.get(r, 0) > 0]
    vals = [counts[r] for r in labels]
    ax.bar(labels, vals, color=[COLORS[r] for r in labels])
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("count")
    ax.set_title("Outcome breakdown")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"outcomes.{ext}"))
    plt.close(fig)

    summary = {
        "iters": n, "accepted": c, "accept_rate": round(c / n, 3),
        "valid_rate": round((n - counts["parse_fail"] - counts["invalid"]) / n, 3),
        "counts": counts,
        "distinct_covered_sets": len(covered_sets),
        "distinct_mutation_kinds": len(kinds),
    }
    print(json.dumps(summary, indent=2))
    print("kinds:", sorted(kinds))
    print(f"figures -> {figdir}/{{yield_curve,outcomes}}.{{pdf,png}}")
    json.dump(summary, open(os.path.join(run, "analysis.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
