"""Compare held-out transfer numbers across multiple checkpoints.

Reads `<save_dir>/heldout_transfer_v1/results.json` for each --runs entry
and prints a single-step + AR-rollout cell-error table per held-out game,
plus a "beats identity" tally.

Usage:
    .venv/bin/python3 nca_wm/scripts/summarize_heldout_transfer.py \\
        --runs nca_wm/logs/multi_scaling_14_v3recipe \\
                nca_wm/logs/multi_scaling_14_mask_v1 \\
                nca_wm/logs/multi_scaling_14_mask_v2_perstep \\
        --out_md nca_wm/figures/heldout_transfer/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _agg(levels, key, sub="random"):
    if not levels:
        return None
    vals = [v[sub][key] for v in levels.values()
            if sub in v and key in v[sub]]
    return sum(vals) / len(vals) if vals else None


def _summarize(run_dir):
    p = os.path.join(run_dir, "heldout_transfer_v1", "results.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    cfg = r.get("config", {})
    out = {
        "name": os.path.basename(run_dir),
        "preset": cfg.get("games"),
        "n_nca_steps": cfg.get("n_nca_steps"),
        "n_nca_repeats": cfg.get("n_nca_repeats"),
        "n_updates": cfg.get("n_updates"),
        "skipped": r.get("skipped", []),
    }
    games = {}
    for g, levels in r.get("heldout", {}).items():
        if isinstance(levels, dict) and len(levels) > 0:
            games[g] = {
                "step1_model": _agg(levels, "model_cell_err_step1"),
                "step1_id": _agg(levels, "identity_cell_err_step1"),
                "ar_mean_model": _agg(levels, "model_cell_err_mean"),
                "ar_mean_id": _agg(levels, "identity_cell_err_mean"),
                "tf_mean": _agg(levels, "model_cell_err_mean", "random_tf"),
                "n_levels": len(levels),
            }
    out["heldout"] = games
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out_md", default=None)
    args = p.parse_args()

    summaries = [s for s in (_summarize(r) for r in args.runs) if s]
    if not summaries:
        print("No completed heldout_transfer_v1 results found.", file=sys.stderr)
        return

    # Union of held-out game names, ordered by first appearance.
    games = []
    for s in summaries:
        for g in s["heldout"]:
            if g not in games:
                games.append(g)

    lines = []
    lines.append("# Heldout transfer comparison\n")
    lines.append(
        "Per held-out game: AR step-1 cell-error rate (model / identity), "
        "with `*` = beats identity at single-step prediction.\n"
    )
    # Per-checkpoint header
    lines.append("## Checkpoints\n")
    lines.append("| run | preset | n_steps | n_repeats | n_updates |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['preset']} | {s['n_nca_steps']} | "
            f"{s['n_nca_repeats']} | {s['n_updates']} |"
        )

    lines.append("\n## AR step-1 cell error (model / identity)\n")
    header = "| game | " + " | ".join(s["name"] for s in summaries) + " |"
    sep = "|---" * (len(summaries) + 1) + "|"
    lines.append(header); lines.append(sep)
    for g in games:
        cells = [g]
        for s in summaries:
            d = s["heldout"].get(g)
            if not d:
                cells.append("—")
                continue
            m = d["step1_model"]; idv = d["step1_id"]
            mark = "*" if m is not None and idv is not None and m < idv else " "
            cells.append(f"{m*100:.2f}% / {idv*100:.2f}%{mark}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## AR rollout mean (over 30 steps)\n")
    lines.append(header); lines.append(sep)
    for g in games:
        cells = [g]
        for s in summaries:
            d = s["heldout"].get(g)
            cells.append(f"{d['ar_mean_model']*100:.2f}% / {d['ar_mean_id']*100:.2f}%"
                         if d else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Beats-identity tally (step-1)\n")
    lines.append(header); lines.append(sep)
    cells = ["count"]
    for s in summaries:
        wins = sum(1 for g, d in s["heldout"].items()
                   if d["step1_model"] < d["step1_id"])
        cells.append(f"{wins}/{len(s['heldout'])}")
    lines.append("| " + " | ".join(cells) + " |")

    out = "\n".join(lines) + "\n"
    print(out)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
