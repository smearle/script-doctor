"""Run `heldout_eval.py` one game at a time, each in a fresh subprocess, so
JAX's JIT compile cache resets between games. This dodges the OOM that
hits when 30 shape-diverse heldouts are evaluated in a single process
(tracked by JIT cache accumulation, see docs/scaling/SCALING_REPORT.md).

Aggregates per-game JSON outputs into a single `results.json` keyed by
heldout name.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 \
        -m nca_wm.scripts.heldout_rollout_per_game \
            --load nca_wm/logs/multi_scaling_gallery_v3 \
            --heldout_file data/heldout_v4_n30.json \
            --max_levels_per_game 3 --n_random_episodes 3 --max_steps 30
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--heldout_file", required=True)
    ap.add_argument("--max_levels_per_game", type=int, default=3)
    ap.add_argument("--n_random_episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--out_subdir", default="heldout_v4_n30_per_game")
    ap.add_argument("--skip_done", action="store_true",
                    help="Skip games whose per-game JSON already exists.")
    args = ap.parse_args()

    with open(args.heldout_file) as f:
        spec = json.load(f)
    names = [h["name"] for h in spec["heldout"]]
    print(f"Running per-game heldout rollout eval on {len(names)} games "
          f"({args.load}) -> {args.out_subdir}")

    out_root = os.path.join(args.load, args.out_subdir)
    os.makedirs(out_root, exist_ok=True)

    aggregated = {}
    failures = []

    t_start = time.time()
    for i, name in enumerate(names):
        per_game_json = os.path.join(out_root, f"{_safe(name)}.json")
        if args.skip_done and os.path.exists(per_game_json):
            try:
                with open(per_game_json) as f:
                    aggregated[name] = json.load(f)
                print(f"  [{i+1}/{len(names)}] {name}: SKIP (cached)")
                continue
            except Exception:
                pass

        print(f"  [{i+1}/{len(names)}] {name}: running...", flush=True)
        with tempfile.TemporaryDirectory() as td:
            sub_subdir = "single_game"
            cmd = [
                ".venv/bin/python3", "-m", "nca_wm.heldout_eval",
                "--load", args.load,
                "--heldout_games", name,
                "--max_levels_per_game", str(args.max_levels_per_game),
                "--n_random_episodes", str(args.n_random_episodes),
                "--max_steps", str(args.max_steps),
                "--include_train_sample", "0",
                "--out_subdir", sub_subdir,
            ]
            t0 = time.time()
            res = subprocess.run(cmd, cwd=os.getcwd(),
                                 capture_output=True, text=True)
            dt = time.time() - t0
            if res.returncode != 0:
                print(f"    FAIL ({dt:.1f}s, rc={res.returncode}). stderr "
                      f"tail:\n      "
                      f"{res.stderr.strip().splitlines()[-1] if res.stderr else ''}")
                failures.append({"name": name, "rc": res.returncode,
                                  "stderr": res.stderr[-2000:]})
                continue

            # Pull the produced per-game JSON out
            inner = os.path.join(args.load, sub_subdir, "results.json")
            if not os.path.exists(inner):
                print(f"    no results.json produced ({dt:.1f}s)")
                failures.append({"name": name, "reason": "no results.json"})
                continue
            with open(inner) as f:
                inner_data = json.load(f)
            # `inner_data` has key "heldout" with one entry — the game.
            game_block = inner_data.get("heldout", {}).get(name)
            if game_block is None:
                # Some heldouts may have been skipped inside heldout_eval
                print(f"    skipped inside heldout_eval ({dt:.1f}s)")
                failures.append({"name": name, "reason": "skipped inside"})
                continue
            aggregated[name] = game_block
            with open(per_game_json, "w") as f:
                json.dump(game_block, f, indent=2)
            # Cleanup the subdir written by the inner call
            shutil.rmtree(os.path.join(args.load, sub_subdir),
                          ignore_errors=True)
            print(f"    OK ({dt:.1f}s)")

    # Final aggregate
    out_path = os.path.join(out_root, "results.json")
    with open(out_path, "w") as f:
        json.dump({"heldout": aggregated, "failures": failures}, f, indent=2)
    print(f"\nDone in {(time.time()-t_start)/60:.1f}m. "
          f"Successes: {len(aggregated)}, failures: {len(failures)}.")
    print(f"Aggregate: {out_path}")

    # Summary table
    rows = []
    for name, block in aggregated.items():
        # block is {level_i: {regime: {...}}} — average across levels and the
        # 'random' regime as the headline metric.
        m_step1, i_step1, m_mean, i_mean = [], [], [], []
        for level_key, regimes in block.items():
            r = regimes.get("random") or regimes.get("random_tf")
            if r is None:
                continue
            m_step1.append(r.get("model_cell_err_step1", float("nan")))
            i_step1.append(r.get("identity_cell_err_step1", float("nan")))
            m_mean.append(r.get("model_cell_err_mean", float("nan")))
            i_mean.append(r.get("identity_cell_err_mean", float("nan")))
        if not m_step1:
            continue
        import numpy as np
        rows.append({
            "name": name,
            "model_step1": float(np.nanmean(m_step1)),
            "identity_step1": float(np.nanmean(i_step1)),
            "model_mean": float(np.nanmean(m_mean)),
            "identity_mean": float(np.nanmean(i_mean)),
            "beats_id_step1": float(np.nanmean(m_step1)) < float(np.nanmean(i_step1)),
        })
    rows.sort(key=lambda r: r["model_step1"] - r["identity_step1"])
    print("\n--- per-game summary (random regime, mean over levels) ---")
    print(f"{'game':<55s} {'m_step1':>9s} {'i_step1':>9s} "
          f"{'m_mean':>9s} {'i_mean':>9s} beats_id")
    for r in rows:
        flag = "✓" if r["beats_id_step1"] else " "
        print(f"{r['name'][:55]:<55s} {r['model_step1']:9.3f} "
              f"{r['identity_step1']:9.3f} {r['model_mean']:9.3f} "
              f"{r['identity_mean']:9.3f}     {flag}")
    n_beats = sum(1 for r in rows if r["beats_id_step1"])
    print(f"\nbeats-identity at step1: {n_beats}/{len(rows)}")

    summary_path = os.path.join(out_root, "summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# Heldout AR rollout summary\n\n")
        f.write(f"Run: `{args.load}`\n\n")
        f.write(f"Heldout: `{args.heldout_file}`\n\n")
        f.write(f"Per-game (random regime, mean over levels):\n\n")
        f.write("| game | model step1 | identity step1 | model mean | "
                "identity mean | beats id |\n")
        f.write("|---|---:|---:|---:|---:|:---:|\n")
        for r in rows:
            flag = "✓" if r["beats_id_step1"] else ""
            f.write(f"| `{r['name']}` | {r['model_step1']:.3f} | "
                    f"{r['identity_step1']:.3f} | {r['model_mean']:.3f} | "
                    f"{r['identity_mean']:.3f} | {flag} |\n")
        f.write(f"\n**beats-identity at step1: {n_beats}/{len(rows)}**\n")
    print(f"Summary: {summary_path}")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


if __name__ == "__main__":
    main()
