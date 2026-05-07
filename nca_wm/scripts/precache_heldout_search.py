#!/usr/bin/env python3
"""Precompute BFS / A* action sequences for every (game, level) in
heldout_v4_n30 so the model-side eval can read them from disk later.

Search is model-independent, so doing this once amortizes across all
checkpoints. Runs on CPU only — leave GPUs free for training.

Usage:
    .venv/bin/python3 nca_wm/scripts/precache_heldout_search.py \\
        --heldout_file data/heldout_v4_n30.json \\
        --max_levels_per_game 2 \\
        --search_timeout_ms 60000

Cached results land in nca_wm/data_cache/heldout_search/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
# Force CPU; some imports trigger JAX init.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from nca_wm.heldout_eval import _get_heldout_search_actions, _build_heldout_game_info
from puzzlescript_jax.utils import init_ps_lark_parser


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout_file", default="data/heldout_v4_n30.json")
    ap.add_argument("--max_levels_per_game", type=int, default=2)
    ap.add_argument("--search_timeout_ms", type=int, default=60_000)
    ap.add_argument("--search_n_steps", type=int, default=100_000)
    ap.add_argument("--algos", default="bfs,astar",
                    help="Comma-separated subset of {bfs, astar}.")
    args = ap.parse_args()

    with open(os.path.join(REPO_ROOT, args.heldout_file)) as f:
        spec = json.load(f)
    heldout = spec["heldout"]
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    parser = init_ps_lark_parser()
    n_total = 0
    n_cached = 0
    n_solved = 0
    n_unsolved = 0
    n_skipped = 0
    t0 = time.time()
    for entry in heldout:
        name = entry["name"]
        info = _build_heldout_game_info(name, parser,
                                        encode_sprites=False, kernel_sep=False)
        if info is None:
            print(f"[skip] {name}: build failed")
            n_skipped += 1
            continue
        n_levels = min(info["n_levels"], args.max_levels_per_game)
        for level_i in range(n_levels):
            for algo in algos:
                n_total += 1
                t1 = time.time()
                actions = _get_heldout_search_actions(
                    name, level_i, algo, info["json_str"],
                    search_n_steps=args.search_n_steps,
                    search_timeout_ms=args.search_timeout_ms,
                )
                dt = time.time() - t1
                if actions is None or len(actions) == 0:
                    n_unsolved += 1
                    print(f"  [{algo}] {name} L{level_i}: no solution "
                          f"(budget {args.search_timeout_ms}ms / "
                          f"{args.search_n_steps} steps; {dt:.1f}s)")
                else:
                    n_solved += 1
                    print(f"  [{algo}] {name} L{level_i}: "
                          f"{len(actions)} actions ({dt:.1f}s)")
        # Best-effort persistence of cumulative status
        elapsed = time.time() - t0
        print(f"  -- progress: solved {n_solved}, unsolved {n_unsolved}, "
              f"skipped_games {n_skipped}, elapsed {elapsed:.0f}s")
    print(f"\n[done] {n_solved} solved / {n_unsolved} unsolved / "
          f"{n_skipped} games skipped (build_failed); "
          f"{time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
