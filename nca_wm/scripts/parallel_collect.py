"""Parallel solver-based transition collection for the gallery.

Each worker compiles a single game once, then iterates its levels and writes
each level's cache file via the same path as `collect_unique_transitions` in
train.py. This lets us populate the per-level cache for all uncached gallery
games in parallel — solver collection is CPU-bound (C++ engine, single-threaded
per call), so a multiprocessing pool ~scales linearly with workers.

Usage:
    python -m nca_wm.scripts.parallel_collect \\
        [--games gallery] \\
        [--workers 16] \\
        [--max-iters 100000 --timeout-ms 60000] \\
        [--skip-existing]   # default
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

# Make sure the repo is on sys.path when run as `python -m`
sys.path.insert(0, "/home/jupyter-smearle/script-doctor")

# Top-level so workers can import (functions defined in __main__ aren't
# picklable on spawn-style fork starts).


def _compile_game(name: str):
    """Compile a single game; returns (json_str, n_levels) or raises."""
    from puzzlescript_jax.utils import init_ps_lark_parser
    from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv

    ps_parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(ps_parser, name)
    env = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
    return json_str, int(env.num_levels)


def _collect_one_game(args):
    """Collect every level for a single game.

    Compiles once, then loops over levels. Each level writes its own cache
    file via collect_unique_transitions (which handles cache-skip itself).
    """
    name, max_iters, timeout_ms, skip_existing = args
    t0 = time.time()
    try:
        from nca_wm.train import collect_unique_transitions, _cache_dir

        json_str, n_levels = _compile_game(name)
        cached_levels = 0
        new_levels = 0
        empty_levels = 0
        total_trans = 0
        for li in range(n_levels):
            cache_path = os.path.join(
                _cache_dir(name, li),
                f"astar_transitions_v4_{max_iters}_{timeout_ms}.npz",
            )
            if skip_existing and os.path.isfile(cache_path):
                cached_levels += 1
                continue
            level_data = collect_unique_transitions(
                json_str, name, level_i=li,
                max_iters=max_iters, timeout_ms=timeout_ms,
                search_algo="astar",
            )
            n = len(level_data["states"])
            total_trans += n
            if n == 0:
                empty_levels += 1
            else:
                new_levels += 1
        elapsed = time.time() - t0
        return {
            "name": name, "n_levels": n_levels,
            "cached": cached_levels, "new": new_levels, "empty": empty_levels,
            "total_trans": total_trans, "elapsed_s": elapsed, "ok": True,
        }
    except Exception as e:
        return {
            "name": name, "ok": False, "error": str(e)[:200],
            "trace": traceback.format_exc()[-2000:],
            "elapsed_s": time.time() - t0,
        }


def _gallery_games() -> list[str]:
    from puzzlescript_jax.utils import get_list_of_games_for_testing
    games = list(get_list_of_games_for_testing(dataset="gallery"))
    NCAWM_EXTRAS = ["nekopuzzle"]
    for g in NCAWM_EXTRAS:
        if g not in games:
            games.append(g)
    return games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="gallery",
                   help="'gallery' or comma-separated game names")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-iters", type=int, default=100_000)
    p.add_argument("--timeout-ms", type=int, default=-1,
                   help="Per-level wall-clock cap in ms; -1 = no cap. We default "
                        "off because cutting search short corrupts our 'as many "
                        "transitions as we aimed for' invariant — the iteration "
                        "cap is the only intended truncation.")
    p.add_argument("--warn-after-min", type=float, default=15.0,
                   help="Print a warning if a single level takes longer than "
                        "this many minutes — early signal of a runaway level.")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="recollect even if cache exists")
    args = p.parse_args()

    if args.games == "gallery":
        games = _gallery_games()
    else:
        games = [g.strip() for g in args.games.split(",")]

    print(f"Parallel collection: {len(games)} games, "
          f"{args.workers} workers, max_iters={args.max_iters}, "
          f"timeout_ms={args.timeout_ms}")

    skip_existing = not args.no_skip_existing
    work = [(g, args.max_iters, args.timeout_ms, skip_existing) for g in games]
    warn_threshold_s = max(0.0, args.warn_after_min * 60.0)

    t0 = time.time()
    n_done = 0
    n_err = 0
    total_new = 0
    total_cached = 0
    # spawn so each worker's C++ engine state is fresh and doesn't share fds
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_collect_one_game, work):
            n_done += 1
            elapsed = time.time() - t0
            if r["ok"]:
                total_new += r["new"]
                total_cached += r["cached"]
                tag = "OK"
                if r["elapsed_s"] > warn_threshold_s:
                    tag = f"OK[!{r['elapsed_s']/60:.0f}m]"
                print(f"[{n_done:>3d}/{len(games)}] {tag:9s} {r['name']:40s}  "
                      f"levels: cached={r['cached']:>2d} new={r['new']:>2d} "
                      f"empty={r['empty']:>2d}  trans={r['total_trans']:>7d}  "
                      f"({r['elapsed_s']:>5.0f}s; {elapsed/60:.1f} min total)")
            else:
                n_err += 1
                print(f"[{n_done:>3d}/{len(games)}] ERR {r['name']:40s}  "
                      f"{r['error']}  ({r['elapsed_s']:.0f}s)")
                # Show short trace for the first couple of errors so we can
                # debug quickly without flooding the log.
                if n_err <= 3:
                    print(r["trace"])
            sys.stdout.flush()

    elapsed = time.time() - t0
    print()
    print(f"All done in {elapsed/60:.1f} min.")
    print(f"  games: ok={n_done-n_err}  errors={n_err}")
    print(f"  levels: pre-cached={total_cached}  newly collected={total_new}")


if __name__ == "__main__":
    main()
