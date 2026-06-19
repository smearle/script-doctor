#!/usr/bin/env python3
"""Pre-collect A* transitions for dedup_pool games in n_per_rule order.

CPU-only; runs `collect_unique_transitions` per (game, level) and writes
the per-game cache files at rollout_data/<game>/level_<i>/astar_*.npz.
Already-cached (game, level) pairs are skipped instantly. The cache
format matches what the bucketed loader in train.py reads, so once a
game is cached here, any future training run that references it will
hit the cache and skip the C++ A* search.

Default settings mirror the canonical training preset:
    n_search_steps=100000, search_timeout_ms=60000,
    max_transitions_per_level=200000

Filters (mirror --n_per_rule_games defaults):
    - Universe: dedup_pool (3,474 games at data/dedup_candidates_v2.json)
    - Heldout-26 names excluded
    - max_level_area <= 30 (avoids the 64x64 OOM on downstream training)
    - n_rules >= 1 (skips template / 0-rule games)

Order: ascending (n_rules, name) so the simplest games are cached first.
This means a long-running pre-collect can be interrupted at any time and
the next n_per_rule_games=N training run still benefits from whatever
landed first.

Usage:
    nohup .venv/bin/python3 nca_wm/scripts/precollect_dedup_pool.py \
        --n_max 500 --workers 1 \
        > /tmp/precollect.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _build_pool(args) -> list[dict]:
    """Return the dedup_pool game list filtered + sorted in n_per_rule order."""
    heldout = set()
    heldout_path = REPO_ROOT / "data" / "heldout_v4_n30.json"
    if heldout_path.is_file():
        heldout = {h["name"] for h in
                   json.loads(heldout_path.read_text())["heldout"]}
    _v3 = REPO_ROOT / "data" / "dedup_candidates_v3.json"
    pool_path = (_v3 if _v3.is_file()
                 else REPO_ROOT / "data" / "dedup_candidates_v2.json")
    raw = json.loads(pool_path.read_text())["candidates"]
    out = []
    for c in raw:
        if c["name"] in heldout:
            continue
        if int(c.get("n_rules", -1)) < 0:
            continue
        if int(c.get("max_level_area", 999)) > args.max_area:
            continue
        out.append(c)
    out.sort(key=lambda c: (c["n_rules"], c["name"]))
    return out[:args.n_max]


def _game_already_cached(game_name: str, n_levels: int, search_algo: str,
                          n_search_steps: int, max_transitions: int,
                          ancestor_closed: bool = False) -> bool:
    """All `n_levels` levels have at least one matching cache file?"""
    cache_dir_root = REPO_ROOT / "rollout_data" / game_name
    if not cache_dir_root.is_dir():
        return False
    for li in range(n_levels):
        # Cap_tag matches collect_unique_transitions naming.
        if max_transitions is None:
            cap_tag = "all"
        elif ancestor_closed:
            cap_tag = f"ac{int(max_transitions)}"
        else:
            cap_tag = str(int(max_transitions))
        # Glob over timeout (any timeout cache for the same algo+iters+cap is OK)
        pattern = (f"{cache_dir_root}/level_{li}/{search_algo}_transitions_v*"
                   f"_{n_search_steps}_*_cap{cap_tag}.npz")
        import glob
        if not glob.glob(pattern):
            return False
    return True


def _collect_one(game_meta: dict, args, repo_root: str) -> tuple[str, str]:
    """Worker: compile via CppPuzzleScriptBackend, then run
    collect_unique_transitions for every level of one game. Returns
    (name, status_str). Imports are inside the worker so the parent
    process doesn't drag in JAX."""
    sys.path.insert(0, repo_root)
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    from nca_wm.data_collection import (
        collect_unique_transitions, TRANSITIONS_CACHE_CAP)
    from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
    from puzzlescript_jax.utils import init_ps_lark_parser

    # ps_parser + backend are per-worker (single-process, no fork-safety
    # concerns). The parser construction is the slow part (~1s); fold it
    # into a module-level cache.
    if not hasattr(_collect_one, "_ps_parser"):
        _collect_one._ps_parser = init_ps_lark_parser()
    ps_parser = _collect_one._ps_parser

    name = game_meta["name"]
    backend = CppPuzzleScriptBackend()
    try:
        json_str = backend.compile_and_serialize(ps_parser, name)
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
    except Exception as e:
        return name, f"FAIL_compile: {type(e).__name__}: {str(e)[:120]}"
    # Re-discover n_levels from the compiled env (the dedup-pool field
    # is not always accurate when the engine post-deduplicates levels).
    n_levels = env0.num_levels
    if n_levels < 1:
        return name, "FAIL_no_levels"

    # Ancestor-closed mode caps each level at TRANSITIONS_CACHE_CAP to match
    # the training assembler (collect_multigame_dataset), so the ac caches
    # written here are the exact files training will load.
    ac = bool(getattr(args, "ancestor_closed", False))
    per_level_cap = (TRANSITIONS_CACHE_CAP if ac
                     else max(1, args.max_transitions_per_game // max(1, n_levels)))
    t0 = time.time()
    n_collected = 0
    for li in range(n_levels):
        try:
            data = collect_unique_transitions(
                json_str, name, level_i=li,
                max_iters=args.n_search_steps,
                timeout_ms=args.search_timeout_ms,
                search_algo=args.search_algo,
                max_transitions=per_level_cap,
                ancestor_closed=ac,
            )
            n_collected += len(data["actions"])
        except Exception as e:
            return name, (f"FAIL_l{li}: {type(e).__name__}: "
                          f"{str(e)[:120]}")
    dt = time.time() - t0
    return name, f"OK n_levels={n_levels} n_trans={n_collected} ({dt:.1f}s)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_max", type=int, default=500,
                    help="Cap how many games to (try to) collect.")
    ap.add_argument("--max_area", type=int, default=30,
                    help="max_level_area filter (mirrors --n_per_rule_max_area).")
    ap.add_argument("--n_search_steps", type=int, default=100000)
    ap.add_argument("--search_timeout_ms", type=int, default=60000)
    ap.add_argument("--search_algo", default="astar", choices=("astar", "bfs"))
    ap.add_argument("--max_transitions_per_game", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel game-collection workers (each game runs "
                         "its levels sequentially). 1 = serial.")
    ap.add_argument("--skip_cached", action=argparse.BooleanOptionalAction, default=True,
                    help="If set, fast-skip games that already have all "
                         "level cache files. Default True.")
    ap.add_argument("--ancestor_closed", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Write ancestor-closed (predecessor-chain-preserving) "
                         "caches (cap tag 'ac{N}', cap=TRANSITIONS_CACHE_CAP per "
                         "level) for the --history training pipeline. Uncapped "
                         "games reuse existing uniform caches (no re-search).")
    ap.add_argument("--n_shards", type=int, default=1,
                    help="Split the (n_max-capped) game list into this many "
                         "interleaved shards; run one process per shard for "
                         "process-level parallelism (the JS-compile bridge does "
                         "not work inside ProcessPoolExecutor workers, so use "
                         "separate processes + --workers 1 instead).")
    ap.add_argument("--shard_idx", type=int, default=0,
                    help="Which shard (0..n_shards-1) this process handles.")
    args = ap.parse_args()
    from nca_wm.data_collection import TRANSITIONS_CACHE_CAP

    games = _build_pool(args)
    if args.n_shards > 1:
        games = games[args.shard_idx::args.n_shards]
        print(f"[shard {args.shard_idx}/{args.n_shards}] handling "
              f"{len(games)} interleaved games")
    print(f"dedup_pool eligible (n_rules>=1, area<={args.max_area}, "
          f"not in Heldout-26): pre-collect order length = {len(games)}")
    print(f"  cap: {args.n_max}; will process up to {min(args.n_max, len(games))}")
    print(f"  worker count: {args.workers}")
    print()

    todo = []
    for g in games:
        nlv = max(1, int(g.get("n_levels", 1)))
        check_cap = (TRANSITIONS_CACHE_CAP if args.ancestor_closed
                     else args.max_transitions_per_game // nlv)
        if args.skip_cached and _game_already_cached(
                g["name"], int(g.get("n_levels", 1)),
                args.search_algo, args.n_search_steps,
                check_cap, ancestor_closed=args.ancestor_closed):
            continue
        todo.append(g)
    n_skipped = len(games) - len(todo)
    print(f"  cached + skipped: {n_skipped} games  "
          f"({len(todo)} need fresh collection)")
    print()

    repo_root = str(REPO_ROOT)
    n_done = 0; n_ok = 0; n_fail = 0
    if args.workers <= 1:
        for g in todo:
            name, status = _collect_one(g, args, repo_root)
            n_done += 1
            tag = "OK" if status.startswith("OK") else "FAIL"
            (n_ok if tag == "OK" else n_fail) and 0  # placeholder
            if tag == "OK": n_ok += 1
            else: n_fail += 1
            print(f"[{n_done}/{len(todo)}] {name}: {status}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_collect_one, g, args, repo_root): g["name"]
                    for g in todo}
            for f in as_completed(futs):
                name, status = f.result()
                n_done += 1
                tag = "OK" if status.startswith("OK") else "FAIL"
                if tag == "OK": n_ok += 1
                else: n_fail += 1
                print(f"[{n_done}/{len(todo)}] {name}: {status}", flush=True)

    print()
    print(f"DONE  total: {n_done}  ok: {n_ok}  fail: {n_fail}  "
          f"(pre-cached: {n_skipped})")


if __name__ == "__main__":
    main()
