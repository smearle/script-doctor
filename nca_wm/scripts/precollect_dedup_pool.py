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
    pool_path = REPO_ROOT / "data" / "dedup_candidates_v2.json"
    raw = json.loads(pool_path.read_text())["candidates"]
    out = []
    for c in raw:
        if c["name"] in heldout:
            continue
        if int(c.get("n_rules", -1)) < 1:
            continue
        if int(c.get("max_level_area", 999)) > args.max_area:
            continue
        out.append(c)
    out.sort(key=lambda c: (c["n_rules"], c["name"]))
    return out[:args.n_max]


def _game_already_cached(game_name: str, n_levels: int, search_algo: str,
                          n_search_steps: int, max_transitions: int) -> bool:
    """All `n_levels` levels have at least one matching cache file?"""
    cache_dir_root = REPO_ROOT / "rollout_data" / game_name
    if not cache_dir_root.is_dir():
        return False
    for li in range(n_levels):
        # Cap_tag matches collect_unique_transitions naming.
        cap_tag = "all" if max_transitions is None else str(int(max_transitions))
        # Glob over timeout (any timeout cache for the same algo+iters+cap is OK)
        pattern = (f"{cache_dir_root}/level_{li}/{search_algo}_transitions_v*"
                   f"_{n_search_steps}_*_cap{cap_tag}.npz")
        import glob
        if not glob.glob(pattern):
            return False
    return True


def _collect_one(game_meta: dict, args, repo_root: str) -> tuple[str, str]:
    """Worker: run collect_unique_transitions for every level of one game.
    Returns (name, status_str). Imports are inside the worker so the parent
    doesn't drag in JAX."""
    sys.path.insert(0, repo_root)
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    from nca_wm.train import collect_unique_transitions
    from puzzlescript_jax.utils import get_game_text

    name = game_meta["name"]
    n_levels = int(game_meta.get("n_levels", 1))
    try:
        json_str = get_game_text(name)
    except Exception as e:
        return name, f"FAIL_text: {type(e).__name__}: {e}"

    # Per-level cap matches train.py's per-level-cap logic
    per_level_cap = (
        max(1, args.max_transitions_per_game // max(1, n_levels))
    )
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
    args = ap.parse_args()

    games = _build_pool(args)
    print(f"dedup_pool eligible (n_rules>=1, area<={args.max_area}, "
          f"not in Heldout-26): pre-collect order length = {len(games)}")
    print(f"  cap: {args.n_max}; will process up to {min(args.n_max, len(games))}")
    print(f"  worker count: {args.workers}")
    print()

    todo = []
    for g in games:
        if args.skip_cached and _game_already_cached(
                g["name"], int(g.get("n_levels", 1)),
                args.search_algo, args.n_search_steps,
                args.max_transitions_per_game // max(1, int(g.get("n_levels", 1)))):
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
