#!/usr/bin/env python3
"""Pre-collect (state, action, next_state) transitions for the deduped master
corpus, sharded for a SLURM array.

Mirrors `precollect_dedup_pool.py` but targets the flat gist master dir
(`puzzlescript-gists`) instead of the curated repo corpora: it registers the
master dir as an extra games dir so the C++ backend resolves gist-hash stems by
name, reads the kept candidates from `dedup_master.json`, and only collects the
parseable ones (`parse_status == "ok"`).

DEFAULT ALGORITHM IS BFS, not A*. For world-model training we want broad,
representative coverage of the dynamics rather than the goal corridor, and BFS
avoids A*'s priority-queue (O(log frontier) per push/pop) + heuristic cost under
the same max_iters/timeout ceiling. Pass --search_algo astar to compare.

Caches land at rollout_data/<stem>/level_<i>/<algo>_transitions_*.npz, the exact
format the bucketed loader in train.py reads. Already-cached (game, level) pairs
are skipped instantly, so a preempted array job resumes for free.

SHARDING: pass --shard K --num-shards M to process only candidates whose index
% M == K. One SLURM array task per shard; all write to the shared rollout_data.

Usage (local smoke):
    .venv/bin/python3 -m nca_wm.scripts.precollect_master \
        --master-dir ../puzzlescript-gists --limit 3 --workers 1

Usage (one shard, as run under SLURM array):
    .venv/bin/python3 -m nca_wm.scripts.precollect_master \
        --master-dir /scratch/se2161/puzzlescript-gists \
        --shard $SLURM_ARRAY_TASK_ID --num-shards 128 --workers 8
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
    """Kept, parseable candidates from dedup_master.json, in a stable order."""
    report = json.loads(Path(args.manifest).read_text())
    out = []
    for c in report["candidates"]:
        if c.get("parse_status") != "ok":
            continue  # only parseable games can be rolled out by the engine
        if args.max_area is not None and c.get("max_level_area") is not None \
                and int(c["max_level_area"]) > args.max_area:
            continue
        out.append(c)
    # Stable order: simplest first (fewest objects, then levels, then name) so a
    # partial run still yields the cheap, high-yield games. Objects drive the
    # channel count / per-transition cost more than level count does.
    out.sort(key=lambda c: (c.get("n_objs") or 0, c.get("n_levels") or 0, c["file"]))
    if args.num_shards > 1:
        out = [c for i, c in enumerate(out) if i % args.num_shards == args.shard]
    if args.limit:
        out = out[:args.limit]
    return out


def _game_already_cached(stem: str, n_levels: int, search_algo: str,
                         n_search_steps: int, per_level_cap) -> bool:
    import glob
    cache_dir_root = REPO_ROOT / "rollout_data" / stem
    if not cache_dir_root.is_dir():
        return False
    cap_tag = "all" if per_level_cap is None else str(int(per_level_cap))
    for li in range(max(1, n_levels)):
        pattern = (f"{cache_dir_root}/level_{li}/{search_algo}_transitions_v*"
                   f"_{n_search_steps}_*_cap{cap_tag}.npz")
        if not glob.glob(pattern):
            return False
    return True


def _collect_one(meta: dict, args, repo_root: str, master_dir: str) -> tuple[str, str]:
    """Worker: register master dir, compile the gist stem, collect every level."""
    sys.path.insert(0, repo_root)
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    from nca_wm.train import collect_unique_transitions
    from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
    from puzzlescript_jax.utils import init_ps_lark_parser
    from puzzlescript_jax.preprocessing import add_extra_games_dir

    # Resolve gist-hash stems (e.g. "9d573d3cfa79cfa1f132") by name.
    add_extra_games_dir(master_dir)

    if not hasattr(_collect_one, "_ps_parser"):
        _collect_one._ps_parser = init_ps_lark_parser()
    ps_parser = _collect_one._ps_parser

    stem = meta["file"][:-4] if meta["file"].endswith(".txt") else meta["file"]
    backend = CppPuzzleScriptBackend()
    try:
        json_str = backend.compile_and_serialize(ps_parser, stem)
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
    except Exception as e:
        return stem, f"FAIL_compile: {type(e).__name__}: {str(e)[:120]}"
    n_levels = env0.num_levels
    if n_levels < 1:
        return stem, "FAIL_no_levels"

    per_level_cap = (None if args.max_transitions_per_game is None
                     else max(1, args.max_transitions_per_game // max(1, n_levels)))
    t0 = time.time()
    n_collected = 0
    for li in range(n_levels):
        try:
            data = collect_unique_transitions(
                json_str, stem, level_i=li,
                max_iters=args.n_search_steps,
                timeout_ms=args.search_timeout_ms,
                search_algo=args.search_algo,
                max_transitions=per_level_cap,
            )
            n_collected += len(data["actions"])
        except Exception as e:
            return stem, f"FAIL_l{li}: {type(e).__name__}: {str(e)[:120]}"
    dt = time.time() - t0
    return stem, f"OK n_levels={n_levels} n_trans={n_collected} ({dt:.1f}s)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-dir", default=str(REPO_ROOT.parent / "puzzlescript-gists"))
    ap.add_argument("--manifest", default=None,
                    help="dedup_master.json (default <master>/dedup_master.json)")
    ap.add_argument("--search_algo", default="bfs", choices=("bfs", "astar"))
    ap.add_argument("--n_search_steps", type=int, default=100000)
    ap.add_argument("--search_timeout_ms", type=int, default=60000)
    ap.add_argument("--max_transitions_per_game", type=int, default=200000)
    ap.add_argument("--max_area", type=int, default=None,
                    help="Optional max_level_area cap (skip giant grids).")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_cached", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    master_dir = str(Path(args.master_dir).resolve())
    if args.manifest is None:
        args.manifest = str(Path(master_dir) / "dedup_master.json")

    games = _build_pool(args)
    print(f"master: {master_dir}")
    print(f"shard {args.shard}/{args.num_shards}: {len(games)} parseable candidates"
          f"  algo={args.search_algo}  workers={args.workers}", flush=True)

    todo = []
    for g in games:
        stem = g["file"][:-4]
        n_levels = int(g.get("n_levels") or 1)
        per_level_cap = (None if args.max_transitions_per_game is None
                         else max(1, args.max_transitions_per_game // max(1, n_levels)))
        if args.skip_cached and _game_already_cached(
                stem, n_levels, args.search_algo, args.n_search_steps, per_level_cap):
            continue
        todo.append(g)
    n_skipped = len(games) - len(todo)
    print(f"  cached+skipped: {n_skipped}   to collect: {len(todo)}", flush=True)

    repo_root = str(REPO_ROOT)
    t0 = time.time()
    n_done = n_ok = n_fail = 0
    if args.workers <= 1:
        for g in todo:
            stem, status = _collect_one(g, args, repo_root, master_dir)
            n_done += 1
            n_ok += status.startswith("OK"); n_fail += not status.startswith("OK")
            print(f"[{n_done}/{len(todo)}] {stem}: {status}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_collect_one, g, args, repo_root, master_dir): g["file"]
                    for g in todo}
            for f in as_completed(futs):
                stem, status = f.result()
                n_done += 1
                n_ok += status.startswith("OK"); n_fail += not status.startswith("OK")
                print(f"[{n_done}/{len(todo)}] {stem}: {status}", flush=True)

    print(f"\nDONE shard {args.shard}/{args.num_shards}  total={n_done} ok={n_ok} "
          f"fail={n_fail} skipped={n_skipped}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
