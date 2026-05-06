"""Pre-warm synthetic-level caches for the per-game x per-arch grid.

Synth-gen is CPU-bound (C++ engine, single-process GA per game). Train.py runs
games sequentially, so caches end up generated serially as each cell starts.
This script populates the same caches up front in parallel, so by the time the
launcher trains a given (game, bucket) cell its synth cache is already warm.

Recipe must match
``nca_wm/scripts/run_per_game_arch_synth_grid.sh`` exactly — cache paths key on
(game, w, h, seed, n_levels, CACHE_VERSION, mode-pop-gen, seed_from_authored,
require_solvable, mi, tmo, ms, k, rc) — so any drift here will miss the cache
at training time.

Usage:
  .venv/bin/python3 nca_wm/scripts/prewarm_synth_caches.py                      # all 5 games, both recipes
  .venv/bin/python3 nca_wm/scripts/prewarm_synth_caches.py --no-baseline        # coverage recipe only
  .venv/bin/python3 nca_wm/scripts/prewarm_synth_caches.py --games Microban Bouncers
  .venv/bin/python3 nca_wm/scripts/prewarm_synth_caches.py --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _REPO)


def _authored_sizes(name: str) -> list[tuple[int, int]]:
    """Return the sorted list of unique (W, H) sizes across this game's
    authored levels — same set --synthetic_multi_grid uses inside train.py."""
    from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(parser, name)
    env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
    authored: set[tuple[int, int]] = set()
    for li in range(int(env0.num_levels)):
        env_li = CppPuzzleScriptEnv(json_str, level_i=li, max_episode_steps=10)
        _, lh, lw = env_li.observation_shape
        authored.add((int(lw), int(lh)))
    if not authored:
        raise RuntimeError(f"No authored levels found for {name}")
    return sorted(authored)


def _gen_one(spec: dict) -> dict:
    """Worker: generate (or read cache) for one (game, recipe, size) triple.

    Multi-grid means one cache file per (game, size). The launcher's
    multi_grid call generates K // n_sizes levels per cache; we mirror that
    so the cache files written here match the ones train.py looks up.
    """
    name = spec["game"]
    w, h = int(spec["w"]), int(spec["h"])
    rc_weight = float(spec["rule_coverage_weight"])
    cstop = bool(spec["coverage_select_topk"])
    track = bool(spec["track_rules_fired"]) or rc_weight != 0.0 or cstop
    seed = int(spec["seed"])
    n_levels = int(spec["n_levels"])
    t0 = time.time()
    try:
        os.chdir(_REPO)
        from nca_wm.synthetic_levels import collect_synthetic_dataset
        ds = collect_synthetic_dataset(
            game_name=name,
            n_levels=n_levels, width=w, height=h, seed=seed,
            mode="evolve",
            require_solvable=True,
            max_attempts_per_level=1000,
            max_iters_search=1500,
            timeout_ms_search=400,
            min_states=5,
            no_a_count_max=5,
            evolve_pop_size=24,
            evolve_max_generations=60,
            evolve_n_mutations_min=1,
            evolve_n_mutations_max=3,
            seed_from_authored=False,
            fallback_dynamics=True,
            track_rules_fired=track,
            rule_coverage_weight=rc_weight,
            coverage_select_topk=cstop,
            verbose=True,
        )
        elapsed = time.time() - t0
        n_states = int(ds["states"].shape[0]) if "states" in ds else 0
        return {
            "game": name, "rc": rc_weight, "cstop": cstop,
            "size": (w, h), "n_transitions": n_states,
            "elapsed_s": elapsed, "ok": True,
        }
    except Exception as e:
        return {
            "game": name, "rc": rc_weight, "cstop": cstop,
            "size": (w, h), "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
            "elapsed_s": time.time() - t0,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--games", nargs="+",
        default=["Microban", "Heroes_of_Sokoban", "Bouncers",
                 "nekopuzzle", "Travelling_salesman"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n_levels", type=int, default=128,
        help="Total K. With multi_grid this is split as K // n_unique_sizes "
        "across the game's authored sizes (matching train.py's logic), so "
        "default 128 gives ~10/size on the worst game (Heroes, 12 sizes).",
    )
    ap.add_argument(
        "--coverage", action=argparse.BooleanOptionalAction, default=True,
        help="Generate the rule-coverage recipe (rc_weight=100, "
        "coverage_select_topk=True). Default on.",
    )
    ap.add_argument(
        "--baseline", action=argparse.BooleanOptionalAction, default=True,
        help="Also generate the iterations-only baseline (rc_weight=0). "
        "Default on for the appendix comparison.",
    )
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel processes; each does its own GA + engine.")
    args = ap.parse_args()

    # Build per-(game, size, recipe) jobs, mirroring train.py's multi_grid
    # cache layout. Each cache file is indexed on (game, w, h, rc); train.py
    # reads them via collect_synthetic_dataset(game, w, h, ...) per size.
    jobs: list[dict] = []
    for g in args.games:
        sizes = _authored_sizes(g)
        per_size_n = max(1, args.n_levels // max(1, len(sizes)))
        for (w, h) in sizes:
            common = {
                "game": g, "w": w, "h": h, "seed": args.seed,
                "n_levels": per_size_n,
            }
            if args.coverage:
                jobs.append({
                    **common,
                    "rule_coverage_weight": 100.0,
                    "coverage_select_topk": True,
                    "track_rules_fired": True,
                })
            if args.baseline:
                jobs.append({
                    **common,
                    "rule_coverage_weight": 0.0,
                    "coverage_select_topk": False,
                    "track_rules_fired": False,
                })

    print(f"Pre-warming {len(jobs)} synth caches across {args.workers} workers")
    for j in jobs:
        tag = "rc100_cstop" if j["rule_coverage_weight"] != 0.0 else "baseline"
        print(f"  - {j['game']:<24s}  {j['w']}x{j['h']:<3d}  {tag}  K={j['n_levels']}")

    t0 = time.time()
    # Use spawn to avoid forking the heavy parent (we'll re-import in workers).
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(_gen_one, jobs)

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.1f}s ===")
    for r in results:
        tag = "rc100_cstop" if r["rc"] != 0.0 else "baseline"
        if r["ok"]:
            print(f"  OK  {r['game']:<24s}  {tag:<12s}  "
                  f"{r['size'][0]}x{r['size'][1]}  "
                  f"{r['n_transitions']:,} trans  "
                  f"{r['elapsed_s']:.1f}s")
        else:
            print(f"  FAIL {r['game']:<24s}  {tag:<12s}  {r['error']}")
            print(f"    {r['trace'].splitlines()[-3:]}")


if __name__ == "__main__":
    main()
