#!/usr/bin/env python3
"""Batch-run the build_dataset pipeline (heuristic + human + seeded search) over many games,
with a concurrency cap. Produces {game}_train_seq.npz per game for the training sweep.

Data generation needs the MARA engine (local); training is engine-free (ships to torch).
Usage: python -m nca_wm.autumn.build_all --games a,b,c --workers 10
"""
import argparse, concurrent.futures as cf, time, traceback
from nca_wm.autumn import build_dataset


def _one(game, kw):
    t0 = time.time()
    try:
        out = build_dataset.build(game, **kw)
        return game, True, f"{time.time()-t0:.0f}s -> {out}"
    except Exception as e:
        return game, False, f"FAIL {time.time()-t0:.0f}s: {e}\n{traceback.format_exc()[-400:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="comma list")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--n_heur", type=int, default=400)
    ap.add_argument("--max_transitions", type=int, default=30000)
    ap.add_argument("--search_seconds", type=int, default=180)
    ap.add_argument("--seed_cap", type=int, default=2000)
    args = ap.parse_args()
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    kw = dict(n_heur=args.n_heur, max_transitions=args.max_transitions,
              search_seconds=args.search_seconds, seed_cap=args.seed_cap)
    print(f"[build_all] {len(games)} games, {args.workers} workers")
    done = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, g, kw): g for g in games}
        for fut in cf.as_completed(futs):
            game, ok, msg = fut.result()
            done += 1
            print(f"[{done}/{len(games)}] {'OK ' if ok else 'ERR'} {game}: {msg}", flush=True)
    print("[build_all] done")


if __name__ == "__main__":
    main()
