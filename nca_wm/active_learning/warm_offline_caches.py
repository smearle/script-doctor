"""Pre-warm the offline BFS/A* transition caches for tree_growth_baseline.

Pure C++ collection, no jax needed. Run from the script-doctor root:

    python -m nca_wm.active_learning.warm_offline_caches \
        --game_json nca_wm/active_learning/heroes_of_sokoban.json --levels 0-13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="heroes_of_sokoban")
    p.add_argument("--game_json", required=True)
    p.add_argument("--levels", default="0-13")
    p.add_argument("--max_iters", type=int, default=100_000)
    p.add_argument("--timeout_ms", type=int, default=300_000)
    args = p.parse_args(argv)

    from nca_wm.data_collection import collect_unique_transitions
    a, b = args.levels.split("-")
    js = Path(args.game_json).read_text()
    for li in range(int(a), int(b) + 1):
        for algo in ("bfs", "astar"):
            d = collect_unique_transitions(
                js, args.game, level_i=li, max_iters=args.max_iters,
                timeout_ms=args.timeout_ms, search_algo=algo)
            print(f"L{li} {algo}: {len(d['states'])} transitions", flush=True)
    print("CACHE_WARM_DONE", flush=True)


if __name__ == "__main__":
    main()
