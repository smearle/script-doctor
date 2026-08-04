"""Which rules CAN fire in these levels at all? Exhaustive-dump replay.

For each level, load the cached BFS transition dump (the offline-collection
regime's data), inject each stored state into the engine via a raw
LevelBackup (dat = W-major cell bitmask, verified byte-identical to
backup_level().dat), step the stored action with rule tracking on, and union
the fired rule indices.

The result is (an underestimate-free sample of) the set of rules FIREABLE in
the level set — the denominator against which tree_growth's
``rules_witnessed`` should be judged. If a rule never fires here either, the
IG search missed nothing.

    python -m nca_wm.active_learning.rule_coverage_check \
        --game_json .../heroes_of_sokoban.json --levels 0-13 --sample 60000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm.active_learning.tree_growth import new_engine


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="heroes_of_sokoban")
    p.add_argument("--game_json", required=True)
    p.add_argument("--levels", default="0-13")
    p.add_argument("--algo", default="bfs")
    p.add_argument("--max_iters", type=int, default=100_000)
    p.add_argument("--sample", type=int, default=60_000,
                   help="max transitions replayed per level (random subsample)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from nca_wm.data_collection import collect_unique_transitions
    from puzzlescript_cpp._puzzlescript_cpp import LevelBackup

    js = Path(args.game_json).read_text()
    rng = np.random.default_rng(args.seed)
    a, b = args.levels.split("-")
    fired_all: dict[int, set] = {}
    mismatches = 0

    for li in range(int(a), int(b) + 1):
        d = collect_unique_transitions(js, args.game, level_i=li,
                                       max_iters=args.max_iters,
                                       timeout_ms=300_000,
                                       search_algo=args.algo)
        W = int(d["W"])
        S = np.unpackbits(d["states"], axis=-1)[..., :W]      # (N,C,H,W)
        A = np.asarray(d["actions"], np.int32)
        n = len(S)
        idx = (np.arange(n) if n <= args.sample
               else rng.choice(n, size=args.sample, replace=False))

        eng = new_engine(js, li)
        h, w = S.shape[2], S.shape[3]
        fired = set()
        for i in idx:
            # multihot (C,H,W) -> per-cell int bitmask (H,W) -> W-major dat
            cell = np.zeros((h, w), np.int64)
            for c in range(S.shape[1]):
                cell |= S[i, c].astype(np.int64) << c
            bak = LevelBackup(cell.T.flatten().astype(np.int32).tolist(), w, h)
            eng.restore_level(bak)
            eng.clear_rules_fired()
            eng.process_input(int(A[i]))
            k = 0
            while eng.is_againing() and k < 50:
                eng.process_input(-1)
                k += 1
            fired.update(int(r) for r in eng.get_rules_fired())
        fired_all[li] = fired
        print(f"L{li}: replayed {len(idx)}/{n}  rules fired: {sorted(fired)}",
              flush=True)

    union = sorted(set().union(*fired_all.values()))
    print(f"\nUNION fireable across levels {args.levels} "
          f"({len(union)} rules): {union}", flush=True)


if __name__ == "__main__":
    main()
