"""Live viewer-index watcher: periodically (incrementally) rebuild the
viewer_index.json of one or more active gp_evolve runs, so the web viewer shows
the latest generations on browser refresh.

Inits the Lark parser ONCE, then loops cheaply (incremental builds only touch new
games). Stops a run's polling once its evolve process is done AND no new games
have appeared for a couple of cycles.

    .venv/bin/python -m game_synth.auto_index --interval 120 \
        game_synth/big_loss_norand game_synth/big_progress_norand
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from game_synth.build_viewer_index import build_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs (relative to repo root)")
    ap.add_argument("--interval", type=int, default=120, help="seconds between rebuilds")
    ap.add_argument("--source", default="gp")
    ap.add_argument("--max-cycles", type=int, default=2000)
    args = ap.parse_args()

    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()
    runs = [r[len(str(_REPO)) + 1:] if r.startswith(str(_REPO)) else r.rstrip("/") for r in args.runs]
    print(f"[auto_index] watching {runs} every {args.interval}s", flush=True)

    stale = {r: 0 for r in runs}   # consecutive cycles with no new games
    for cycle in range(args.max_cycles):
        active = []
        for r in runs:
            try:
                total, new = build_index(r, parser, source=args.source, verbose=True)
            except Exception as e:
                print(f"[auto_index] {r}: build error {e}", flush=True)
                new = 1  # keep trying
            stale[r] = stale[r] + 1 if new == 0 else 0
            # a run is "done" once its wm.pt exists (written at the very end) and
            # two cycles pass with no new games
            done = (_REPO / r / "wm.pt").exists() and stale[r] >= 2
            if not done:
                active.append(r)
        if not active:
            print("[auto_index] all runs finished and indices final; exiting", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
