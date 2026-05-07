"""Dump engine-rendered (s_0, s_1) PNGs for a list of games + level 0.

s_1 follows the cached BFS-optimal first action so the figure shows a
move that meaningfully changes the level. Output:

    <out_dir>/<game>/engine_t0.png
    <out_dir>/<game>/engine_t1.png

Used by figures/id_ood_teaser/teaser.tex to populate the in-distribution
training row of the teaser figure.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_cpp import CppPuzzleScriptBackend


def _bfs_first_action(name: str, level: int) -> int | None:
    candidates = [
        _REPO_ROOT / "rollout_data" / name / f"level_{level}"
            / "search_bfs_100000_60000.npz",
        _REPO_ROOT / "rollout_data" / name / f"level_{level}"
            / "search_bfs_100000_-1.npz",
        _REPO_ROOT / "nca_wm" / "data_cache" / "heldout_search"
            / f"{name}_L{level}_bfs_100000_60000.npz",
    ]
    for p in candidates:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if "actions" in d.files and len(d["actions"]) > 0:
                return int(d["actions"][0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True,
                    help="comma-separated game names")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    parser = init_ps_lark_parser()
    out_root = Path(args.out_dir)

    for name in [g.strip() for g in args.games.split(",") if g.strip()]:
        backend = CppPuzzleScriptBackend()
        backend.compile_game(parser, name)
        backend.cpp_engine.load_level(args.level)
        f0 = backend.render_frame()

        a = _bfs_first_action(name, args.level)
        if a is None:
            a = 2  # right; fallback
        backend.cpp_engine.process_input(int(a))
        again = 0
        while backend.cpp_engine.againing and again < 8:
            backend.cpp_engine.process_input(-1)
            again += 1
        f1 = backend.render_frame()

        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(d / "engine_t0.png"), f0)
        imageio.imwrite(str(d / "engine_t1.png"), f1)
        print(f"  {name}  action={a}  saved {d}/engine_t0.png + engine_t1.png")


if __name__ == "__main__":
    main()
