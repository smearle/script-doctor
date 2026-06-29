"""Bridge the two Mario variants into the mature recurrent-NCA pipeline.

This file intentionally does not implement a new trainer. It only:

1. registers ``custom_games/autumn`` with the PuzzleScript lookup path,
2. precollects solver transition caches via ``nca_wm.data_collection``, and
3. writes a games-list file consumed by ``nca_wm.train_recurrent``.

Example:
    .venv/bin/python -m nca_wm.active_learning.mario_recurrent_pipeline --collect

    .venv/bin/python -m nca_wm.train_recurrent \
        --games_list_file nca_wm/active_learning/_mario_recurrent_games.txt \
        --heldout_game_frac 0 --k 8 --n_hid 288 --n_steps 8 \
        --n_updates 50000 --batch_size 32 --val_frac 0.1 \
        --save_dir nca_wm/logs/mario2_recurrent
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from nca_wm.data_collection import collect_unique_transitions

ROOT = Path(__file__).resolve().parents[2]
AUTUMN_GAMES_DIR = ROOT / "custom_games" / "autumn"
GAMES = ("mario", "mario_breakable")
DEFAULT_GAMES_LIST = Path(__file__).resolve().parent / "_mario_recurrent_games.txt"


def _register_autumn_games_dir() -> None:
    from puzzlescript_jax.preprocessing import add_extra_games_dir

    add_extra_games_dir(str(AUTUMN_GAMES_DIR))


def _compile_game(name: str) -> tuple[str, int]:
    from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
    from puzzlescript_jax.utils import init_ps_lark_parser

    _register_autumn_games_dir()
    parser = init_ps_lark_parser()
    json_str = CppPuzzleScriptBackend().compile_and_serialize(parser, name)
    env = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
    return json_str, int(env.num_levels)


def write_games_list(path: Path = DEFAULT_GAMES_LIST) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(GAMES) + "\n", encoding="utf-8")
    return path


def collect(max_iters: int, timeout_ms: int, search_algo: str) -> None:
    write_games_list()
    for name in GAMES:
        print(f"[mario_recurrent_pipeline] compiling {name}", flush=True)
        json_str, n_levels = _compile_game(name)
        for level_i in range(n_levels):
            print(
                f"[mario_recurrent_pipeline] collecting {name} level {level_i} "
                f"({search_algo}, max_iters={max_iters}, timeout_ms={timeout_ms})",
                flush=True,
            )
            d = collect_unique_transitions(
                json_str,
                name,
                level_i=level_i,
                max_iters=max_iters,
                timeout_ms=timeout_ms,
                search_algo=search_algo,
                ancestor_closed=True,
            )
            print(
                f"[mario_recurrent_pipeline] {name} level {level_i}: "
                f"{len(d['actions']):,} transitions",
                flush=True,
            )


def train(args: argparse.Namespace) -> None:
    games_list = write_games_list()
    cmd = [
        sys.executable,
        "-m",
        "nca_wm.train_recurrent",
        "--games_list_file",
        str(games_list),
        "--heldout_game_frac",
        "0",
        "--k",
        str(args.k),
        "--n_hid",
        str(args.n_hid),
        "--n_steps",
        str(args.n_steps),
        "--n_updates",
        str(args.n_updates),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--eval_interval",
        str(args.eval_interval),
        "--val_frac",
        str(args.val_frac),
        "--max_transitions_per_game",
        str(args.max_transitions_per_game),
        "--max_grid_dim",
        str(args.max_grid_dim),
        "--seed",
        str(args.seed),
        "--save_dir",
        str(args.save_dir),
    ]
    if args.bptt_window:
        cmd += ["--bptt_window", str(args.bptt_window)]
    print("[mario_recurrent_pipeline] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--max-iters", type=int, default=100_000)
    p.add_argument("--timeout-ms", type=int, default=60_000)
    p.add_argument("--search-algo", choices=["astar", "bfs"], default="astar")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n-hid", type=int, default=288)
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--n-updates", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=2_500)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--max-transitions-per-game", type=int, default=20_000)
    p.add_argument("--max-grid-dim", type=int, default=30)
    p.add_argument("--bptt-window", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="nca_wm/logs/mario2_recurrent")
    args = p.parse_args()

    if args.collect:
        collect(args.max_iters, args.timeout_ms, args.search_algo)
    else:
        write_games_list()
    if args.train:
        train(args)
    if not (args.collect or args.train):
        path = write_games_list()
        print(f"[mario_recurrent_pipeline] wrote {path}")


if __name__ == "__main__":
    main()
