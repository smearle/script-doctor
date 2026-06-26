"""Tiny shim: run nca_wm.train with an extra game-search directory registered.

Usage:
    python -m nca_wm.scripts.train_with_extra_games_dir \\
        --extra_games_dir nca_wm/logs/<run>/games \\
        -- --games <name> --n_updates 2000 [...other train.py flags...]

Everything after the ``--`` is forwarded verbatim to ``nca_wm.train.main()``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra_games_dir", required=True, action="append",
                    help="Dir(s) to prepend to the game-search path. Repeatable.")
    args, train_args = ap.parse_known_args()
    # argparse leaves the separating '--' in the remainder; drop one leading
    # '--' so the flags after it reach train.py as options rather than being
    # swallowed as positionals (which made train.py report --game missing).
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    from puzzlescript_jax.preprocessing import add_extra_games_dir
    for d in args.extra_games_dir:
        add_extra_games_dir(d)
        print(f"[train_shim] registered games dir: {d}")

    sys.argv = ["nca_wm.train"] + train_args
    from nca_wm.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
