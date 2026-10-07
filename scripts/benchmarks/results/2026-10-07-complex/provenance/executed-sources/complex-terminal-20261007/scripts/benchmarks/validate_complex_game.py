"""Replay a deterministic action sample against the original JS engine."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-on-win", action="store_true",
                        help="Validate through the first terminal state, as used by autoresetting RL rollouts.")
    args = parser.parse_args()
    from backends import NodeJSPuzzleScriptBackend
    from puzzlejax.validate_actions import run_test_case
    from puzzlescript_jax.globals import TREES_DIR
    from puzzlescript_jax.utils import init_ps_lark_parser

    Path(TREES_DIR).mkdir(parents=True, exist_ok=True)
    actions = list(range(5)) * 8 + np.random.default_rng(42).integers(0, 5, 88).tolist()
    try:
        ok, message = run_test_case(args.game, 0, actions, NodeJSPuzzleScriptBackend(), init_ps_lark_parser(),
                                    stop_on_win=args.stop_on_win)
        status = "passed" if ok else "mismatch"
    except Exception as error:
        traceback.print_exc()
        ok, message, status = False, f"{type(error).__name__}: {error}", "error"
    root = Path(__file__).resolve().parents[2]
    result = {"game": args.game, "level": 0, "passed": ok, "status": status, "message": message,
              "actions": actions, "stop_on_win": args.stop_on_win,
              "comparison": "Original JS versus JAX: board, heuristic and winning flag after every replayed step",
              "source_hashes": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                                ["puzzlejax/validate_actions.py", "puzzlejax/validate_sols_jax.py", "puzzlescript_jax/env.py",
                                 "PuzzleScript/src/js/compiler.js", "PuzzleScript/src/js/engine.js",
                                 "puzzlescript_nodejs/puzzlescript/engine.js", str(Path(__file__).relative_to(root))]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
