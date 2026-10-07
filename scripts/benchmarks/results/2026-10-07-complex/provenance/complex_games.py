"""Frozen candidate list and structural metadata for the complex-game sweep."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COMPLEX_GAMES = [
    "Beam_Islands", "Boxes_&_Balloons", "Caramelban", "Crate_Assembler",
    "Heroes_of_Sokoban_III__The_Bard_and_The_Druid", "IceCrates", "Indigestion",
    "Memories_Of_Castlemouse", "ParaLands", "Sokoboros", "SwapBot", "Symbolism",
    "Transition", "Unconventional_Guns", "Vacuum", "castlecloset",
]


def inspect(game, output_dir):
    from scripts.benchmarks.benchmark_cpp_scoring import compile_game
    from scripts.benchmarks.profile_rand_nodejs import _load_nodejs_native_game_text
    from puzzlescript_jax.utils import init_ps_env
    from puzzlescript_jax.globals import TREES_DIR

    Path(TREES_DIR).mkdir(parents=True, exist_ok=True)
    env = init_ps_env(game, level_i=0, max_episode_steps=np.iinfo(np.int32).max)
    compiled = compile_game(game)
    data = json.loads(compiled)
    (output_dir / "compiled-games").mkdir(parents=True, exist_ok=True)
    (output_dir / "compiled-games" / f"{game}.json").write_text(compiled + "\n")
    groups = data["rules"] + data.get("lateRules", [])
    rules = [rule for group in groups for rule in group]
    source = _load_nodejs_native_game_text(game)
    result = {
        "game": game, "objects": env.n_objs, "layers": len(env.collision_layers),
        "level_shape": list(env.get_level(0).shape),
        "compiled_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "compiled_rules": len(rules), "compiled_groups": len(groups),
        "loop_points": len(data.get("loopPoint", {})) + len(data.get("lateLoopPoint", {})),
        "source_rules": sum(len(block.rules) for block in env.tree.rules),
        "multirow_rules": sum(len(rule["patterns"]) > 1 for rule in rules),
        "max_pattern_rows": max((len(rule["patterns"]) for rule in rules), default=0),
        "ellipsis_rules": sum(any(rule["ellipsisCount"]) for rule in rules),
        "max_pattern_span": max((len(row) for rule in rules for row in rule["patterns"]), default=0),
        "board_cells": int(np.prod(env.get_level(0).shape[1:])),
        "force_channels": 5 * len(env.collision_layers),
    }
    try:
        blocks = env._gen_rule_blocks(env.get_level(0).shape[1:])
        result["jax_rule_functions"] = sum(len(fns) for _, groups in blocks for fns, _ in groups) - 1
        result["jax_rule_groups"] = sum(len(groups) for _, groups in blocks) - 1
    except Exception as error:
        result["jax_structure_error"] = f"{type(error).__name__}: {error}"
    (output_dir / f"{game}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inspect(args.game, args.output_dir)


if __name__ == "__main__":
    main()
