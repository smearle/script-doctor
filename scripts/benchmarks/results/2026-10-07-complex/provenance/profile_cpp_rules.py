"""Count rule pruning and matching work in a separately instrumented C++ build.

These are work counts, not throughput timings. The instrumented extension is
never used for the published performance curves.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from scripts.benchmarks.benchmark_paper_throughput import BENCHMARK_GAMES
from scripts.benchmarks.complex_games import COMPLEX_GAMES

COUNTERS = ["rule_match_calls", "board_object_mask_rejections", "board_row_mask_rejections",
            "row_column_mask_skips", "cell_start_checks", "tuple_attempts", "changed_rules", "group_passes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--games", nargs="+", default=BENCHMARK_GAMES + COMPLEX_GAMES)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1042])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = "puzzlescript_cpp._puzzlescript_cpp"
    spec = importlib.util.spec_from_file_location(name, args.library)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    result = {"library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "steps": args.steps, "seeds": args.seeds, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for game in args.games:
        compiled = (args.compiled_dir / f"{game}.json").read_text().strip()
        engine = module.Engine()
        assert engine.load_from_json(compiled)
        for seed in args.seeds:
            engine.load_level(0, "profile")
            engine.clear_benchmark_counters()
            actions = np.random.default_rng(seed).integers(0, 5, args.steps)
            again_ticks = wins = 0
            for action in actions:
                engine.process_input(int(action))
                count = 0
                while engine.is_againing() and count < 50:
                    engine.process_input(-1)
                    count += 1
                again_ticks += count
                if engine.is_winning() or engine.check_win():
                    wins += 1
                    engine.load_level(0, "profile")
            counters = dict(zip(COUNTERS, engine.benchmark_counters()))
            row = {"game": game, "seed": seed, "compiled_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
                   "again_ticks": again_ticks, "wins": wins, **counters}
            row["board_pruned_fraction"] = ((counters["board_object_mask_rejections"] + counters["board_row_mask_rejections"])
                                            / counters["rule_match_calls"] if counters["rule_match_calls"] else None)
            result["results"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
