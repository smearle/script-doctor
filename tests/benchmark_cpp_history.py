"""Small reproducible history-on/off stepping benchmark (no learner or LLM).

Run each extension in a separate process:
  python tests/benchmark_cpp_history.py --extension path/to/_puzzlescript_cpp.so
The report includes all timing repeats; it is not an end-to-end training benchmark.
"""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from test_cpp_history import ROOT, fixtures, load_extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    ps = load_extension(args.extension.resolve())
    f = fixtures()[0]
    f["actions"] = []
    result = subprocess.run(["node", str(ROOT / "tests/cpp_history_reference.cjs")],
                            input=json.dumps([f]), text=True, capture_output=True, check=True)
    compiled = json.loads(result.stdout)[0]["compiled"]
    e = ps.Engine()
    e.load_from_json(compiled)
    # Walk a loop in the open lower row; every episode has 128 input steps.
    actions = [2] + ([3, 3, 3, 3, 1, 1, 1, 1] * 16)[:127]
    seconds = []
    for rep in range(args.repeats + 1):
        start = time.perf_counter()
        for _ in range(args.episodes):
            e.load_level(0, "benchmark")
            for a in actions:
                e.process_input(a)
        elapsed = time.perf_counter() - start
        if rep:
            seconds.append(elapsed)
    snapshot_bytes = len(e.serialize_state()) if ps.HISTORY_ENABLED else None
    print(json.dumps({"history_enabled": ps.HISTORY_ENABLED,
                      "steps": args.episodes * len(actions),
                      "seconds": seconds, "median_seconds": statistics.median(seconds),
                      "snapshot_bytes_at_128_inputs": snapshot_bytes,
                      "undo_depth": e.undo_depth if ps.HISTORY_ENABLED else None}))


if __name__ == "__main__":
    main()
