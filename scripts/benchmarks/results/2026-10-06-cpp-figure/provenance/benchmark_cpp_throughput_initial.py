"""Measure optimized C++ paper curves with the original CPU rollout workload.

Random actions, complete RL outputs, and win counting are inside the timer.
Reset, JSON compilation, environment construction, and warmup are excluded.
Each trial resets the board. Thread count is min(batch, --max-threads), rather
than selecting the fastest of many thread counts or timing samples.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

import numpy as np

from scripts.benchmarks.benchmark_cpp_scoring import compile_game, request

PAPER_GAMES = ["sokoban_basic", "notsnake", "Zen_Puzzle_Garden", "Slidings",
               "limerick", "kettle", "Take_Heart_Lass", "atlas shrank"]


def stop_reason(rows, min_gain=0.03, patience=2, regression=0.10):
    tail = [r for r in rows if r["batch"] >= 64]
    if len(tail) < 2:
        return None
    best, stalled = tail[0]["median_fps"], 0
    for row in tail[1:]:
        fps = row["median_fps"]
        if fps < best * (1 - regression):
            return "regression"
        stalled = 0 if fps > best * (1 + min_gain) else stalled + 1
        best = max(best, fps)
    return "plateau" if stalled >= patience else None


def save(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--games", nargs="+", default=PAPER_GAMES)
    parser.add_argument("--compiled-dir", type=Path)
    parser.add_argument("--max-threads", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=8192)
    parser.add_argument("--base-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_threads, args.max_batch, args.base_steps, args.min_steps, args.trials) < 1:
        parser.error("thread, batch, step and trial counts must be positive")
    if args.max_threads > len(os.sched_getaffinity(0)):
        parser.error("thread count exceeds the available CPU affinity")
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "scripts/benchmarks/benchmark_cpp_scoring.py",
             root / "puzzlescript_cpp/__init__.py", root / "puzzlescript_nodejs/puzzlescript/engine.js"]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    cpu_model = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                      if s.startswith("model name")), platform.processor())
    config = {"library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "source_hashes": hashes, "host": platform.node(), "cpu_model": cpu_model,
              "python_version": platform.python_version(), "numpy_version": np.__version__,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "max_threads": args.max_threads,
              "thread_policy": "min(batch, max_threads)", "trials": args.trials, "seed": args.seed,
              "base_steps": args.base_steps, "min_steps": args.min_steps, "level": 0,
              "thread_environment": {k: os.environ.get(k) for k in
                                     ("OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS")},
              "timing": "median/IQR; random actions, full RL outputs and win counting timed; reset before each trial excluded",
              "adaptive": {"min_batch": 64, "factor": 2, "min_gain": 0.03, "patience": 2,
                           "regression": 0.10, "max_batch": args.max_batch}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([sys.executable, "-m", "scripts.benchmarks.benchmark_cpp_scoring",
                                "--worker", str(args.library.resolve())],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        for game in args.games:
            compiled = ((args.compiled_dir / f"{game}.json").read_text().strip()
                        if args.compiled_dir else compile_game(game))
            data = {**config, "game": game, "compiled_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
                    "results": []}
            output = args.output_dir / f"{game}.json"
            if output.exists():
                if not args.resume:
                    raise ValueError(f"{output} exists; use --resume")
                previous = json.loads(output.read_text())
                for key in data:
                    if key != "results" and previous[key] != data[key]:
                        raise ValueError(f"Cannot resume changed {key}: {game}")
                data = previous
            data.pop("stop_reason", None)
            batch = 2 * data["results"][-1]["batch"] if data["results"] else 1
            while True:
                reason = stop_reason(data["results"])
                if reason or batch > args.max_batch:
                    data["stop_reason"] = reason or "max_batch"
                    save(output, data)
                    break
                steps = args.base_steps if batch == 1 else max(args.base_steps // batch, args.min_steps)
                threads = min(batch, args.max_threads)
                data["pending_batch"] = batch
                save(output, data)
                ready = request(process, {"command": "init", "compiled": compiled, "level": 0,
                                         "batch": batch, "threads": threads, "steps": steps, "seed": args.seed,
                                         "max_episode_steps": steps + 1, "random_actions": True})
                for warmup in range(2):
                    request(process, {"command": "run", "seed": args.seed + 10000 + warmup})
                samples = [request(process, {"command": "run", "seed": args.seed + trial})
                           for trial in range(args.trials)]
                times = np.asarray([r["seconds"] for r in samples])
                fps = batch * steps / times
                row = {"batch": batch, "threads": threads, "steps": steps,
                       "max_episode_steps": steps + 1, "observation_shape": ready["observation_shape"],
                       "samples_s": times.tolist(), "wins": [r["wins"] for r in samples],
                       "fps_samples": fps.tolist(), "median_fps": float(np.median(fps)),
                       "q25_fps": float(np.percentile(fps, 25)), "q75_fps": float(np.percentile(fps, 75))}
                for path in paths:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(root))]:
                        raise RuntimeError(f"Source changed during measurement: {path}")
                data["results"].append(row)
                data.pop("pending_batch", None)
                save(output, data)
                print(json.dumps({"game": game, **row}), flush=True)
                batch *= 2
    finally:
        process.stdin.close()
        if process.poll() is None:
            os.kill(process.pid, signal.SIGCONT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
