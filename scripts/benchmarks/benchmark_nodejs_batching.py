"""Audit per-step Node IPC separately from native, engine-only rollouts.

JSON/advanced trials use identical pre-generated actions, reset outside timing,
and compare every observation, reward, terminal flag and info field. The native
pool is a different workload (no observations/rewards or per-step Python IPC).
It is reported as context, never as an equivalent serialization comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from puzzlescript_nodejs.rl_env import _NodeJSBatchedController
from scripts.benchmarks.profile_rand_nodejs import (
    _load_nodejs_native_game_text, _get_nodejs_native_game_path, _run_nodejs_native_pool,
)


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ["puzzlescript_nodejs/rl_env.py", "puzzlescript_nodejs/puzzlescript/batched_env_controller.js",
             "puzzlescript_nodejs/puzzlescript/batched_env_worker.js",
             "scripts/benchmarks/benchmark_nodejs_batching.py"]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def rollout(controller, actions, *, digest=False):
    controller.reset()
    snapshots = []
    start = time.perf_counter()
    for action in actions:
        output = controller.step(action)
        if digest:
            obs, rewards, dones, truncated, info = output
            snapshots.extend((obs, rewards, dones, truncated, *[info[k] for k in sorted(info)]))
    elapsed = time.perf_counter() - start
    if digest:
        return hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in snapshots)).hexdigest()
    return elapsed


def benchmark(game, batch, steps, trials, experiment):
    text = _load_nodejs_native_game_text(game)
    actions = np.random.default_rng(42).integers(0, 5, (steps, batch), dtype=np.int32)
    controllers = {}
    try:
        modes = {"json": ("json", False), "advanced": ("advanced", False)} if experiment == "serialization" else {
            "previous": ("json", False), "reuse_score": ("json", True)}
        for mode, (serialization, reuse_score) in modes.items():
            controllers[mode] = _NodeJSBatchedController(
                game_text=text, level_i=0, n_envs=batch, max_episode_steps=2**31 - 1,
                ipc_serialization=serialization, reuse_score=reuse_score)
        digests = {mode: rollout(controller, actions, digest=True)
                   for mode, controller in controllers.items()}
        if len(set(digests.values())) != 1:
            raise AssertionError(f"IPC changes output: {digests}")
        for controller in controllers.values():
            rollout(controller, actions)
        samples = {mode: [] for mode in controllers}
        for trial in range(trials):
            order = list(controllers) if trial % 2 == 0 else list(reversed(controllers))
            for mode in order:
                samples[mode].append(rollout(controllers[mode], actions))
    finally:
        for controller in controllers.values():
            controller.close()
    native = _run_nodejs_native_pool(
        game_path=_get_nodejs_native_game_path(game), level_i=0, n_envs=batch,
        n_steps=steps, timeout_ms=60000, repeats=trials + 2, execution_mode="nodejs_native_pool")
    return {"game": game, "batch": batch, "steps": steps, "output_sha256": digests,
            "game_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "samples_s": samples,
            "median_fps": {mode: float(batch * steps / np.median(times)) for mode, times in samples.items()},
            "native_engine_only": {"runs": native, "median_fps": float(np.median([r["fps"] for r in native[2:]]))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=["sokoban_basic", "blocks", "Zen_Puzzle_Garden"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=("serialization", "score"), default="serialization")
    args = parser.parse_args()
    if min([*args.batches, args.steps, args.trials]) < 1:
        parser.error("batch, steps and trials must be positive")
    result = {"host": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "source_hashes": source_hashes(), "load_average_at_start": os.getloadavg(),
              "node_version": subprocess.check_output(["node", "--version"], text=True).strip(),
              "trials": args.trials, "seed": 42,
              "experiment": args.experiment,
              "timing": "warmed alternating modes; reset and action generation outside timing",
              "native_note": "different engine-only workload; original pool includes per-run game compilation",
              "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for game in args.games:
        for batch in args.batches:
            row = benchmark(game, batch, args.steps, args.trials, args.experiment)
            if source_hashes() != result["source_hashes"]:
                raise RuntimeError("Benchmark source changed during the run; use a frozen snapshot")
            result["results"].append(row)
            temp = args.output.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, indent=2) + "\n")
            temp.replace(args.output)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
