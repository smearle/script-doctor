"""Reproduce the paper's random-rollout workload with explicit warm timing.

Like profile_rand_jax, generate random actions inside a compiled scan, retain
only the final carry, use level zero and no practical episode-length cutoff,
and continue the carry across repetitions. Report median/IQR of warmed trials
instead of choosing a maximum. Compilation and warmup are excluded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from puzzlescript_jax.env import PJParams
from puzzlescript_jax.utils import init_ps_env


PAPER_GAMES = ["sokoban_basic", "notsnake", "Zen_Puzzle_Garden", "Slidings",
               "limerick", "kettle", "Take_Heart_Lass", "atlas shrank"]
BENCHMARK_GAMES = PAPER_GAMES + [
    "blocks", "nekopuzzle", "sokoban_match3", "Travelling_salesman",
    "Multi-word_Dictionary_Game", "Microban", "Magnet_Jack", "HyperMaze",
]


def scaling_stop(rows, *, min_gain=0.03, patience=2, regression=0.10):
    """Stop after a clear regression or consecutive doublings without progress.

    Compare medians with the best earlier median, not the fastest timing sample.
    Consider only the high-batch tail; small batches can have launch noise.
    """
    tail = sorted((r for r in rows if r["batch"] >= 4096), key=lambda r: r["batch"])
    if len(tail) < 2:
        return None
    best = tail[0]["median_fps"]
    stalled = 0
    for row in tail[1:]:
        fps = row["median_fps"]
        if fps < best * (1 - regression):
            return "regression"
        if fps > best * (1 + min_gain):
            stalled = 0
        else:
            stalled += 1
        best = max(best, fps)
    return "plateau" if stalled >= patience else None


def save_result(path, result):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def make_random_rollout(env, params, batch, steps):
    def rollout(carry):
        def step(carry, unused):
            state, key = carry
            key, step_key = jax.random.split(key)
            # Preserve the existing paper profiler's RNG convention.
            action = jax.random.randint(step_key, (batch,), 0, env.action_space.n)
            keys = jax.random.split(step_key, batch)
            result = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, state, action, params)
            return (result[1], key), None
        return jax.lax.scan(step, carry, None, steps)[0]
    return rollout


def benchmark(env, batch, steps, trials, seed, *, params=None):
    if params is None:
        params = PJParams(level=env.get_level(0), level_i=0)
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    _, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(jax.random.split(reset_key, batch), params)
    carry = jax.block_until_ready((state, key))
    print(f"Compiling batch={batch}, steps={steps}", flush=True)
    start = time.perf_counter()
    runner = jax.jit(make_random_rollout(env, params, batch, steps)).lower(carry).compile()
    compile_s = time.perf_counter() - start
    print(f"Compiled in {compile_s:.2f}s; warming up", flush=True)
    carry = jax.block_until_ready(runner(carry))
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        carry = jax.block_until_ready(runner(carry))
        times.append(time.perf_counter() - start)
    fps = batch * steps / np.asarray(times)
    memory = runner.memory_analysis()
    return {"batch": batch, "steps": steps, "compile_s": compile_s,
            "host": platform.node(),
            "argument_bytes": memory.argument_size_in_bytes if memory else None,
            "output_bytes": memory.output_size_in_bytes if memory else None,
            "temporary_bytes": memory.temp_size_in_bytes if memory else None,
            "samples_s": times, "fps_samples": fps.tolist(),
            "median_fps": float(np.median(fps)),
            "q25_fps": float(np.percentile(fps, 25)), "q75_fps": float(np.percentile(fps, 75))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    game_arg = parser.add_mutually_exclusive_group(required=True)
    game_arg.add_argument("--game")
    game_arg.add_argument("--game-index", type=int, choices=range(len(BENCHMARK_GAMES)))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 256, 1024, 4096, 16384])
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-params", action="store_true",
                        help="Precompute deterministic reset state outside timed rollouts.")
    parser.add_argument("--adaptive", action="store_true", help="Double batches until plateau or regression.")
    parser.add_argument("--max-batch", type=int, default=1048576)
    parser.add_argument("--min-gain", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--regression", type=float, default=0.10)
    args = parser.parse_args()
    if min([*args.batches, args.n_steps, args.min_steps, args.trials]) < 1:
        parser.error("batches, step counts and trials must be positive")
    if args.max_batch < max(args.batches) or args.patience < 1 or not 0 <= args.min_gain < 1 or not 0 < args.regression < 1:
        parser.error("invalid adaptive limits")
    if sorted(set(args.batches)) != args.batches:
        parser.error("batches must be unique and increasing")
    game = args.game if args.game is not None else BENCHMARK_GAMES[args.game_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{game}.json"
    env = init_ps_env(game, level_i=0, max_episode_steps=np.iinfo(np.int32).max)
    engine_path = Path(__file__).resolve().parents[2] / "puzzlescript_jax" / "env.py"
    result = {"game": game, "level": 0, "jax_version": jax.__version__,
              "devices": [d.device_kind for d in jax.devices()], "host": platform.node(),
              "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
              "requested_batches": args.batches,
              "trials": args.trials, "seed": args.seed, "base_steps": args.n_steps,
              "min_steps": args.min_steps, "max_episode_steps": int(np.iinfo(np.int32).max),
              "timing": "median/IQR of synchronized warmed calls; carry continues across trials",
              "output_mode": "final_carry", "action_generation": "inside timed scan",
              "results": []}
    if output.exists():
        if not args.resume:
            parser.error(f"{output} exists; use --resume or a new output directory")
        previous = json.loads(output.read_text())
        if previous.get("prepare_params", False) != args.prepare_params:
            parser.error("cannot resume with changed reset preparation")
        for field in ("game", "engine_sha256", "jax_version", "devices", "level", "trials", "seed",
                      "base_steps", "min_steps", "max_episode_steps", "output_mode", "action_generation"):
            if previous[field] != result[field]:
                parser.error(f"cannot resume with changed {field}")
        result = previous
        for row in result["results"]:
            row.setdefault("host", previous["host"])
    result["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["prepare_params"] = args.prepare_params
    if args.adaptive:
        result["adaptive"] = {"factor": 2, "max_batch": args.max_batch, "min_gain": args.min_gain,
                              "patience": args.patience, "regression": args.regression}
    result.pop("stop_reason", None)
    save_result(output, result)
    prepared_params = None
    if args.prepare_params:
        start = time.perf_counter()
        prepared_params = env.prepare_params(PJParams(level=env.get_level(0), level_i=0))
        jax.block_until_ready(prepared_params)
        result["reset_preparation_s"] = time.perf_counter() - start
        result["reset_cache_used"] = prepared_params.reset_cache is not None
        save_result(output, result)
    pending = [b for b in args.batches if b not in {r["batch"] for r in result["results"]}]
    while True:
        if pending:
            batch = pending.pop(0)
        elif args.adaptive:
            reason = scaling_stop(result["results"], min_gain=args.min_gain,
                                  patience=args.patience, regression=args.regression)
            batch = max(r["batch"] for r in result["results"]) * 2
            if reason or batch > args.max_batch:
                result["stop_reason"] = reason or "max_batch"
                break
        else:
            result["stop_reason"] = "fixed_batches_complete"
            break
        steps = args.n_steps if batch == 1 else max(args.n_steps // batch, args.min_steps)
        result["pending_batch"] = batch
        save_result(output, result)
        try:
            row = benchmark(env, batch, steps, args.trials, args.seed, params=prepared_params)
        except Exception as error:
            result["stop_reason"] = "error"
            result["error"] = {"batch": batch, "type": type(error).__name__, "message": str(error)}
            save_result(output, result)
            raise
        result["results"].append(row)
        result["requested_batches"] = sorted(set(result["requested_batches"] + [batch]))
        result.pop("pending_batch", None)
        result.pop("error", None)
        save_result(output, result)
        print(json.dumps(row), flush=True)
        jax.clear_caches()
    save_result(output, result)
    print(f"Stopped: {result['stop_reason']}", flush=True)


if __name__ == "__main__":
    main()
