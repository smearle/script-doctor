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


def benchmark(env, batch, steps, trials, seed):
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
            "temporary_bytes": memory.temp_size_in_bytes if memory else None,
            "samples_s": times, "fps_samples": fps.tolist(),
            "median_fps": float(np.median(fps)),
            "q25_fps": float(np.percentile(fps, 25)), "q75_fps": float(np.percentile(fps, 75))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    game_arg = parser.add_mutually_exclusive_group(required=True)
    game_arg.add_argument("--game")
    game_arg.add_argument("--game-index", type=int, choices=range(len(PAPER_GAMES)))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 256, 1024, 4096, 16384])
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--min-steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min([*args.batches, args.n_steps, args.min_steps, args.trials]) < 1:
        parser.error("batches, step counts and trials must be positive")
    game = args.game if args.game is not None else PAPER_GAMES[args.game_index]
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
    output.write_text(json.dumps(result, indent=2) + "\n")
    for batch in args.batches:
        steps = args.n_steps if batch == 1 else max(args.n_steps // batch, args.min_steps)
        row = benchmark(env, batch, steps, args.trials, args.seed)
        result["results"].append(row)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
