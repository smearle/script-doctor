"""Compare histogram-based argwhere with stable prefix/scatter compaction.

Both whole-engine variants retain independent-cell rules and all earlier
optimizations. Complete reset/rollout outputs must match before paired timing.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import puzzlescript_jax.env as env_module
from puzzlescript_jax.env import PJParams
from puzzlescript_jax.utils import init_ps_env
from scripts.benchmarks.benchmark_manhattan_distance import assert_equal, compile_runner, time_pair
from scripts.benchmarks.benchmark_rollout_scaling import make_rollout


def argwhere_coords(mask, size):
    return jnp.argwhere(mask, size=size, fill_value=-1)


COMPACT = env_module._movement_coords
VARIANTS = (("argwhere", argwhere_coords), ("compact", COMPACT))
NAMES = tuple(name for name, _ in VARIANTS)


def benchmark_coordinates(size, batch, trials):
    rng = np.random.default_rng(42)
    masks = rng.random((batch, size, size, 12)) < 0.01
    if batch > 1:
        masks[0] = False
        masks[1] = True
    capacity = size * size * 3 + 1
    args = (jnp.asarray(masks),)
    runners, outputs, stats = [], [], {}
    for name, fn in VARIANTS:
        runner, output, stats[name] = compile_runner(jax.vmap(lambda mask: fn(mask, capacity)), args)
        runners.append(runner)
        outputs.append(output)
    assert_equal(*outputs)
    time_pair(runners, [args, args], stats, trials, names=NAMES)
    return {"kind": "coordinates", "shape": [size, size, 12], "batch": batch, **stats}


def benchmark_rollout(game, batch, steps, trials):
    runners, arguments, outputs, stats = [], [], [], {}
    try:
        for name, fn in VARIANTS:
            env_module._movement_coords = fn
            print(f"Compiling {game}, batch={batch}, variant={name}", flush=True)
            env = init_ps_env(game, level_i=0, max_episode_steps=100)
            params = PJParams(level=env.get_level(0), level_i=0)
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(0), i))(jnp.arange(batch))
            reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)
            actions = jnp.asarray(np.random.default_rng(42).integers(
                0, env.action_space.n, (batch, steps), dtype=np.int32).T.copy())
            args = (reset[1], jax.random.PRNGKey(1), actions)
            runner, output, stats[name] = compile_runner(make_rollout(env, params, batch, "full"), args)
            runners.append(runner)
            arguments.append(args)
            outputs.append((reset, output))
        print(f"Comparing and timing {game}, batch={batch}", flush=True)
        assert_equal(*outputs)
        time_pair(runners, arguments, stats, trials, names=NAMES)
        for values in stats.values():
            values["env_steps_per_s"] = batch * steps / values["median_s"]
    finally:
        env_module._movement_coords = COMPACT
    return {"kind": "rollout", "game": game, "batch": batch, "steps": steps, **stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="*", default=["sokoban_basic", "blocks", "Zen_Puzzle_Garden"])
    parser.add_argument("--sizes", nargs="*", type=int, default=[7, 12, 16])
    parser.add_argument("--batches", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min([*args.sizes, *args.batches, args.steps, args.trials]) < 1:
        parser.error("sizes, batches, steps and trials must be positive")
    results = {"jax_version": jax.__version__, "devices": [d.device_kind for d in jax.devices()],
               "trials": args.trials, "seed": 42, "timing": "alternating A/B after compilation and warmup",
               "results": []}

    def record(row):
        row["speedup"] = row["argwhere"]["median_s"] / row["compact"]["median_s"]
        results["results"].append(row)
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        jax.clear_caches()

    for size in args.sizes:
        for batch in args.batches:
            record(benchmark_coordinates(size, batch, args.trials))
    for game in args.games:
        for batch in args.batches:
            record(benchmark_rollout(game, batch, args.steps, args.trials))


if __name__ == "__main__":
    main()
