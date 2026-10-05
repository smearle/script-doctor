"""Compare sort-based match coordinates with direct row/column enumeration.

Times synchronized calls with runtime inputs, then full evolving game rollouts.
Every output leaf must match exactly before A/B timing. Both variants use the
current heuristic and turn logic, isolating the coordinate collection change.
"""

import argparse
import json
import re

import jax
import jax.numpy as jnp
import numpy as np

import puzzlescript_jax.env as env_module
from puzzlescript_jax.env import PJParams
from puzzlescript_jax.utils import init_ps_env
from scripts.benchmarks.benchmark_manhattan_distance import (
    assert_equal, compile_runner, time_pair,
)


def sorted_match_coords(kernel_activations, kernel_order_is_col):
    """Original production coordinate path, retained as the A/B baseline."""
    height, width = kernel_activations.shape[1:]
    coords = jnp.stack([
        jnp.argwhere(mask, size=height * width, fill_value=-1)
        for mask in kernel_activations
    ])
    rows, cols = coords[:, :, 0], coords[:, :, 1]
    keys = jnp.where(jnp.array(kernel_order_is_col)[:, None],
                     cols * height + rows, rows * width + cols)
    keys = jnp.where(rows != -1, keys, jnp.iinfo(jnp.int32).max)
    indices = jnp.argsort(keys, axis=1)
    return jnp.take_along_axis(coords, indices[:, :, None], axis=1)


DIRECT = env_module._ordered_match_coords
VARIANTS = (("sorted", sorted_match_coords), ("direct", DIRECT))
NAMES = tuple(name for name, _ in VARIANTS)


def compile_with_sort_count(fn, args):
    runner, output, stats = compile_runner(fn, args)
    # Count operations, not references in metadata or names. A zero confirms
    # that the replacement does not introduce another sort in its lowering.
    stats["hlo_sort_ops"] = len(re.findall(r"\bsort\(", runner.as_text()))
    return runner, output, stats


def benchmark_coordinates(size, batch, trials):
    rng = np.random.default_rng(42)
    masks = rng.random((batch, 2, size, size)) < 0.1
    if batch > 1:
        masks[0] = False
        masks[1] = True
    args = (jnp.asarray(masks),)
    runners, outputs, stats = [], [], {}
    for name, collect in VARIANTS:
        runner, output, stats[name] = compile_with_sort_count(
            jax.vmap(lambda masks: collect(masks, (False, True))), args)
        runners.append(runner)
        outputs.append(output)
    assert_equal(*outputs)
    time_pair(runners, [args, args], stats, trials, names=NAMES)
    return {"kind": "coordinates", "shape": [size, size], "batch": batch,
            "kernels": 2, "column_major": [False, True], **stats}


def benchmark_rollout(game, batch, steps, trials):
    outputs, runners, arguments, stats = [], [], [], {}
    try:
        for name, collect in VARIANTS:
            env_module._ordered_match_coords = collect
            env = init_ps_env(game, level_i=0, max_episode_steps=100)
            params = PJParams(level=env.get_level(0), level_i=0)
            keys = jax.random.split(jax.random.PRNGKey(0), batch)
            _, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)
            rng = np.random.default_rng(42)
            actions = jnp.asarray(rng.integers(0, env.action_space.n,
                                             (steps, batch), dtype=np.int32))

            def rollout(state, key, actions):
                def step(carry, action):
                    state, key = carry
                    key, step_key = jax.random.split(key)
                    keys = jax.random.split(step_key, batch)
                    result = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                        keys, state, action, params)
                    return (result[1], key), result
                return jax.lax.scan(step, (state, key), actions)

            args = (state, jax.random.PRNGKey(1), actions)
            runner, output, stats[name] = compile_with_sort_count(rollout, args)
            runners.append(runner)
            arguments.append(args)
            outputs.append(output)
        assert_equal(*outputs)
        time_pair(runners, arguments, stats, trials, names=NAMES)
        for values in stats.values():
            values["env_steps_per_s"] = batch * steps / values["median_s"]
    finally:
        env_module._ordered_match_coords = DIRECT
    return {"kind": "rollout", "game": game, "batch": batch, "steps": steps, **stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=[8, 16, 32])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 64, 256])
    parser.add_argument("--games", nargs="*", default=["sokoban_basic", "blocks", "Zen_Puzzle_Garden"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min([*args.sizes, *args.batches, args.steps, args.trials]) < 1:
        parser.error("sizes, batches, steps, and trials must be positive")
    results = {
        "jax_version": jax.__version__,
        "devices": [d.device_kind for d in jax.devices()],
        "timing": "alternating A/B after both compilations and warmup",
        "trials": args.trials, "seed": 42, "results": [],
    }
    print(json.dumps({k: v for k, v in results.items() if k != "results"}), flush=True)

    def record(row):
        row["speedup"] = row["sorted"]["median_s"] / row["direct"]["median_s"]
        results["results"].append(row)
        print(json.dumps(row), flush=True)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
        jax.clear_caches()

    for size in args.sizes:
        for batch in args.batches:
            record(benchmark_coordinates(size, batch, args.trials))
    for game in args.games:
        for batch in args.batches:
            record(benchmark_rollout(game, batch, args.steps, args.trials))


if __name__ == "__main__":
    main()
