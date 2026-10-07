"""Compare exact nearest-target heuristics and optional full game rollouts.

Run from the repository root, for example::

    python -m scripts.benchmarks.benchmark_manhattan_distance \
        --sizes 8 16 32 --batches 1 64 --games sokoban_basic Microban \
        --steps 100 --trials 7 --output /tmp/manhattan-benchmark.json

Each timed call is synchronized. Inputs are runtime arguments; there is no
loop of identical calls for XLA to hoist. Rollouts carry evolving states and
compare every returned observation, state, reward, done flag, and info field.
"""

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import puzzlescript_jax.env as env_module
from puzzlescript_jax.env import PJParams
from puzzlescript_jax.utils import init_ps_env


def pairwise_sum_nearest(src_channel, trg_channel):
    """Pre-optimization implementation, including the current empty fallback."""
    max_dist = sum(src_channel.shape)
    src_coords = jnp.argwhere(src_channel, size=src_channel.size, fill_value=-1)
    is_real_src = ~jnp.all(src_coords == -1, axis=-1)
    dists = env_module.compute_manhattan_dists_from_channels(src_channel, trg_channel)
    dists = jnp.nanmin(dists, axis=1)
    dists = jnp.where(jnp.isnan(dists), jnp.where(is_real_src, max_dist, 0), dists)
    return jnp.sum(dists).astype(jnp.int32)


def pairwise_min_nearest(src_channel, trg_channel):
    max_dist = sum(src_channel.shape)
    dists = env_module.compute_manhattan_dists_from_channels(src_channel, trg_channel)
    return jnp.min(jnp.where(jnp.isnan(dists), max_dist, dists)).astype(jnp.int32)


PAIRWISE = (pairwise_sum_nearest, pairwise_min_nearest)
TRANSFORM = (
    env_module.compute_sum_of_manhattan_dists_from_channels,
    env_module.compute_min_manhattan_dist_from_channels,
)


def compile_runner(fn, args):
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - start
    output = jax.block_until_ready(compiled(*args))
    memory = compiled.memory_analysis()
    return compiled, output, {
        "compile_s": compile_s,
        "temporary_bytes": memory.temp_size_in_bytes if memory else None,
    }


def time_pair(runners, arguments, stats, trials, names=("pairwise", "transform")):
    """Alternate A/B order after both versions compile to limit timing drift."""
    for _ in range(2):
        for fn, args in zip(runners, arguments):
            jax.block_until_ready(fn(*args))
    durations = [[], []]
    for trial in range(trials):
        for i in ((0, 1) if trial % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            jax.block_until_ready(runners[i](*arguments[i]))
            durations[i].append(time.perf_counter() - start)
    for name, samples in zip(names, durations):
        stats[name].update(
            median_s=statistics.median(samples),
            min_s=min(samples),
            max_s=max(samples),
        )


def assert_equal(before, after):
    assert jax.tree.structure(before) == jax.tree.structure(after)
    for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def benchmark_metrics(size, batch, trials):
    rng = np.random.default_rng(42)
    sources = rng.random((batch, size, size)) < 0.1
    targets = rng.random((batch, size, size)) < 0.1
    # Include empty channels in larger batches, too.
    if batch > 1:
        sources[0] = False
        targets[1] = False
    args = (jnp.asarray(sources), jnp.asarray(targets))
    outputs, runners, stats = [], [], {}
    for name, (sum_fn, min_fn) in [("pairwise", PAIRWISE), ("transform", TRANSFORM)]:
        fn = jax.vmap(lambda src, trg: (sum_fn(src, trg), min_fn(src, trg)))
        runner, output, stats[name] = compile_runner(fn, args)
        runners.append(runner)
        outputs.append(output)
    assert_equal(*outputs)
    time_pair(runners, [args, args], stats, trials)
    return {"kind": "metrics", "shape": [size, size], "batch": batch, **stats}


def benchmark_rollout(game, batch, steps, trials):
    outputs, runners, arguments, stats = [], [], [], {}
    try:
        for name, funcs in [("pairwise", PAIRWISE), ("transform", TRANSFORM)]:
            (env_module.compute_sum_of_manhattan_dists_from_channels,
             env_module.compute_min_manhattan_dist_from_channels) = funcs
            env = init_ps_env(game, level_i=0, max_episode_steps=100)
            params = PJParams(level=env.get_level(0), level_i=0)
            keys = jax.random.split(jax.random.PRNGKey(0), batch)
            _, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)
            rng = np.random.default_rng(42)
            actions = jnp.asarray(rng.integers(0, env.action_space.n, (steps, batch), dtype=np.int32))

            def rollout(state, key, actions):
                def step(carry, action):
                    state, key = carry
                    key, step_key = jax.random.split(key)
                    keys = jax.random.split(step_key, batch)
                    result = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, state, action, params)
                    return (result[1], key), result
                return jax.lax.scan(step, (state, key), actions)

            args = (state, jax.random.PRNGKey(1), actions)
            runner, output, stats[name] = compile_runner(rollout, args)
            runners.append(runner)
            arguments.append(args)
            outputs.append(output)
        assert_equal(*outputs)
        time_pair(runners, arguments, stats, trials)
        for values in stats.values():
            values["env_steps_per_s"] = batch * steps / values["median_s"]
    finally:
        (env_module.compute_sum_of_manhattan_dists_from_channels,
         env_module.compute_min_manhattan_dist_from_channels) = TRANSFORM
    return {"kind": "rollout", "game": game, "batch": batch, "steps": steps, **stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=[8, 16, 32])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--games", nargs="*", default=[])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min([*args.sizes, *args.batches, args.steps, args.trials]) < 1:
        parser.error("sizes, batches, steps, and trials must be positive")
    results = {
        "jax_version": jax.__version__,
        "devices": [d.device_kind for d in jax.devices()],
        "timing": "alternating A/B after both compilations and warmup",
        "trials": args.trials,
        "seed": 42,
        "results": [],
    }
    print(json.dumps({k: v for k, v in results.items() if k != "results"}), flush=True)

    def record(row):
        row["speedup"] = row["pairwise"]["median_s"] / row["transform"]["median_s"]
        results["results"].append(row)
        print(json.dumps(row), flush=True)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
        jax.clear_caches()

    for size in args.sizes:
        for batch in args.batches:
            record(benchmark_metrics(size, batch, args.trials))
    for game in args.games:
        for batch in args.batches:
            record(benchmark_rollout(game, batch, args.steps, args.trials))


if __name__ == "__main__":
    main()
