"""Compare one large environment batch with sequential smaller chunks.

Each environment gets identical runtime actions and fold_in PRNG streams in
both variants. This is a controlled collection workload, separate from the
paper profiler's continuing random-action rollouts. Both modes check every
retained output exactly before alternating synchronized timings.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import puzzlescript_jax.env as engine
from puzzlescript_jax.env import PJParams
from puzzlescript_jax.utils import init_ps_env
from scripts.benchmarks.benchmark_manhattan_distance import compile_runner, time_pair


def make_rollout(env, params, chunk_size=None, mode="full"):
    if mode not in ("full", "final"):
        raise ValueError(f"Unknown output mode: {mode}")

    def segment(state, key, actions, offset):
        indices = offset + jnp.arange(actions.shape[1])

        def step(carry, action):
            state, key = carry
            key, step_key = jax.random.split(key)
            keys = jax.vmap(lambda i: jax.random.fold_in(step_key, i))(indices)
            result = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, state, action, params)
            return (result[1], key), result if mode == "full" else None

        return jax.lax.scan(step, (state, key), actions)

    def rollout(state, key, actions):
        if chunk_size is None:
            return segment(state, key, actions, 0)
        steps, batch = actions.shape
        if chunk_size < 1 or batch % chunk_size:
            raise ValueError("chunk_size must be positive and divide the batch")
        n_chunks = batch // chunk_size
        states = jax.tree.map(lambda x: x.reshape(n_chunks, chunk_size, *x.shape[1:]), state)
        acts = actions.reshape(steps, n_chunks, chunk_size).swapaxes(0, 1)
        offsets = jnp.arange(n_chunks) * chunk_size
        (states, keys), outputs = jax.lax.map(
            lambda entry: segment(entry[0], key, entry[1], entry[2]), (states, acts, offsets))
        state = jax.tree.map(lambda x: x.reshape(batch, *x.shape[2:]), states)
        outputs = jax.tree.map(
            lambda x: x.swapaxes(0, 1).reshape(steps, batch, *x.shape[3:]), outputs)
        # The master key advances once per step, independently of chunk size.
        return (state, keys[0]), outputs

    return rollout


def assert_equal_on_device(before, after):
    """Compare large trajectories without copying both to host RAM."""
    assert jax.tree.structure(before) == jax.tree.structure(after)
    compare = jax.jit(lambda a, b: jnp.array_equal(a, b, equal_nan=True))
    for i, (left, right) in enumerate(zip(jax.tree.leaves(before), jax.tree.leaves(after))):
        assert left.shape == right.shape and left.dtype == right.dtype, i
        assert bool(compare(left, right)), f"Mismatch in output leaf {i}"


def benchmark(game, batch, chunks, steps, trials, mode):
    print(f"Initializing {game}, batch={batch}", flush=True)
    env = init_ps_env(game, level_i=0, max_episode_steps=100)
    params = PJParams(level=env.get_level(0), level_i=0)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(0), i))(jnp.arange(batch))
    _, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)
    actions = jnp.asarray(np.random.default_rng(42).integers(
        0, env.action_space.n, (batch, steps), dtype=np.int32).T.copy())
    args = (state, jax.random.PRNGKey(1), actions)
    print("Compiling monolithic rollout", flush=True)
    baseline, expected, baseline_stats = compile_runner(make_rollout(env, params, mode=mode), args)
    baseline_stats["output_bytes"] = baseline.memory_analysis().output_size_in_bytes
    for chunk in chunks:
        print(f"Compiling chunk_size={chunk}", flush=True)
        candidate, actual, candidate_stats = compile_runner(make_rollout(env, params, chunk, mode), args)
        candidate_stats["output_bytes"] = candidate.memory_analysis().output_size_in_bytes
        assert_equal_on_device(expected, actual)
        del actual
        print("Exact outputs match; timing", flush=True)
        stats = {"monolithic": dict(baseline_stats), "chunked": candidate_stats}
        time_pair([baseline, candidate], [args, args], stats, trials, names=("monolithic", "chunked"))
        for value in stats.values():
            value["env_steps_per_s"] = batch * steps / value["median_s"]
        yield {"game": game, "batch": batch, "chunk_size": chunk, "steps": steps,
               "mode": mode, **stats,
               "speedup": stats["monolithic"]["median_s"] / stats["chunked"]["median_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=["sokoban_basic", "blocks", "Zen_Puzzle_Garden", "atlas shrank"])
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--chunks", nargs="+", type=int, default=[4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--mode", choices=("full", "final"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min([args.batch, *args.chunks, args.steps, args.trials]) < 1 or any(args.batch % c for c in args.chunks):
        parser.error("positive batch, chunks, steps and trials required; chunks must divide batch")
    result = {"jax_version": jax.__version__, "devices": [d.device_kind for d in jax.devices()],
              "engine_sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
              "trials": args.trials, "seed": 42, "max_episode_steps": 100,
              "timing": "alternating synchronized calls after compilation and warmup",
              "rng": "fold_in(master step key, global environment index)",
              "actions": "identical precomputed runtime inputs in both variants", "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for game in args.games:
        for row in benchmark(game, args.batch, args.chunks, args.steps, args.trials, args.mode):
            result["results"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
