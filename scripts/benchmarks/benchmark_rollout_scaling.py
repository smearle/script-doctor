"""Measure evolving batched rollouts with full and learner-facing outputs.

The transitions mode retains observations, actions, rewards and done flags,
plus the final state. It excludes per-step engine state and diagnostic info.
This measures environment collection, without a policy network or training.
Optional offline GPU traces capture a warmed call, excluding compilation.
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
from scripts.benchmarks.benchmark_manhattan_distance import (
    assert_equal, compile_runner, time_pair,
)


def make_rollout(env, params, batch, mode):
    def rollout(state, key, actions):
        def step(carry, action):
            state, key = carry
            key, step_key = jax.random.split(key)
            keys = jax.vmap(lambda i: jax.random.fold_in(step_key, i))(
                jnp.arange(batch))
            result = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                keys, state, action, params)
            # Observation is the returned next observation, including autoreset.
            output = result if mode == "full" else (
                result[0], action, result[2], result[3])
            return (result[1], key), output
        return jax.lax.scan(step, (state, key), actions)
    return rollout


def benchmark(game, batch, steps, trials, profile_dir=None):
    env = init_ps_env(game, level_i=0, max_episode_steps=100)
    params = PJParams(level=env.get_level(0), level_i=0)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(0), i))(
        jnp.arange(batch))
    _, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)
    # Batch prefixes share identical action sequences at every batch size.
    actions = jnp.asarray(np.random.default_rng(42).integers(
        0, env.action_space.n, (batch, steps), dtype=np.int32).T.copy())
    arguments = (state, jax.random.PRNGKey(1), actions)
    runners, outputs, stats = [], [], {}
    names = ("full", "transitions")
    for mode in names:
        runner, output, stats[mode] = compile_runner(
            make_rollout(env, params, batch, mode), arguments)
        memory = runner.memory_analysis()
        stats[mode].update(
            argument_bytes=memory.argument_size_in_bytes if memory else None,
            output_bytes=memory.output_size_in_bytes if memory else None,
        )
        runners.append(runner)
        outputs.append(output)
    full_carry, (obs, _, rewards, dones, _) = outputs[0]
    assert_equal((full_carry, (obs, actions, rewards, dones)), outputs[1])
    time_pair(runners, [arguments, arguments], stats, trials, names=names)
    for mode in names:
        stats[mode]["env_steps_per_s"] = batch * steps / stats[mode]["median_s"]
    if profile_dir:
        destination = Path(profile_dir) / f"{game}-{batch}"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "optimized_hlo.txt").write_text(runners[1].as_text())
        with jax.profiler.trace(str(destination), create_perfetto_trace=True):
            with jax.profiler.TraceAnnotation("warmed_transitions_rollout"):
                jax.block_until_ready(runners[1](*arguments))
        stats["trace_directory"] = str(destination)
    return {"game": game, "batch": batch, "steps": steps, **stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=[
        "sokoban_basic", "blocks", "Zen_Puzzle_Garden"])
    parser.add_argument("--batches", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-batch", type=int, default=4096)
    parser.add_argument("--sequential-cell-rules", action="store_true",
                        help="Disable independent-cell optimization to reproduce the baseline sweep.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min([*args.batches, args.steps, args.trials]) < 1:
        parser.error("batches, steps and trials must be positive")
    results = {"jax_version": jax.__version__,
               "devices": [d.device_kind for d in jax.devices()],
               "independent_cell_rules": not args.sequential_cell_rules,
               "trials": args.trials, "seed": 42,
               "timing": "alternating full/transitions after compilation and warmup",
               "results": []}
    original = env_module._is_independent_cell_rule
    try:
        if args.sequential_cell_rules:
            env_module._is_independent_cell_rule = lambda rule: False
        for game in args.games:
            for batch in args.batches:
                row = benchmark(game, batch, args.steps, args.trials,
                                args.profile_dir if batch == args.profile_batch else None)
                results["results"].append(row)
                print(json.dumps(row), flush=True)
                Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
                jax.clear_caches()
    finally:
        env_module._is_independent_cell_rule = original


if __name__ == "__main__":
    main()
