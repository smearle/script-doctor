"""A/B deterministic single-cell rules against sequential match application."""

import argparse
import json
from pathlib import Path
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
from scripts.benchmarks.benchmark_rollout_scaling import make_rollout


INDEPENDENT = env_module._is_independent_cell_rule


def benchmark(game, batch, steps, trials):
    runners, arguments, outputs, stats = [], [], [], {}
    names = ("sequential", "parallel")
    try:
        for name in names:
            env_module._is_independent_cell_rule = (
                INDEPENDENT if name == "parallel" else lambda rule: False)
            env = init_ps_env(game, level_i=0, max_episode_steps=100)
            params = PJParams(level=env.get_level(0), level_i=0)
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(0), i))(
                jnp.arange(batch))
            reset_output = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)
            actions = jnp.asarray(np.random.default_rng(42).integers(
                0, env.action_space.n, (batch, steps), dtype=np.int32).T.copy())
            args = (reset_output[1], jax.random.PRNGKey(1), actions)
            runner, output, stats[name] = compile_runner(
                make_rollout(env, params, batch, "full"), args)
            stats[name]["hlo_while_ops"] = len(re.findall(r"\bwhile\(", runner.as_text()))
            runners.append(runner)
            arguments.append(args)
            outputs.append((reset_output, output))
        assert_equal(*outputs)
        time_pair(runners, arguments, stats, trials, names=names)
        for values in stats.values():
            values["env_steps_per_s"] = batch * steps / values["median_s"]
    finally:
        env_module._is_independent_cell_rule = INDEPENDENT
    return {"game": game, "batch": batch, "steps": steps, **stats,
            "speedup": stats["sequential"]["median_s"] / stats["parallel"]["median_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=[
        "sokoban_basic", "blocks", "Zen_Puzzle_Garden"])
    parser.add_argument("--batches", nargs="+", type=int, default=[256, 4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min([*args.batches, args.steps, args.trials]) < 1:
        parser.error("batches, steps and trials must be positive")
    results = {"jax_version": jax.__version__,
               "devices": [d.device_kind for d in jax.devices()],
               "trials": args.trials, "seed": 42,
               "timing": "alternating A/B after compilation and warmup",
               "results": []}
    for game in args.games:
        for batch in args.batches:
            row = benchmark(game, batch, args.steps, args.trials)
            results["results"].append(row)
            print(json.dumps(row), flush=True)
            Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
            jax.clear_caches()


if __name__ == "__main__":
    main()
