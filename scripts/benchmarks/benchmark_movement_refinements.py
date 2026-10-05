"""Exact-output A/B experiments for movement cleanup, layout, and compaction.

Baseline implementations are retained here so later production changes do not
silently change the comparison. All whole-engine tests include reset, every
rollout output, and automatic episode resets. Timing excludes compilation.
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
from puzzlescript_jax.env_utils import N_FORCES, N_MOVEMENTS, ACTION
from puzzlescript_jax.utils import init_ps_env
from scripts.benchmarks.benchmark_manhattan_distance import assert_equal, compile_runner, time_pair
from scripts.benchmarks.benchmark_rollout_scaling import make_rollout


def sequential_cleanup(lvl, coords, layer_masks):
    """The original cleanup loop, for valid force lists with suffix padding."""
    n_objs = layer_masks.shape[1]

    def body(carry):
        level, i = carry
        y, x, c = coords[i]
        layer = c // (N_FORCES - 1)
        mask = jnp.asarray(layer_masks)[layer]
        obj = engine._first_true_idx_large(jnp.where(mask, level[0, :n_objs, x, y], False))
        cleared = jax.lax.dynamic_update_slice(
            level, jnp.zeros((1, N_FORCES, 1, 1), dtype=bool),
            (0, n_objs + layer * N_FORCES, x, y))
        return jax.lax.select(obj == -1, cleared, level), i + 1

    return jax.lax.while_loop(lambda carry: coords[carry[1], 0] != -1, body, (lvl, 0))[0]


def gather_force_array(force_arr):
    mask = np.ones(force_arr.shape[0], dtype=bool)
    mask[ACTION::N_FORCES] = False
    return force_arr[mask].transpose(2, 1, 0)


def sliced_force_array(force_arr):
    _, height, width = force_arr.shape
    forces = force_arr.reshape(-1, N_FORCES, height, width)[:, :N_MOVEMENTS]
    return forces.transpose(3, 2, 0, 1).reshape(width, height, -1)


def compact_coords(mask, size, *, mask_barrier=False, prefix_barrier=False):
    if mask.size == 0 or size == 0:
        return jnp.full((size, mask.ndim), -1, dtype=jnp.int32)
    flat = mask.reshape(-1)
    if mask_barrier:
        flat = jax.lax.optimization_barrier(flat)
    offsets = jnp.cumsum(flat, dtype=jnp.int32) - 1
    if prefix_barrier:
        offsets = jax.lax.optimization_barrier(offsets)
    destinations = jnp.where(flat, offsets, size)
    indices = jnp.full((size,), -1, dtype=jnp.int32).at[destinations].set(
        jnp.arange(flat.size, dtype=jnp.int32), mode="drop")
    strides = np.cumprod(mask.shape[::-1])[::-1] // np.array(mask.shape)
    coords = (indices[:, None] // jnp.asarray(strides, dtype=jnp.int32)) % jnp.array(mask.shape)
    return jnp.where(indices[:, None] >= 0, coords, -1)


def prefix_barrier_coords(mask, size):
    return compact_coords(mask, size, prefix_barrier=True)


def mask_prefix_barrier_coords(mask, size):
    return compact_coords(mask, size, mask_barrier=True, prefix_barrier=True)


EXPERIMENTS = {
    "cleanup": ("_remove_invalid_forces", sequential_cleanup,
                {"parallel": engine._remove_invalid_forces}),
    "layout": ("_movement_force_array", gather_force_array, {"slice": sliced_force_array}),
    "compaction": ("_movement_coords", compact_coords,
                   {"prefix_barrier": prefix_barrier_coords,
                    "mask_prefix_barrier": mask_prefix_barrier_coords}),
    "combined": (
        ("_remove_invalid_forces", "_movement_force_array", "_movement_coords"),
        (sequential_cleanup, gather_force_array, compact_coords),
        {"cleanup_barriers": (engine._remove_invalid_forces, gather_force_array,
                              mask_prefix_barrier_coords),
         "cleanup_slice": (engine._remove_invalid_forces, sliced_force_array, compact_coords),
         "all": (engine._remove_invalid_forces, sliced_force_array, mask_prefix_barrier_coords)}),
}


def benchmark_pair(experiment, candidate, game, batch, steps, trials):
    attribute, baseline, candidates = EXPERIMENTS[experiment]
    attributes = (attribute,) if isinstance(attribute, str) else attribute
    original = tuple(getattr(engine, attr) for attr in attributes)
    runners, arguments, outputs, stats = [], [], [], {}
    try:
        for name, fn in [("baseline", baseline), (candidate, candidates[candidate])]:
            functions = (fn,) if isinstance(attribute, str) else fn
            for attr, function in zip(attributes, functions):
                setattr(engine, attr, function)
            print(f"{experiment}/{name}: compiling {game}, batch={batch}", flush=True)
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
        assert_equal(*outputs)
        print(f"Exact outputs match; timing {game}, batch={batch}", flush=True)
        time_pair(runners, arguments, stats, trials, names=("baseline", candidate))
        for row in stats.values():
            row["env_steps_per_s"] = batch * steps / row["median_s"]
    finally:
        for attr, function in zip(attributes, original):
            setattr(engine, attr, function)
    return {"experiment": experiment, "candidate": candidate, "game": game,
            "batch": batch, "steps": steps, **stats,
            "speedup": stats["baseline"]["median_s"] / stats[candidate]["median_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--candidates", nargs="+")
    parser.add_argument("--games", nargs="+", default=["sokoban_basic", "blocks", "Zen_Puzzle_Garden"])
    parser.add_argument("--batches", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min([*args.batches, args.steps, args.trials]) < 1:
        parser.error("batches, steps and trials must be positive")
    candidates = args.candidates or list(EXPERIMENTS[args.experiment][2])
    if not set(candidates) <= set(EXPERIMENTS[args.experiment][2]):
        parser.error("unknown candidate for this experiment")
    result = {"jax_version": jax.__version__, "devices": [d.device_kind for d in jax.devices()],
              "engine_sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
              "trials": args.trials, "seed": 42,
              "timing": "alternating A/B after compilation and warmup; exact full outputs",
              "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        for game in args.games:
            for batch in args.batches:
                row = benchmark_pair(args.experiment, candidate, game, batch, args.steps, args.trials)
                result["results"].append(row)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                jax.clear_caches()


if __name__ == "__main__":
    main()
