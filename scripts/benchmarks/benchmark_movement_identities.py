"""Try caching moving-object identities while retaining sequential movement.

The cache is valid only for disjoint object-to-layer assignments with complete
within-layer collisions. Movement never creates forces, and moving an object
clears its source layer's forces. While a force remains, its original object
cannot leave or be displaced by another object in that layer. Other layers
cannot change its identity. Overlapping masks/custom collisions use live reads.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import puzzlescript_jax.env as engine
from puzzlescript_jax.env import PJParams, RuleState
from puzzlescript_jax.env_utils import N_FORCES, N_MOVEMENTS
from puzzlescript_jax.utils import init_ps_env
from scripts.benchmarks.benchmark_manhattan_distance import compile_runner
from scripts.benchmarks.benchmark_rollout_scaling import make_rollout
from scripts.benchmarks.benchmark_chunked_rollouts import assert_equal_on_device


def can_cache_objects(layer_masks, coll_mat):
    if not isinstance(coll_mat, np.ndarray):
        return False
    if np.any(np.sum(layer_masks, axis=0) > 1):
        return False
    return all(np.all(coll_mat[np.ix_(ids, ids)])
               for ids in (np.flatnonzero(mask) for mask in layer_masks))


def object_indices(level, layer_masks):
    maps = []
    for mask in layer_masks:
        ids = np.flatnonzero(mask)
        if not len(ids):
            maps.append(jnp.full(level.shape[2:], -1, dtype=jnp.int32))
            continue
        present = level[0, ids]
        first = jnp.argmax(present, axis=0)
        maps.append(jnp.where(jnp.any(present, axis=0), jnp.asarray(ids, dtype=jnp.int32)[first], -1))
    return jnp.stack(maps)


def movement(self, rng, lvl, coll_mat, n_objs, obj_force_masks, jit=True, *, cache_objects):
    identities = object_indices(lvl, self.layer_masks) if (
        cache_objects and can_cache_objects(self.layer_masks, coll_mat)) else None
    coll_mat = jnp.asarray(coll_mat, dtype=bool)
    capacity = len(self.collision_layers) * lvl.shape[2] * lvl.shape[3] + 1
    force_arr = lvl[0, n_objs:-2 if self._is_multi_level else -1]
    coords = engine._movement_coords(engine._movement_force_array(force_arr), size=capacity)
    lvl = engine._remove_invalid_forces(lvl, coords, self.layer_masks)

    def attempt(carry):
        level, applied, key, i = carry
        y, x, channel = coords[i]
        layer = channel // N_MOVEMENTS
        force_present = (x != -1) & jnp.any(jax.lax.dynamic_slice(
            level, (0, n_objs + layer * N_FORCES, x, y), (1, N_MOVEMENTS, 1, 1)))
        if identities is None:
            mask = jnp.asarray(self.layer_masks)[layer]
            obj = engine._first_true_idx_large(jnp.where(mask, level[0, :self.n_objs, x, y], False))
        else:
            obj = identities[layer, x, y]
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]])[channel % N_MOVEMENTS]
        x1, y1 = x + delta[0], y + delta[1]
        collision = jnp.any(level[0, :n_objs, x1, y1] & coll_mat[obj])
        outside = (x1 < 0) | (x1 >= level.shape[2]) | (y1 < 0) | (y1 >= level.shape[3])
        valid = level[0, -1, x1, y1] if self._is_multi_level else True
        can_move = (obj != -1) & force_present & ~collision & ~outside & valid
        updated = level.at[0, obj, x, y].set(False).at[0, obj, x1, y1].set(True)
        updated = jax.lax.dynamic_update_slice(updated, jnp.zeros((1, N_FORCES, 1, 1), bool),
                                              (0, n_objs + layer * N_FORCES, x, y))
        return jax.lax.select(can_move, updated, level), applied | can_move, key, i + 1

    carry = (lvl, False, rng, 0)
    if jit:
        carry = jax.lax.while_loop(lambda c: coords[c[3], 0] != -1, attempt, carry)
    else:
        while coords[carry[3], 0] != -1 and not carry[1]:
            carry = attempt(carry)
    level, applied, key, _ = carry
    return RuleState(lvl=level, applied=applied, cancelled=False, restart=False,
                     again=False, win=False, rng=key)


def cached_movement(self, *args, **kwargs):
    return movement(self, *args, **kwargs, cache_objects=True)


def uncached_movement(self, *args, **kwargs):
    return movement(self, *args, **kwargs, cache_objects=False)


def benchmark(game, level, batch, steps, trials, seeds):
    original = engine.PuzzleJaxEnv.apply_movement
    runners, resets, stats = [], [], {}
    try:
        for name, fn in (("live", uncached_movement), ("cached", cached_movement)):
            engine.PuzzleJaxEnv.apply_movement = fn
            print(f"Compiling {game} level={level} batch={batch} {name}", flush=True)
            env = init_ps_env(game, level_i=level, max_episode_steps=100)
            params = PJParams(level=env.get_level(level), level_i=level)
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(0), i))(jnp.arange(batch))
            reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)
            actions = jnp.zeros((steps, batch), dtype=jnp.int32)
            runner, _, stats[name] = compile_runner(make_rollout(env, params, batch, "full"),
                                                   (reset[1], jax.random.PRNGKey(1), actions))
            runners.append(runner)
            resets.append(reset)
        assert_equal_on_device(*resets)
        for seed in seeds:
            actions = jnp.asarray(np.random.default_rng(seed).integers(
                0, env.action_space.n, (batch, steps), dtype=np.int32).T.copy())
            args = [(reset[1], jax.random.PRNGKey(seed), actions) for reset in resets]
            outputs = [jax.block_until_ready(fn(*a)) for fn, a in zip(runners, args)]
            assert_equal_on_device(*outputs)
            del outputs
            for _ in range(2):
                for fn, argument in zip(runners, args):
                    jax.block_until_ready(fn(*argument))
            durations = [[], []]
            for trial in range(trials):
                for i in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    start = time.perf_counter()
                    jax.block_until_ready(runners[i](*args[i]))
                    durations[i].append(time.perf_counter() - start)
            row = {"game": game, "level": level, "batch": batch, "steps": steps, "seed": seed,
                   "cache_guard": bool(can_cache_objects(env.layer_masks, env.coll_mat))}
            for name, times in zip(("live", "cached"), durations):
                row[name] = {**stats[name], "samples_s": times, "median_s": statistics.median(times),
                             "env_steps_per_s": batch * steps / statistics.median(times)}
            row["speedup"] = row["live"]["median_s"] / row["cached"]["median_s"]
            yield row
    finally:
        engine.PuzzleJaxEnv.apply_movement = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", nargs="+", default=["sokoban_basic", "blocks", "Zen_Puzzle_Garden"])
    p.add_argument("--levels", nargs="+", type=int, default=[0])
    p.add_argument("--batches", nargs="+", type=int, default=[256, 4096])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--trials", type=int, default=9)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1042])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min([*a.batches, a.steps, a.trials]) < 1 or min(a.levels) < 0:
        p.error("positive sizes and nonnegative levels required")
    result = {"jax_version": jax.__version__, "devices": [d.device_kind for d in jax.devices()],
              "engine_sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "trials": a.trials, "seeds": a.seeds, "max_episode_steps": 100,
              "output_mode": "full", "results": []}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for game in a.games:
        for level in a.levels:
            for batch in a.batches:
                for row in benchmark(game, level, batch, a.steps, a.trials, a.seeds):
                    result["results"].append(row)
                    a.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                jax.clear_caches()


if __name__ == "__main__":
    main()
