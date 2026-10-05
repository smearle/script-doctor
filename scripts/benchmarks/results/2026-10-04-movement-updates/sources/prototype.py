"""A/B sparse movement updates and an explicitly batched movement loop.

Each environment still visits the same ordered coordinate list and reads live
objects, forces, and collisions. Rejected moves scatter outside the channel
dimension with mode='drop'. A shared batched loop stops after the longest list;
sentinel coordinates make all updates no-ops for already-finished environments.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import jax
from jax.custom_batching import custom_vmap
import jax.numpy as jnp
import numpy as np

import puzzlescript_jax.env as engine
from puzzlescript_jax.env import PJParams, RuleState
from puzzlescript_jax.env_utils import N_FORCES, N_MOVEMENTS
from puzzlescript_jax.utils import init_ps_env
from scripts.benchmarks.benchmark_chunked_rollouts import assert_equal_on_device
from scripts.benchmarks.benchmark_manhattan_distance import compile_runner
from scripts.benchmarks.benchmark_movement_identities import uncached_movement
from scripts.benchmarks.benchmark_rollout_scaling import make_rollout


def movement(self, rng, lvl, coll_mat, n_objs, obj_force_masks, jit=True,
             *, batched_loop, sparse_updates):
    if not jit:
        return uncached_movement(self, rng, lvl, coll_mat, n_objs, obj_force_masks, jit=False)
    capacity = len(self.collision_layers) * lvl.shape[2] * lvl.shape[3] + 1
    forces = lvl[0, n_objs:-2 if self._is_multi_level else -1]
    coords = engine._movement_coords(engine._movement_force_array(forces), size=capacity)
    lvl = engine._remove_invalid_forces(lvl, coords, self.layer_masks)
    multi_level = self._is_multi_level

    def attempt(level, applied, coord, collisions, masks):
        y, x, channel = coord
        layer = channel // N_MOVEMENTS
        force_present = (x != -1) & jnp.any(jax.lax.dynamic_slice(
            level, (0, n_objs + layer * N_FORCES, x, y), (1, N_MOVEMENTS, 1, 1)))
        obj = engine._first_true_idx_large(jnp.where(masks[layer], level[0, :n_objs, x, y], False))
        delta = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]])[channel % N_MOVEMENTS]
        x1, y1 = x + delta[0], y + delta[1]
        collision = jnp.any(level[0, :n_objs, x1, y1] & collisions[obj])
        outside = (x1 < 0) | (x1 >= level.shape[2]) | (y1 < 0) | (y1 >= level.shape[3])
        valid = level[0, -1, x1, y1] if multi_level else True
        can_move = (obj != -1) & force_present & ~collision & ~outside & valid
        if sparse_updates:
            # A positive, out-of-bounds channel drops the entire update, even
            # when sentinel coordinates have negative spatial components.
            target_obj = jnp.where(can_move, obj, level.shape[1])
            force_channels = jnp.where(can_move,
                                      n_objs + layer * N_FORCES + jnp.arange(N_FORCES),
                                      level.shape[1])
            level = level.at[0, target_obj, x, y].set(False, mode='drop')
            level = level.at[0, target_obj, x1, y1].set(True, mode='drop')
            level = level.at[0, force_channels, x, y].set(False, mode='drop')
        else:
            updated = level.at[0, obj, x, y].set(False).at[0, obj, x1, y1].set(True)
            updated = jax.lax.dynamic_update_slice(updated, jnp.zeros((1, N_FORCES, 1, 1), bool),
                                                  (0, n_objs + layer * N_FORCES, x, y))
            level = jax.lax.select(can_move, updated, level)
        return level, applied | can_move

    def single_loop(level, coordinates, collisions, masks):
        def body(carry):
            updated, applied = attempt(carry[0], carry[1], coordinates[carry[2]], collisions, masks)
            return updated, applied, carry[2] + 1
        updated, applied, _ = jax.lax.while_loop(
            lambda carry: coordinates[carry[2], 0] != -1, body, (level, False, 0))
        return updated, applied

    if batched_loop:
        run = custom_vmap(single_loop)

        @run.def_vmap
        def run_batched(axis_size, in_batched, level, coordinates, collisions, masks):
            # All dynamic values are explicit arguments: the rule also works
            # with mapped collision matrices and non-leading/nested vmaps.
            level, coordinates, collisions, masks = [
                value if mapped else jnp.broadcast_to(value, (axis_size, *value.shape))
                for value, mapped in zip((level, coordinates, collisions, masks), in_batched)]
            def body(carry):
                updated, applied = jax.vmap(attempt)(
                    carry[0], carry[1], coordinates[:, carry[2]], collisions, masks)
                return updated, applied, carry[2] + 1
            updated, applied, _ = jax.lax.while_loop(
                lambda carry: jnp.any(coordinates[:, carry[2], 0] != -1), body,
                (level, jnp.zeros(axis_size, bool), 0))
            return (updated, applied), (True, True)
    else:
        run = single_loop
    level, applied = run(lvl, coords, jnp.asarray(coll_mat, bool), jnp.asarray(self.layer_masks))
    return RuleState(lvl=level, applied=applied, cancelled=False, restart=False,
                     again=False, win=False, rng=rng)


def masked_movement(self, *args, **kwargs):
    return movement(self, *args, **kwargs, batched_loop=False, sparse_updates=True)


def batched_movement(self, *args, **kwargs):
    return movement(self, *args, **kwargs, batched_loop=True, sparse_updates=True)


def batched_select_movement(self, *args, **kwargs):
    return movement(self, *args, **kwargs, batched_loop=True, sparse_updates=False)


CANDIDATES = {'masked': masked_movement, 'batched': batched_movement,
              'batched_select': batched_select_movement}


def benchmark(candidate, game, level, batch, steps, trials, seeds):
    original = engine.PuzzleJaxEnv.apply_movement
    runners, resets, stats = [], [], {}
    try:
        for name, fn in [('baseline', uncached_movement), ('candidate', CANDIDATES[candidate])]:
            engine.PuzzleJaxEnv.apply_movement = fn
            print(f'{candidate}/{name}: {game} level={level} batch={batch}', flush=True)
            env = init_ps_env(game, level_i=level, max_episode_steps=100)
            params = PJParams(level=env.get_level(level), level_i=level)
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(0), i))(jnp.arange(batch))
            reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)
            runner, _, stats[name] = compile_runner(make_rollout(env, params, batch, 'full'),
                (reset[1], jax.random.PRNGKey(1), jnp.zeros((steps, batch), jnp.int32)))
            runners.append(runner)
            resets.append(reset)
        assert_equal_on_device(*resets)
        for seed in seeds:
            actions = jnp.asarray(np.random.default_rng(seed).integers(
                0, env.action_space.n, (batch, steps), dtype=np.int32).T.copy())
            args = [(reset[1], jax.random.PRNGKey(seed), actions) for reset in resets]
            outputs = [jax.block_until_ready(fn(*argument)) for fn, argument in zip(runners, args)]
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
            row = {'candidate_name': candidate, 'game': game, 'level': level,
                   'batch': batch, 'steps': steps, 'seed': seed}
            for name, times in zip(('baseline', 'candidate'), durations):
                row[name] = {**stats[name], 'samples_s': times, 'median_s': statistics.median(times),
                             'env_steps_per_s': batch * steps / statistics.median(times)}
            row['speedup'] = row['baseline']['median_s'] / row['candidate']['median_s']
            yield row
    finally:
        engine.PuzzleJaxEnv.apply_movement = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', nargs='+', choices=CANDIDATES, default=['masked', 'batched'])
    p.add_argument('--games', nargs='+', default=['sokoban_basic', 'blocks', 'Zen_Puzzle_Garden'])
    p.add_argument('--levels', nargs='+', type=int, default=[0])
    p.add_argument('--batches', nargs='+', type=int, default=[256, 4096])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--trials', type=int, default=9)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 1042])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min([*a.batches, a.steps, a.trials]) < 1 or min([*a.levels, *a.seeds]) < 0:
        p.error('positive sizes and nonnegative levels/seeds required')
    result = {'jax_version': jax.__version__, 'devices': [d.device_kind for d in jax.devices()],
              'engine_sha256': hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'trials': a.trials, 'seeds': a.seeds, 'max_episode_steps': 100,
              'output_mode': 'full', 'results': []}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for candidate in a.candidates:
        for game in a.games:
            for level in a.levels:
                for batch in a.batches:
                    for row in benchmark(candidate, game, level, batch, a.steps, a.trials, a.seeds):
                        result['results'].append(row)
                        temporary = a.output.with_suffix('.tmp')
                        temporary.write_text(json.dumps(result, indent=2) + '\n')
                        temporary.replace(a.output)
                        print(json.dumps(row), flush=True)
                    jax.clear_caches()


if __name__ == '__main__':
    main()
