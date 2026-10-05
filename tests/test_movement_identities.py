"""Cached identities must preserve ordered moves, fallback, and padded maps."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puzzlescript_jax.env import PuzzleJaxEnv
from puzzlescript_jax.env_utils import N_FORCES
from scripts.benchmarks.benchmark_movement_identities import (
    cached_movement, uncached_movement, can_cache_objects, object_indices)
from scripts.benchmarks.benchmark_manhattan_distance import assert_equal


def fixture(overlap=False, collision_hole=False, multi=False):
    masks = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1]], bool)
    if overlap:
        masks[1, 0] = True
    coll = np.einsum('ij,ik->jk', masks, masks, dtype=bool)
    if collision_hole:
        coll[0, 1] = False
    env = SimpleNamespace(layer_masks=masks, collision_layers=list(range(3)), n_objs=6,
                          n_objs_per_layer=masks.sum(1), n_objs_prior_to_layer=np.array([0, 2, 5]),
                          _is_multi_level=multi)
    return env, coll


def test_cache_guard_requires_stable_identity():
    for overlap, hole, expected in [(False, False, True), (True, False, False), (False, True, False)]:
        env, coll = fixture(overlap, hole)
        assert can_cache_objects(env.layer_masks, coll) == expected
    assert not can_cache_objects(env.layer_masks, jnp.asarray(coll))


def test_identity_map_empty_layers_and_first_object():
    masks = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]], bool)
    level = np.array([[[[0, 1, 1]], [[0, 0, 1]], [[1, 0, 0]]]], bool)
    result = jax.jit(lambda x: object_indices(x, masks))(level)
    np.testing.assert_array_equal(result, [[[-1, 0, 0]], [[-1, -1, -1]], [[2, -1, -1]]])


@pytest.mark.parametrize('shape', [(1, 5), (4, 5), (9, 11)])
@pytest.mark.parametrize('mode', ['ordinary', 'padding', 'overlap', 'collision_hole'])
def test_random_ordered_movement_matches_original(shape, mode):
    env, coll = fixture(mode == 'overlap', mode == 'collision_hole', mode == 'padding')
    rng = np.random.default_rng(27)
    # Include empty cells, invalid orphan forces, and multiple objects per cell.
    levels = rng.random((64, 1, 6 + 3 * N_FORCES + 1 + int(env._is_multi_level), *shape)) < 0.3
    forces = np.zeros((64, 3, N_FORCES, *shape), bool)
    directions = rng.integers(0, N_FORCES + 1, (64, 3, *shape))
    for direction in range(N_FORCES):
        forces[:, :, direction] = directions == direction
    levels[:, 0, 6:21] = forces.reshape(64, 15, *shape)
    levels[0] = False
    levels[1, 0, :6] = False
    if env._is_multi_level:
        levels[:, 0, -1] = rng.random((64, *shape)) > 0.3
    def run(fn, level):
        def step(level, unused):
            result = fn(env, jax.random.PRNGKey(5), level, coll, 6, None)
            return result.lvl, result
        return jax.lax.scan(step, level, None, 4)
    outputs = [jax.jit(jax.vmap(lambda level: run(fn, level)))(levels)
               for fn in (PuzzleJaxEnv.apply_movement, uncached_movement, cached_movement)]
    assert_equal(outputs[0], outputs[1])
    assert_equal(outputs[0], outputs[2])


@pytest.mark.parametrize('overlap', [False, True])
def test_debug_path_keeps_first_move_behavior(overlap):
    env, coll = fixture(overlap=overlap)
    level = jnp.zeros((1, 22, 1, 4), bool)
    level = level.at[0, 0, 0, 0].set(True).at[0, 8, 0, 0].set(True)
    level = level.at[0, 2, 0, 2].set(True).at[0, 13, 0, 2].set(True)
    results = [fn(env, jax.random.PRNGKey(7), level, coll, 6, None, jit=False)
               for fn in (PuzzleJaxEnv.apply_movement, uncached_movement, cached_movement)]
    assert_equal(results[0], results[1])
    assert_equal(results[0], results[2])
    assert bool(results[2].lvl[0, 0, 0, 1])
    assert bool(results[2].lvl[0, 2, 0, 2])
