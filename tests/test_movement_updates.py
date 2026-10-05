"""Sparse movement preserves live collision reads and inactive batch members."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puzzlescript_jax.env import PuzzleJaxEnv
from puzzlescript_jax.env_utils import N_FORCES
from scripts.benchmarks.benchmark_manhattan_distance import assert_equal
from scripts.benchmarks.benchmark_movement_updates import CANDIDATES
from scripts.benchmarks.benchmark_movement_identities import uncached_movement


def make_fixture(shape, mode, count=24):
    masks = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1]], bool)
    if mode == 'overlap':
        masks[1, 0] = True
    coll = np.einsum('ij,ik->jk', masks, masks, dtype=bool)
    if mode == 'collision_hole':
        coll[0, 1] = coll[2, 2] = False
    env = SimpleNamespace(layer_masks=masks, collision_layers=list(range(3)), n_objs=6,
                          n_objs_per_layer=masks.sum(1), n_objs_prior_to_layer=np.array([0, 2, 5]),
                          _is_multi_level=mode == 'padding')
    rng = np.random.default_rng(83)
    levels = rng.random((count, 1, 22 + int(env._is_multi_level), *shape)) < 0.3
    forces = np.zeros((count, 3, N_FORCES, *shape), bool)
    directions = rng.integers(0, N_FORCES + 1, (count, 3, *shape))
    for direction in range(N_FORCES):
        forces[:, :, direction] = directions == direction
    levels[:, 0, 6:21] = forces.reshape(count, 15, *shape)
    # Empty lists, orphan forces, ACTION-only cells, and dense force lists must
    # coexist: some batch members finish long before the shared loop stops.
    levels[0] = False
    levels[1, 0, :6] = False
    levels[2, 0, 6:21] = False
    levels[2, 0, 10] = True
    if env._is_multi_level:
        levels[:, 0, -1] = rng.random((count, *shape)) > 0.3
    return env, coll, levels


@pytest.mark.parametrize('shape', [(1, 5), (4, 5), (9, 11)])
@pytest.mark.parametrize('mode', ['ordinary', 'padding', 'overlap', 'collision_hole'])
def test_ordered_movement_and_unequal_batch_completion(shape, mode):
    env, coll, levels = make_fixture(shape, mode)
    def run(fn, level):
        def step(board, _):
            result = fn(env, jax.random.PRNGKey(5), board, coll, 6, None)
            return result.lvl, result
        return jax.lax.scan(step, level, None, 4)
    baseline = jax.jit(jax.vmap(lambda level: run(uncached_movement, level)))(levels)
    for candidate in CANDIDATES.values():
        actual = jax.jit(jax.vmap(lambda level: run(candidate, level)))(levels)
        assert_equal(baseline, actual)


@pytest.mark.parametrize('candidate', CANDIDATES.values(), ids=CANDIDATES)
def test_unbatched_debug_and_nested_mapping(candidate):
    env, coll, levels = make_fixture((2, 4), 'ordinary')
    key = jax.random.PRNGKey(9)
    original = lambda board, matrix: uncached_movement(env, key, board, matrix, 6, None)
    optimized = lambda board, matrix: candidate(env, key, board, matrix, 6, None)
    assert_equal(jax.jit(original)(levels[3], coll), jax.jit(optimized)(levels[3], coll))
    assert_equal(jax.jit(jax.vmap(original, in_axes=(0, None)))(levels[3:4], coll),
                 jax.jit(jax.vmap(optimized, in_axes=(0, None)))(levels[3:4], coll))
    for board in levels[:4]:
        assert_equal(uncached_movement(env, key, jnp.asarray(board), coll, 6, None, jit=False),
                     candidate(env, key, jnp.asarray(board), coll, 6, None, jit=False))
    # A mapped collision matrix prevents the custom rule from treating dynamic
    # metadata as a closed-over constant. Nonleading and nested maps exercise
    # composition with callers' existing vmap conventions.
    matrices = np.broadcast_to(coll, (24, *coll.shape)).copy()
    matrices[::2, 0, 1] = False
    matrices[::3, 2, 2] = False
    args = (np.swapaxes(levels, 0, 1), matrices)
    assert_equal(jax.jit(jax.vmap(original, in_axes=(1, 0)))(*args),
                 jax.jit(jax.vmap(optimized, in_axes=(1, 0)))(*args))
    nested_args = (levels.reshape(4, 6, *levels.shape[1:]), matrices.reshape(4, 6, 6, 6))
    assert_equal(jax.jit(jax.vmap(jax.vmap(original)))(*nested_args),
                 jax.jit(jax.vmap(jax.vmap(optimized)))(*nested_args))
    # Unmapped boards (e.g. a shared initial level) must broadcast correctly.
    assert_equal(jax.jit(jax.vmap(original, in_axes=(None, 0)))(levels[3], matrices),
                 jax.jit(jax.vmap(optimized, in_axes=(None, 0)))(levels[3], matrices))


def test_mapped_switch_with_mixed_movement_and_identity_branches():
    env, collisions, levels = make_fixture((2, 4), 'padding')
    def run(method):
        def move(board):
            return method(env, jax.random.PRNGKey(7), board, collisions, 6, None).lvl
        def dispatch(board, branch):
            return jax.lax.switch(branch, (move, lambda board: board), board)
        with jax.checking_leaks():
            return jax.jit(jax.vmap(dispatch))(levels, jnp.arange(len(levels)) % 2)
    assert_equal(run(uncached_movement), run(PuzzleJaxEnv.apply_movement))
