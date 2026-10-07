"""Prepared parameters must retain reset semantics and invalidate stale caches."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puzzlescript_jax.env import PJParams, PuzzleJaxEnv
from puzzlescript_jax.preprocessing import get_tree_from_txt
from puzzlescript_jax.utils import init_ps_lark_parser


def assert_trees_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(left, right)


def make_env(game, level_i=0, jit=True):
    tree, _, error = get_tree_from_txt(init_ps_lark_parser(), game, test_env_init=False)
    assert tree is not None, error
    return PuzzleJaxEnv(tree, level_i=level_i, print_score=False, max_steps=3, jit=jit)


@pytest.mark.parametrize('game', ['sokoban_basic', 'test_run_rules_on_level_start_score',
                                  'test_cancel_again', 'test_again'])
def test_prepared_rollouts_match_all_outputs_including_automatic_resets(game):
    env = make_env(game)
    ordinary = PJParams(level=env.get_level(0))
    prepared = env.prepare_params(ordinary)
    assert prepared.reset_cache is not None
    keys = jax.random.split(jax.random.PRNGKey(42), 3)

    def rollout(params):
        reset = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
        def step(state, action):
            output = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                keys, state, jnp.full((3,), action), params)
            return output[1], output
        result = jax.lax.scan(step, reset[1], jnp.array([2, 4, 1, 0, 3, 4, 2, 2], jnp.int32))
        return reset, result

    with jax.checking_leaks():
        expected = jax.jit(rollout)(ordinary)
        actual = jax.jit(rollout)(prepared)
    assert_trees_equal(expected, actual)
    assert np.any(np.asarray(actual[1][1][3]))


def test_changed_level_and_other_environment_do_not_reuse_stale_state():
    env = make_env('test_run_rules_on_level_start_score')
    ordinary = PJParams(level=env.get_level(0))
    prepared = env.prepare_params(ordinary)
    # Flip a non-background object at one cell, preserving shape. The cached
    # board and score must not hide the replacement parameter's contents.
    changed = ordinary.level.at[-1, 0, 0].set(~ordinary.level[-1, 0, 0])
    key = jax.random.PRNGKey(1042)
    assert_trees_equal(env.reset(key, ordinary.replace(level=changed)),
                       env.reset(key, prepared.replace(level=changed)))
    other = make_env('test_run_rules_on_level_start_score')
    poisoned = prepared.replace(reset_cache=prepared.reset_cache.replace(
        state=prepared.reset_cache.state.replace(score=jnp.array(12345))))
    assert_trees_equal(other.reset(key, ordinary), other.reset(key, poisoned))


def test_mutable_numpy_level_cannot_change_the_cache_snapshot():
    env = make_env('test_run_rules_on_level_start_score')
    level = np.array(env.get_level(0), copy=True)
    params = PJParams(level=level)
    prepared = env.prepare_params(params)
    snapshot = np.array(prepared.reset_cache.level, copy=True)
    level[-1, 0, 0] = ~level[-1, 0, 0]
    np.testing.assert_array_equal(prepared.reset_cache.level, snapshot)
    key = jax.random.PRNGKey(42)
    assert_trees_equal(env.reset(key, params), env.reset(key, prepared))


@pytest.mark.parametrize('game,level_i,jit', [('test_random', 0, True),
                                             ('test_padding_mask', -1, True),
                                             ('sokoban_basic', 0, False)])
def test_unsupported_cache_modes_keep_original_parameters(game, level_i, jit):
    env = make_env(game, level_i, jit)
    params = PJParams(level=env.get_level(0), level_i=level_i)
    assert env.prepare_params(params).reset_cache is None
