"""Parallel cell application must preserve metadata, forces, commands and RNG."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import puzzlescript_jax.env as env_module
from puzzlescript_jax.env import PuzzleJaxEnv
from puzzlescript_jax.ps_game import Rule
from puzzlescript_jax.preprocessing import get_tree_from_txt
from puzzlescript_jax.utils import init_ps_lark_parser
from scripts.benchmarks.benchmark_manhattan_distance import assert_equal


@pytest.fixture(scope="module")
def env():
    tree, _, error = get_tree_from_txt(init_ps_lark_parser(), "test_moving", test_env_init=False)
    assert tree is not None, error
    return PuzzleJaxEnv(tree, level_i=-1, print_score=False)


@pytest.mark.parametrize("left,right,prefixes,expected", [
    ([[["Player"]]], [[["Block"]]], [], True),
    ([[[]]], [[["Player"]]], [], True),
    ([[["Player"]]], [[[]]], [], True),
    ([[["Player"]]], [], [], False),
    ([[["Player"], ["Block"]]], [[["Player"], ["Block"]]], [], False),
    ([[["Player"]], [["Block"]]], [[["Player"]], [["Block"]]], [], False),
    ([[["Player"]]], [[["randomdir", "Player"]]], [], False),
    ([[["Player"]]], [[["random", "Moveable"]]], [], False),
    ([[["Player"]]], [[["Block"]]], ["random"], False),
])
def test_independence_guard(left, right, prefixes, expected):
    assert env_module._is_independent_cell_rule(Rule(left, right, prefixes)) == expected


@pytest.mark.parametrize("left,right,command", [
    (["Player"], ["Block"], None),
    (["no", "Player"], ["Player"], None),
    ([], ["Player"], None),
    (["Moveable"], ["Moveable"], None),
    (["moving", "Moveable"], ["moving", "Moveable"], None),
    ([">", "Moveable"], ["<", "Moveable"], None),
    (["stationary", "Player"], ["right", "Block"], None),
    (["Player"], [], None),
    (["Player"], ["Block"], "again"),
    (["Player"], ["Player"], "win"),
])
def test_batched_cell_rules_match_sequential(env, monkeypatch, left, right, command):
    rule = Rule([[left]], [[right]], command=command)
    eligible = env_module._is_independent_cell_rule
    # Include dense, sparse, empty and padded cells, mixed property bindings,
    # and all movement bit patterns. Both paths receive the same runtime keys.
    rng = np.random.default_rng(7)
    channels = env.n_objs + len(env.collision_layers) * env_module.N_FORCES + 2
    levels = rng.random((8, 1, channels, 3, 5)) < 0.35
    levels[0] = False
    levels[1] = True
    # One collision-layer occupant per cell, while leaving arbitrary forces.
    for layer in env.collision_layers:
        idxs = [env.objs_to_idxs[name] for name in layer]
        chosen = rng.integers(-1, len(idxs), (8, 3, 5))
        for i, idx in enumerate(idxs):
            levels[:, 0, idx] = chosen == i
    outputs = []
    for enabled in (False, True):
        monkeypatch.setattr(env_module, "_is_independent_cell_rule",
                            eligible if enabled else lambda rule: False)
        fns = env.gen_subrules_meta(rule, str(rule), (3, 5))
        outputs.append([
            jax.jit(jax.vmap(fn))(jax.random.split(jax.random.PRNGKey(11), 8), jnp.asarray(levels))
            for fn in fns
        ])
    assert_equal(*outputs)
    jax.clear_caches()
