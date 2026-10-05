"""Switch dispatch must preserve reset, batching, and complete step outputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puzzlescript_jax.env import PJParams, PuzzleJaxEnv, RuleState
from puzzlescript_jax.env_switch import PuzzleJaxEnvSwitch
from puzzlescript_jax.preprocessing import get_tree_from_txt
from puzzlescript_jax.utils import init_ps_lark_parser


@pytest.fixture(scope="module")
def parser():
    return init_ps_lark_parser()


@pytest.mark.parametrize("intercept,expected", [(True, 10), (False, 2)])
def test_group_reconverges_after_every_rule_has_run(intercept, expected):
    # First rule increments once; a following rule can intercept that state.
    # Repeating the first rule alone would skip 1. Forgetting its effect when
    # the second rule does nothing would stop early at 1 instead of 2.
    def rule(update):
        def apply(rng, level):
            new_level = update(level)
            return RuleState(lvl=new_level, applied=jnp.any(new_level != level),
                             cancelled=False, restart=False, again=False,
                             win=False, rng=rng)
        return apply

    rules = [rule(lambda x: jnp.where(x < 2, x + 1, x)),
             rule(lambda x: jnp.where((x == 1) & intercept, 10, x))]
    for cls in (PuzzleJaxEnv, PuzzleJaxEnvSwitch):
        env = object.__new__(cls)
        env.jit = True
        dispatcher = env._gen_rule_blocks_fn([(False, [(rules, False)])])
        carry = (jnp.array([0]), False, False, False, False, False,
                 jax.random.PRNGKey(0), 0)
        result = jax.jit(dispatcher)(carry)
        np.testing.assert_array_equal(result[0], [expected])


def test_group_remembers_changes_that_cancel_each_other(monkeypatch):
    import puzzlescript_jax.env as standard_module
    import puzzlescript_jax.env_switch as switch_module

    # A -> B -> A still counts as a changed pass. Limit the intentional cycle
    # to two passes per group/block so the regression stays small and bounded.
    monkeypatch.setattr(standard_module, "MAX_LOOPS", 2)
    monkeypatch.setattr(switch_module, "MAX_LOOPS", 2)

    def set_to(value):
        def rule(rng, level):
            new_level = jnp.full_like(level, value)
            return RuleState(lvl=new_level, applied=jnp.any(new_level != level),
                             cancelled=False, restart=False, again=False,
                             win=False, rng=rng)
        return rule

    for cls in (PuzzleJaxEnv, PuzzleJaxEnvSwitch):
        env = object.__new__(cls)
        env.jit = True
        dispatcher = env._gen_rule_blocks_fn([(True, [([set_to(1), set_to(0)], False)])])
        carry = (jnp.array([0]), False, False, False, False, False,
                 jax.random.PRNGKey(0), 0)
        result = jax.jit(dispatcher)(carry)
        np.testing.assert_array_equal(result[0], [0])
        assert bool(result[1]), "Transient changes must cause the block to repeat"


@pytest.mark.parametrize("game,level_i", [
    ("sokoban_basic", 0),
    ("test_padding_mask", -1),
    ("test_run_rules_on_level_start_score", 0),
    ("test_cancel_again", 0),
    ("test_random", 0),
])
def test_jitted_batched_reset_and_autoreset_match_standard(parser, game, level_i):
    tree, _, error = get_tree_from_txt(parser, game, test_env_init=False)
    assert tree is not None, error
    outputs = []
    for cls in (PuzzleJaxEnv, PuzzleJaxEnvSwitch):
        env = cls(tree, level_i=level_i, print_score=False, max_steps=3)
        params = PJParams(level=env.get_level(0), level_i=level_i)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        # Reset is often first called inside jit/vmap; retained dispatch tables
        # must not capture tracers from that reset's compilation.
        with jax.checking_leaks():
            obs, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, params)

            def rollout(state):
                def step(state, action):
                    result = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                        keys, state, jnp.full((2,), action), params)
                    return result[1], result
                return jax.lax.scan(step, state, jnp.array([2, 4, 1, 0], jnp.int32))

            result = jax.jit(rollout)(state)
        outputs.append((obs, state, result))
    assert jax.tree.structure(outputs[0]) == jax.tree.structure(outputs[1])
    for standard, switch in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1])):
        np.testing.assert_array_equal(standard, switch)
    jax.clear_caches()
