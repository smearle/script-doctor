from pathlib import Path

import jax
import jax.numpy as jnp
from lark import Lark

from puzzlescript_jax.env import PJState, PuzzleJaxEnv
from puzzlescript_jax.gen_tree import GenPSTree
from puzzlescript_jax.globals import LARK_SYNTAX_PATH
from puzzlescript_jax.preprocessing import StripPuzzleScript, preprocess_ps


TEST_GAME = """title Force Sentinel
run_rules_on_level_start

========
OBJECTS
========

Background .
black

Player P
white
00000
0...0
0...0
0...0
00000

=======
LEGEND
=======

. = Background
P = Player and Background

=======
SOUNDS
=======

================
COLLISIONLAYERS
================

Background
Player

======
RULES
======

==============
WINCONDITIONS
==============

some Player

=======
LEVELS
=======

...
.P.
...
"""


def _build_env() -> PuzzleJaxEnv:
    grammar = Path(LARK_SYNTAX_PATH).read_text(encoding="utf-8")
    parser = Lark(grammar, start="ps_game", maybe_placeholders=False)
    parse_tree = parser.parse(preprocess_ps(TEST_GAME))
    min_tree = StripPuzzleScript().transform(parse_tree)
    tree = GenPSTree().transform(min_tree)
    return PuzzleJaxEnv(tree, jit=False)


def test_apply_player_force_ignores_reset_sentinel_action():
    env = _build_env()
    level = env.get_level(0)
    state = PJState(
        multihot_level=level,
        win=jnp.array(False),
        score=jnp.array(0, dtype=jnp.int32),
        heuristic=jnp.array(0, dtype=jnp.int32),
        restart=jnp.array(False),
        step_i=jnp.array(0, dtype=jnp.int32),
        init_heuristic=jnp.array(0, dtype=jnp.int32),
        prev_heuristic=jnp.array(0, dtype=jnp.int32),
        rng=jax.random.PRNGKey(0),
        view_bounds=env._get_default_view_bounds(level.shape[1:]),
    )

    forced = env.apply_player_force(-1, state)

    assert jnp.array_equal(forced[: env.n_objs], level)
    assert not bool(forced[env.n_objs :].any())
