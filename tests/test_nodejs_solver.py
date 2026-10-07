import time

import pytest

from backends.nodejs import NodeJSPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser, level_to_int_arr
from puzzlescript_nodejs.utils import replay_actions_js


def make_game(rules: str, winconditions: str, level: str) -> str:
    return f"""title regression

========
OBJECTS
========

Background
black

Wall
grey

Player
blue

Coin
yellow

Crate
orange

Target
green

=======
LEGEND
=======

. = Background
# = Wall
P = Player
C = Coin
* = Crate
T = Target

=======
SOUNDS
=======

================
COLLISIONLAYERS
================

Background
Target
Player, Wall, Coin, Crate

======
RULES
======

{rules}

==============
WINCONDITIONS
==============

{winconditions}

=======
LEVELS
=======

{level}
"""


# A "no" wincondition; the coin is walled off, so every MCTS simulation ends in the heuristic.
NO_COIN_UNREACHABLE = make_game(
    "[ > Player | Coin ] -> [ > Player | ]",
    "no Coin",
    "#########\n#P....#C#\n#########",
)
# No winconditions at all: the level is won by a rule.
WIN_BY_RULE_CORRIDOR = make_game(
    "late [ Player Target ] -> win",
    "",
    "############\n#P........T#\n############",
)
# Crates in an open room and an unreachable target: a large state space and no solution.
CRATE_ROOM = make_game(
    "[ > Player | Crate ] -> [ > Player | > Crate ]",
    "all Target on Crate",
    "#############\n#P..........#\n#..*....*...#\n#...........#\n#.....*.....#\n#...........#\n"
    "#..*.....*..#\n#...........#\n#############\n#T###########\n#############",
)

SEARCH_ALGOS = ["bfs", "astar", "gbfs", "mcts", "random"]


def test_bfs_find_girlfriend_level_14_replayable_solution():
    backend = NodeJSPuzzleScriptBackend()
    parser = init_ps_lark_parser()

    try:
        game_text = backend.compile_game(parser, "Find_Girlfriend!")
        result = backend.run_search(
            "bfs",
            game_text=game_text,
            level_i=14,
            n_steps=10_000,
            timeout_ms=10_000,
        )

        assert result.solved
        assert len(result.actions) >= 2

        backend.load_level(game_text, 14)
        engine = backend.engine
        first_action = result.actions[0]
        changed = engine.processInput(first_action)
        while engine.getAgaining():
            changed = engine.processInput(-1) or changed
        assert changed
        assert not engine.getWinning()

        for action in result.actions[1:]:
            changed = engine.processInput(action)
            while engine.getAgaining():
                changed = engine.processInput(-1) or changed

        assert engine.getWinning()
    finally:
        backend.unload_game()


def test_successful_solver_state_matches_replayed_actions():
    backend = NodeJSPuzzleScriptBackend()
    parser = init_ps_lark_parser()

    def assert_state_matches(algo_name, raw_result, game_text, level_i):
        assert raw_result[0], f"{algo_name} did not solve the level"
        actions = list(raw_result[1])
        returned_state = raw_result[5]
        obj_list = list(raw_result[7])

        _, js_states = replay_actions_js(
            backend.engine,
            backend.solver,
            actions,
            game_text,
            level_i,
        )

        returned_arr = level_to_int_arr(returned_state, len(obj_list))
        replayed_arr = level_to_int_arr(js_states[-1], len(obj_list))
        assert (returned_arr == replayed_arr).all(), algo_name

    try:
        game_text = backend.compile_game(parser, "Slidings")
        level_i = 0

        backend.load_level(game_text, level_i)
        assert_state_matches(
            "astar",
            backend.solver.solveAStar(backend.engine, 10_000),
            game_text,
            level_i,
        )

        backend.load_level(game_text, level_i)
        assert_state_matches(
            "gbfs",
            backend.solver.solveGBFS(backend.engine, 10_000),
            game_text,
            level_i,
        )

        backend.load_level(game_text, level_i)
        assert_state_matches(
            "mcts",
            backend.solver.solveMCTS(
                backend.engine,
                {"max_iterations": 5_000, "max_sim_length": 50},
            ),
            game_text,
            level_i,
        )
    finally:
        backend.unload_game()


def test_mcts_scores_no_wincondition_levels():
    # getScoreNormalized read `_o10`, an engine global that solver.js cannot see.
    backend = NodeJSPuzzleScriptBackend()
    try:
        result = backend.run_search(
            "mcts", game_text=NO_COIN_UNREACHABLE, level_i=0, n_steps=50, timeout_ms=-1,
        )
        assert not result.solved
        assert result.iterations == 50
    finally:
        backend.unload_game()


def test_mcts_solves_level_without_winconditions():
    # With no winconditions the normalized score was 0 / 0 = NaN, which froze UCB
    # selection on the first child: this corridor went unsolved in 0 of 10 runs.
    backend = NodeJSPuzzleScriptBackend()
    try:
        result = backend.run_search(
            "mcts", game_text=WIN_BY_RULE_CORRIDOR, level_i=0, n_steps=5_000, timeout_ms=-1,
        )
        assert result.solved
    finally:
        backend.unload_game()


@pytest.mark.parametrize("algo", SEARCH_ALGOS)
def test_run_search_stops_at_n_steps(algo):
    # solveMCTS takes an options object, so a positional n_steps was ignored.
    backend = NodeJSPuzzleScriptBackend()
    try:
        result = backend.run_search(algo, game_text=CRATE_ROOM, level_i=0, n_steps=50, timeout_ms=-1)
        assert not result.solved
        assert not result.timeout
        assert result.iterations == 50
    finally:
        backend.unload_game()


@pytest.mark.parametrize("algo", SEARCH_ALGOS)
def test_run_search_stops_at_timeout_ms(algo):
    # A*, GBFS, MCTS and random rollouts ignored timeout_ms.
    backend = NodeJSPuzzleScriptBackend()
    try:
        start = time.perf_counter()
        result = backend.run_search(
            algo, game_text=CRATE_ROOM, level_i=0, n_steps=1_000_000, timeout_ms=200,
        )
        elapsed = time.perf_counter() - start
        assert not result.solved
        assert result.timeout
        assert 0 < result.iterations < 1_000_000
        assert elapsed < 20
    finally:
        backend.unload_game()
