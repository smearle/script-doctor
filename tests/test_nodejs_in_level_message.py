"""An in-level ``message`` must not stop the headless JS engine from detecting wins.

Upstream ``processOutputCommands`` shows an in-level message with
``showTempMessage()``, which sets ``textMode = true``. ``processInput`` only
checks the win condition while ``textMode`` is false, and only the browser's
dismissal handler clears it, so a headless caller that never dismisses the
message would see no win for the rest of the level. The wrapper dismisses each
message as it is raised and queues its text for ``takeMessages()``.
"""

import pytest

from backends.nodejs import NodeJSPuzzleScriptBackend

RIGHT = 3

GAME_TEMPLATE = """title In-Level Message Win

========
OBJECTS
========

Background
black

Player
white

Goal
yellow

Mark
red

=======
LEGEND
=======

. = Background
P = Player
G = Goal
M = Mark

=======
SOUNDS
=======

================
COLLISIONLAYERS
================

Background
Goal, Mark
Player

======
RULES
======

{rule}

==============
WINCONDITIONS
==============

all Player on Goal

=======
LEVELS
=======

P.M.G
"""

# Walking right, the player crosses the Mark on its way to the Goal.
# Each case: (rule, index of the turn whose message fires, message text).
CASES = {
    # The message fires on turn 3, one turn before the winning move.
    "message_before_win": ("[ Player Mark ] -> message halfway", 2, "halfway"),
    # The message fires on the winning turn itself.
    "message_on_win": ("late [ Player Goal ] -> message arrived", 3, "arrived"),
}
RULES = {name: rule for name, (rule, _, _) in CASES.items()}


@pytest.fixture
def backend():
    backend = NodeJSPuzzleScriptBackend()
    yield backend
    backend.unload_game()


def _step(engine, action):
    engine.processInput(action)
    while engine.getAgaining():
        engine.processInput(-1)
    return bool(engine.getWinning())


@pytest.mark.parametrize(("rule", "message_turn", "text"), CASES.values(), ids=CASES.keys())
def test_win_detected_after_in_level_message(backend, rule, message_turn, text):
    backend.load_level(GAME_TEMPLATE.format(rule=rule), 0)
    wins, messages = [], []
    for _ in range(4):
        wins.append(_step(backend.engine, RIGHT))
        messages.append(list(backend.engine.takeMessages()))
    assert wins == [False, False, False, True]
    assert messages == [[text] if t == message_turn else [] for t in range(4)]


@pytest.mark.parametrize("rule", RULES.values(), ids=RULES.keys())
@pytest.mark.parametrize("algo", ["bfs", "astar", "gbfs"])
def test_solvers_solve_level_with_in_level_message(backend, rule, algo):
    backend.load_level(GAME_TEMPLATE.format(rule=rule), 0)
    engine = backend.engine
    if algo == "bfs":
        result = backend.solver.solveBFS(engine, 1_000, -1)
    elif algo == "astar":
        result = backend.solver.solveAStar(engine, 1_000)
    else:
        result = backend.solver.solveGBFS(engine, 1_000)
    assert result[0], f"{algo} found no solution"
    assert list(result[1]) == [RIGHT] * 4
