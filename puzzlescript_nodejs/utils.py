
import os
from typing import List

from javascript.proxy import Proxy

from puzzlescript_jax.preprocessing import SIMPLIFIED_GAMES_DIR, get_tree_from_txt


def compile_game(parser, engine, game, level_i):
    game_path = os.path.join(SIMPLIFIED_GAMES_DIR, f'{game}.txt')
    if not os.path.isfile(game_path):
        get_tree_from_txt(parser=parser, game=game, test_env_init=False, overwrite=True)
    with open(f'{game_path[:-4]}_simplified.txt', 'r') as f:
        game_text = f.read()
    engine.compile(['restart'], game_text)
    return game_text


def replay_actions_js(
    engine: Proxy,
    solver: Proxy,
    actions: List[int],
    game_text: str,
    level_i: int,
    *,
    stop_on_win: bool = True,
    max_again: int = 50,
    return_winning: bool = False,
):
    """Faithfully replay gameplay actions against the JS engine.

    This intentionally avoids ``solver.takeAction()``, which is search-oriented
    and auto-restarts the engine after a win. For validation and regression
    tests we want direct gameplay semantics from ``engine.processInput()``.
    """
    engine.compile(['loadLevel', level_i], game_text)
    solver.precalcDistances(engine)
    scores = [solver.getScore(engine)]
    states = [engine.backupLevel()]
    winning = [bool(engine.getWinning())]
    for action in actions:
        if action == 5:
            engine.DoUndo(False, True)
        elif action == 6:
            engine.DoRestart()
        else:
            engine.processInput(action)
            again_steps = 0
            while bool(engine.getAgaining()) and again_steps < max_again:
                engine.processInput(-1)
                again_steps += 1

        scores.append(solver.getScore(engine))
        states.append(engine.backupLevel())
        winning.append(bool(engine.getWinning()))

        if stop_on_win and bool(engine.getWinning()):
            break
    if return_winning:
        return scores, states, winning
    return scores, states
