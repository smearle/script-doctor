
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


def check_compile(engine: Proxy, game_text: str) -> tuple[bool, bool, list[str]]:
    """Compile raw ``game_text`` and read the verdict from the engine's messages.

    PuzzleScript's ``compile`` never throws on compile errors: it logs them and,
    whenever it can salvage a state, loads that state anyway, so a broken game
    can still serialize cleanly. Compiled = the engine logged "Successful
    Compilation" and no "Errors detected"; playable = compiled with at least
    one non-message level. Returns (compiled, playable, messages), where
    messages are the captured engine messages other than the success line:
    errors and line-numbered warnings (the capture strips the markup that
    tells them apart).
    """
    engine.unloadGame()
    engine.clearCapturedErrors()
    try:
        engine.compile(['restart'], game_text)
    except Exception as e:  # the compiler itself crashed on this input
        return False, False, [f"{type(e).__name__}: {e}"]
    msgs = [str(m) for m in engine.getCapturedErrors()]
    compiled = (any('Successful Compilation' in m for m in msgs)
                and not any('Errors detected' in m for m in msgs))
    playable = compiled and any(lv.type == 'level' for lv in engine.getLevelInfo())
    return compiled, playable, [m for m in msgs if 'Successful Compilation' not in m]


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
    random_seed: str = None,
):
    """Faithfully replay gameplay actions against the JS engine.

    This intentionally avoids ``solver.takeAction()``, which is search-oriented
    and auto-restarts the engine after a win. For validation and regression
    tests we want direct gameplay semantics from ``engine.processInput()``.
    """
    if random_seed is not None:
        engine.compile(['loadLevel', level_i], game_text, random_seed, timeout=120)
    else:
        engine.compile(['loadLevel', level_i], game_text, timeout=120)
    solver.precalcDistances(engine, timeout=120)
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
