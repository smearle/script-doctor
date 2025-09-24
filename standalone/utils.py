
import os
from typing import List

from javascript.proxy import Proxy

from preprocess_games import SIMPLIFIED_GAMES_DIR, get_tree_from_txt


def compile_game(parser, engine, game, level_i):
    game_path = os.path.join(SIMPLIFIED_GAMES_DIR, f'{game}.txt')
    if not os.path.isfile(game_path):
        get_tree_from_txt(parser=parser, game=game, test_env_init=False, overwrite=True)
    with open(f'{game_path[:-4]}_simplified.txt', 'r') as f:
        game_text = f.read()
    engine.compile(['restart'], game_text)
    return game_text


def replay_actions_js(engine: Proxy, solver: Proxy, actions: List[int], game_text: str, level_i: int):
    engine.compile(['loadLevel', level_i], game_text)
    solver.precalcDistances(engine)
    scores = [solver.getScore(engine)]
    states = [solver.getState(engine)]
    for action in actions:
        _, _, _, _, score, state, _, _ = solver.takeAction(engine, action)
        scores.append(score)
        states.append(state)
    return scores, states
