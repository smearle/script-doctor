import os

import random
from timeit import default_timer as timer

from javascript import require

from parse_lark import GAMES_DIR
from utils import get_list_of_games_for_testing

ps = require('./standalone/puzzlescript/engine.js')
solver = require('./standalone/puzzlescript/solver.js')

n_steps = 20_000

def compile_game(game, level_i):
    game_path = os.path.join(GAMES_DIR, f'{game}.txt')
    with open(game_path) as f:
        game_text = f.read()
    ps.compile(game_text, level_i)

def main():
    games_to_test = get_list_of_games_for_testing(all_games=True)
    for game in games_to_test:
        compile_game(game, 0)

        for algo in [solver.solveBFS, solver.solveAStar, solver.solveMCTS]:
            result = algo(ps)
            print(result)



if __name__ == "__main__":
    main()