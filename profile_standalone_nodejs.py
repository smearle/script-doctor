import json
import os

import random
from timeit import default_timer as timer

from javascript import require

from parse_lark import GAMES_DIR
from utils import get_list_of_games_for_testing

ps = require('./standalone/puzzlescript/engine.js')
solver = require('./standalone/puzzlescript/solver.js')

STANDALONE_NODEJS_RESULTS_PATH = os.path.join('data', 'standalone_nodejs_results.json')

# algos = [solver.solveBFS, solver.solveAStar, solver.solveMCTS]:
algos = [solver.solveBFS]
n_steps = 20_000

def compile_game(game, level_i):
    game_path = os.path.join(GAMES_DIR, f'{game}.txt')
    with open(game_path) as f:
        game_text = f.read()
    ps.compile(game_text, level_i)

def get_algo_name(algo):
    return str(algo).split(' ')[1].strip(']')

def main():
    games_to_test = get_list_of_games_for_testing(all_games=True)
    results = {get_algo_name(algo): {game: {} for game in games_to_test} for algo in algos }
    print(results)
    for game in games_to_test:

        print(f'Game: {game}')
        # TODO: How to get the available number of levels from nodejs?
        level_i = 0
        print(f'Level: {level_i}')
        for algo in [solver.solveBFS]:
            algo_name = get_algo_name(algo)
            print(f'Algorithm: {algo_name}')
            compile_game(game, 0)
            result = algo(ps, timeout=1000)
            result = {
                # 'game': game,
                # 'level': level_i,
                # 'algo': algo.__name__,
                'solved': result[0],
                'actions': result[1],
                'iterations': result[2],
                'time': result[3],
                'FPS': result[2] / result[3],
            }
            print(result)
            if level_i not in results[algo_name][game]:
                results[algo_name][level_i] = result

            with open(STANDALONE_NODEJS_RESULTS_PATH, 'w') as f:
                json.dump(results, f, indent=4)



if __name__ == "__main__":
    main()