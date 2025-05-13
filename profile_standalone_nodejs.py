import json
import os

import random
import shutil
from timeit import default_timer as timer
import traceback

import hydra
from javascript import require

from conf.config import ProfileEnvConfig, ProfileStandalone
from parse_lark import GAMES_DIR
from utils import get_list_of_games_for_testing

ps = require('./standalone/puzzlescript/engine.js')
solver = require('./standalone/puzzlescript/solver.js')

STANDALONE_NODEJS_RESULTS_PATH = os.path.join('data', 'standalone_nodejs_results.json')

# algos = [solver.solveBFS, solver.solveAStar, solver.solveMCTS]:
algos = [solver.solveBFS]

def compile_game(game, level_i):
    game_path = os.path.join(GAMES_DIR, f'{game}.txt')
    with open(game_path, 'r') as f:
        game_text = f.read()
    ps.compile(game_text, level_i)

def get_algo_name(algo):
    return str(algo).split(' ')[1].strip(']')


@hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone')
def main(cfg: ProfileStandalone):
    if cfg.game is None:
        games_to_test = get_list_of_games_for_testing(all_games=True)
    else:
        games_to_test = [cfg.game]
    results = {get_algo_name(algo): {} for algo in algos}
    if os.path.isfile(STANDALONE_NODEJS_RESULTS_PATH):
        if not cfg.overwrite:
            with open(STANDALONE_NODEJS_RESULTS_PATH, 'r') as f:
                results = json.load(f)
        else:
            shutil.copyfile(STANDALONE_NODEJS_RESULTS_PATH, STANDALONE_NODEJS_RESULTS_PATH[:-5] + '_bkp.json')

    for game in games_to_test:

        print(f'\nGame: {game}')
        # TODO: How to get the available number of levels from nodejs?
        level_i = 0
        print(f'Level: {level_i}')
        for algo in [solver.solveBFS]:
            algo_name = get_algo_name(algo)
            print(f'Algorithm: {algo_name}')
            if game not in results[algo_name]:
                results[algo_name][game] = {}
            if str(level_i) in results[get_algo_name(algo)][game]:
                print(f'Already solved {game} level {level_i} with {get_algo_name(algo)}, skipping.')
                continue
            try:
                compile_game(game, level_i)
            except Exception as e:
                print(f'Error compiling game {game} level {level_i}: {e}')
                print(traceback.print_exc())
                results[algo_name][game][level_i] = {"Error": traceback.print_exc()}
                continue

            result = algo(ps, timeout=1000)
            result = {
                'solved': result[0],
                'actions': tuple(result[1]),
                'iterations': result[2],
                'time': result[3],
                'FPS': result[2] / (result[3] if result[3] > 0 else 1e4),
            }
            print(json.dumps(result))
            results[algo_name][game][level_i] = result

            with open(STANDALONE_NODEJS_RESULTS_PATH, 'w') as f:
                json.dump(results, f, indent=4)



if __name__ == "__main__":
    main()