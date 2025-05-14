import json
import os

import random
import shutil
from timeit import default_timer as timer
import traceback

import hydra
from javascript import require

from conf.config import ProfileJaxRandConfig, ProfileStandalone
from parse_lark import GAMES_DIR, MIN_GAMES_DIR
from utils import get_list_of_games_for_testing

engine = require('./standalone/puzzlescript/engine.js')
solver = require('./standalone/puzzlescript/solver.js')

STANDALONE_NODEJS_RESULTS_PATH = os.path.join('data', 'standalone_nodejs_results.json')

def compile_game(game, level_i):
    game_path = os.path.join(MIN_GAMES_DIR, f'{game}.txt')
    with open(game_path, 'r') as f:
        game_text = f.read()
    engine.compile(game_text, level_i)

def get_algo_name(algo):
    return str(algo).split(' ')[1].strip(']')

actions = ["LEFT", "RIGHT", "UP", "DOWN", "ACTION"]

def rand_rollout_from_python(engine, solver, n_steps, timeout=1000):
    start_time = timer()
    for i in range(n_steps):
        # if (i % 1_000) and (timer() - start_time > timeout):
        #     fps = i / (timer() - start_time)
        #     return False, [], i, fps
        action = random.randint(0, 5)
        winning = solver.takeAction(engine, action)
        if winning:
            return True, [], i, start_time - timer()
    return False, [], i, timer() - start_time
        

# algos = [solver.solveBFS, solver.solveAStar, solver.solveMCTS]:
algos = [solver.randomRollout, rand_rollout_from_python, solver.solveBFS]


# @hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone')
@hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone_config')
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
        for algo in algos:
            algo_name = get_algo_name(algo)
            print(f'Algorithm: {algo_name}')
            if algo_name not in results:
                results[algo_name] = {}
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

            if algo == rand_rollout_from_python:
                result = rand_rollout_from_python(engine, solver, timeout=1000, n_steps=cfg.n_profile_steps)
            else:
                result = algo(engine, timeout=1000)
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