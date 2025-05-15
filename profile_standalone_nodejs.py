import json
import multiprocessing as mp
import os

import random
import shutil
from timeit import default_timer as timer
import traceback

import hydra
from javascript import require

from conf.config import ProfileStandalone
from globals import STANDALONE_NODEJS_RESULTS_PATH
from preprocess_games import SIMPLIFIED_GAMES_DIR
from utils import get_list_of_games_for_testing


engine = require('./standalone/puzzlescript/engine.js')
solver = require('./standalone/puzzlescript/solver.js')


def compile_game(engine, game, level_i):
    game_path = os.path.join(SIMPLIFIED_GAMES_DIR, f'{game}.txt')
    with open(f'{game_path[:-4]}_simplified.txt', 'r') as f:
        game_text = f.read()
    engine.compile(game_text, level_i)
    return game_text

def get_algo_name(algo):
    return str(algo).split(' ')[1].strip(']')

actions = ["LEFT", "RIGHT", "UP", "DOWN", "ACTION"]

def rand_rollout_from_python(engine, solver, game_text, level_i, n_steps, timeout=1000):
    start_time = timer()
    for i in range(n_steps):
        if (i % 1_000) and (timer() - start_time > timeout):
            fps = i / (timer() - start_time)
            return False, [], i, fps
        action = random.randint(0, 5)
        solver.takeAction(engine, action)
    return False, [], i, timer() - start_time
        

# algos = [solver.solveBFS, solver.solveAStar, solver.solveMCTS]:
algos = [solver.randomRollout, rand_rollout_from_python]


# @hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone')
@hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone_config')
def main(cfg: ProfileStandalone):
    if cfg.game is None:
        games_to_test = get_list_of_games_for_testing(all_games=cfg.all_games, include_random=cfg.include_randomness)
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
                game_text = compile_game(engine, game, level_i)
            except Exception as e:
                print(f'Error compiling game {game} level {level_i}: {e}')
                results[algo_name][game][level_i] = {"Error": traceback.print_exc()}
                continue

            if algo == rand_rollout_from_python:
                result = rand_rollout_from_python(engine, solver, game_text, level_i, timeout=1000, n_steps=cfg.n_profile_steps)
            else:
                result = algo(engine, timeout=10000)
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