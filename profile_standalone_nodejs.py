<<<<<<< Updated upstream
import os
import json
import multiprocessing as mp
import random
import re
import shutil
from timeit import default_timer as timer
import traceback

import cpuinfo
import hydra
from javascript import require
from javascript.proxy import Proxy
import numpy as np

from conf.config import ProfileStandalone
from globals import STANDALONE_NODEJS_RESULTS_PATH
from preprocess_games import SIMPLIFIED_GAMES_DIR
from utils import get_list_of_games_for_testing
from validate_sols import JS_SOLS_DIR


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

def rand_rollout_from_python(engine, solver, game_text, level_i, n_steps, timeout):
    start_time = timer()
    for i in range(n_steps):
        if (i % 1_000) and (timer() - start_time > timeout):
            fps = i / (timer() - start_time)
            return False, [], i, fps, score, state
        action = random.randint(0, 5)
        _, _, _, _, score, state = solver.takeAction(engine, action)
    return False, [], i, timer() - start_time, score, state

    
def get_standalone_run_name(cfg: ProfileStandalone, algo_name, cpu_name):
    return f'algo-{algo_name}_{cfg.n_steps}-steps_{cpu_name}'

def get_standalone_run_params_from_name(run_name: str):
    # use regex to extract the parameters from the run name
    groups = re.match(r'algo-(.*)_(\d+)-steps_(.*)', run_name)
    algo_name, n_steps, device_name = groups.groups()
    return algo_name, n_steps, device_name



# @hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone')
@hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone_config')
def main(cfg: ProfileStandalone):
    timeout_ms = cfg.timeout * 1_000 if cfg.timeout > 0 else -1
    print(f'Timeout: {timeout_ms} ms')

    if cfg.algo == 'bfs':
        algos = [solver.solveBFS]
        # algos = [solver.solveBFS, solver.solveAStar, solver.solveMCTS]:
    elif cfg.algo == 'random':
        algos = [solver.randomRollout, rand_rollout_from_python]
        

    cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    if cfg.game is None:
        games_to_test = get_list_of_games_for_testing(all_games=cfg.all_games, include_random=cfg.include_randomness,
                                                      random_order=cfg.random_order)
    else:
        games_to_test = [cfg.game]
    results = {get_algo_name(algo): {} for algo in algos}
    if os.path.isfile(STANDALONE_NODEJS_RESULTS_PATH):
        shutil.copyfile(STANDALONE_NODEJS_RESULTS_PATH, STANDALONE_NODEJS_RESULTS_PATH[:-5] + '_bkp.json')
        with open(STANDALONE_NODEJS_RESULTS_PATH, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for game in games_to_test:

        print(f'\nGame: {game}')
        # TODO: How to get the available number of levels from nodejs?
        for algo in algos:
            algo_name = get_algo_name(algo)
            run_name = get_standalone_run_name(cfg, algo_name, cpu_name)
            print(f'Algorithm: {algo_name}')
            if run_name not in results:
                results[run_name] = {}
            if game not in results[run_name]:
                results[run_name][game] = {}
            try:
                game_text = compile_game(engine, game, 0)
            except Exception as e:
                print(f'Error compiling game {game} level {0}: {e}')
                results[run_name][game][level_i] = {"Error": traceback.print_exc()}
                continue

            n_levels = engine.getNumLevels()
            game_js_sols_dir = os.path.join(JS_SOLS_DIR, game)
            os.makedirs(game_js_sols_dir, exist_ok=True)

            for level_i in range(n_levels):
                level_js_sol_path = os.path.join(game_js_sols_dir, f'level-{level_i}.json')
                print(f'Level: {level_i}')
                if cfg.gen_solutions_for_validation and not cfg.overwrite and os.path.isfile(level_js_sol_path):
                    print(f'Already solved (for validation) {game} level {level_i} with {run_name}, skipping.')
                    continue
                if not cfg.gen_solutions_for_validation and not cfg.overwrite and str(level_i) in results[run_name][game]:
                    print(f'Already solved (for profiling) {game} level {level_i} with {run_name}, skipping.')
                    continue
                engine.compile(game_text, level_i)
                if algo == rand_rollout_from_python:
                    result = rand_rollout_from_python(engine, solver, game_text, level_i, timeout=timeout_ms, n_steps=cfg.n_steps)
                else:
                    # Make the javascript timeout longer so that we can timeout from inside JS and return stats properly
                    result = algo(engine,
                                  cfg.n_steps, timeout_ms,
                                  timeout=timeout_ms*1.5 if timeout_ms > 0 else None,)
                result = {
                    'solved': result[0],
                    'actions': tuple(result[1]),
                    'iterations': result[2],
                    'time': result[3],
                    'FPS': result[2] / (result[3] if result[3] > 0 else 1e4),
                    'score': result[4],
                    'state': result[5],
                    'timeout': result[6],
                    'objs': result[7],
                }

                if os.path.isfile(level_js_sol_path):
                    with open(level_js_sol_path, 'r') as f:
                        level_js_sol_dict = json.load(f)
                    best_solve = level_js_sol_dict['won']
                    best_score = level_js_sol_dict['score']
                else:
                    best_solve = False
                    best_score = -np.inf
                
                if result['solved'] > best_solve or result['score'] > best_score:
                    result_dict = {
                        'won': result['solved'],
                        'score': result['score'],
                        'timeout': None,  # TODO
                        'objs': result['objs'],
                        'state': result['state'],
                    }
                    with open(level_js_sol_path, 'w') as f:
                        json.dump(result_dict, f, indent=4)
                    print(f"Saved solution to {level_js_sol_path}")


                print(json.dumps(result))
                results[run_name][game][level_i] = result

                with open(STANDALONE_NODEJS_RESULTS_PATH, 'w') as f:
                    json.dump(results, f, indent=4)



if __name__ == "__main__":
=======
import os
import json
import multiprocessing as mp
import random
import re
import shutil
from timeit import default_timer as timer
import traceback

import cpuinfo
import hydra
from javascript import require
from javascript.proxy import Proxy
import numpy as np

from conf.config import ProfileStandalone
from globals import STANDALONE_NODEJS_RESULTS_PATH
from preprocess_games import SIMPLIFIED_GAMES_DIR
from utils import get_list_of_games_for_testing
from validate_sols import JS_SOLS_DIR


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

def rand_rollout_from_python(engine, solver, game_text, level_i, n_steps, timeout):
    start_time = timer()
    for i in range(n_steps):
        if (i % 1_000) and (timer() - start_time > timeout):
            fps = i / (timer() - start_time)
            return False, [], i, fps, score, state
        action = random.randint(0, 5)
        _, _, _, _, score, state = solver.takeAction(engine, action)
    return False, [], i, timer() - start_time, score, state

    
def get_standalone_run_name(cfg: ProfileStandalone, algo_name, cpu_name):
    return f'algo-{algo_name}_{cfg.n_steps}-steps_{cpu_name}'

def get_standalone_run_params_from_name(run_name: str):
    # use regex to extract the parameters from the run name
    groups = re.match(r'algo-(.*)_(\d+)-steps_(.*)', run_name)
    algo_name, n_steps, device_name = groups.groups()
    return algo_name, n_steps, device_name



# @hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone')
@hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone_config')
def main(cfg: ProfileStandalone):
    timeout_ms = cfg.timeout * 1_000 if cfg.timeout > 0 else -1
    print(f'Timeout: {timeout_ms} ms')

    if cfg.algo == 'bfs':
        algos = [solver.solveBFS]
        # algos = [solver.solveBFS, solver.solveAStar, solver.solveMCTS]:
    elif cfg.algo == 'random':
        algos = [solver.randomRollout, rand_rollout_from_python]
        

    cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    if cfg.game is None:
        games_to_test = get_list_of_games_for_testing(all_games=cfg.all_games, include_random=cfg.include_randomness,
                                                      random_order=cfg.random_order)
    else:
        games_to_test = [cfg.game]
    results = {get_algo_name(algo): {} for algo in algos}
    if os.path.isfile(STANDALONE_NODEJS_RESULTS_PATH):
        shutil.copyfile(STANDALONE_NODEJS_RESULTS_PATH, STANDALONE_NODEJS_RESULTS_PATH[:-5] + '_bkp.json')
        with open(STANDALONE_NODEJS_RESULTS_PATH, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for game in games_to_test:

        print(f'\nGame: {game}')
        # TODO: How to get the available number of levels from nodejs?
        for algo in algos:
            algo_name = get_algo_name(algo)
            run_name = get_standalone_run_name(cfg, algo_name, cpu_name)
            print(f'Algorithm: {algo_name}')
            if run_name not in results:
                results[run_name] = {}
            if game not in results[run_name]:
                results[run_name][game] = {}
            try:
                game_text = compile_game(engine, game, 0)
            except Exception as e:
                print(f'Error compiling game {game} level {0}: {e}')
                results[run_name][game][level_i] = {"Error": traceback.print_exc()}
                continue

            n_levels = engine.getNumLevels()
            game_js_sols_dir = os.path.join(JS_SOLS_DIR, game)
            os.makedirs(game_js_sols_dir, exist_ok=True)

            for level_i in range(n_levels):
                level_js_sol_path = os.path.join(game_js_sols_dir, f'level-{level_i}.json')
                print(f'Level: {level_i}')
                if cfg.gen_solutions_for_validation and not cfg.overwrite and os.path.isfile(level_js_sol_path):
                    print(f'Already solved (for validation) {game} level {level_i} with {run_name}, skipping.')
                    continue
                if not cfg.gen_solutions_for_validation and not cfg.overwrite and str(level_i) in results[run_name][game]:
                    print(f'Already solved (for profiling) {game} level {level_i} with {run_name}, skipping.')
                    continue
                engine.compile(game_text, level_i)
                if algo == rand_rollout_from_python:
                    result = rand_rollout_from_python(engine, solver, game_text, level_i, timeout=timeout_ms, n_steps=cfg.n_steps)
                else:
                    # Make the javascript timeout longer so that we can timeout from inside JS and return stats properly
                    result = algo(engine,
                                  cfg.n_steps, timeout_ms,
                                  timeout=timeout_ms*1.5 if timeout_ms > 0 else None,)
                result = {
                    'solved': result[0],
                    'actions': tuple(result[1]),
                    'iterations': result[2],
                    'time': result[3],
                    'FPS': result[2] / (result[3] if result[3] > 0 else 1e4),
                    'score': result[4],
                    'state': result[5],
                    'timeout': result[6],
                    'objs': result[7],
                }

                if os.path.isfile(level_js_sol_path):
                    with open(level_js_sol_path, 'r') as f:
                        level_js_sol_dict = json.load(f)
                    best_solve = level_js_sol_dict['won']
                    best_score = level_js_sol_dict['score']
                else:
                    best_solve = False
                    best_score = -np.inf
                
                if result['solved'] > best_solve or result['score'] > best_score:
                    result_dict = {
                        'won': result['solved'],
                        'score': result['score'],
                        'timeout': None,  # TODO
                        'objs': result['objs'],
                        'state': result['state'],
                    }
                    with open(level_js_sol_path, 'w') as f:
                        json.dump(result_dict, f, indent=4)
                    print(f"Saved solution to {level_js_sol_path}")


                print(json.dumps(result))
                results[run_name][game][level_i] = result

                with open(STANDALONE_NODEJS_RESULTS_PATH, 'w') as f:
                    json.dump(results, f, indent=4)



if __name__ == "__main__":
>>>>>>> Stashed changes
    main()