import copy
import math
import os
import json
import multiprocessing as mp
import random
import re
import shutil
from timeit import default_timer as timer
import traceback
from typing import List, Optional

import cpuinfo
import hydra
from javascript import require
from javascript.proxy import Proxy
import numpy as np
import submitit

from conf.config import ProfileNodeJS
from globals import STANDALONE_NODEJS_RESULTS_PATH
from preprocess_games import SIMPLIFIED_GAMES_DIR, get_tree_from_txt
from utils import get_list_of_games_for_testing, level_to_int_arr, init_ps_lark_parser
from validate_sols import JS_SOLS_DIR


def compile_game(parser, engine, game, level_i):
    game_path = os.path.join(SIMPLIFIED_GAMES_DIR, f'{game}.txt')
    if not os.path.isfile(game_path):
        get_tree_from_txt(parser=parser, game=game, test_env_init=False, overwrite=True)
    with open(f'{game_path[:-4]}_simplified.txt', 'r') as f:
        game_text = f.read()
    engine.compile(['restart'], game_text)
    return game_text

def get_algo_name(algo):
    return str(algo).split(' ')[1].strip(']')

actions = ["LEFT", "RIGHT", "UP", "DOWN", "ACTION"]

def rand_rollout_from_python(engine, solver, game_text, level_i, n_steps, timeout):
    start_time = timer()
    for i in range(n_steps):
        if timeout > 0 and (i % 1_000 == 0) and (timer() - start_time > timeout):
            fps = i / (timer() - start_time)
            return False, [], i, fps, score, state, False, []
        action = random.randint(0, 5)
        _, _, _, _, score, state, _, objects = solver.takeAction(engine, action)
    return False, [], i, timer() - start_time, score, state, False, list(objects)

    
def get_standalone_run_name(cfg: ProfileNodeJS, algo_name, cpu_name):
    return f'algo-{algo_name}_{cfg.n_steps}-steps_{cpu_name}'

def get_standalone_run_params_from_name(run_name: str):
    # use regex to extract the parameters from the run name
    groups = re.match(r'algo-(.*)_(\d+)-steps_(.*)', run_name)
    algo_name, n_steps, device_name = groups.groups()
    return algo_name, n_steps, device_name


# @hydra.main(version_base="1.3", config_path='./', config_name='profile_standalone')
@hydra.main(version_base="1.3", config_path='./', config_name='profile_nodejs_config')
def main_launch(cfg: ProfileNodeJS):
    if cfg.slurm:
        games = get_list_of_games_for_testing(
            all_games=cfg.all_games, include_random=cfg.include_randomness, random_order=cfg.random_order)
        # Get sub-lists of batches of games to distribute across nodes.
        n_jobs = math.ceil(len(games) / cfg.n_games_per_job)
        game_sublists = [games[i::n_jobs] for i in range(n_jobs)]
        assert np.sum([len(g) for g in game_sublists]) == len(games), "Not all games are assigned to a job."
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "profile_nodejs"))
        executor.update_parameters(
            slurm_job_name=f"profile_nodejs",
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
            slurm_account='pr_174_tandon_advanced', 
            slurm_array_parallelism=n_jobs,
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: ProfileNodeJS, games: Optional[List[str]] = None):
    if cfg.for_profiling:
        cfg.n_steps = 5_000
        cfg.algo = 'random'

    engine = require('./standalone/puzzlescript/engine.js')
    solver = require('./standalone/puzzlescript/solver.js')
    timeout_ms = cfg.timeout * 1_000 if cfg.timeout > 0 else -1
    parser = init_ps_lark_parser()
    print(f'Timeout: {timeout_ms} ms')

    if cfg.algo == 'bfs':
        algos = [solver.solveBFS]
        # algos = [solver.solveBFS, solver.solveAStar, solver.solveMCTS]:
    elif cfg.algo == 'random':
        algos = [solver.randomRollout, rand_rollout_from_python]
    cpu_name = cpuinfo.get_cpu_info()['brand_raw']

    if games is not None:
        games_to_test = games
    elif cfg.game is None:
        games_to_test = get_list_of_games_for_testing(
            all_games=cfg.all_games, include_random=cfg.include_randomness, random_order=cfg.random_order)
    else:
        games_to_test = [cfg.game]
    results = {get_algo_name(algo): {} for algo in algos}
    if os.path.isfile(STANDALONE_NODEJS_RESULTS_PATH) and not cfg.overwrite and not cfg.for_validation \
            and not cfg.for_solution:
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
                engine.unloadGame()
                game_text = compile_game(parser, engine, game, 0)
            except Exception as e:
                print(f'Error compiling game {game} level {0}: {e}')
                results[run_name][game] = {"Error": traceback.print_exc()}
                continue

            if cfg.for_profiling:
                n_levels = 1
            else:
                n_levels = engine.getNumLevels()
            game_js_sols_dir = os.path.join(JS_SOLS_DIR, game)
            os.makedirs(game_js_sols_dir, exist_ok=True)

            for level_i in range(n_levels):
                level_js_sol_path = os.path.join(game_js_sols_dir, f'{cfg.n_steps}-steps_level-{level_i}.json')
                print(f'Level: {level_i}')
                if cfg.for_validation or cfg.for_solution and not cfg.overwrite and os.path.isfile(level_js_sol_path):
                    print(f'Already solved (for validation) {game} level {level_i}.')
                    continue
                if not cfg.for_validation and not cfg.for_solution and not cfg.overwrite and str(level_i) in results[run_name][game]:
                    print(f'Already solved (for profiling) {game} level {level_i} with {run_name}, skipping.')
                    continue
                engine.compile(['loadLevel', level_i], game_text)
                if algo == rand_rollout_from_python:
                    result = rand_rollout_from_python(engine, solver, game_text, level_i, timeout=timeout_ms, n_steps=cfg.n_steps)
                else:
                    # Make the javascript timeout longer so that we can timeout from inside JS and return stats properly
                    def call_algo():
                        result = algo(engine,
                                    cfg.n_steps, timeout_ms,
                                    timeout=timeout_ms*1.5 if timeout_ms > 0 else None,)
                        return result
                    # If profiling, let the nodejs engine warm up (the same way we do for JAX). Maybe we should tweak
                    # the number of warmup loops? I don't think warming up really applies for the python-nodejs bridge,
                    # so we don't do it in that case above.
                    if cfg.for_profiling:
                        for _ in range(3):
                            result = call_algo()
                    else:
                        result = call_algo()

                n_objs = len(list(result[7]))
                end_level_state = level_to_int_arr(result[5], n_objs).tolist()
                result = {
                    'solved': result[0],
                    'actions': tuple(result[1]),
                    'iterations': result[2],
                    'time': result[3],
                    'FPS': result[2] / (result[3] if result[3] > 0 else 1e4),
                    'score': result[4],
                    'state': end_level_state,
                    'timeout': result[6],
                    'objs': list(result[7]),
                }

                if os.path.isfile(level_js_sol_path):
                    with open(level_js_sol_path, 'r') as f:
                        level_js_sol_dict = json.load(f)
                    best_solve = level_js_sol_dict['won']
                    best_score = level_js_sol_dict['score']
                    solution_exists = 'sol' in level_js_sol_dict or 'actions' in level_js_sol_dict
                else:
                    best_solve = False
                    best_score = -np.inf
                    solution_exists = False
                
                if cfg.overwrite or not solution_exists or result['solved'] > best_solve or result['score'] > best_score:
                    result_dict = {
                        'won': result['solved'],
                        'actions': result['actions'],
                        'score': result['score'],
                        'timeout': result['timeout'],  # TODO
                        'iterations': result['iterations'],
                        'FPS': result['FPS'],
                        'time': result['time'],
                        'objs': result['objs'],
                        'state': result['state'],
                    }
                    with open(level_js_sol_path, 'w') as f:
                        json.dump(result_dict, f, indent=4)
                    print(f"Saved solution to {level_js_sol_path}")


                # print(json.dumps(result))
                results[run_name][game][level_i] = result

        if not cfg.for_validation or cfg.for_solution:
            with open(STANDALONE_NODEJS_RESULTS_PATH, 'w') as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    main_launch()