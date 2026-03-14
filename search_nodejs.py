import math
import os
import json
import re
import shutil
import traceback
from typing import List, Optional

import cpuinfo
import dotenv
import hydra
import numpy as np
import submitit

from conf.config import SearchNodeJSConfig
from backends import NodeJSPuzzleScriptBackend
from puzzlescript_jax.globals import STANDALONE_NODEJS_RESULTS_PATH, JS_SOLS_DIR
from puzzlescript_jax.utils import get_list_of_games_for_testing, init_ps_lark_parser


dotenv.load_dotenv()


OOM_ERROR_PATTERNS = (
    "out of memory",
    "oom",
    "heap out of memory",
    "allocation failed",
    "memory limit",
)


def get_standalone_run_name(cfg: SearchNodeJSConfig, algo_name, cpu_name):
    return f'algo-{algo_name}_{cfg.n_steps}-steps_{cpu_name}'


def get_standalone_run_params_from_name(run_name: str):
    # use regex to extract the parameters from the run name
    groups = re.match(r'algo-(.*)_(\d+)-steps_(.*)', run_name)
    algo_name, n_steps, device_name = groups.groups()
    return algo_name, n_steps, device_name


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    error_text = " ".join(
        part for part in (
            str(exc),
            repr(exc),
            traceback.format_exc(),
        ) if part
    ).lower()
    return any(pattern in error_text for pattern in OOM_ERROR_PATTERNS)


def write_level_error_log(path: str, error_type: str, error_message: str) -> dict:
    result_dict = {
        'won': False,
        'actions': [],
        'score': None,
        'timeout': False,
        'iterations': 0,
        'FPS': 0,
        'time': 0,
        'objs': [],
        'state': [],
        'error': error_type,
        'error_message': error_message,
    }
    with open(path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict


def should_skip_existing_level_result(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, 'r') as f:
            json.load(f)
    except Exception:
        return True
    return True


@hydra.main(version_base="1.3", config_path='./', config_name='search_nodejs_config')
def main_launch(cfg: SearchNodeJSConfig):
    if cfg.slurm:
        games = get_list_of_games_for_testing(
            dataset=cfg.dataset, include_random=cfg.include_randomness, random_order=cfg.random_order)
        # Get sub-lists of batches of games to distribute across nodes.
        n_jobs = math.ceil(len(games) / cfg.n_games_per_job)
        game_sublists = [games[i::n_jobs] for i in range(n_jobs)]
        assert np.sum([len(g) for g in game_sublists]) == len(games), "Not all games are assigned to a job."
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "search_nodejs"))
        executor.update_parameters(
            slurm_job_name=f"search_nodejs",
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
            slurm_array_parallelism=n_jobs,
            slurm_account=os.environ.get("SLURM_ACCOUNT")
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: SearchNodeJSConfig, games: Optional[List[str]] = None):

    backend = NodeJSPuzzleScriptBackend()
    timeout_ms = cfg.timeout * 1_000 if cfg.timeout > 0 else -1
    parser = init_ps_lark_parser()
    print(f'Timeout: {timeout_ms} ms')
    print(f'Max node budget: {cfg.n_steps}')

    if cfg.algo == 'bfs':
        algos = ['bfs']
    elif cfg.algo == 'astar':
        algos = ['astar']
    elif cfg.algo == 'gbfs':
        algos = ['gbfs']
    elif cfg.algo == 'mcts':
        algos = ['mcts']
    elif cfg.algo == 'random':
        algos = ['random', 'python_random']
    else:
        raise ValueError(f"Invalid search algorithm: {cfg.algo}")
    cpu_name = cpuinfo.get_cpu_info()['brand_raw']

    if games is not None:
        games_to_test = games
    elif cfg.game is None:
        games_to_test = get_list_of_games_for_testing(
            dataset=cfg.dataset, include_random=cfg.include_randomness, random_order=cfg.random_order)
    else:
        games_to_test = [cfg.game]
    results = {algo: {} for algo in algos}
    if os.path.isfile(STANDALONE_NODEJS_RESULTS_PATH) and not cfg.overwrite:
        shutil.copyfile(STANDALONE_NODEJS_RESULTS_PATH, STANDALONE_NODEJS_RESULTS_PATH[:-5] + '_bkp.json')
        with open(STANDALONE_NODEJS_RESULTS_PATH, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for game in games_to_test:

        print(f'\nGame: {game}')
        # TODO: How to get the available number of levels from nodejs?
        for algo in algos:
            run_name = get_standalone_run_name(cfg, algo, cpu_name)
            print(f'Algorithm: {algo}')
            if run_name not in results:
                results[run_name] = {}
            if game not in results[run_name]:
                results[run_name][game] = {}
            try:
                backend.unload_game()
                game_text = backend.compile_game(parser, game)
            except Exception as e:
                print(f'Error compiling game {game} level {0}: {e}')
                results[run_name][game] = {"Error": traceback.print_exc()}
                continue

            n_levels = backend.get_num_levels()
            game_js_sols_dir = os.path.join(JS_SOLS_DIR, game)
            os.makedirs(game_js_sols_dir, exist_ok=True)

            levels_to_test = range(n_levels) if cfg.level is None else [cfg.level]

            for level_i in levels_to_test:
                algo_prefix = f'{algo}_'
                level_js_sol_path = os.path.join(
                    game_js_sols_dir, f'{algo_prefix}{cfg.n_steps}-steps_level-{level_i}.json')
                print(f'Level: {level_i}')
                if not cfg.overwrite and should_skip_existing_level_result(level_js_sol_path):
                    print(f'Already solved {game} level {level_i}.')
                    continue
                try:
                    result = backend.run_search(
                        algo,
                        game_text=game_text,
                        level_i=level_i,
                        n_steps=cfg.n_steps,
                        timeout_ms=timeout_ms,
                        warmup=False,
                    ).to_dict()
                except Exception as e:
                    if not is_oom_error(e):
                        raise
                    error_message = str(e) or repr(e)
                    print(f'OOM during {game} level {level_i} with {algo}: {error_message}')
                    result = write_level_error_log(
                        level_js_sol_path,
                        error_type='oom',
                        error_message=error_message,
                    )
                    results[run_name][game][level_i] = result
                    continue

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

                    if cfg.render:
                        gif_path = os.path.join(
                            game_js_sols_dir, f'{algo_prefix}{cfg.n_steps}-steps_level-{level_i}.gif')
                        backend.render_gif(
                            game_text=game_text,
                            level_i=level_i,
                            actions=result['actions'],
                            gif_path=gif_path,
                            frame_duration_s=0.2,
                            scale=1,
                        )

                # print(json.dumps(result))
                results[run_name][game][level_i] = result

        with open(STANDALONE_NODEJS_RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main_launch()
