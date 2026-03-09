import json
import math
import os
import re
import shutil
import traceback
from typing import List, Optional

import cpuinfo
import dotenv
import hydra
import numpy as np
import submitit

from conf.config import SearchCppConfig
from puzzlejax.globals import CPP_SOLS_DIR, STANDALONE_CPP_RESULTS_PATH
from puzzlejax.utils import get_list_of_games_for_testing, init_ps_lark_parser
from puzzlescript_cpp import CppPuzzleScriptBackend


dotenv.load_dotenv()


def get_standalone_run_name(cfg: SearchCppConfig, algo_name, cpu_name):
    return f'algo-{algo_name}_{cfg.n_steps}-steps_{cpu_name}'


def get_standalone_run_params_from_name(run_name: str):
    groups = re.match(r'algo-(.*)_(\d+)-steps_(.*)', run_name)
    algo_name, n_steps, device_name = groups.groups()
    return algo_name, n_steps, device_name


@hydra.main(version_base="1.3", config_path='./', config_name='search_cpp_config')
def main_launch(cfg: SearchCppConfig):
    if cfg.slurm:
        games = get_list_of_games_for_testing(
            all_games=cfg.all_games, include_random=cfg.include_randomness, random_order=cfg.random_order)
        n_jobs = math.ceil(len(games) / cfg.n_games_per_job)
        game_sublists = [games[i::n_jobs] for i in range(n_jobs)]
        assert np.sum([len(g) for g in game_sublists]) == len(games), "Not all games are assigned to a job."
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "search_cpp"))
        executor.update_parameters(
            slurm_job_name="search_cpp",
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
            slurm_array_parallelism=n_jobs,
            slurm_account=os.environ.get("SLURM_ACCOUNT"),
            slurm_setup=["export JAX_PLATFORMS=cpu"],
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: SearchCppConfig, games: Optional[List[str]] = None):
    backend = CppPuzzleScriptBackend()
    timeout_ms = cfg.timeout * 1_000 if cfg.timeout > 0 else -1
    parser = init_ps_lark_parser()
    print(f'Timeout: {timeout_ms} ms')

    if cfg.algo == 'bfs':
        algos = ['bfs']
    elif cfg.algo == 'astar':
        algos = ['astar']
    elif cfg.algo == 'gbfs':
        algos = ['gbfs']
    elif cfg.algo == 'mcts':
        algos = ['mcts']
    elif cfg.algo == 'random':
        algos = ['random']
    else:
        raise ValueError(f"Invalid search algorithm: {cfg.algo}")
    cpu_name = cpuinfo.get_cpu_info()['brand_raw']

    if games is not None:
        games_to_test = games
    elif cfg.game is None:
        games_to_test = get_list_of_games_for_testing(
            all_games=cfg.all_games, include_random=cfg.include_randomness, random_order=cfg.random_order)
    else:
        games_to_test = [cfg.game]

    if os.path.isfile(STANDALONE_CPP_RESULTS_PATH) and not cfg.overwrite:
        shutil.copyfile(STANDALONE_CPP_RESULTS_PATH, STANDALONE_CPP_RESULTS_PATH[:-5] + '_bkp.json')
        with open(STANDALONE_CPP_RESULTS_PATH, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for game in games_to_test:
        print(f'\nGame: {game}')
        for algo in algos:
            run_name = get_standalone_run_name(cfg, algo, cpu_name)
            print(f'Algorithm: {algo}')
            results.setdefault(run_name, {})
            results[run_name].setdefault(game, {})
            try:
                backend.unload_game()
                game_text = backend.compile_game(parser, game)
            except Exception as e:
                print(f'Error compiling game {game} level {0}: {e}')
                traceback.print_exc()
                results[run_name][game] = {"Error": traceback.format_exc()}
                continue

            n_levels = backend.get_num_levels()
            game_cpp_sols_dir = os.path.join(CPP_SOLS_DIR, game)
            os.makedirs(game_cpp_sols_dir, exist_ok=True)

            for level_i in range(n_levels):
                algo_prefix = f'{algo}_'
                level_cpp_sol_path = os.path.join(
                    game_cpp_sols_dir, f'{algo_prefix}{cfg.n_steps}-steps_level-{level_i}.json')
                print(f'Level: {level_i}')
                if not cfg.overwrite and os.path.isfile(level_cpp_sol_path):
                    print(f'Already solved {game} level {level_i}.')
                    continue
                result = backend.run_search(
                    algo,
                    game_text=game_text,
                    level_i=level_i,
                    n_steps=cfg.n_steps,
                    timeout_ms=timeout_ms,
                    warmup=False,
                ).to_dict()

                if os.path.isfile(level_cpp_sol_path):
                    with open(level_cpp_sol_path, 'r') as f:
                        level_cpp_sol_dict = json.load(f)
                    best_solve = level_cpp_sol_dict['won']
                    best_score = level_cpp_sol_dict['score']
                    solution_exists = 'sol' in level_cpp_sol_dict or 'actions' in level_cpp_sol_dict
                else:
                    best_solve = False
                    best_score = -np.inf
                    solution_exists = False

                if cfg.overwrite or not solution_exists or result['solved'] > best_solve or result['score'] > best_score:
                    result_dict = {
                        'won': result['solved'],
                        'actions': result['actions'],
                        'score': result['score'],
                        'timeout': result['timeout'],
                        'iterations': result['iterations'],
                        'FPS': result['FPS'],
                        'time': result['time'],
                        'objs': result['objs'],
                        'state': result['state'],
                    }
                    with open(level_cpp_sol_path, 'w') as f:
                        json.dump(result_dict, f, indent=4)
                    print(f"Saved solution to {level_cpp_sol_path}")

                results[run_name][game][level_i] = result

        with open(STANDALONE_CPP_RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=4)

    if cfg.render:
        print("Rendering C++ solutions is not wired yet. Search results were saved without GIF rendering.")


if __name__ == "__main__":
    main_launch()
