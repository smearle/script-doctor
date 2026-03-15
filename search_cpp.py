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
from puzzlescript_jax.globals import CPP_SOLS_DIR, STANDALONE_CPP_RESULTS_PATH
from puzzlescript_jax.utils import get_list_of_games_for_testing, init_ps_lark_parser, distribute_slurm_jobs
from puzzlescript_cpp import CppPuzzleScriptBackend
from search_nodejs import _classify_error, write_level_error_log


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
            dataset=cfg.dataset, include_random=cfg.include_randomness, random_order=cfg.random_order)
        game_sublists = distribute_slurm_jobs(games, cfg.n_games_per_job)
        n_jobs = len(game_sublists)
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "search_cpp"))
        executor.update_parameters(
            slurm_job_name="search_cpp",
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=cfg.slurm_timeout_min,
            slurm_array_parallelism=n_jobs,
            slurm_account=os.environ.get("SLURM_ACCOUNT"),
            slurm_setup=["export JAX_PLATFORMS=cpu"],
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: SearchCppConfig, games: Optional[List[str]] = None):
    backend = CppPuzzleScriptBackend()
    if cfg.timeout > 0:
        timeout_ms = cfg.timeout * 1_000
    elif cfg.slurm:
        safe_seconds = max(int((cfg.slurm_timeout_min - 2) * 60 * 0.9), 60)
        timeout_ms = safe_seconds * 1_000
        print(f'Derived per-level timeout from SLURM wall-time: {safe_seconds}s')
    else:
        timeout_ms = -1
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
            dataset=cfg.dataset, include_random=cfg.include_randomness, random_order=cfg.random_order)
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
                    with open(level_cpp_sol_path, 'r') as f:
                        result = json.load(f)
                    results[run_name][game][level_i] = result

                else:
                    score_initial = backend.get_initial_score(game_text, level_i)
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
                        error_type = _classify_error(e)
                        if error_type == 'unknown':
                            raise
                        error_message = str(e) or repr(e)
                        print(f'{error_type.upper()} during {game} level {level_i} with {algo}: {error_message}')
                        result = write_level_error_log(
                            level_cpp_sol_path,
                            error_type=error_type,
                            error_message=error_message,
                        )
                        results[run_name][game][level_i] = result
                        continue

                    # Replay solution to get per-step heuristic trajectory
                    score_trajectory = None
                    if result['actions']:
                        try:
                            score_trajectory = backend.replay_score_trajectory(
                                game_text, level_i, result['actions'],
                            )
                        except Exception:
                            pass

                    result_dict = {
                        'won': result['solved'],
                        'actions': result['actions'],
                        'score': result['score'],
                        'score_initial': score_initial,
                        'score_trajectory': score_trajectory,
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

                if cfg.render:
                    level_cpp_gif_path = os.path.splitext(level_cpp_sol_path)[0] + "_sol.gif"
                    try:
                        backend.render_gif(
                            game_text=game_text,
                            level_i=level_i,
                            actions=result['actions'],
                            gif_path=level_cpp_gif_path,
                            frame_duration_s=0.5,
                        )
                        print(f"Saved GIF to {level_cpp_gif_path}")
                    except Exception as e:
                        print(f"Error rendering game {game} level {level_i}: {e}")
                        traceback.print_exc()


        with open(STANDALONE_CPP_RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=4)
if __name__ == "__main__":
    main_launch()
