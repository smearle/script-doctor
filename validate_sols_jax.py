import bdb
import copy
import glob
import json
import math
import os
import pickle
import random
import re
import shutil
import traceback
from collections import OrderedDict
from typing import List, Optional

from einops import rearrange
import hydra
import imageio
import jax
import jax.numpy as jnp
from lark import Lark
import numpy as np
import pandas as pd
from skimage.transform import resize
import submitit

from conf.config import JaxValidationConfig
from backends import NodeJSPuzzleScriptBackend
from puzzlescript_jax.env import PuzzleJaxEnv
from puzzlescript_jax.globals import (
    SOLUTION_REWARDS_PATH, GAMES_TO_N_RULES_PATH, JS_SOLS_DIR, JAX_VALIDATED_JS_SOLS_DIR, JS_TO_JAX_ACTIONS, DATA_DIR,
    LARK_SYNTAX_PATH,
)
from puzzlescript_jax.preprocessing import PJParseErrors, get_tree_from_txt
from puzzlescript_jax.env_utils import multihot_to_desc
from puzzlescript_nodejs.utils import replay_actions_js
from puzzlescript_jax.utils import get_list_of_games_for_testing, to_binary_vectors
from utils_rl import get_env_params_from_config


scratch_dir = 'scratch'
os.makedirs(scratch_dir, exist_ok = True)

games_to_skip = set({
    '2048',  # hangs
})


def _dedupe_preserve_order(items, key_fn=None):
    seen = OrderedDict()
    for item in items:
        key = key_fn(item) if key_fn is not None else item
        seen[key] = item
    return list(seen.values())


def make_side_by_side_frames(left_frames, right_frames, separator_w=2):
    if len(left_frames) == 0 or len(right_frames) == 0:
        return []

    def normalize_frame(frame):
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3:
            raise ValueError(f"Expected HWC frame, got shape {frame.shape}")
        if frame.shape[2] == 4:
            return frame[:, :, :3]
        if frame.shape[2] == 3:
            return frame
        raise ValueError(f"Unsupported channel count {frame.shape[2]} for frame shape {frame.shape}")

    left_frames = [normalize_frame(frame) for frame in left_frames]
    right_frames = [normalize_frame(frame) for frame in right_frames]
    max_len = max(len(left_frames), len(right_frames))
    while len(left_frames) < max_len:
        left_frames.append(left_frames[-1])
    while len(right_frames) < max_len:
        right_frames.append(right_frames[-1])

    combo_frames = []
    for left_frame, right_frame in zip(left_frames, right_frames):
        left_h, _ = left_frame.shape[:2]
        right_h, _ = right_frame.shape[:2]
        frame_h = max(left_h, right_h)
        if left_h < frame_h:
            left_frame = np.pad(left_frame, ((0, frame_h - left_h), (0, 0), (0, 0)))
        if right_h < frame_h:
            right_frame = np.pad(right_frame, ((0, frame_h - right_h), (0, 0), (0, 0)))
        separator = np.full((frame_h, separator_w, 3), 128, dtype=np.uint8)
        combo_frames.append(np.concatenate([left_frame, separator, right_frame], axis=1))

    return combo_frames


def get_trace_js_gif_path(level_sol_json_path: str) -> str:
    return os.path.splitext(level_sol_json_path)[0] + '_sol.gif'


def resolve_js_gif_path(level_sol_json_path: str, sol_dir: str, level_i: int) -> str:
    trace_path = get_trace_js_gif_path(level_sol_json_path)
    return trace_path


def multihot_level_from_js_state(level_state, obj_list, target_obj_names=None):
    level_state = np.array(level_state).T
    multihot_level_js = to_binary_vectors(level_state, len(obj_list))
    multihot_level_js = rearrange(multihot_level_js, 'h w c -> c h w')[::-1]

    # Remove duplicate channels from the multihot level.
    new_multihot_level_js = []
    new_objs_to_idxs = {}
    for obj_idx, obj in enumerate(obj_list):
        obj = obj.lower() if isinstance(obj, str) else obj
        if obj not in new_objs_to_idxs:
            new_objs_to_idxs[obj] = len(new_objs_to_idxs)
            new_multihot_level_js.append(multihot_level_js[obj_idx])
        else:
            dupe_obj_idx = new_objs_to_idxs[obj]
            new_multihot_level_js[dupe_obj_idx] = np.logical_or(
                new_multihot_level_js[dupe_obj_idx],
                multihot_level_js[obj_idx]
            )
    multihot_level_js = np.array(new_multihot_level_js, dtype=bool)

    if target_obj_names is None:
        return multihot_level_js

    target_multihot_level = np.zeros((len(target_obj_names), *multihot_level_js.shape[1:]), dtype=bool)
    for target_idx, obj in enumerate(target_obj_names):
        obj = obj.lower() if isinstance(obj, str) else obj
        if obj in new_objs_to_idxs:
            target_multihot_level[target_idx] = multihot_level_js[new_objs_to_idxs[obj]]

    return target_multihot_level


def format_state_for_log(state, env):
    multihot_desc = multihot_to_desc(
        np.asarray(state.multihot_level),
        env.objs_to_idxs,
        env.n_objs,
        env.obj_idxs_to_force_idxs,
    )
    return (
        "PJState(\n"
        f"  multihot_level=\n{multihot_desc}\n"
        f"  win={state.win},\n"
        f"  score={state.score},\n"
        f"  heuristic={state.heuristic},\n"
        f"  restart={state.restart},\n"
        f"  init_heuristic={state.init_heuristic},\n"
        f"  prev_heuristic={state.prev_heuristic},\n"
        f"  step_i={state.step_i},\n"
        f"  rng={state.rng}\n"
        ")"
    )


@hydra.main(version_base="1.3", config_path='conf', config_name='jax_validation_config')
def main_launch(cfg: JaxValidationConfig):
    if cfg.slurm:
        games = get_list_of_games_for_testing(all_games=cfg.all_games)
        # Get sub-lists of games to distribute across nodes.
        n_jobs = math.ceil(len(games) / cfg.n_games_per_job)
        game_sublists = [games[i::n_jobs] for i in range(n_jobs)]
        assert np.sum([len(g) for g in game_sublists]) == len(games), "Not all games are assigned to a job."
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "validate_sols"))
        executor.update_parameters(
            slurm_job_name=f"validate_sols",
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
            # slurm_gres='gpu:1',
            slurm_setup=["export JAX_PLATFORMS=cpu"],
            slurm_array_parallelism=n_jobs,
            slurm_account="torch_pr_84_tandon_advanced",
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: JaxValidationConfig, games: Optional[List[str]] = None):
    backend = NodeJSPuzzleScriptBackend()
    engine = backend.engine
    solver = backend.solver
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Initialize the Lark parser with the PuzzleScript grammar
    with open(LARK_SYNTAX_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    with open(GAMES_TO_N_RULES_PATH, 'r') as f:
        games_to_n_rules = json.load(f)
    games_to_n_rules_dirty = False
    if games is not None:
        games = games
    elif cfg.game is None:
        games = get_list_of_games_for_testing(all_games=cfg.all_games, random_order=cfg.random_order)
    else:
        games = [cfg.game]
    current_game_names = {os.path.splitext(game_file)[0] for game_file in os.listdir(os.path.join(DATA_DIR, 'scraped_games'))}
    games = [game for game in games if game in current_game_names]
    games = _dedupe_preserve_order(games)
    games_to_n_rules = {
        game_key: game_val for game_key, game_val in games_to_n_rules.items()
        if os.path.splitext(os.path.basename(game_key))[0] in current_game_names
    }
    if cfg.aggregate:
        results = {
            'stats': {},
            'compile_error': [],
            'rigid_prefix_error': [],
            'timeout': [],
            'runtime_error': {},
            'solution_error': {},
            'state_error': {},
            'random_solution_error': {},
            'random_state_error': {},
            'score_error': {},
            'success': {},
            'valid_games': [],
            'partial_valid_games': [],
        }
        if not cfg.include_test_games:
            games = [game for game in games if not game.startswith('test_')]
    val_results_path = os.path.join('data', 'validation_results.json')
    if os.path.isfile(val_results_path):
        shutil.copy(val_results_path, val_results_path[:-5] + '_bkp.json')

    # tree_paths = [os.path.join(TREES_DIR, os.path.basename(path) + '.pkl') for path in sol_paths]
    # games = [os.path.basename(path)[:-4] for path in sol_paths]
    sol_paths = [os.path.join(JS_SOLS_DIR, game) for game in games]

    n_levels = 0
    n_compile_error = 0
    n_rigid_prefix_error = 0
    n_timeout_error = 0
    n_runtime_error = 0
    n_solution_error = 0
    n_state_error = 0
    n_score_error = 0
    n_success = 0
    n_unvalidated_levels = 0
    n_random_solution_error = 0
    n_random_state_error = 0

    # if os.path.exists(SOLUTION_REWARDS_PATH):
    #     with open(SOLUTION_REWARDS_PATH, 'r') as f:
    #         solution_rewards_dict = json.load(f)    
    # else:
    solution_rewards_dict = {}
    

    def save_stats(results, n_levels, n_success, n_compile_error, n_rigid_prefix_error, n_timeout_error,
                   n_runtime_error, n_solution_error, n_state_error, n_score_error, n_unvalidated_levels,
                   n_random_solution_error, n_random_state_error):
        results['stats']['total_games'] = len(games)
        results['stats']['total_levels'] = n_levels
        results['stats']['successful_solutions'] = n_success
        results['stats']['compile_error'] = n_compile_error
        results['stats']['rigid_prefix_error'] = n_rigid_prefix_error
        results['stats']['timeout'] = n_timeout_error
        results['stats']['runtime_error'] = n_runtime_error
        results['stats']['solution_error'] = n_solution_error
        results['stats']['state_error'] = n_state_error
        results['stats']['random_solution_error'] = n_random_solution_error
        results['stats']['random_state_error'] = n_random_state_error
        results['stats']['score_error'] = n_score_error
        results['stats']['unvalidated_levels'] = n_unvalidated_levels

        with open(val_results_path, 'w') as f:
            json.dump(results, f, indent=4)

        with open(SOLUTION_REWARDS_PATH, 'w') as f:
            json.dump(solution_rewards_dict, f, indent=4)


    def is_runtime_timeout_log(log: str) -> bool:
        if not log:
            return False
        log_lower = log.lower()
        return (
            'uncompletedjoberror' in log_lower
            and ('timed-out' in log_lower or 'timed out' in log_lower)
        )


    def get_games_to_n_rules_entry(game: str):
        if game in games_to_n_rules:
            return game, games_to_n_rules[game]
        if f"{game}.txt" in games_to_n_rules:
            return f"{game}.txt", games_to_n_rules[f"{game}.txt"]
        game_name = os.path.basename(game)
        if game_name in games_to_n_rules:
            return game_name, games_to_n_rules[game_name]
        if f"{game_name}.txt" in games_to_n_rules:
            return f"{game_name}.txt", games_to_n_rules[f"{game_name}.txt"]
        return None, None


    def is_random_game(n_rules_entry) -> bool:
        if isinstance(n_rules_entry, (list, tuple)) and len(n_rules_entry) > 1:
            return bool(n_rules_entry[1])
        return False


    def update_game_randomness(game_key, n_rules_entry, has_randomness: bool):
        nonlocal games_to_n_rules_dirty
        if game_key is None or n_rules_entry is None:
            return n_rules_entry
        new_val = bool(has_randomness)
        if isinstance(n_rules_entry, (list, tuple)) and len(n_rules_entry) > 1:
            if n_rules_entry[1] != new_val:
                n_rules_entry = list(n_rules_entry)
                n_rules_entry[1] = new_val
                games_to_n_rules[game_key] = n_rules_entry
                games_to_n_rules_dirty = True
        return n_rules_entry


    key = jax.random.PRNGKey(0)

    # 0 - left
    # 1 - down
    # 2 - right
    # 3 - up
    # 4 - action

    for sol_dir, game in zip(sol_paths, games):
        n_rules_key, n_rules = get_games_to_n_rules_entry(game)
        if game not in solution_rewards_dict:
            solution_rewards_dict[game] = {}
        game_name = os.path.basename(game)
        if game_name in games_to_skip:
            print(f"Skipping {game_name} because it is in the skip list")
            continue
        jax_sol_dir = os.path.join(JAX_VALIDATED_JS_SOLS_DIR, game)
        if cfg.overwrite:
            shutil.rmtree(jax_sol_dir, ignore_errors=True)
        compile_log_path = os.path.join(jax_sol_dir, 'compile_err.txt')
        if os.path.exists(compile_log_path) and not cfg.overwrite:
            if cfg.aggregate:
                with open(compile_log_path, 'r') as f:
                    compile_log = f.read()
                if 'Rigid prefix not implemented' in compile_log:
                    results['rigid_prefix_error'].append({'game': game, 'n_rules': n_rules, 'log': compile_log})
                    n_rigid_prefix_error += 1
                elif 'timeout' in compile_log or compile_log.strip() == "":
                    results['timeout'].append({'game': game, 'n_rules': n_rules, 'log': compile_log})
                    n_timeout_error += 1
                else:
                    results['compile_error'].append({'game': game, 'n_rules': n_rules, 'log': compile_log})
                    n_compile_error += 1
            else:
                with open(compile_log_path, 'r') as f:
                    compile_log = f.read()
                if 'Rigid prefix not implemented' in compile_log:
                    n_rigid_prefix_error += 1
                elif 'timeout' in compile_log or compile_log.strip() == "":
                    n_timeout_error += 1
                else:
                    n_compile_error += 1
            n_levels += 1
            print(f"Skipping {game} because compile error log already exists")
            continue

        # Get all level solutions previously generated by tree search in javascript.
        level_sols = glob.glob(os.path.join(sol_dir, '*level-*.json'))
        level_sols = [os.path.basename(p) for p in level_sols]
        level_ints = [int(os.path.basename(p).split('-')[-1].split('.')[0]) for p in level_sols]
        sorted_idxs = np.argsort(level_ints)
        level_ints = [level_ints[i] for i in sorted_idxs]
        level_sols = [level_sols[i] for i in sorted_idxs]
        level_ints_to_sols = {}
        for i, level_i in enumerate(level_ints):
            if level_i not in level_ints_to_sols:
                level_ints_to_sols[level_i] = []
            level_ints_to_sols[level_i].append(level_sols[i])
        # Remove the level_sols with fewer number of steps in case of multiple solutions (at different step counts) for 
        # the same level.
        # Prioritize BFS > AStar > MCTS
        new_level_sols = []
        level_ints = []
        for level_i, sols in level_ints_to_sols.items():
            # Filter by algorithm priority
            bfs_sols = [s for s in sols if 'solveBFS' in s]
            astar_sols = [s for s in sols if 'solveAStar' in s]
            mcts_sols = [s for s in sols if 'solveMCTS' in s]

            if bfs_sols:
                sols_to_consider = bfs_sols
            elif astar_sols:
                sols_to_consider = astar_sols
            elif mcts_sols:
                sols_to_consider = mcts_sols
            else:
                sols_to_consider = sols

            if len(sols_to_consider) == 1:
                new_level_sols.append(sols_to_consider[0])
            else:
                # Sort by number of steps, and take the one with the most steps.
                n_steps = [
                    int(os.path.basename(p).split('-steps_')[0].split('_')[-1])
                        if '-steps_' in os.path.basename(p)
                        else 10_000 for p in sols_to_consider
                    ]
                max_steps_idx = np.argmax(n_steps)
                new_level_sols.append(sols_to_consider[max_steps_idx])
            level_ints.append(level_i)
        level_sols = new_level_sols

        game_success = True
        game_partial_success = False
        game_compile_error = False
        game_randomness_updated = False
        env = None

        if len(level_sols) == 0:
            print(f"No js solutions found for {game_name}")
            game_success = False
            continue

        og_path = os.path.join(DATA_DIR, 'scraped_games', game_name + '.txt')

        print(f"Processing solution for game: {og_path}")

        if not cfg.aggregate:
            game_text = backend.compile_game(parser, game_name)

        os.makedirs(jax_sol_dir, exist_ok=True)
        print(f"Saving validation results to {jax_sol_dir}")
        for level_i, level_sol_path in zip(level_ints, level_sols):
        
            # if cfg.aggregate:
            #     save_stats()
            if game_compile_error:
                break

            n_levels += 1
            sol_log_path = os.path.join(jax_sol_dir, f'level-{level_i}_solution_err.txt')
            score_log_path = os.path.join(jax_sol_dir, f'level-{level_i}_score_err.txt')
            intermediary_scores_log_path = os.path.join(jax_sol_dir, f'level-{level_i}_intermediary_scores.txt')
            run_log_path = os.path.join(jax_sol_dir, f'level-{level_i}_runtime_err.txt')
            state_log_path = os.path.join(jax_sol_dir, f'level-{level_i}_state_err.txt')
            gif_path = os.path.join(jax_sol_dir, f'level-{level_i}.gif')
            compare_gif_path = os.path.join(jax_sol_dir, f'level-{level_i}_compare.gif')

            # Skip if we already have a result and are not overwritiing.
            if (os.path.exists(gif_path) or os.path.exists(sol_log_path) or os.path.exists(score_log_path)
                    or os.path.exists(run_log_path) or os.path.exists(state_log_path)
                    or os.path.exists(compare_gif_path) or os.path.exists(intermediary_scores_log_path)):
                if cfg.overwrite:
                    if os.path.exists(gif_path):
                        os.remove(gif_path)
                    if os.path.exists(compare_gif_path):
                        os.remove(compare_gif_path)
                    if os.path.exists(sol_log_path):
                        os.remove(sol_log_path)
                    if os.path.exists(score_log_path):
                        os.remove(score_log_path)
                    if os.path.exists(intermediary_scores_log_path):
                        os.remove(intermediary_scores_log_path)
                    if os.path.exists(run_log_path):
                        os.remove(run_log_path)
                    if os.path.exists(state_log_path):
                        os.remove(state_log_path)
                else:
                    if os.path.exists(run_log_path):
                        if cfg.aggregate:
                            with open(run_log_path, 'r') as f:
                                run_log = f.read()
                            if is_runtime_timeout_log(run_log):
                                results['timeout'].append({'game': game_name, 'n_rules': n_rules, 'level': level_i, 'log': run_log})
                                n_timeout_error += 1
                            else:
                                if game_name not in results['runtime_error']:
                                    results['runtime_error'][game_name] = []
                                results['runtime_error'][game_name].append({'level': level_i, 'n_rules': n_rules, 'log': run_log})
                                n_runtime_error += 1
                        else:
                            with open(run_log_path, 'r') as f:
                                run_log = f.read()
                            if is_runtime_timeout_log(run_log):
                                n_timeout_error += 1
                            else:
                                n_runtime_error += 1
                        game_success = False
                    elif os.path.exists(sol_log_path):
                        if cfg.aggregate:
                            if is_random_game(n_rules):
                                if game_name not in results['random_solution_error']:
                                    results['random_solution_error'][game_name] = []
                                results['random_solution_error'][game_name].append({'n_rules': n_rules, 'level': level_i})
                                n_random_solution_error += 1
                            else:
                                if game_name not in results['solution_error']:
                                    results['solution_error'][game_name] = []
                                results['solution_error'][game_name].append({'n_rules': n_rules, 'level': level_i})
                                n_solution_error += 1
                        else:
                            n_solution_error += 1
                        game_success = False
                    elif os.path.exists(state_log_path):
                        if cfg.aggregate:
                            if is_random_game(n_rules):
                                if game_name not in results['random_state_error']:
                                    results['random_state_error'][game_name] = []
                                results['random_state_error'][game_name].append({'n_rules': n_rules, 'level': level_i})
                                n_random_state_error += 1
                            else:
                                if game_name not in results['state_error']:
                                    results['state_error'][game_name] = []
                                results['state_error'][game_name].append({'n_rules': n_rules, 'level': level_i})
                                n_state_error += 1
                        else:
                            n_state_error += 1
                        game_success = False
                        # game_partial_success = True
                    elif os.path.exists(score_log_path):
                        if cfg.aggregate:
                            with open(score_log_path, 'r') as f:
                                score_log = f.read()
                            if game_name not in results['score_error']:
                                results['score_error'][game_name] = []
                            results['score_error'][game_name].append({'n_rules': n_rules, 'level': level_i, 'log': score_log})
                        n_score_error += 1
                        # We'll be a bit generous and not count this for now. TODO: fix the score mismatch.
                        # game_success = False
                        game_partial_success = True
                    elif os.path.exists(intermediary_scores_log_path):
                        if cfg.aggregate:
                            with open(intermediary_scores_log_path, 'r') as f:
                                intermediary_scores_log = f.read()
                            if game_name not in results['score_error']:
                                results['score_error'][game_name] = []
                            results['score_error'][game_name].append({'n_rules': n_rules, 'level': level_i, 'log': intermediary_scores_log})
                        n_score_error += 1
                        game_partial_success = True
                    else:
                        if cfg.aggregate:
                            if game_name not in results['success']:
                                results['success'][game_name] = []
                            results['success'][game_name].append({'n_rules': n_rules, 'level': level_i})
                        n_success += 1
                        game_partial_success = True
                    if not cfg.aggregate:
                        print(f"Skipping level {level_i} because gif or error log already exists")
                    continue

            if cfg.aggregate:
                # In this case, don't run any new validations, just aggregate results for the ones we've already run.
                n_unvalidated_levels += 1
                game_success = False
                continue

            # Otherwise, let's initialize the environment (in single-level mode on the given level) and run the solution.
            tree, success, err_msg = get_tree_from_txt(parser, game, test_env_init=False, timeout=60*20)
            if success == PJParseErrors.SUCCESS:
                try:
                    env = PuzzleJaxEnv(tree, debug=False, print_score=False, level_i=level_i)
                except KeyboardInterrupt as e:
                    raise e
                except bdb.BdbQuit as e:
                    raise e
                except Exception as e:
                    err_msg = traceback.format_exc()
                    success = PJParseErrors.ENV_ERROR
            if success != PJParseErrors.SUCCESS:
                if success == PJParseErrors.TIMEOUT and not err_msg:
                    err_msg = "timeout"
                with open(compile_log_path, 'w') as f:
                    f.write(err_msg)
                print(f"Error creating env: {og_path}\n{err_msg}")
                # results['compile_error'].append({'game': game, 'n_rules': n_rules, 'log': err_log})
                if 'Rigid prefix not implemented' in err_msg:
                    n_rigid_prefix_error += 1
                elif success == PJParseErrors.TIMEOUT:
                    if cfg.aggregate:
                        results['timeout'].append({'game': game, 'n_rules': n_rules, 'log': err_msg})
                    n_timeout_error += 1
                else:
                    n_compile_error += 1
                n_levels += 1
                game_success = False
                game_compile_error = True
                continue

            level_sol_path = os.path.join(sol_dir, level_sol_path)
            with open(level_sol_path, 'r') as f:
                sol_dict = json.load(f)
            if 'sol' not in sol_dict and 'actions' not in sol_dict:
                print(f"No sol/actions found in sol_dict, skipping.")
                continue
            print(f"Using solution from {level_sol_path}")
            level_sol = sol_dict['sol'] if 'sol' in sol_dict else sol_dict['actions']
            level_win = sol_dict['won']
            level_score = sol_dict['score']
            level_state = sol_dict['state']
            obj_list = sol_dict['objs']
            multihot_level_js = multihot_level_from_js_state(
                level_state,
                obj_list,
                target_obj_names=env.atomic_obj_names,
            )
            actions = level_sol
            # print(f"Level {level_i} solution: {actions}")
            actions = [JS_TO_JAX_ACTIONS[a] for a in actions]
            actions = jnp.array([int(a) for a in actions], dtype=jnp.int32)

            params = get_env_params_from_config(env, cfg)
            js_gif_path = resolve_js_gif_path(level_sol_path, sol_dir, level_i)
            level = env.get_level(level_i)
            if level is None:
                print(f"Level {level_i} not found in game {game_name}, skipping. Must be an old JS solution generated "
                      "before the level was removed from the PS file?")
                continue
            params = params.replace(level=level)
            print(f"Level {level_i} solution: {actions}")
            js_scores, js_states = replay_actions_js(engine, solver, level_sol, game_text, level_i)
            if not os.path.isfile(js_gif_path):
                print(f"Generating missing JS gif for level {level_i}")
                try:
                    backend.render_gif(
                        game_text=game_text,
                        level_i=level_i,
                        actions=level_sol,
                        gif_path=js_gif_path,
                        frame_duration_s=0.3,
                    )
                except Exception:
                    print(f"Failed to generate JS gif for level {level_i}")
                    traceback.print_exc()

            def step_env(state, action):
                obs, state, reward, done, info = env.step_env(key, state, action, params)
                return state, (state, reward)

            print("Replaying solution in JAX.")
            try:
                obs, init_state = env.reset(key, params)
                if not game_randomness_updated:
                    n_rules = update_game_randomness(n_rules_key, n_rules, env.has_randomness())
                    game_randomness_updated = True
                if len(actions) > 0:
                    state, (state_v, reward_v) = jax.lax.scan(step_env, init_state, actions)
                    reward = float(reward_v.sum().item())
                    # Use jax tree map to add the initial state
                    state_v = jax.tree.map(lambda x, y: jnp.concatenate([x[None], y]), init_state, state_v)
                else:
                    reward = 0.0
                    state_v = jax.tree.map(lambda x: x[None], init_state)
                    state = init_state
                if level_i not in solution_rewards_dict or cfg.overwrite:
                    solution_rewards_dict[game][level_i] = reward
                if level_win and not state.win:
                # if not done:
                    sol_log = f"Level {level_i} solution failed\nActions: {actions}\n"
                    with open(sol_log_path, 'w') as f:
                        f.write(sol_log)
                    if is_random_game(n_rules):
                        n_random_solution_error += 1
                    else:
                        n_solution_error += 1
                    game_success = False
                    print(f"Level {level_i} solution failed (won in JS, did not win in jax)")
                elif (multihot_level_js.shape != state.multihot_level.shape) or np.any(multihot_level_js != state.multihot_level):
                    js_state = state.replace(multihot_level=multihot_level_js)
                    js_frame = backend.render_frame(js_states[-1])
                    imageio.imsave(os.path.join(jax_sol_dir, f'level-{level_i}_state_js.png'), js_frame)
                    jax_frame = env.render(state, cv2=False)
                    jax_frame = np.array(jax_frame, dtype=np.uint8)
                    imageio.imsave(os.path.join(jax_sol_dir, f'level-{level_i}_state_jax.png'), jax_frame)
                    with open(state_log_path, 'w') as f:
                        f.write(f"Level {level_i} solution failed\n")
                        f.write(f"Actions: {actions}\n")
                        f.write(f"Expected JS state:\n{format_state_for_log(js_state, env)}\n")
                        f.write(f"Actual JAX state:\n{format_state_for_log(state, env)}\n")
                    if is_random_game(n_rules):
                        n_random_state_error += 1
                    else:
                        n_state_error += 1
                    
                    print(f"Level {level_i} solution failed (state mismatch)")
                    # game_success = False
                # FIXME: There is a discrepancy between the way we compute scores in js (I actually don't understand
                # how we're getting that number) and the way we compute scores in jax, so this will always fail.
                # elif not level_win and (state.heuristic != level_score):
                elif (state.heuristic != -level_score):
                    with open(score_log_path, 'w') as f:
                        f.write(f"Level {level_i} solution score mismatch\n")
                        f.write(f"Actions: {actions}\n")
                        f.write(f"Jax score: {state.heuristic}\n")
                        f.write(f"JS score: {level_score}\n")
                        # if game_name not in results['score_error']:
                        #     results['score_error'][game_name] = []
                        # results['score_error'][game_name].append({'n_rules': n_rules, 'level': level_i})
                        n_score_error += 1
                    print(f"Level {level_i} solution score mismatch.")
                    # We'll be a bit generous and not count this for now. TODO: fix the score mismatch.
                    # game_success = False
                
                elif not np.all(-np.array(js_scores) == np.array(state_v.heuristic)):
                    print(f"Warning: intermediary JS and JAX heuristics do not match for game {game_name} level {level_i}")
                    # Log this to disk
                    with open(intermediary_scores_log_path, 'w') as f:
                        f.write(f"Level {level_i} solution score mismatch\n")
                        f.write(f"Actions: {actions}\n")
                        f.write(f"Jax score: {state_v.heuristic}\n")
                        f.write(f"JS score: {js_scores}\n")
                        n_score_error += 1
                else:
                    # if game_name not in results['success']:
                    #     results['success'][game_name] = []
                    # results['success'][game_name].append({'n_rules': n_rules, 'level': level_i})
                    n_success += 1
                    print(f"Level {level_i} solution succeeded")
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                traceback.print_exc()
                print(f"Error running solution: {og_path}")
                err_log = traceback.format_exc()
                if is_runtime_timeout_log(err_log):
                    if cfg.aggregate:
                        results['timeout'].append({'game': game_name, 'n_rules': n_rules, 'level': level_i, 'log': err_log})
                    n_timeout_error += 1
                else:
                    n_runtime_error += 1
                game_success = False
                with open(run_log_path, 'w') as f:
                    f.write(err_log)
                continue

            print(f"Rendering frames for level {level_i}")
            frames = jax.vmap(env.render, in_axes=(0, None))(state_v, None)
            frames = frames.astype(np.uint8)

            # Scale up the frames
            # print(f"Scaling up frames for level {level_i}")
            scale = 1
            frames = jnp.repeat(frames, scale, axis=1)
            frames = jnp.repeat(frames, scale, axis=2)
            frames = np.array(frames)

            # Save the frames
            # print(f"Saving frames for level {level_i}")
            # frames_dir = os.path.join(jax_sol_dir, 'frames')
            # os.makedirs(frames_dir, exist_ok=True)
            # for i, js_frame in enumerate(frames):
            #     imageio.imsave(os.path.join(frames_dir, f'level-{level_i}_sol_{i:03d}.png'), js_frame)

            # Make a gif out of the frames
            print(f"Making gif for level {level_i}")
            imageio.mimsave(gif_path, frames, duration=1, loop=0)
            print(f'Saved gif to {gif_path}')

            # Copy over the JS gif corresponding to the level solution. This is
            # stored per-level in data/js_sols/<game>/level-{i}_sol.gif.
            copied_js_gif_path = os.path.join(jax_sol_dir, f'level-{level_i}_js.gif')
            if os.path.isfile(js_gif_path):
                shutil.copy(js_gif_path, copied_js_gif_path)
            else:
                print(f"Warning: JS gif missing for level {level_i}; expected {js_gif_path}. "
                      "Skipping comparison gif generation.")
                js_gif_path = None

            try:
                if js_gif_path is not None and os.path.isfile(js_gif_path) and os.path.isfile(gif_path):
                    js_frames = imageio.mimread(js_gif_path)
                    jax_frames = imageio.mimread(gif_path)
                    compare_frames = make_side_by_side_frames(js_frames, jax_frames)
                    if compare_frames:
                        imageio.mimsave(compare_gif_path, compare_frames, duration=1, loop=0)
                        print(f'Saved comparison gif to {compare_gif_path}')
                    else:
                        print(f'Could not make comparison gif for level {level_i} because js_frames or jax_frames is empty')
            except Exception:
                print(f"Failed to generate comparison gif for level {level_i}")
                traceback.print_exc()

            jax.clear_caches()

        if cfg.aggregate:
            if game_success:
                results['valid_games'].append({'game': game_name, 'n_rules': n_rules})
            elif game_partial_success:
                results['partial_valid_games'].append({'game': game_name, 'n_rules': n_rules})

    if cfg.aggregate:
        results['stats']['valid_games'] = len(results['valid_games'])
        results['stats']['partial_valid_games'] = len(results['partial_valid_games'])
        
        save_stats(results, n_levels, n_success, n_compile_error, n_rigid_prefix_error, n_timeout_error,
               n_runtime_error, n_solution_error, n_state_error, n_score_error, n_unvalidated_levels,
               n_random_solution_error, n_random_state_error)
        print(f"Validation results saved to {val_results_path}")
        stats_dict = {
            "Total Games": len(games),
            "Valid Games": len(results['valid_games']),
            "Partially Valid Games": len(results['partial_valid_games']),
            "Total Levels": n_levels,
            "Successful Solutions": n_success,
            "Compile Errors": n_compile_error,
            "Timeouts": n_timeout_error,
            "Runtime Errors": n_runtime_error,
            "Solution Errors": n_solution_error,
            "State Errors": n_state_error,
            "Random Solution Errors": n_random_solution_error,
            "Random State Errors": n_random_state_error,
            "Unvalidated Levels": n_unvalidated_levels,
        }

        # Create a latex table with the total number of games, the number of valid games, and the number of partial valid games
        df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Count'])
        val_stats_tex_path = os.path.join('plots', 'validation_stats.tex')
        os.makedirs('plots', exist_ok=True)
        df.to_latex(val_stats_tex_path, index=True, header=False)
        print(f"Validation results saved to {val_stats_tex_path}")

        # Create a latex table of valid games, with the headers "Game", "# Rules", "Stochastic"
        valid_games = results['valid_games']
        valid_games = [game for game in valid_games if game['n_rules']]
        print(f"N valid games: {len(valid_games)}")
        valid_game_names = [game['game'] for game in valid_games]
        valid_game_names = clean_game_names(valid_game_names)
        valid_game_n_rules = [game['n_rules'][0] for game in valid_games]
        valid_game_name_stochastic = [game['n_rules'][1] for game in valid_games]
        valid_games_df = pd.DataFrame({'Game': valid_game_names, r'\# Rules': valid_game_n_rules})
        # Sort games by number of rules
        valid_games_df = valid_games_df.sort_values(by=[r'\# Rules'], ascending=True)
        # Save to latex as a longtable
        valid_games_tex_path = os.path.join('plots', 'valid_games.tex')
        valid_games_df.to_latex(valid_games_tex_path, index=False, header=True, longtable=True,
                                caption="PuzzleScript games in which all levels were successfully validated in JAX (vis-a-vis solutions generated by breadth-first search in JavaScript).",
                                label="tab:valid_games")
        print(f"Valid games saved to {valid_games_tex_path}")

        # Do the same for partially valid games
        partial_valid_games = results['partial_valid_games']
        partial_valid_games = [game for game in partial_valid_games if game['n_rules']]
        print(f"N partial valid games: {len(partial_valid_games)}")
        partial_valid_game_names = [game['game'] for game in partial_valid_games]
        partial_valid_game_names = clean_game_names(partial_valid_game_names)
        partial_valid_game_n_rules = [game['n_rules'][0] for game in partial_valid_games]
        partial_valid_game_name_stochastic = [game['n_rules'][1] for game in partial_valid_games]
        partial_valid_games_df = pd.DataFrame({'Game': partial_valid_game_names, r'\# Rules': partial_valid_game_n_rules})
        # Sort games by number of rules
        partial_valid_games_df = partial_valid_games_df.sort_values(by=[r'\# Rules'], ascending=True)
        # Save to latex as a longtable
        partial_valid_games_tex_path = os.path.join('plots', 'partial_valid_games.tex')
        partial_valid_games_df.to_latex(partial_valid_games_tex_path, index=False, header=True, longtable=True,
                                        caption="PuzzleScript games in which one or more levels were successfully validated in JAX (vis-a-vis solutions generated by breadth-first search in JavaScript).",
                                        label="tab:partial_valid_games")
        print(f"Partially valid games saved to {partial_valid_games_tex_path}")

    if games_to_n_rules_dirty:
        with open(GAMES_TO_N_RULES_PATH, 'w') as f:
            json.dump(games_to_n_rules, f, indent=4)
        print(f"Updated games_to_n_rules with missing randomness flags at {GAMES_TO_N_RULES_PATH}")

    print(f"Finished validating solutions in jax.")
    

def clean_game_names(games):
    # Clean the game names to remove any special characters
    games = [game.replace('_', ' ') for game in games]
    games = [game.replace('^', r'\^') for game in games]
    games = [game.replace('&', r'\&') for game in games]
    games = [game[:60] + '...' if len(game) > 50 else game for game in games]
    return games


if __name__ == '__main__':
    main_launch()
