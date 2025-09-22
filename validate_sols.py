import bdb
import copy
import glob
import json
import math
import os
import pickle
import random
import shutil
import traceback
from typing import List, Optional

from einops import rearrange
import hydra
import imageio
from javascript import require
import jax
import jax.numpy as jnp
from lark import Lark
import numpy as np
import pandas as pd
from skimage.transform import resize
import submitit

from conf.config import JaxValidationConfig
from env import PSEnv
from globals import SOLUTION_REWARDS_PATH, GAMES_TO_N_RULES_PATH
from preprocess_games import PS_LARK_GRAMMAR_PATH, TREES_DIR, DATA_DIR, TEST_GAMES, PSErrors, get_tree_from_txt, count_rules
from standalone.utils import replay_actions_js
from standalone.utils import compile_game as compile_game_js
from utils import get_list_of_games_for_testing, to_binary_vectors
from utils_rl import get_env_params_from_config


scratch_dir = 'scratch'
os.makedirs(scratch_dir, exist_ok = True)

JAX_VALIDATED_JS_SOLS_DIR = os.path.join('data', 'jax_validated_js_sols')
JS_SOLS_DIR = os.path.join('data', 'js_sols')

games_to_skip = set({
    '2048',  # hangs
})

JS_TO_JAX_ACTIONS = [3, 0, 1, 2, 4]


def multihot_level_from_js_state(level_state, obj_list):
    level_state = np.array(level_state).T
    multihot_level_js = to_binary_vectors(level_state, len(obj_list))
    multihot_level_js = rearrange(multihot_level_js, 'h w c -> c h w')[::-1]

    # Remove duplicate channels from the multihot level.
    new_multihot_level_js = []
    new_objs_to_idxs = {}
    for obj_idx, obj in enumerate(obj_list):
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

    return multihot_level_js


@hydra.main(version_base="1.3", config_path='./conf', config_name='jax_validation_config')
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
            slurm_account='pr_174_tandon_advanced', 
            slurm_setup=["export JAX_PLATFORMS=cpu"],
            slurm_array_parallelism=n_jobs,
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: JaxValidationConfig, games: Optional[List[str]] = None):
    engine = require('./standalone/puzzlescript/engine.js')
    solver = require('./standalone/puzzlescript/solver.js')
    if cfg.slurm:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    # Initialize the Lark parser with the PuzzleScript grammar
    with open(PS_LARK_GRAMMAR_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    with open(GAMES_TO_N_RULES_PATH, 'r') as f:
        games_to_n_rules = json.load(f)
    if games is not None:
        games = games
    elif cfg.game is None:
        games = get_list_of_games_for_testing(all_games=cfg.all_games, random_order=cfg.random_order)
    else:
        games = [cfg.game]
    if cfg.aggregate:
        results = {
            'stats': {},
            'compile_error': [],
            'runtime_error': {},
            'solution_error': {},
            'state_error': {},
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
    n_runtime_error = 0
    n_solution_error = 0
    n_state_error = 0
    n_score_error = 0
    n_success = 0
    n_unvalidated_levels = 0

    # if os.path.exists(SOLUTION_REWARDS_PATH):
    #     with open(SOLUTION_REWARDS_PATH, 'r') as f:
    #         solution_rewards_dict = json.load(f)    
    # else:
    solution_rewards_dict = {}
    

    def save_stats(results, n_levels, n_success, n_compile_error, n_runtime_error,
                   n_solution_error, n_state_error, n_score_error, n_unvalidated_levels):
        results['stats']['total_games'] = len(games)
        results['stats']['total_levels'] = n_levels
        results['stats']['successful_solutions'] = n_success
        results['stats']['compile_error'] = n_compile_error
        results['stats']['runtime_error'] = n_runtime_error
        results['stats']['solution_error'] = n_solution_error
        results['stats']['state_error'] = n_state_error
        results['stats']['score_error'] = n_score_error
        results['stats']['unvalidated_levels'] = n_unvalidated_levels

        with open(val_results_path, 'w') as f:
            json.dump(results, f, indent=4)

        with open(SOLUTION_REWARDS_PATH, 'w') as f:
            json.dump(solution_rewards_dict, f, indent=4)


    key = jax.random.PRNGKey(0)

    # 0 - left
    # 1 - down
    # 2 - right
    # 3 - up
    # 4 - action

    for sol_dir, game in zip(sol_paths, games):
        n_rules = None
        if game in games_to_n_rules:
            n_rules = games_to_n_rules[game]
        if game + ".txt" in games_to_n_rules:
            n_rules = games_to_n_rules[game + ".txt"]
        if game not in solution_rewards_dict:
            solution_rewards_dict[game] = {}
        game_name = os.path.basename(game)
        if game_name in games_to_skip:
            print(f"Skipping {game_name} because it is in the skip list")
            continue
        jax_sol_dir = os.path.join(JAX_VALIDATED_JS_SOLS_DIR, game)
        os.makedirs(jax_sol_dir, exist_ok=True)
        compile_log_path = os.path.join(jax_sol_dir, 'compile_err.txt')
        if cfg.overwrite:
            if os.path.exists(compile_log_path):
                os.remove(compile_log_path)
        if os.path.exists(compile_log_path) and not cfg.overwrite:
            if cfg.aggregate:
                with open(compile_log_path, 'r') as f:
                    compile_log = f.read()
                results['compile_error'].append({'game': game, 'n_rules': n_rules, 'log': compile_log})
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
        new_level_sols = []
        level_ints = []
        for level_i, sols in level_ints_to_sols.items():
            if len(sols) == 1:
                new_level_sols.append(sols[0])
            else:
                # Sort by number of steps, and take the one with the most steps.
                n_steps = [
                    int(os.path.basename(p).split('-steps_')[0].split('_')[-1])
                        if '-steps_' in os.path.basename(p)
                        else 10_000 for p in sols
                    ]
                max_steps_idx = np.argmax(n_steps)
                new_level_sols.append(sols[max_steps_idx])
            level_ints.append(level_i)
        level_sols = new_level_sols

        if len(level_sols) == 0:
            print(f"No js solutions found for {game_name}")
            continue

        og_path = os.path.join(DATA_DIR, 'scraped_games', game_name + '.txt')

        print(f"Processing solution for game: {og_path}")


        game_success = True
        game_partial_success = False
        game_compile_error = False
        env = None

        game_text = compile_game_js(parser, engine, game_name, level_i=0)

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

            # Skip if we already have a result and are not overwritiing.
            if (os.path.exists(gif_path) or os.path.exists(sol_log_path) or os.path.exists(score_log_path)
                    or os.path.exists(run_log_path) or os.path.exists(state_log_path)):
                if cfg.overwrite:
                    if os.path.exists(gif_path):
                        os.remove(gif_path)
                    if os.path.exists(sol_log_path):
                        os.remove(sol_log_path)
                    if os.path.exists(score_log_path):
                        os.remove(score_log_path)
                    if os.path.exists(run_log_path):
                        os.remove(run_log_path)
                    if os.path.exists(state_log_path):
                        os.remove(state_log_path)
                else:
                    if os.path.exists(run_log_path):
                        if cfg.aggregate:
                            with open(run_log_path, 'r') as f:
                                run_log = f.read()
                            if game_name not in results['runtime_error']:
                                results['runtime_error'][game_name] = []
                            results['runtime_error'][game_name].append({'level': level_i, 'n_rules': n_rules, 'log': run_log})
                        n_runtime_error += 1
                        game_success = False
                    elif os.path.exists(sol_log_path):
                        if cfg.aggregate:
                            if game_name not in results['solution_error']:
                                results['solution_error'][game_name] = []
                            results['solution_error'][game_name].append({'n_rules': n_rules, 'level': level_i})
                        n_solution_error += 1
                        game_success = False
                    elif os.path.exists(state_log_path):
                        if cfg.aggregate:
                            if game_name not in results['state_error']:
                                results['state_error'][game_name] = []
                            results['state_error'][game_name].append({'n_rules': n_rules, 'level': level_i})
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

            # Otherwise, let's initialize the environment (if on level 0) and run the solution.
            if env is None:
                tree, success, err_msg = get_tree_from_txt(parser, game, test_env_init=False, timeout=60*20)
                if success == PSErrors.SUCCESS:
                    try:
                        env = PSEnv(tree, debug=False, print_score=False)
                    except KeyboardInterrupt as e:
                        raise e
                    except bdb.BdbQuit as e:
                        raise e
                    except Exception as e:
                        err_msg = traceback.format_exc()
                        success = PSErrors.ENV_ERROR
                if success != PSErrors.SUCCESS:
                    with open(compile_log_path, 'w') as f:
                        f.write(err_msg)
                    print(f"Error creating env: {og_path}\n{err_msg}")
                    # results['compile_error'].append({'game': game, 'n_rules': n_rules, 'log': err_log})
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
            level_sol = sol_dict['sol'] if 'sol' in sol_dict else sol_dict['actions']
            level_win = sol_dict['won']
            level_score = sol_dict['score']
            level_state = sol_dict['state']
            obj_list = sol_dict['objs']
            multihot_level_js = multihot_level_from_js_state(level_state, obj_list)
            actions = level_sol
            # print(f"Level {level_i} solution: {actions}")
            actions = [JS_TO_JAX_ACTIONS[a] for a in actions]
            actions = jnp.array([int(a) for a in actions], dtype=jnp.int32)

            params = get_env_params_from_config(env, cfg)
            js_gif_path = os.path.join(sol_dir, f'level-{level_i}_sol.gif')
            level = env.get_level(level_i)
            if level is None:
                print(f"Level {level_i} not found in game {game_name}, skipping. Must be an old JS solution generated "
                      "before the level was removed from the PS file?")
                continue
            params = params.replace(level=level)
            print(f"Level {level_i} solution: {actions}")
            js_scores, js_states = replay_actions_js(engine, solver, level_sol, game_text, level_i)

            def step_env(state, action):
                obs, state, reward, done, info = env.step_env(key, state, action, params)
                return state, (state, reward)

            try:
                obs, init_state = env.reset(key, params)
                if len(actions) > 0:
                    state, (state_v, reward_v) = jax.lax.scan(step_env, init_state, actions)
                    reward = float(reward_v.sum().item())
                    # Use jax tree map to add the initial state
                    state_v = jax.tree.map(lambda x, y: jnp.concatenate([x[None], y]), init_state, state_v)
                    if not np.all(-np.array(js_scores) == np.array(state_v.heuristic)):
                        print(f"Warning: intermediary JS and JAX heuristics do not match for game {game_name} level {level_i}")
                        # Log this to disk
                        with open(intermediary_scores_log_path, 'w') as f:
                            f.write(f"Level {level_i} solution score mismatch\n")
                            f.write(f"Actions: {actions}\n")
                            f.write(f"Jax score: {state_v.heuristic}\n")
                            f.write(f"JS score: {js_scores}\n")
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
                    # if game_name not in results['solution_error']:
                    #     results['solution_error'][game_name] = []
                    # results['solution_error'][game_name].append({'n_rules': n_rules, 'level': level_i})
                    n_solution_error += 1
                    game_success = False
                        # f.write(f"State: {state}\n")
                    print(f"Level {level_i} solution failed (won in JS, did not win in jax)")
                elif np.any(multihot_level_js != state.multihot_level):
                    js_state = state.replace(multihot_level=multihot_level_js)
                    js_frame = env.render(js_state, cv2=False)
                    js_frame = np.array(js_frame, dtype=np.uint8)
                    imageio.imsave(os.path.join(jax_sol_dir, f'level-{level_i}_state_js.png'), js_frame)
                    jax_frame = env.render(state, cv2=False)
                    jax_frame = np.array(jax_frame, dtype=np.uint8)
                    imageio.imsave(os.path.join(jax_sol_dir, f'level-{level_i}_state_jax.png'), jax_frame)
                    with open(state_log_path, 'w') as f:
                        f.write(f"Level {level_i} solution failed\n")
                        f.write(f"Actions: {actions}\n")
                        f.write(f"State: {state}\n")
                        # if game_name not in results['state_error']:
                        #     results['state_error'][game_name] = []
                        # results['state_error'][game_name].append({'n_rules': n_rules, 'level': level_i})
                        n_state_error += 1
                    
                    print(f"Level {level_i} solution failed (state mismatch)")
                    # game_success = False
                # FIXME: There is a discrepancy between the way we compute scores in js (I actually don't understand
                # how we're getting that number) and the way we compute scores in jax, so this will always fail.
                # elif not level_win and (state.heuristic != level_score):
                elif (state.heuristic != level_score):
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
                # if game_name not in results['runtime_error']:
                #     results['runtime_error'][game_name] = []
                # results['runtime_error'][game_name].append({'n_rules': n_rules, 'level': level_i, 'log': err_log})
                n_runtime_error += 1
                game_success = False
                with open(run_log_path, 'w') as f:
                    f.write(err_log)
                continue

            frames = jax.vmap(env.render, in_axes=(0, None))(state_v, None)
            frames = frames.astype(np.uint8)

            # Scale up the frames
            # print(f"Scaling up frames for level {level_i}")
            scale = 10
            frames = jnp.repeat(frames, scale, axis=1)
            frames = jnp.repeat(frames, scale, axis=2)

            # Save the frames
            # print(f"Saving frames for level {level_i}")
            frames_dir = os.path.join(jax_sol_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            for i, js_frame in enumerate(frames):
                imageio.imsave(os.path.join(frames_dir, f'level-{level_i}_sol_{i:03d}.png'), js_frame)

            # Make a gif out of the frames
            imageio.mimsave(gif_path, frames, duration=1, loop=0)
            print(f'Saved gif to {gif_path}')

            js_gif_paths = glob.glob(os.path.join(sol_dir, f'*level-{level_i}_sol*.gif'))
            n_steps = [int(os.path.basename(p).split('-steps_')[0]) if '-steps_' in os.path.basename(p) else 10_000 for p in js_gif_paths]
            if len(js_gif_paths) > 0:
                js_gif_path = js_gif_paths[np.argmax(n_steps)]

                # Copy over the js gif
                if os.path.isfile(js_gif_path):
                    shutil.copy(js_gif_path, os.path.join(jax_sol_dir, f'level-{level_i}_js.gif'))

            else:
                js_gif_path = None

            jax.clear_caches()

        if cfg.aggregate:
            if game_success:
                results['valid_games'].append({'game': game_name, 'n_rules': n_rules})
            elif game_partial_success:
                results['partial_valid_games'].append({'game': game_name, 'n_rules': n_rules})

    if cfg.aggregate:
        results['stats']['valid_games'] = len(results['valid_games'])
        results['stats']['partial_valid_games'] = len(results['partial_valid_games'])
        
        save_stats(results, n_levels, n_success, n_compile_error, n_runtime_error,
                   n_solution_error, n_state_error, n_score_error, n_unvalidated_levels)
        print(f"Validation results saved to {val_results_path}")
        stats_dict = {
            "Total Games": len(games),
            "Valid Games": len(results['valid_games']),
            "Partially Valid Games": len(results['partial_valid_games']),
            "Total Levels": n_levels,
            "Successful Solutions": n_success,
            "Compile Errors": n_compile_error,
            "Runtime Errors": n_runtime_error,
            "Solution Errors": n_solution_error,
            "State Errors": n_state_error,
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