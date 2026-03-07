"""Profile environment speed while taking random actions."""
import functools
import glob
import itertools
import logging
import os
import traceback

import hydra
import imageio
import jax
from jax.experimental import profiler
import json
import jaxlib
import numpy as np
import pandas as pd
from timeit import default_timer as timer

from conf.config import ProfileJaxRandConfig
from puzzlejax.env import PJState
from puzzlejax.env_switch import PuzzleJaxEnvSwitch
from puzzlejax.globals import JAX_PROFILING_RESULTS_DIR
from puzzlejax.utils import get_list_of_games_for_testing, load_games_n_rules_sorted, init_ps_lark_parser, get_tree_from_txt
from utils_rl import get_env_params_from_config, init_ps_env


# game_paths = glob.glob(os.path.join('data', 'scraped_games', '*.txt'))
# games = [os.path.basename(p) for p in game_paths]

BATCH_SIZES = [
    1,
    10,
    # 50,
    100,
    # 200,
    # 400,
    # 600,
    1_000,
    # 1_200,
    # 1_500,
    # 1_800,
    # 2_000,
    # 3_500,
    # 5_000,
    # 7_500,
    # 8_000,
    10_000,
    # 20_000,
    100_000,
]
# batch_sizes = batch_sizes[::-1]
VMAPS = [
    True,
    # False,
]

def get_step_str(s):
    return f'{s}-step_rollout'

def get_step_int(step_str):
    return int(step_str.split('-')[0])

def get_level_str(level_i, vmap):
    # return f'level-{level_i}-vmap-{vmap}'
    if vmap:
        # For backward compatibility TODO: Clean this up (as above)
        return f'level-{level_i}'
    else:
        return f'level-{level_i}-vmap-{vmap}'

def get_level_int(level_str):
    return int(level_str.split('-')[1])

def get_vmap(level_str):
    # return level_str.split('-')[-1]
    # For backward compatibility. TODO: Clean this up (as above)
    return 'vmap-False' not in level_str

def save_results(results, results_path):
    results_dir = os.path.dirname(results_path)
    os.makedirs(results_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

@hydra.main(version_base="1.3", config_path='./conf', config_name='profile_jax')
def profile(cfg: ProfileJaxRandConfig):
    logging.getLogger().setLevel(logging.WARNING)
    devices = jax.devices()
    assert len(devices) == 1, f'JAX is not using a single device. Found {len(devices)} devices: {devices}. This is unexpected.'
    device_name = devices[0].device_kind
    device_name = device_name.replace(' ', '_')

    if cfg.game is None:
        games = get_list_of_games_for_testing(all_games=cfg.all_games)
    else:
        games = [cfg.game]

    global BATCH_SIZES, VMAPS
    hparams = itertools.product(
        games,
        BATCH_SIZES,
        VMAPS,
    )
    games, BATCH_SIZES, VMAPS = zip(*hparams)

    vids_dir = 'vids'
    if cfg.render:
        os.makedirs(vids_dir, exist_ok=True)

    # if config.overwrite:

    rng = jax.random.PRNGKey(42)
    step_str = get_step_str(cfg.n_steps)

    device_dir = os.path.join(JAX_PROFILING_RESULTS_DIR, device_name)
    steps_dir = os.path.join(device_dir, step_str)
    env_type_str = 'switch' if cfg.use_switch_env else 'standard'
    env_dir = os.path.join(steps_dir, env_type_str)
    os.makedirs(env_dir, exist_ok=True)
    last_game = None
    compiled_env_cache = {}

    for (game, n_envs, vmap) in zip(games, BATCH_SIZES, VMAPS):

        cfg.game = game
        cfg.vmap = vmap

        print(f'\nGame: {game}, n_envs: {n_envs}, vmap: {vmap}.')

        # Only profiling the first level for now.
        for level_i in range(1):
        # for level_i in range(len(env.levels)):

            level_str = get_level_str(level_i, vmap=vmap)
            results_path = os.path.join(env_dir, game, level_str + '.json')
            if os.path.exists(results_path):
                n_envs_to_fps = json.load(open(results_path, 'r'))
            else:
                n_envs_to_fps = {}

            if str(n_envs) in n_envs_to_fps and not cfg.overwrite:
                print(f'Skipping {game} level {level_i} with n_envs={n_envs} vmap={vmap} as results already exists.')
                continue

            if last_game != game:
                if cfg.use_switch_env:
                    parser = init_ps_lark_parser()
                    tree, success, err_msg = get_tree_from_txt(parser, cfg.game, test_env_init=False)
                    env = PuzzleJaxEnvSwitch(
                        tree, jit=True, level_i=cfg.level, max_steps=cfg.max_episode_steps,
                        print_score=False, debug=False, vmap=cfg.vmap,
                    )
                    print(f'  Using switch-based env (PuzzleJaxEnvSwitch)')
                else:
                    env = init_ps_env(cfg)
                    print(f'  Using standard env (PuzzleJaxEnv)')

            last_game = game

            # jax.clear_caches()

            env_params = get_env_params_from_config(env, cfg)
            cache_key = (game, level_i, vmap, cfg.use_switch_env)
            cached_env = compiled_env_cache.get(cache_key)
            if cached_env is None:
                print('  Compiling single-environment reset/step for cache reuse across n_envs...')
                single_env_reset = jax.jit(functools.partial(env.reset, params=env_params))
                single_env_step = jax.jit(functools.partial(env.step, params=env_params))

                rng, reset_compile_rng = jax.random.split(rng)
                _, sample_env_state = single_env_reset(reset_compile_rng)
                sample_env_state.multihot_level.block_until_ready()

                rng, step_compile_rng = jax.random.split(rng)
                env_compile_start = timer()
                _, sample_env_state, _, _, _ = single_env_step(step_compile_rng, sample_env_state, 0)
                sample_env_state.multihot_level.block_until_ready()
                env_compile_time = timer() - env_compile_start
                print(f'  Finished single-environment compile in {env_compile_time:.3f} seconds.')

                cached_env = {
                    'reset': single_env_reset,
                    'step': single_env_step,
                    'env_compile_time': env_compile_time,
                }
                compiled_env_cache[cache_key] = cached_env
                env_compile_reused = False
            else:
                print('  Reusing cached single-environment compilation.')
                env_compile_reused = True

            single_env_reset = cached_env['reset']
            single_env_step = cached_env['step']
            env_compile_time = 0.0 if env_compile_reused else cached_env['env_compile_time']
            batched_reset = jax.vmap(single_env_reset, in_axes=(0,))
            batched_step = jax.vmap(single_env_step, in_axes=(0, 0, 0))

            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, n_envs)

            def _env_step(carry, unused):
                env_state, rng = carry
                rng, _rng = jax.random.split(rng)
                rand_act = jax.random.randint(_rng, (n_envs,), 0, env.action_space.n)
                action = rand_act

                # STEP ENV
                rng_step = jax.random.split(_rng, n_envs)
                obsv, env_state, reward, done, info = batched_step(rng_step, env_state, action)
                carry = (env_state, rng)
                return carry, None

            _env_step_jitted = jax.jit(_env_step)

            try:
                # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                obsv, env_state = batched_reset(reset_rng)
                carry = (env_state, rng)
                # jax.config.update('jax_log_compiles', True)

                compile_start = timer()
                carry, _ = _env_step_jitted(carry, None)
                carry[0].multihot_level.block_until_ready()
                compile_time = timer() - compile_start
                print(f'Finished 1st step (compile + execute) in {compile_time:.3f} seconds.')
                
                start = timer()
                carry, _ = _env_step_jitted(carry, None)
                carry[0].multihot_level.block_until_ready()
                exec_time_2nd = timer() - start
                print(f'Finished 2nd step (execute only) in {exec_time_2nd:.3f} seconds.')
                batch_compile_time_est = compile_time - exec_time_2nd
                total_compile_time_est = env_compile_time + batch_compile_time_est
                print(f'Estimated batched-wrapper compile time: {batch_compile_time_est:.3f} seconds.')
                print(f'Estimated total incremental compile time: {total_compile_time_est:.3f} seconds.')

                n_env_steps = cfg.n_steps * n_envs
                times = []
                for i in range(3):
                    start = timer()
                    # carry, env_states = jax.lax.scan(
                    carry, _ = jax.lax.scan(
                        _env_step_jitted, carry, None, cfg.n_steps
                    )
                    env_state: PJState = carry[0]
                    # Otherwise, when running on CPU, the state may not be ready yet
                    env_state.multihot_level.block_until_ready()
                    times.append(timer() - start)
                    print(f'Loop {i} ran {n_env_steps} steps in {times[-1]} seconds. FPS: {n_env_steps / times[-1]:,.2f}')

            except Exception as e:
                err_msg = traceback.format_exc()
                print(f'Error in first step: {err_msg}')
                n_envs_to_fps[str(n_envs)] = {
                    'error': str(e),
                    'error_traceback': err_msg,
                }
                save_results(n_envs_to_fps, results_path)
                continue

            fpss = tuple(n_env_steps / np.array(times))
            # print(f'Finished {n_env_steps} steps in {end - start} seconds.')
            # print(f'Average steps per second: {fps}')

            # if cfg.render:
            #     print('Rendering gif...')
            #     start = timer()
            #     env_states_0 = jax.tree.map(lambda x: x[:, 0], env_states)
            #     frames = jax.vmap(env.render, in_axes=(0,))(env_states_0)
            #     print(f'Finished rendering frames in {timer() - start} seconds.')
            #     start = timer()
            #     gif_path = os.path.join(vids_dir, f'{game}_{n_envs}_randAct.gif')
            #     imageio.mimsave(gif_path, frames, duration=cfg.gif_frame_duration)
            #     print(f'Finished saving gif in {timer() - start} seconds.')

            n_envs_to_fps[str(n_envs)] = {
                'fps': fpss,
                'compile_time': compile_time,
                'compile_time_est': total_compile_time_est,
                'batch_compile_time_est': batch_compile_time_est,
                'env_compile_time': env_compile_time,
                'env_compile_reused': env_compile_reused,
                'use_switch_env': cfg.use_switch_env,
            }
            save_results(n_envs_to_fps, results_path)

    else:
        # Load from json
        with open(results_path, 'r') as f:
            results = json.load(f)

if __name__ == '__main__':
    # with jax.numpy_dtype_promotion('strict'):
    profile() 
