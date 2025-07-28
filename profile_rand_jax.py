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
from env import PSState
from utils import get_list_of_games_for_testing, load_games_n_rules_sorted
from utils_rl import get_env_params_from_config, init_ps_env


# game_paths = glob.glob(os.path.join('data', 'scraped_games', '*.txt'))
# games = [os.path.basename(p) for p in game_paths]

batch_sizes = [
    1,
    10,
    50,
    100,
    200,
    400,
    600,
    1_200,
    1_500,
    1_800,
]
batch_sizes = batch_sizes[::-1]

JAX_N_ENVS_TO_FPS_PATH = os.path.join('data', 'jax_n_envs_to_fps.json')


def get_step_str(s):
    return f'{s}-step_rollout'

def get_step_int(step_str):
    return int(step_str.split('-')[0])

def get_level_str(level_i):
    return f'level-{level_i}'

def get_level_int(level_str):
    return int(level_str.split('-')[1])

def save_results(results):
    with open(JAX_N_ENVS_TO_FPS_PATH, 'w') as f:
        json.dump(results, f, indent=4)

@hydra.main(version_base="1.3", config_path='./conf', config_name='profile_jax')
def profile(cfg: ProfileJaxRandConfig):
    logging.getLogger().setLevel(logging.WARNING)
    devices = jax.devices()
    assert len(devices) == 1, f'JAX is not using a single device. Found {len(devices)} devices: {devices}. This is unexpected.'
    device_name = devices[0].device_kind

    if cfg.game is None:
        games = get_list_of_games_for_testing(all_games=cfg.all_games)
    else:
        games = [cfg.game]

    global batch_sizes
    hparams = itertools.product(
        games,
        batch_sizes,
    )
    games, batch_sizes = zip(*hparams)

    vids_dir = 'vids'
    if cfg.render:
        os.makedirs(vids_dir, exist_ok=True)

    # if config.overwrite:

    rng = jax.random.PRNGKey(42)
    step_str = get_step_str(cfg.n_steps)


    if not os.path.isfile(JAX_N_ENVS_TO_FPS_PATH):
        results = {}
        game_n_envs_to_fps = {}
        step_str = get_step_str(cfg.n_steps)
        results[device_name] = {}
        results[device_name][step_str] = game_n_envs_to_fps
    else:
        with open(JAX_N_ENVS_TO_FPS_PATH, 'r') as f:
            results = json.load(f)
        if device_name not in results:
            results[device_name] = {}
        if step_str not in results[device_name]:
            results[device_name][step_str] = {}
        game_n_envs_to_fps = results[device_name][step_str]

    for (game, n_envs) in zip(games, batch_sizes):

        # if n_envs >= 1_200 and game in ['limerick']:
        #     continue

        cfg.game = game

        print(f'\nGame: {game}, n_envs: {n_envs}.')
        env = init_ps_env(cfg)

        # for level_i in range(len(env.levels[:1])):
        for level_i in range(len(env.levels)):

            # jax.clear_caches()
            level_str = get_level_str(level_i)
            if game not in game_n_envs_to_fps:
                game_n_envs_to_fps[game] = {}

            if level_str not in game_n_envs_to_fps[game]:
                game_n_envs_to_fps[game][level_str] = {}

            if not cfg.overwrite and str(n_envs) in game_n_envs_to_fps[game][level_str]:
                continue

            env_params = get_env_params_from_config(env, cfg)

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
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                carry = (env_state, rng)
                return carry, None

            _env_step_jitted = jax.jit(_env_step)

            start = timer()

            try:
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
                carry = (env_state, rng)

                carry, _ = _env_step_jitted(carry, None)
                print(f'Finished 1st step in {(timer() - start)} seconds.')
                
                start = timer()
                carry, _ = _env_step_jitted(carry, None)
                print(f'Finished 2nd step in {(timer() - start)} seconds.')

                start = timer()
                # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                carry, _ = _env_step_jitted(carry, None)
                # carry[0].multihot_level.block_until_ready()
                print(f'Finished 3rd step in {(timer() - start)} seconds.')

                n_env_steps = cfg.n_steps * n_envs
                times = []
                for i in range(3):
                    start = timer()
                    # carry, env_states = jax.lax.scan(
                    carry, _ = jax.lax.scan(
                        _env_step_jitted, carry, None, cfg.n_steps
                    )
                    env_state: PSState = carry[0]
                    # Otherwise, when running on CPU, the state may not be ready yet
                    env_state.multihot_level.block_until_ready()
                    times.append(timer() - start)
                    print(f'Loop {i} ran {n_env_steps} steps in {times[-1]} seconds. FPS: {n_env_steps / times[-1]:,.2f}')

            except jaxlib.xla_extension.XlaRuntimeError as e:
                err_msg = traceback.format_exc()
                print(f'Error in first step: {err_msg}')
                results[device_name][step_str][game][level_str][n_envs] = None
                save_results(results)
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

            game_n_envs_to_fps[game][level_str][n_envs] = fpss
            save_results(results)

    else:
        # Load from json
        with open(JAX_N_ENVS_TO_FPS_PATH, 'r') as f:
            results = json.load(f)

if __name__ == '__main__':
    # with jax.numpy_dtype_promotion('strict'):
    profile() 
