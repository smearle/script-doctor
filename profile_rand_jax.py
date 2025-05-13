"""Profile environment speed while taking random actions."""
import functools
import glob
import itertools
import os
import hydra
import imageio
import jax
import json
import pandas as pd
from timeit import default_timer as timer

from conf.config import ProfileJaxRandConfig
from env import PSState
from utils import get_list_of_games_for_testing, load_games_n_rules_sorted
from utils_rl import get_env_params_from_config, init_ps_env


# game_paths = glob.glob(os.path.join('data', 'scraped_games', '*.txt'))
# games = [os.path.basename(p) for p in game_paths]

batch_sizes = [
    1_200,
    600,
    400,
    200,
    100,
    50,
    10,
    1,
]

JAX_N_ENVS_TO_FPS_PATH = os.path.join('data', 'jax_n_envs_to_fps.json')


@hydra.main(version_base="1.3", config_path='./conf', config_name='profile_jax')
def profile(cfg: ProfileJaxRandConfig):
    devices = jax.devices()
    assert len(devices) == 1, f'JAX is not using a single device. Found {len(devices)} devices: {devices}. This is unexpected.'
    device_name = devices[0].device_kind

    games = get_list_of_games_for_testing(all_games=cfg.all_games)

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

    results = {}
    game_n_envs_to_fps = {}
    if device_name not in results:
        results[device_name] = {}
    results[device_name][f'{cfg.n_profile_steps}-step_rollout'] = game_n_envs_to_fps

    for (game, n_envs) in zip(games, batch_sizes):
        cfg.game = game

        print(f'\nGame: {game}, n_envs: {n_envs}.')
        start = timer()
        env = init_ps_env(cfg)

        for level_i in range(len(env.levels[:-1])):

            if not cfg.overwrite and game in game_n_envs_to_fps and level_i in game_n_envs_to_fps[game]:
                continue

            env_params = get_env_params_from_config(env, cfg)

            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, n_envs)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

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
                return carry, env_state

            _env_step_jitted = jax.jit(_env_step)

            print(f'Initialized and reset jitted PSEnv in {(timer() - start)} seconds.')

            start = timer()
            carry = (env_state, rng)
            carry, _ = _env_step_jitted(carry, None)
            print(f'Finished 1st step in {(timer() - start)} seconds.')

            start = timer()
            carry, _ = _env_step_jitted(carry, None)
            print(f'Finished 2nd step in {(timer() - start)} seconds.')

            start = timer()
            carry, _ = _env_step_jitted(carry, None)
            print(f'Finished 3rd step in {(timer() - start)} seconds.')

            start = timer()
            carry, env_states = jax.lax.scan(
                _env_step_jitted, carry, None, cfg.n_profile_steps
            )

            n_env_steps = cfg.n_profile_steps * n_envs

            end = timer()
            fps = n_env_steps / (end - start)
            print(f'Finished {n_env_steps} steps in {end - start} seconds.')
            print(f'Average steps per second: {fps}')

            if cfg.render:
                print('Rendering gif...')
                start = timer()
                env_states_0 = jax.tree.map(lambda x: x[:, 0], env_states)
                frames = jax.vmap(env.render, in_axes=(0,))(env_states_0)
                print(f'Finished rendering frames in {timer() - start} seconds.')
                start = timer()
                gif_path = os.path.join(vids_dir, f'{game}_{n_envs}_randAct.gif')
                imageio.mimsave(gif_path, frames, duration=cfg.gif_frame_duration)
                print(f'Finished saving gif in {timer() - start} seconds.')

            if game not in game_n_envs_to_fps:
                game_n_envs_to_fps[game] = {}
            level_str = f'level-{level_i}'
            if level_str not in game_n_envs_to_fps[game]:
                game_n_envs_to_fps[game][level_str] = {}
            game_n_envs_to_fps[game][level_str][n_envs] = fps

            # Save as json
            with open(JAX_N_ENVS_TO_FPS_PATH, 'w') as f:
                json.dump(results, f, indent=4)

    else:
        # Load from json
        with open(JAX_N_ENVS_TO_FPS_PATH, 'r') as f:
            results = json.load(f)

if __name__ == '__main__':
    profile() 
