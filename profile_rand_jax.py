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

from conf.config import ProfileEnvConfig
from env import PSState
from utils import load_games_n_rules_sorted
from utils_rl import get_env_params_from_config, init_ps_env


priority_games = [
    # 'castlemouse',
    # 'atlas shrank',
    'sokoban_basic',
    'sokoban_match3',
    'limerick',
    'slidings',
    'tiny treasure hunt',
    'test',
]
# game_paths = glob.glob(os.path.join('data', 'scraped_games', '*.txt'))
# games = [os.path.basename(p) for p in game_paths]

batch_sizes = [
    600,
    400,
    200,
    1,
    10,
    50,
    100,
]

JAX_N_ENVS_TO_FPS_PATH = os.path.join('data', 'jax_n_envs_to_fps.json')


@hydra.main(version_base=None, config_path='./', config_name='profile_pcgrl')
def profile(config: ProfileEnvConfig):
    if config.all_games:
        games_n_rules = load_games_n_rules_sorted()
        games = [game for game, n_rules, has_randomness in games_n_rules]
        games = priority_games + [game for game in games if game not in priority_games]
    else:
        games = priority_games

    global batch_sizes
    hparams = itertools.product(
        batch_sizes,
        games,
    )
    batch_sizes, games = zip(*hparams)

    vids_dir = 'vids'
    if config.render:
        os.makedirs(vids_dir, exist_ok=True)

    if config.reevaluate:

        rng = jax.random.PRNGKey(42)

        results = {}
        game_n_envs_to_fps = {}
        results[f'{config.n_profile_steps}-step_rollout'] = game_n_envs_to_fps

        for (game, n_envs) in zip(games, batch_sizes):
            config.game = game
            config.n_envs= n_envs

            print(f'\nGame: {game}, n_envs: {config.n_envs}.')
            start = timer()
            env = init_ps_env(config)

            for level_i in range(len(env.levels)):
                env_params = get_env_params_from_config(env, config)

                # INIT ENV
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config.n_envs)
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

                def _env_step(carry, unused):
                    env_state, rng = carry
                    rng, _rng = jax.random.split(rng)
                    rand_act = jax.random.randint(_rng, (config.n_envs,), 0, env.action_space.n)
                    action = rand_act

                    # STEP ENV
                    rng_step = jax.random.split(_rng, config.n_envs)
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
                    _env_step_jitted, carry, None, config.n_profile_steps
                )

                n_env_steps = config.n_profile_steps * config.n_envs

                end = timer()
                fps = n_env_steps / (end - start)
                print(f'Finished {n_env_steps} steps in {end - start} seconds.')
                print(f'Average steps per second: {fps}')

                if config.render:
                    print('Rendering gif...')
                    start = timer()
                    env_states_0 = jax.tree.map(lambda x: x[:, 0], env_states)
                    frames = jax.vmap(env.render, in_axes=(0,))(env_states_0)
                    print(f'Finished rendering frames in {timer() - start} seconds.')
                    start = timer()
                    gif_path = os.path.join(vids_dir, f'{game}_{n_envs}_randAct.gif')
                    imageio.mimsave(gif_path, frames, duration=config.gif_frame_duration)
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

    # Turn into a dataframe, where rows are games and columns are different n_envs
    fps_df = pd.DataFrame(game_n_envs_to_fps)
    # Swap rows and columns
    fps_df = fps_df.T
    # Round to 2 decimal places
    fps_df = fps_df.round(2)
    print(fps_df)
    # Save as markdown
    fps_df.to_markdown(f'n_envs_to_fps.md')
    # latex format 2 decimal places
    styled_fps_df = fps_df.style.format("{:.2f}")
    with open(f'n_envs_to_fps.tex', 'w') as f:
        f.write(styled_fps_df.to_latex())

if __name__ == '__main__':
    profile() 
