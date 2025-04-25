"""Profile environment speed while taking random actions."""
import glob
import os
import hydra
import jax
import json
import pandas as pd
from timeit import default_timer as timer

from conf.config import ProfileEnvConfig
from env import init_ps_env
from utils_rl import init_config, get_env_params_from_config


games = [
    'atlas shrank',
    'sokoban_basic',
    'sokoban_match3',
    'slidings',
]
# game_paths = glob.glob(os.path.join('data', 'scraped_games', '*.txt'))
# games = [os.path.basename(p) for p in game_paths]

n_envss= [1, 10, 50, 100, 200, 400, 600]


@hydra.main(version_base=None, config_path='./', config_name='profile_pcgrl')
def profile(config: ProfileEnvConfig):
    if config.reevaluate:
        # config = init_config(config)
        # exp_dir = config.exp_dir

        rng = jax.random.PRNGKey(42)
        n_steps = 0

        game_n_envs_to_fps = {}
        
        for n_envs in n_envss:
            config.n_envs= n_envs
            for game in games:
                start_time = timer()
                env_params = get_env_params_from_config(config)
                env = init_ps_env(config, env_params)
                config.game = game
                game_n_envs_to_fps[game] = {}

                # INIT ENV
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config.n_envs)
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

                def _env_step(carry, unused):
                    env_state, rng = carry
                    rng, _rng = jax.random.split(rng)
                    rng_act = jax.random.split(_rng, config.n_envs)
                    rand_act = jax.random.randint(_rng, config.n_envs, 0, env.action_space.n)
                    action = rand_act

                    # STEP ENV
                    rng_step = jax.random.split(_rng, config.n_envs)
                    obsv, env_state, reward, done, info = jax.vmap(
                        env.step, in_axes=(0, 0, 0)
                    )(rng_step, env_state, action)
                    carry = (env_state, rng)
                    return carry, None

                _env_step_jitted = jax.jit(_env_step)

                print(f'Initialized and reset jitted PSEnv in {(timer() - start_time) / 1000} seconds.')

                start = timer()
                carry = (env_state, rng)
                carry, _ = jax.lax.scan(
                    _env_step_jitted, carry, None, config.N_PROFILE_STEPS
                )

                n_env_steps = config.N_PROFILE_STEPS * config.n_envs

                end = timer()
                print(f'Game: {game}, n_envs: {config.n_envs}. Finished {n_env_steps} steps in {end - start} seconds.')
                fps = n_env_steps / (end - start)
                print(f'Average steps per second: {fps}')

                game_n_envs_to_fps[game][n_envs] = fps

        # Save as json
        with open(f'n_envs_to_fps.json', 'w') as f:
            json.dump(game_n_envs_to_fps, f)

    else:
        # Load from json
        with open(f'n_envs_to_fps.json', 'r') as f:
            game_n_envs_to_fps = json.load(f)

    # Turn into a dataframe, where rows are problems and columns are different n_envs
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
