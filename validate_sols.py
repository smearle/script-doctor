import bdb
import glob
import json
import os
import pickle
import random
import shutil
import traceback

import hydra
import imageio
import jax
import jax.numpy as jnp
from lark import Lark
import numpy as np

from conf.config import Config
from env import PSEnv
from gen_tree import GenPSTree
from parse_lark import TREES_DIR, DATA_DIR, TEST_GAMES, get_tree_from_txt
from ps_game import PSGameTree
from utils_rl import get_env_params_from_config


scratch_dir = 'scratch'
os.makedirs(scratch_dir, exist_ok = True)


@hydra.main(version_base="1.3", config_path='./conf', config_name='config')
def main(config: Config):
    sol_paths = glob.glob(os.path.join('sols', '*'))
    random.shuffle(sol_paths)
    games = [os.path.basename(path) for path in sol_paths]
    # tree_paths = [os.path.join(TREES_DIR, os.path.basename(path) + '.pkl') for path in sol_paths]
    # games = [os.path.basename(path)[:-4] for path in sol_paths]
    sols_dir = os.path.join('vids', 'jax_sols')
    shutil.rmtree(sols_dir, ignore_errors=True)

    for sol_dir, game in zip(sol_paths, games):

        traj_dir = os.path.join('vids', 'jax_sols', game)
        os.makedirs(traj_dir)

        with open("syntax.lark", "r", encoding='utf-8') as file:
            puzzlescript_grammar = file.read()
        # Initialize the Lark parser with the PuzzleScript grammar
        parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
        # min_parser = Lark(min_puzzlescript_grammar, start="ps_game")
        tree = get_tree_from_txt(parser, game)
        og_path = os.path.join(DATA_DIR, 'scraped_games', os.path.basename(game) + '.txt')

        print(f"Processing solution for game: {og_path}")

        try:
            env = PSEnv(tree)
        except KeyboardInterrupt as e:
            raise e
        except bdb.BdbQuit as e:
            raise e
        except Exception as e:
            err_log = traceback.format_exc()
            with open(os.path.join(traj_dir, 'error.txt'), 'w') as f:
                f.write(err_log)
            traceback.print_exc()
            print(f"Error creating env: {og_path}")
            log_path = os.path.join(traj_dir)
            continue

        key = jax.random.PRNGKey(0)
        params = get_env_params_from_config(env, config)
        obs, state = env.reset(key, params)

        # 0 - left
        # 1 - down
        # 2 - right
        # 3 - up
        # 4 - action
        action_remap = [3, 0, 1, 2, 4]

        key = jax.random.PRNGKey(0)

        level_sols = glob.glob(os.path.join(sol_dir, 'level-*.json'))

        for level_sol_path in level_sols:
            with open(level_sol_path, 'r') as f:
                level_sol = json.load(f)
            actions = level_sol
            actions = [action_remap[a] for a in actions]
            actions = jnp.array([int(a) for a in actions])

            level_i = int(os.path.basename(level_sol_path).split('-')[1].split('.')[0])
            level = env.get_level(level_i)
            params = params.replace(level=level)
            print(f"Level {level_i} solution: {actions}")

            def step_env(carry, action):
                state, _ = carry
                obs, state, reward, done, info = env.step(key, state, action, params)
                return (state, done), state

            try:
                obs, state = env.reset(key, params)
                (state, done), state_v = jax.lax.scan(step_env, (state, False), actions)
                # if not state.win:
                if not done:
                    log_path = os.path.join(traj_dir, f'level-{level_i}_solution_err.txt')
                    with open(log_path, 'w') as f:
                        f.write(f"Level {level_i} solution failed\n")
                        f.write(f"Actions: {actions}\n")
                        # f.write(f"State: {state}\n")
                    print(f"Level {level_i} solution failed")
            except Exception as e:
                traceback.print_exc()
                print(f"Error running solution: {og_path}")
                err_log = traceback.format_exc()
                log_path = os.path.join(traj_dir, f'level-{level_i}_runtime_err.txt')
                with open(log_path, 'w') as f:
                    f.write(err_log)
                continue

            # Use jax tree map to add the initial state
            state_v = jax.tree_map(lambda x, y: jnp.concatenate([x[None], y]), state, state_v)

            frames = jax.vmap(env.render, in_axes=(0, None))(state_v, None)
            frames = frames.astype(np.uint8)

            frames_dir = os.path.join(traj_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                imageio.imsave(os.path.join(frames_dir, f'level-{level_i}_sol_{i:03d}.png'), frame)

            # Make a gif out of the frames
            gif_path = os.path.join(traj_dir, f'level-{level_i}.gif')
            imageio.mimsave(gif_path, frames, duration=0.1, loop=1)
            # exit()


if __name__ == '__main__':
    main()