import glob
import json
import os
import pickle
import random
import traceback

import imageio
import jax
import jax.numpy as jnp
import numpy as np

from env import PSEnv
from gen_tree import GenPSTree
from parse_lark import TREES_DIR, DATA_DIR, TEST_GAMES
from ps_game import PSGame


scratch_dir = 'scratch'
os.makedirs(scratch_dir, exist_ok = True)
if __name__ == '__main__':
    sol_paths = glob.glob(os.path.join('sols', '*'))
    tree_paths = [os.path.join(TREES_DIR, os.path.basename(path) + '.pkl') for path in sol_paths]
    trees = []
    for sol_dir, tree_path in zip(sol_paths, tree_paths):
        print(f"Processing solution for game: {tree_path}")
        og_game_path = os.path.join(DATA_DIR, 'scraped_games', os.path.basename(tree_path)[:-3] + 'txt')
        print(f"Parsing {og_game_path}")
        with open(tree_path, 'rb') as f:
            tree = pickle.load(f)
        trees.append(tree)

        try:
            tree: PSGame = GenPSTree().transform(tree)
        except Exception as e:
            traceback.print_exc()
            print(f"Error parsing tree: {tree_path}")
            continue

        try:
            env = PSEnv(tree)
        except Exception as e:
            traceback.print_exc()
            print(f"Error creating env: {tree_path}")
            continue

        state = env.reset(0)

        # 0 - left
        # 1 - down
        # 2 - right
        # 3 - up
        action_remap = [3, 0, 1, 2]

        key = jax.random.PRNGKey(0)

        level_sols = glob.glob(os.path.join(sol_dir, 'level-*.json'))

        for level_sol_path in level_sols:
            with open(level_sol_path, 'r') as f:
                level_sol = json.load(f)
            actions = level_sol
            actions = [action_remap[a] for a in actions]
            actions = jnp.array([int(a) for a in actions])

            level_i = int(os.path.basename(level_sol_path).split('-')[1].split('.')[0])
            print(f"Level {level_i} solution: {actions}")
            traj_dir = os.path.join('vids', 'jax_sols', os.path.basename(tree_path)[:-3])

            state = env.reset(level_i)

            def step_env(state, action):
                state = env.step(action, state)
                return state, state

            state, state_v = jax.lax.scan(step_env, state, actions)
            if not state.win:
                log_path = os.path.join(traj_dir, f'level-{level_i}_err.txt')
                with open(log_path, 'w') as f:
                    f.write(f"Level {level_i} solution failed\n")
                    f.write(f"Actions: {actions}\n")
                    f.write(f"State: {state}\n")
                print(f"Level {level_i} solution failed")

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
            imageio.mimsave(gif_path, frames, duration=0.1)
            # exit()