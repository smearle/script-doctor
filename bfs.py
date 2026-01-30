import glob
import json
import os
import random

import cv2
import gym
import hydra
import jax
from jax import numpy as jnp
from lark import Lark
import numpy as np
from timeit import default_timer as timer

from conf.config import BFSConfig, RLConfig
from puzzlejax.env import PuzzleJaxEnv, PJParams, PJState
from globals import PRIORITY_GAMES
from human_env import SCALING_FACTOR
from jax_utils import stack_leaves
from preprocess_games import LARK_SYNTAX_PATH, get_tree_from_txt
from sort_games_by_n_rules import GAMES_N_RULES_SORTED_PATH
from puzzlejax.utils import save_gif_from_states
from utils_rl import get_env_params_from_config
from validate_sols import JS_SOLS_DIR

JAX_BFS_SOLS_DIR = os.path.join('data', 'jax_bfs_sols')

def hash_state(state: PJState):
    """Hash the state to a string."""
    byte_string = state.multihot_level.tobytes()
    hash_value_builtin = hash(byte_string)
    return hash_value_builtin
    # print(f"Hash value (built-in): {hash_value_builtin}")
    # return str(state.multihot_level.flatten())

def bfs(env: PuzzleJaxEnv, state: PJState, params: PJParams,
          max_steps: int = np.inf, render: bool = False, max_episode_steps: int = 100, n_best_to_keep: int = 1):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    rng = jax.random.PRNGKey(0)
    frontier = [(state, [], 0)]
    visited = {hash_state(state): -np.inf}
    best_action_seq = []
    best_heuristic_score = -np.inf
    n_iter_best = 0
    n_iter = 0

    possible_actions = jnp.arange(env.action_space.n)
    start_time = timer()

    while len(frontier) > 0:
        if n_iter > max_steps:
            break
        # Find the idx of the best state in the frontier
        # best_idx = np.argmax([f[2] for f in frontier])
        # parent_state, parent_action_seq, parent_rew = frontier.pop(best_idx)
        parent_state, parent_action_seq, parent_rew = frontier.pop(0)
        
        for action in possible_actions:
            obs, state, rew, done, info = \
                env.step_env(rng=rng, action=action, state=parent_state, params=params)
            state: PJState
            child_score = state.heuristic
            # if render:
            #     im = env.render(state, cv2=True)
            #     im = np.array(im, dtype=np.uint8)
            #     # Resize the image by a scaling factor
            #     new_h, new_w = tuple(np.array(im.shape[:2]) * SCALING_FACTOR)
            #     im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            #     # Display the image in an OpenCV window
            #     cv2.imshow(env.title, im)
            #     # Add a short waitKey here to allow the window to update.
            #     cv2.waitKey(1)  # 1 ms delay; adjust as necessary

            action_seq = parent_action_seq + [action]
            hashed_state = hash_state(state)
            if (hashed_state in visited) and ((child_score < visited[hashed_state]) or (len(action_seq) >= len(best_action_seq))):
                # print(f'already visited {hashed_state}')
                continue
            visited[hashed_state] = child_score
            if child_score > best_heuristic_score:
                best_heuristic_score = child_score
                best_action_seq = action_seq
                n_iter_best = n_iter
            if not jnp.all(done):
                # Add this state to the frontier so can we can continue searching from it later
                frontier.append((state, action_seq, child_score))
            if n_iter % 1_000 == 0:
                print(f'\nn_iter: {n_iter}')
                print(f'frontier size: {len(frontier)}')
                print(f'visited size: {len(visited)}')
                print(f'best heuristic score: {best_heuristic_score}')
                print(f'FPS: {n_iter / (timer() - start_time):.2f}')
            if state.win:
                return action_seq, True, child_score, n_iter, n_iter
            n_iter += 1

    return best_action_seq, False, best_heuristic_score, n_iter_best, n_iter


@hydra.main(version_base="1.3", config_path='./conf', config_name='bfs_config')
def main(cfg: BFSConfig):
    os.makedirs(JAX_BFS_SOLS_DIR, exist_ok=True)
    with open(LARK_SYNTAX_PATH, 'r', encoding='utf-8') as f:
        puzzlescript_grammar = f.read()
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    if cfg.game is not None:
        games = [cfg.game]
    elif cfg.all_games:
        with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
            games_n_rules = json.load(f)
        games_n_rules = sorted(games_n_rules, key=lambda x: x[1])
        games = [game for game, n_rules, has_randomness in games_n_rules if not has_randomness]
        # Throw these ones at the top to analyze first
        games = PRIORITY_GAMES + [game for game in games if game not in PRIORITY_GAMES]
    else:
        games = PRIORITY_GAMES
    js_sols_dirs = [os.path.join(JS_SOLS_DIR, game) for game in games]

    for js_sol_dir, game in zip(js_sols_dirs, games):
        game_name = os.path.basename(game)
        tree, success, err_msg = get_tree_from_txt(parser, game, test_env_init=False)
        env = PuzzleJaxEnv(tree, debug=False, print_score=False)

        # level_sol_paths = glob.glob(os.path.join(js_sol_dir, 'level-*.json'))
        # level_idxs = [int(os.path.basename(level_sol_path).split('-')[1].split('.')[0]) for level_sol_path in level_sol_paths]
        # level_sorted_idxs = sorted(range(len(level_idxs)), key=lambda i: level_idxs[i])
        # level_sol_paths = [level_sol_paths[i] for i in level_sorted_idxs]
        # level_idxs = sorted(level_idxs)
        # for level_i, level_sol_path in zip(level_idxs, level_sol_paths):
        for level_i in range(len(env.levels)):
            print(f"Loading solution for level {level_i} of {game_name}")
            # with open(level_sol_path, 'r') as f:
            #     js_level_sol = json.load(f)

            rng = jax.random.PRNGKey(0)
            
            level = env.get_level(level_i)
            params = PJParams(
                level=level,
            )

            print(f"Searching with BrFS in jax for level {level_i} of {game_name}.")
            obs, state = env.reset(rng=rng, params=params)

            # Take two steps to get compilation out of the way
            for _ in range(2):
                obs, state, rew, done, info = env.step_env(rng=rng, action=0, state=state, params=params)
            
            obs, state = env.reset(rng=rng, params=params)

            start_time = timer()
            best_action_seq, win, best_reward, n_iter_best, n_iter = bfs(
                env=env,
                state=state,
                params=params,
                max_steps=cfg.max_steps,
                render=cfg.render_live,
                n_best_to_keep=cfg.n_best_to_keep
            )
            time_elapsed = timer() - start_time
            print(f"Found {'winning' if win else 'non-winning'} solution {[int(i) for i in best_action_seq]} with in {time_elapsed} seconds.")
            print(f"FPS: {n_iter / time_elapsed}")

            def step_env(state, action):
                obs, state, reward, done, info = env.step_env(rng, state, action, params)
                return state, state

            if cfg.render_gif:
                obs, state = env.reset(rng, params)
                state, states = jax.lax.scan(step_env, state, jnp.array(best_action_seq))

                save_path = os.path.join(JAX_BFS_SOLS_DIR, f"{game_name}_{level_i}")
                save_gif_from_states(env, states, save_path=save_path)

        
if __name__ == '__main__':
    main()