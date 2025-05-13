import glob
import json
import os
import random

import gym
import hydra
import jax
from jax import numpy as jnp
from lark import Lark
import numpy as np
from timeit import default_timer as timer

from conf.config import BFSConfig, RLConfig
from env import PSEnv, PSParams, PSState
from jax_utils import stack_leaves
from parse_lark import PS_LARK_GRAMMAR_PATH, get_tree_from_txt
from sort_games_by_n_rules import GAMES_N_RULES_SORTED_PATH
from utils import save_gif_from_states
from utils_rl import get_env_params_from_config
from validate_sols import JS_SOLS_DIR

JAX_BFS_SOLS_DIR = os.path.join('data', 'jax_bfs_sols')

def hash_state(state: PSState):
    """Hash the state to a string."""
    return str(state.multihot_level.flatten())

def bfs(env: PSEnv, state: PSState, params: PSParams,
          max_steps: int = np.inf, render: bool = False, max_episode_steps: int = 100, n_best_to_keep: int = 1):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    # height, width = env.height, env.width
    # state = env.get_state()
    rng = jax.random.PRNGKey(0)
    frontier = [(state, [], 0)]
    visited = {hash_state(state): -np.inf}
    # visited = {env.player_pos: state}
    # visited = {type(env).hashable(state): state}
    # visited_0 = [state]
    best_action_seq = []
    best_score = -np.inf
    n_iter_best = 0
    n_iter = 0

    possible_actions = jnp.arange(env.action_space.n)

    while len(frontier) > 0:
        n_iter += 1
        if n_iter > max_steps:
            break
        # parent_state, parent_action_seq, parent_rew = frontier.pop(0)
        # Find the idx of the best state in the frontier
        best_idx = np.argmax([f[2] for f in frontier])
        parent_state, parent_action_seq, parent_rew = frontier.pop(best_idx)
        
        # FIXME: Redundant, remove me
        # env.set_state(parent_state)

        # visited[env.player_pos] = env.get_state()
        # if type(env).hashable(parent_state) in visited:
            # continue
        # visited[hash(env, parent_state)] = parent_rew
        # print(visited.keys())
        # random.shuffle(possible_actions)
        for action in possible_actions:
            # env.set_state(parent_state)
            # print('set frontier state')
            obs, state, rew, done, info = \
                env.step_env(rng=rng, action=action, state=parent_state, params=params)
            state: PSState
            child_score = state.score
            if render:
                env.render()
            # print(f'action: {action}')

            # FIXME: Redundant, remove me

            # map_arr = state['map_arr']
            action_seq = parent_action_seq + [action]
            # if env.player_pos in visited:
            hashed_state = hash_state(state)
            # if hashed_state in visited and child_rew > visited[hashed_state]:
            #     breakpoint()
            if (hashed_state in visited) and (child_score <= visited[hashed_state]):
                # print(f'already visited {hashed_state}')
                continue
            # visited[env.player_pos] = state
            # visited[tuple(action_seq)] = state
            visited[hashed_state] = child_score
            # print(len(visited))
            # visited_0.append(state)
            # print([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()])
            # if not np.all([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()]):
                # TT()
            if child_score > best_score:
                best_score = child_score
                best_action_seq = action_seq
                n_iter_best = n_iter
                # print(f'new best score: {best_score}')
                # print(f'new best action seq: {best_action_seq}')
                # print(f'new best state: {state}')
            if not jnp.all(done):
                # Add this state to the frontier so can we can continue searching from it later
                frontier.append((state, action_seq, child_score))

    return best_action_seq, best_score, n_iter_best, n_iter


@hydra.main(version_base="1.3", config_path='./conf', config_name='bfs_config')
def main(cfg: BFSConfig):
    os.makedirs(JAX_BFS_SOLS_DIR, exist_ok=True)
    with open(PS_LARK_GRAMMAR_PATH, 'r', encoding='utf-8') as f:
        puzzlescript_grammar = f.read()
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
        games_n_rules = json.load(f)
    games_n_rules = sorted(games_n_rules, key=lambda x: x[1])
    games = [game for game, n_rules in games_n_rules]
    js_sols_dirs = [os.path.join(JS_SOLS_DIR, game) for game in games]

    for js_sol_dir, game in zip(js_sols_dirs, games):
        game_name = os.path.basename(game)
        tree, success, err_msg = get_tree_from_txt(parser, game, test_env_init=False)
        env = PSEnv(tree, debug=False, print_score=False)

        level_sol_paths = glob.glob(os.path.join(js_sol_dir, 'level-*.json'))
        for level_sol_path in level_sol_paths:
            level_i = int(os.path.basename(level_sol_path).split('-')[1].split('.')[0])
            print(f"Loading solution for level {level_i} of {game_name}")
            with open(level_sol_path, 'r') as f:
                js_level_sol = json.load(f)

            rng = jax.random.PRNGKey(0)
            
            level = env.get_level(level_i)
            params = PSParams(
                level=level,
            )

            print(f"Searching with BrFS in jax for level {level_i} of {game_name}.")
            obs, state = env.reset(rng=rng, params=params)

            # Take two steps to get compilation out of the way
            for _ in range(2):
                obs, state, rew, done, info = env.step_env(rng=rng, action=0, state=state, params=params)
            
            obs, state = env.reset(rng=rng, params=params)

            start_time = timer()
            best_action_seq, best_reward, n_iter_best, n_iter = bfs(
                env=env,
                state=state,
                params=params,
                max_steps=cfg.max_steps,
                render=False,
                n_best_to_keep=cfg.n_best_to_keep
            )
            time_elapsed = timer() - start_time
            print(f"Found solution best states in {time_elapsed} seconds.")
            print(f"FPS: {n_iter / time_elapsed}")

            def step_env(state, action):
                obs, state, reward, done, info = env.step_env(rng, state, action, params)
                return state, state

            if cfg.render:
                obs, state = env.reset(rng, params)
                state, states = jax.lax.scan(step_env, state, jnp.array(best_action_seq))

                save_path = os.path.join(JAX_BFS_SOLS_DIR, f"{game_name}_{level_i}")
                save_gif_from_states(env, states, save_path=save_path)

        
if __name__ == '__main__':
    main()