import json
import os
import random

import gym
import hydra
import jax
from lark import Lark
import numpy as np

from conf.config import RLConfig
from env import PSEnv, PSParams, PSState
from parse_lark import PS_LARK_GRAMMAR_PATH, get_tree_from_txt
from utils_rl import get_env_params_from_config
from validate_sols import JS_SOLS_DIR


def bfs(env: PSEnv, state: PSState, params: PSParams,
          max_steps: int = np.inf, render: bool = False, max_episode_steps: int = 100, n_best_to_keep: int = 1):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    # height, width = env.height, env.width
    # state = env.get_state()
    key = jax.random.PRNGKey(0)
    frontier = [(state, [], 0)]
    visited = {hash(env, state): -np.inf}
    # visited = {env.player_pos: state}
    # visited = {type(env).hashable(state): state}
    # visited_0 = [state]
    best_state_actionss = [None] * n_best_to_keep
    best_rewards = [-np.inf] * n_best_to_keep
    n_iter_bests = [0] * n_best_to_keep
    n_iter = 0

    if isinstance(env.action_space, gym.spaces.Discrete):
        possible_actions = list(range(env.action_space.n))
    else:
        raise NotImplementedError

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
        random.shuffle(possible_actions)
        for action in possible_actions:
            # env.set_state(parent_state)
            # print('set frontier state')
            obs, state, rew, done, info = \
                env.step_env(key=key, action=action, state=parent_state, params=params)
            child_rew = state.ep_rew.item()
            if render:
                env.render()
            # print(f'action: {action}')

            # FIXME: Redundant, remove me

            # map_arr = state['map_arr']
            action_seq = parent_action_seq + [action]
            # if env.player_pos in visited:
            hashed_state = hash(env, state)
            # if hashed_state in visited and child_rew > visited[hashed_state]:
            #     breakpoint()
            if (hashed_state in visited) and (child_rew <= visited[hashed_state]):
                # print(f'already visited {hashed_state}')
                continue
            # visited[env.player_pos] = state
            # visited[tuple(action_seq)] = state
            visited[hashed_state] = child_rew
            # print(len(visited))
            # visited_0.append(state)
            # print([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()])
            # if not np.all([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()]):
                # TT()
            for i in range(n_best_to_keep):
                if child_rew > best_rewards[i]:
                    best_state_actionss = best_state_actionss[:i] + [(state, action_seq)] + best_state_actionss[i+1:]
                    best_rewards = best_rewards[:i] + [child_rew] + best_rewards[i+1:]
                    n_iter_bests = n_iter_bests[:i] + [n_iter] + n_iter_bests[i+1:]
                    # print(f'found new best: {best_reward} at {n_iter_best} iterations step {state.n_step} action sequence length {len(action_seq)}')
                    break
            if not jnp.all(done):
                # Add this state to the frontier so can we can continue searching from it later
                frontier.append((state, action_seq, child_rew))

    return best_state_actionss, best_rewards, n_iter_bests, n_iter


@hydra.main(version_base="1.3", config_path='./conf', config_name='config')
def main(config: RLConfig):
    with open(PS_LARK_GRAMMAR_PATH, 'r', encoding='utf-8') as f:
        puzzlescript_grammar = f.read()
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    with open(os.path.join('data', 'games_n_rules.json'), 'r') as f:
        games_n_rules = json.load(f)
    games_n_rules = sorted(games_n_rules, key=lambda x: x[1])
    games = [game for game, n_rules in games_n_rules]

    for js_sol_dir, game in zip(JS_SOLS_DIR, game):
        game_name = os.path.basename(game)
        env = PSEnv(tree, debug=False, print_score=False)
        tree, success, err_msg = get_tree_from_txt(parser, game)
        key = jax.random.PRNGKey(0)
        params = get_env_params_from_config(env, config)

        obs, state = env.reset(key=key, params=params)

        best_state_actionss, best_rewards, n_iter_bests, n_iter = bfs(
            env=env,
            state=state,
            params=params,
            max_steps=config.max_steps,
            render=False,
            max_episode_steps=config.max_episode_steps,
            n_best_to_keep=config.n_best_to_keep
        )