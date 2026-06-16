#!/usr/bin/env python
"""Debug the portal force-clearing behavior."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from lark import Lark

from puzzlescript_jax.env import PuzzleJaxEnv, PJParams
from puzzlescript_jax.globals import LARK_SYNTAX_PATH, JS_TO_JAX_ACTIONS
from puzzlescript_jax.preprocessing import get_tree_from_txt
from puzzlescript_jax.env_utils import multihot_to_desc

with open(LARK_SYNTAX_PATH, 'r', encoding='utf-8') as f:
    grammar = f.read()
parser = Lark(grammar, start='ps_game', maybe_placeholders=False)
tree, err, msg = get_tree_from_txt(parser, 'test_gdd301_lab')

env = PuzzleJaxEnv(tree, jit=False, max_steps=100, debug=False)
level = env.get_level(0)
params = PJParams(level=level, level_i=0)
rng = jax.random.PRNGKey(0)
_, state = env.reset(rng, params)

# JAX actions: 0=left, 1=down, 2=right, 3=up, 4=action
# JS actions: 0=up, 1=left, 2=down, 3=right, 4=action
# JS_TO_JAX_ACTIONS = [3, 0, 1, 2, 4]
# So JS right=3 -> JAX 2

player_idx = env.objs_to_idxs['player']
force_start = env.obj_idxs_to_force_idxs[player_idx]
force_names = ['left', 'down', 'right', 'up', 'action']

def show_state(label, state):
    level = np.array(state.multihot_level)
    desc = multihot_to_desc(level, env.objs_to_idxs, env.n_objs, env.obj_idxs_to_force_idxs)
    print(f"\n=== {label} ===")
    print(desc)
    # level shape is (n_channels, H, W)
    player_positions = list(zip(*np.where(level[player_idx])))
    print(f"  Player positions: {player_positions}")

show_state("Initial", state)

for step, act in enumerate([2, 2]):  # RIGHT, RIGHT (JAX action 2)
    print(f"\n--- Step {step+1}: action={act} (right) ---")
    try:
        _, state, reward, done, info = env.step_env(rng, state, act, params)
    except Exception as e:
        print(f"Error: {e}")
        break
    show_state(f"After step {step+1}", state)
