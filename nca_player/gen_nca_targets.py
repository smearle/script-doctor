from dataclasses import dataclass
import glob
import json
import os
import shutil
import traceback

import PIL
import hydra
import imageio
from javascript import require
import jax
from jax import numpy as jnp
from lark import Lark
import numpy as np
import pickle

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conf.config import ProfileNodeJS
from env import PSState
from globals import NCA_DATA_DIR
from preprocess_games import PSErrors, get_env_from_ps_file
from profile_nodejs import compile_game
from utils import collect_best_solutions, get_list_of_games_for_testing, init_ps_lark_parser, level_to_int_arr
from validate_sols import JS_SOLS_DIR, multihot_level_from_js_state

@dataclass
class PSStateData:
    multihot_levels: jnp.ndarray
    target_frames: jnp.ndarray
    level_ims: np.ndarray
    player_coords: jnp.ndarray

@hydra.main(version_base="1.3", config_path='../conf', config_name='profile_nodejs_config')
def main(cfg: ProfileNodeJS):
    parser = init_ps_lark_parser()
    engine = require('../standalone/puzzlescript/engine.js')
    solver = require('../standalone/puzzlescript/solver.js')
    if cfg.game is None:
        if cfg.all_games:
            game_sols_dirs = glob.glob(f"{JS_SOLS_DIR}/*")
        else:
            game_sols_dirs = get_list_of_games_for_testing(all_games=cfg.all_games)
            game_sols_dirs = [os.path.join(JS_SOLS_DIR, game) for game in game_sols_dirs]
    else:
        game_sols_dirs = [os.path.join(JS_SOLS_DIR, cfg.game)]
    # A dummy state, only the multihot_level is used.
    state = PSState(
        multihot_level=None,
        win=False, score=0, heuristic=0, restart=False, init_heuristic=0, prev_heuristic=0,
        step_i=0, rng=jax.random.PRNGKey(0),
    )
    for game_dir in game_sols_dirs:
        game_name = game_dir.split(os.path.sep)[-1]
        level_ints, level_sol_jsons = collect_best_solutions(game_dir)
        game_text = None
        env = None
        game_nca_data_dir = os.path.join(NCA_DATA_DIR, game_name)
        os.makedirs(game_nca_data_dir, exist_ok=True)
        for level_sol_json in level_sol_jsons:
            level_sol_json = os.path.join(game_dir, level_sol_json)
            level_i = level_sol_json.split(os.path.sep)[-1].split(".")[0].split('-')[-1]
            if '-steps' in level_sol_json:
                n_steps = level_sol_json.split(os.path.sep)[-1].split(".")[0].split('-steps')[0]
            else:
                n_steps = 10_000
            print(f"Game: {game_name}, Level: {level_i}, Steps: {n_steps}")
            level_sol_gif_path = os.path.join(game_dir, f"{n_steps}-steps_level-{level_i}_sol.gif")
            if os.path.isfile(level_sol_gif_path) and not cfg.overwrite:
                print(f"Level {level_i} data already saved")
                continue

            if game_text is None:
                try:
                    engine.unloadGame()
                    game_text = compile_game(parser, engine, game_name, level_i=level_i)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error compiling game {game_name} level {level_i}: {e}")
                    continue
                env, tree, success, err_msg = get_env_from_ps_file(parser, game_name)
                if success != PSErrors.SUCCESS:
                    print(f"Error parsing game {game_name} level {level_i}: {err_msg}")
                    break
            try:
                engine.compile(['loadLevel', level_i], game_text)
            except Exception as e:
                traceback.print_exc()
                print(f"Error compiling game {game_name} level {level_i}: {e}")
                break
            solver.precalcDistances(engine)

            with open(level_sol_json, "r") as f:
                data = json.load(f)
            if 'actions' in data:
                actions = data['actions']
            elif 'sol' in data:
                actions = data['sol']    

            multihot_levels = []
            player_coords = []
            level_ims = []
            level_state = solver.getState(engine)
            n_objs = len(list(data['objs']))
            if n_objs > 64:
                continue
            level_state = level_to_int_arr(level_state, n_objs).tolist()
            multihot_level = multihot_level_from_js_state(level_state, data['objs'])
            # Pad the level with 0s
            # multihot_level = jnp.pad(multihot_level, ((0,0),(1,1),(1,1)), mode='constant', constant_values=0)
            target_frames = jnp.zeros((len(actions), 5, multihot_level.shape[1], multihot_level.shape[2]), dtype=jnp.uint8)
            state = state.replace(multihot_level=multihot_level)
            multihot_levels.append(multihot_level)
            # Ones where any player index is active
            player_pos_mask = jnp.where(multihot_level[env.player_idxs] == 1, 1, 0)
            # Take 'or' over player indices
            player_pos_mask = jnp.clip(jnp.sum(player_pos_mask, axis=0), 0, 1)
            player_coord = jnp.argwhere(player_pos_mask == 1)
            player_coords.append(player_coord)
            target_frames = target_frames.at[0, actions[0]].set(jnp.where(player_pos_mask == 1, 1, 0))
            for i, action in enumerate(actions):
                _, _, _, _, _, level_state, _, _ = solver.takeAction(engine, action)
                level_state = level_to_int_arr(level_state, n_objs).tolist()
                multihot_level = multihot_level_from_js_state(level_state, data['objs'])
                multihot_levels.append(multihot_level)
                state = state.replace(multihot_level=multihot_level)
                player_pos_mask = jnp.where(multihot_level[env.player_idxs] == 1, 1, 0)
                player_pos_mask = jnp.clip(jnp.sum(player_pos_mask, axis=0), 0, 1)
                player_coord = jnp.argwhere(player_pos_mask == 1)
                player_coords.append(player_coord)
                target_frames = target_frames.at[i+1, action].set(jnp.where(player_pos_mask == 1, 1, 0))
                level_ims.append(env.render(state, cv2=False))

            if len(multihot_levels) == 0:
                print(f"No states to render for level {level_i}")
                continue

            states = PSStateData(
                multihot_levels=jnp.stack(multihot_levels),
                target_frames=target_frames,
                level_ims=np.array(level_ims),
                player_coords=jnp.stack(player_coords),
            )

            # Save multihot levels and target frames in a pickle file for later use in training.
            pkl_path = os.path.join(game_nca_data_dir, f"level-{level_i}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(states, f)

            print(f"Level {level_i} data rendered and saved to {pkl_path}")

            

if __name__ == '__main__':
    main()

    