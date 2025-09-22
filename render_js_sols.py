import glob
import json
import os
import shutil
import traceback

import hydra
import imageio
from javascript import require
import jax
from lark import Lark

from conf.config import ProfileNodeJS
from env import PSState
from preprocess_games import PS_LARK_GRAMMAR_PATH, PSErrors, get_env_from_ps_file
from profile_nodejs import compile_game
from utils import get_list_of_games_for_testing, init_ps_lark_parser, level_to_int_arr
from validate_sols import JS_SOLS_DIR, multihot_level_from_js_state, JAX_VALIDATED_JS_SOLS_DIR


@hydra.main(version_base="1.3", config_path='./conf', config_name='profile_nodejs_config')
def main(cfg: ProfileNodeJS):
    parser = init_ps_lark_parser()
    engine = require('./standalone/puzzlescript/engine.js')
    solver = require('./standalone/puzzlescript/solver.js')
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
        level_sol_jsons = glob.glob(f"{game_dir}/*.json")
        game_text = None
        env = None
        for level_sol_json in level_sol_jsons:
            level_i = level_sol_json.split(os.path.sep)[-1].split(".")[0].split('-')[-1]
            if '-steps' in level_sol_json:
                n_steps = level_sol_json.split(os.path.sep)[-1].split(".")[0].split('-steps')[0]
            else:
                n_steps = 10_000
            print(f"Game: {game_name}, Level: {level_i}, Steps: {n_steps}")
            level_sol_gif_path = os.path.join(game_dir, f"{n_steps}-steps_level-{level_i}_sol.gif")
            if os.path.isfile(level_sol_gif_path) and not cfg.overwrite:
                print(f"Level {level_i} already rendered")
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

            frames = []
            level_state = solver.getState(engine)
            n_objs = len(list(data['objs']))
            if n_objs > 64:
                continue
            level_state = level_to_int_arr(level_state, n_objs).tolist()
            multihot_level = multihot_level_from_js_state(level_state, data['objs'])
            state = state.replace(multihot_level=multihot_level)
            frames.append(env.render(state, cv2=False))
            for action in actions:
                _, _, _, _, _, level_state, _, _ = solver.takeAction(engine, action)
                level_state = level_to_int_arr(level_state, n_objs).tolist()
                multihot_level = multihot_level_from_js_state(level_state, data['objs'])
                state = state.replace(multihot_level=multihot_level)
                frames.append(env.render(state, cv2=False))

            if len(frames) == 0:
                print(f"No frames to render for level {level_i}")
                continue
            imageio.mimsave(level_sol_gif_path, frames, duration=1, loop=0)

            # jax_val_game_dir = os.path.join(JAX_VALIDATED_JS_SOLS_DIR, game_name)
            # os.makedirs(jax_val_game_dir, exist_ok=True)
            # level_sol_jax_val_path = os.path.join(jax_val_game_dir, f"level-{level_i}_js.gif")
            # shutil.copyfile(level_sol_gif_path, level_sol_jax_val_path)
            # print(f"Level {level_i} rendered and saved to {level_sol_gif_path}, and copied to {level_sol_jax_val_path}")
            print(f"Level {level_i} rendered and saved to {level_sol_gif_path}")

            

if __name__ == '__main__':
    main()

    