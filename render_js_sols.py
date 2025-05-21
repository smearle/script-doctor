import glob
import json
import os
import shutil

import imageio
from javascript import require
import jax
from lark import Lark

from env import PSState
from preprocess_games import PS_LARK_GRAMMAR_PATH, get_env_from_ps_file
from profile_standalone_nodejs import compile_game
from utils import level_to_int_arr
from validate_sols import JS_SOLS_DIR, multihot_level_from_js_state, JAX_VALIDATED_JS_SOLS_DIR


if __name__ == '__main__':
    with open(PS_LARK_GRAMMAR_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    engine = require('./standalone/puzzlescript/engine.js')
    solver = require('./standalone/puzzlescript/solver.js')
    game_sols_dir = glob.glob(f"{JS_SOLS_DIR}/*")
    # A dummy state, only the multihot_level is used.
    state = PSState(
        multihot_level=None,
        win=False, score=0, heuristic=0, restart=False, init_heuristic=0, prev_heuristic=0,
        step_i=0, rng=jax.random.PRNGKey(0),
    )
    for game_dir in game_sols_dir:
        game_name = game_dir.split("/")[-1]
        level_sol_jsons = glob.glob(f"{game_dir}/*.json")
        game_text = None
        for level_sol_json in level_sol_jsons:
            level_i = level_sol_json.split("/")[-1].split(".")[0].split('-')[1]
            print(f"Game: {game_name}, Level: {level_i}")
            level_sol_gif_path = os.path.join(game_dir, f"level-{level_i}_sol.gif")
            if os.path.isfile(level_sol_gif_path):
                print(f"Level {level_i} already rendered")
                continue

            if game_text is None:
                game_text = compile_game(engine, game_name, level_i=level_i)
                env, tree, success, err_msg = get_env_from_ps_file(parser, game_name)
            else:
                engine.compile(game_text, level_i)
                solver.precalcDistances(engine)

            with open(level_sol_json, "r") as f:
                data = json.load(f)
            if 'actions' in data:
                actions = data['actions']
            elif 'sol' in data:
                actions = data['sol']    

            frames = []
            for action in actions:
                _, _, _, _, _, level_state = solver.takeAction(engine, action)
                level_state = level_to_int_arr(level_state).tolist()
                multihot_level = multihot_level_from_js_state(level_state, data['objs'])
                state = state.replace(multihot_level=multihot_level)
                frames.append(env.render(state))

            if len(frames) == 0:
                print(f"No frames to render for level {level_i}")
                continue
            imageio.mimsave(level_sol_gif_path, frames, duration=0.1, loop=0)

            jax_val_game_dir = os.path.join(JAX_VALIDATED_JS_SOLS_DIR, game_name)
            os.makedirs(jax_val_game_dir, exist_ok=True)
            level_sol_jax_val_path = os.path.join(jax_val_game_dir, f"level-{level_i}_js.gif")
            # shutil.copyfile(level_sol_gif_path, level_sol_jax_val_path)
            # print(f"Level {level_i} rendered and saved to {level_sol_gif_path}, and copied to {level_sol_jax_val_path}")
            print(f"Level {level_i} rendered and saved to {level_sol_gif_path}")

            

