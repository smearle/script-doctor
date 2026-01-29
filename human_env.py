from dataclasses import dataclass
import glob
import logging
import os
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
import cv2
import jax
import jax.numpy as jnp
from lark import Lark
import numpy as np

from globals import LARK_SYNTAX_PATH, TEST_GAMES, TREES_DIR, DATA_DIR
from puzzlejax.env import PuzzleJaxEnv, PJParams, multihot_to_desc
from preprocess_games import get_tree_from_txt
from preprocess_games import PSErrors

logger = logging.getLogger(__name__)

@dataclass
class Config:
    jit: bool = True
    game: Optional[str] = None
    profile: bool = False
    debug: bool = False
    level: int = 0


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

SCALING_FACTOR = 10

def human_loop(env: PuzzleJaxEnv, level: int = 0, profile=False):
    lvl_i = level
    level = env.get_level(lvl_i)
    params = PJParams(level=level)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, params)
    im = env.render(state)
    im = np.array(im, dtype=np.uint8)
    
    state_hist = []
    
    # Display the image in an OpenCV window
    cv2.imshow(env.title, im)
    print("Press an arrow key or 'x' (ESC to exit).")
    done = False
    
    # Loop waiting for key presses
    while True:
        rng, _ = jax.random.split(rng)
        # cv2.waitKey(0) waits indefinitely until a key is pressed.
        key = cv2.waitKey(0)
        action = None
        do_reset = False
        print("\n\n========= STEP =========\n")
        print(multihot_to_desc(state.multihot_level, env.objs_to_idxs, env.n_objs, obj_idxs_to_force_idxs=env.obj_idxs_to_force_idxs))

        # If the user presses ESC (ASCII 27), exit the loop.
        print("Player input:")
        if key == 27:
            break
        elif key == ord('x'):
            print("x")
            action = 4
        elif key == 97:
            print("left arrow")
            action = 0
        elif key == 119:
            print("up arrow")
            action = 3
        elif key == 100:
            print("right arrow")
            action = 2
        elif key == 115:
            print("down arrow")
            action = 1
        elif key == ord('r'):
            print("Restarting level...")
            do_reset = True
        elif key == ord('n'):
            print("Advancing level...")
            lvl_i += 1
            if lvl_i >= len(env.levels):
                print("No more levels!")
                break
            params = params.replace(level=env.get_level(lvl_i))
            do_reset = True
        elif key == ord('b'):
            print("Going back a level...")
            if lvl_i > 0:
                lvl_i -= 1
            params = params.replace(level=env.get_level(lvl_i))
            do_reset = True
        elif key == ord('z'):
            print("Undoing last action...")
            if len(state_hist) > 1:
                state_hist.pop()
                state = state_hist[-1]
            im = env.render(state)
            im = np.array(im, dtype=np.uint8)
            # new_h, new_w = tuple(np.array(im.shape[:2]) * SCALING_FACTOR)
            # im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)

        else:
            print("Other key pressed:", key)

        if lvl_i >= len(env.levels):
            print("No more levels!")
            break

        elif action is not None:
            action = jnp.array(action, dtype=jnp.int32)
            if profile:
                with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                    obs, state, reward, done, info = env.step(rng, state, action, params)
                    obs.multihot_level.block_until_ready()
                    print("JAX profiling complete.")
            else:
                obs, state, reward, done, info = env.step(rng, state, action, params)
            win = state.win
            print(multihot_to_desc(state.multihot_level, env.objs_to_idxs, env.n_objs, obj_idxs_to_force_idxs=env.obj_idxs_to_force_idxs))
            im = env.render(state)
            im = np.array(im, dtype=np.uint8)
            # new_h, new_w = tuple(np.array(im.shape[:2]) * SCALING_FACTOR)
            # im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)
            # Add a short waitKey here to allow the window to update.
            cv2.waitKey(1)  # 1 ms delay; adjust as necessary
            if win:
                print("You win!")
            else:
                print("You don't win yet!")
            state_hist.append(state)
            do_reset = state.restart

        if do_reset:
            obs, state = env.reset(rng, params)
            print(multihot_to_desc(state.multihot_level, env.objs_to_idxs, env.n_objs, obj_idxs_to_force_idxs=env.obj_idxs_to_force_idxs))
            done = False
            state_hist.append(state)
            im = env.render(state)
            im = np.array(im, dtype=np.uint8)
            cv2.imshow(env.title, im)
    

        if done:
        # if state.win:
            lvl_i += 1
            if lvl_i >= len(env.levels):
                print("No more levels!")
                break
            level = env.get_level(lvl_i)
            params = params.replace(level=level)
            obs, state = env.reset(rng, params)
            im = env.render(state)
            im = np.array(im, dtype=np.uint8)
            cv2.imshow(env.title, im)
    # Close the image window
    cv2.destroyAllWindows()


def play_game(game: str, level: int = 0, jit: bool = False, profile: bool = False, debug: bool = False):
    with open(LARK_SYNTAX_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    # min_parser = Lark(min_puzzlescript_grammar, start="ps_game")
    print(f"""Parsing game: \"{game}\"""")
    tree, success, err_msg = get_tree_from_txt(parser, game, overwrite=True, test_env_init=False)
    if success != PSErrors.SUCCESS:
        print(f"Error parsing game: {err_msg}")
        return
    print(f"Initializing environment for game: {game}")
    env = PuzzleJaxEnv(tree, jit=jit, debug=debug, print_score=True)
    print(f"Playing game: {game}")
    human_loop(env, profile=profile, level=level)

@hydra.main(config_name="config", version_base="1.3")
def main(cfg: Config):

    # Using this line to play games with characters (e.g. ` ) that don't agree with CL
    # TODO: Fix this (by removing this character from filenames?)
    # cfg.game = "-=lost=-"

    if cfg.game is not None:
        play_game(cfg.game, level=cfg.level, jit=cfg.jit, profile=cfg.profile, debug=cfg.debug)

    else:
        tree_paths = glob.glob(os.path.join(TREES_DIR, '*'))
        tree_paths = sorted(tree_paths, reverse=True)
        test_game_paths = [os.path.join(TREES_DIR, tg + '.pkl') for tg in TEST_GAMES]
        tree_paths = test_game_paths + tree_paths
        game_paths = [os.path.basename(tree_path)[:-4] for tree_path in tree_paths]
        for tree_path in game_paths:
            play_game(tree_path, jit=cfg.jit, debug=cfg.debug)
    


if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')
    main()