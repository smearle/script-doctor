from dataclasses import dataclass
import glob
import os
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
import cv2
import jax
from lark import Lark
import numpy as np

from env import PSEnv, PSParams, multihot_to_desc
from parse_lark import TREES_DIR, DATA_DIR, TEST_GAMES, get_tree_from_txt


@dataclass
class Config:
    jit: bool = True
    game: Optional[str] = None
    profile: bool = False
    debug: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def human_loop(env: PSEnv, profile=False):
    lvl_i = 0 
    level = env.get_level(lvl_i)
    params = PSParams(level=level)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, params)
    im = env.render(state)
    # print(multihot_to_desc(state.multihot_level, env.obj_to_idxs))
    im = np.array(im, dtype=np.uint8)
    
    # Resize the image by a factor of 5
    new_h, new_w = tuple(np.array(im.shape[:2]) * 10)
    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    state_hist = []
    
    # Display the image in an OpenCV window
    cv2.imshow(env.title, im)
    print("Press an arrow key or 'x' (ESC to exit).")
    done = False
    
    # Loop waiting for key presses
    while True:
        rng, _ = jax.random.split(rng)
        # cv2.waitKey(0) waits indefinitely until a key is pressed.
        # Mask with 0xFF to get the lowest 8 bits (common practice).
        key = cv2.waitKey(0)
        action = None
        do_reset = False
        print("\n\n========= STEP =========\n")
        print(multihot_to_desc(state.multihot_level, env.obj_to_idxs, env.n_objs))

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
            params = params.replace(level=env.get_level(lvl_i))
            do_reset = True
        elif key == ord('b'):
            print("Going back a level...")
            if lvl_i > 0:
                lvl_i -= 1
            do_reset = True
        elif key == ord('z'):
            print("Undoing last action...")
            if len(state_hist) > 1:
                state_hist.pop()
                state = state_hist[-1]
            im = env.render(state)
            im = np.array(im, dtype=np.uint8)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)

        else:
            print("Other key pressed:", key)

        if lvl_i >= len(env.levels):
            print("No more levels!")
            break

        elif action is not None:
            lvl = env.apply_player_force(action, state)
            vis_lvl = lvl[:env.n_objs]
            lvl_changed = True
            n_vis_apps = 0
            if profile:
                with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                    obs, state, reward, done, info = env.step(rng, state, action, params)
                    obs.multihot_level.block_until_ready()
                    print("JAX profiling complete.")
            else:
                obs, state, reward, done, info = env.step(rng, state, action, params)
            win = state.win
            print(multihot_to_desc(state.multihot_level, env.obj_to_idxs, env.n_objs))
            im = env.render(state)
            im = np.array(im, dtype=np.uint8)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
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
            print(multihot_to_desc(state.multihot_level, env.obj_to_idxs, env.n_objs))
            state_hist.append(state)
            im = env.render(state)
            im = np.array(im, dtype=np.uint8)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
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
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)
    # Close the image window
    cv2.destroyAllWindows()


def play_game(game: str, jit: bool = False, profile: bool = False, debug: bool = False):
    with open("syntax.lark", "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    # min_parser = Lark(min_puzzlescript_grammar, start="ps_game")
    print(f"""Parsing game: \"{game}\"""")
    tree = get_tree_from_txt(parser, game, overwrite=True)
    print(f"Initializing environment for game: {game}")
    env = PSEnv(tree, jit=jit, debug=debug)
    print(f"Playing game: {game}")
    human_loop(env, profile=profile)

@hydra.main(config_name="config", version_base="1.3")
def main(cfg: Config):

    if cfg.game is not None:
        play_game(cfg.game, jit=cfg.jit, profile=cfg.profile, debug=cfg.debug)

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