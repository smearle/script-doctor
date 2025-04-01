
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
import glob
import os
import pickle
import cv2
import numpy as np

from env import PSEnv, multihot_to_desc
from gen_trees import GenPSTree
from parse_lark import TREES_DIR, DATA_DIR, TEST_GAMES
from ps_game import PSGame



@dataclass
class Config:
    jit: bool = True


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def human_loop(env: PSEnv):
    lvl_i = 0 
    state = env.reset(lvl_i)
    im = env.render(state)
    print(multihot_to_desc(state.multihot_level, env.obj_to_idxs))
    im = np.array(im, dtype=np.uint8)
    
    # Resize the image by a factor of 5
    new_h, new_w = tuple(np.array(im.shape[:2]) * 10)
    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    state_hist = []
    
    # Display the image in an OpenCV window
    cv2.imshow(env.title, im)
    print("Press an arrow key or 'x' (ESC to exit).")
    
    # Loop waiting for key presses
    while True:
        # cv2.waitKey(0) waits indefinitely until a key is pressed.
        # Mask with 0xFF to get the lowest 8 bits (common practice).
        key = cv2.waitKey(0)
        action = None
        do_reset = False

        # If the user presses ESC (ASCII 27), exit the loop.
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
            do_reset = True
        elif key == ord('z'):
            print("Undoing last action...")
            if len(state_hist) > 1:
                state_hist.pop()
                state = state_hist[-1]
            im = env.render(state)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            im = np.array(im, dtype=np.uint8)
            cv2.imshow(env.title, im)

        else:
            print("Other key pressed:", key)

        if lvl_i >= len(env.levels):
            print("No more levels!")
            break

        elif do_reset:
            state = env.reset(lvl_i)
            print(multihot_to_desc(state.multihot_level, env.obj_to_idxs))
            state_hist.append(state)
            im = env.render(state)
            im = np.array(im, dtype=np.uint8)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)
    
        elif action is not None:
            lvl = env.apply_player_force(action, state)
            vis_lvl = lvl[:env.n_objs]
            lvl_changed = True
            n_vis_apps = 0
            state = env.step(action, state)
            print(multihot_to_desc(state.multihot_level, env.obj_to_idxs))
            im = env.render(state)
            im = np.array(im, dtype=np.uint8)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)
            # Add a short waitKey here to allow the window to update.
            cv2.waitKey(1)  # 1 ms delay; adjust as necessary
            win = env.check_win(vis_lvl)
            if win:
                print("You win!")
            else:
                print("You don't win yet!")
            state_hist.append(state)

        if state.win:
            lvl_i += 1
            state = env.reset(lvl_i)
            im = env.render(state)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)
    # Close the image window
    cv2.destroyAllWindows()


@hydra.main(config_name="config", version_base="1.3")
def main(cfg: Config):
    tree_paths = glob.glob(os.path.join(TREES_DIR, '*'))
    trees = []
    tree_paths = sorted(tree_paths, reverse=True)
    test_game_paths = [os.path.join(TREES_DIR, tg + '.pkl') for tg in TEST_GAMES]
    tree_paths = test_game_paths + tree_paths
    for tree_path in tree_paths:
        print(tree_path)
        og_game_path = os.path.join(DATA_DIR, 'scraped_games', os.path.basename(tree_path)[:-3] + 'txt')
        print(f"Parsing {og_game_path}")
        with open(tree_path, 'rb') as f:
            tree = pickle.load(f)
        trees.append(tree)

        tree: PSGame = GenPSTree().transform(tree)

        env = PSEnv(tree, jit=cfg.jit)
        human_loop(env)


if __name__ == '__main__':
    main()