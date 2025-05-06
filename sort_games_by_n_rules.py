import glob
import json
import os
import pickle
import traceback

from env import PSEnv
from gen_tree import GenPSTree
from parse_lark import TREES_DIR, GAMES_DIR
from ps_game import PSGameTree


if __name__ == '__main__':
    game_tree_paths = glob.glob(os.path.join(TREES_DIR, '*'))
    games_n_rules = []
    for game_tree_path in game_tree_paths:
        game_name = os.path.basename(game_tree_path)[:-4]
        og_game_path = os.path.join(GAMES_DIR, game_name + '.txt')
        parse_tree = pickle.load(open(game_tree_path, 'rb'))
        try:
            breakpoint()
            tree: PSGameTree = GenPSTree().transform(parse_tree)
            env = PSEnv(tree, level_i=0)
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error parsing {game_name}: {e}")
            continue
        n_rules = len(tree.rules)
        games_n_rules.append((game_name, n_rules))
        
    games_n_rules = sorted(games_n_rules, key=lambda x: x[1])

    # Save the sorted list to a json
    with open('games_n_rules.json', 'w') as f:
        json.dump(games_n_rules, f, indent=4)
    



