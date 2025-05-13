import glob
import json
import os
import pickle
import traceback

from lark import Lark

from env import PSEnv
from gen_tree import GenPSTree
from parse_lark import TREES_DIR, GAMES_DIR, count_rules, get_tree_from_txt
from ps_game import PSGameTree
from utils import GAMES_N_RULES_SORTED_PATH


GAMES_N_RULES_SORTED_PATH = os.path.join('data', 'games_n_rules.json')


def main():
    with open("syntax.lark", "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    game_tree_paths = glob.glob(os.path.join(TREES_DIR, '*'))
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    games_n_rules = []
    for game_tree_path in game_tree_paths:
        print(game_tree_path)
        game_name = os.path.basename(game_tree_path)[:-4]
        og_game_path = os.path.join(GAMES_DIR, game_name + '.txt')
        try:
            # with open(game_tree_path, "rb") as f:
            #     parse_tree = pickle.load(f)
            # ps_tree: PSGameTree = GenPSTree().transform(parse_tree)
            ps_tree, err_msg, success = get_tree_from_txt(parser, game_name, test_env_init=False)
            env = PSEnv(ps_tree)
            has_randomness = env.has_randomness()
            n_rules = count_rules(ps_tree)
            games_n_rules.append((game_name, n_rules, has_randomness))
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error parsing/initializing {og_game_path}: {e}")
            pass
        
    print(f"Total games: {len(games_n_rules)}")
    games_n_rules = sorted(games_n_rules, key=lambda x: x[1])

    # Save the sorted list to a json
    with open(GAMES_N_RULES_SORTED_PATH, 'w') as f:
        json.dump(games_n_rules, f, indent=4)


if __name__ == '__main__':
    main()