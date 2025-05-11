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
            with open(game_tree_path, "rb") as f:
                parse_tree = pickle.load(f)
            # parse_tree = get_tree_from_txt(parser, game_name)
            tree: PSGameTree = GenPSTree().transform(parse_tree)
            env = PSEnv(tree, level_i=0)
            n_rules = count_rules(tree)
            games_n_rules.append((game_name, n_rules))
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error parsing {game_name}: {e}")
            pass
        
    print(f"Total games: {len(games_n_rules)}")
    games_n_rules = sorted(games_n_rules, key=lambda x: x[1])

    # Save the sorted list to a json
    with open('data', 'games_n_rules.json', 'w') as f:
        json.dump(games_n_rules, f, indent=4)


if __name__ == '__main__':
    main()