"""
DEPRECATED

This all happens in the preprocess_games.py file now.
"""
import glob
import json
import os
import pickle
import traceback

import hydra
from lark import Lark

from conf.config import PreprocessConfig
from puzzlejax.env import PuzzleJaxEnv
from puzzlejax.gen_tree import GenPSTree
from preprocess_games import TREES_DIR, GAMES_DIR, count_rules, get_tree_from_txt
from puzzlejax.ps_game import PSGameTree
from puzzlejax.utils import GAMES_N_RULES_SORTED_PATH
from globals import GAMES_TO_N_RULES_PATH, GAMES_N_RULES_SORTED_PATH, LARK_SYNTAX_PATH


@hydra.main(version_base="1.3", config_path="conf", config_name="preprocess_config")
def main(cfg: PreprocessConfig):
    with open(LARK_SYNTAX_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    scraped_games_paths = glob.glob(os.path.join(GAMES_DIR, '*'))
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    if cfg.overwrite:
        games_n_rules = []
    else:
        with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
            games_n_rules = json.load(f)
        games_n_rules_dict = {game: (n_rules, has_randomness) for game, n_rules, has_randomness in games_n_rules}
        print(f"Loaded {len(games_n_rules)} games from {GAMES_N_RULES_SORTED_PATH}")
    for game_path in scraped_games_paths:
        print(f"Processing {game_path}")
        game_name = os.path.basename(game_path)[:-4]
        if game_name in games_n_rules_dict:
            print(f"Skipping {game_name}: already in the list")
            continue
        og_game_path = os.path.join(GAMES_DIR, game_name + '.txt')
        try:
            # with open(game_tree_path, "rb") as f:
            #     parse_tree = pickle.load(f)
            # ps_tree: PSGameTree = GenPSTree().transform(parse_tree)
            ps_tree, err_msg, success = get_tree_from_txt(parser, game_name, test_env_init=False)
            env = PuzzleJaxEnv(ps_tree)
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

    games_to_n_rules = {game: (n_rules, has_randomness) for game, n_rules, has_randomness in games_n_rules}
    with open(GAMES_TO_N_RULES_PATH, 'w') as f:
        json.dump(games_to_n_rules, f, indent=4)


if __name__ == '__main__':
    main()