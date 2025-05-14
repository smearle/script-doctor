PRIORITY_GAMES = [
    # 'blank',
    'sokoban_basic',
    'sokoban_match3',
    'limerick',
    'blocks',
    'slidings',
    'notsnake',
    'Travelling_salesman',
    'Zen_Puzzle_Garden',
    'tiny treasure hunt',
    # 'Sokodig',
    # 'Watch_Your_Step',
    'Multi-word_Dictionary_Game',
    'Take_Heart_Lass',
    'atlas shrank',
    'castlemouse',
    'Indigestion',
]

import os
GAMES_TO_SKIP = set({'easyenigma', 'A_Plaid_Puzzle'})
PLOTS_DIR = 'plots'
GAMES_N_RULES_SORTED_PATH = os.path.join('data', 'games_n_rules.json')
GAMES_TO_N_RULES_PATH = os.path.join('data', 'games_to_n_rules.json')

