import os


PRIORITY_GAMES = [
    # 'sokoban_basic',
    # 'test_sokoban_objs_10',
    # 'test_sokoban_objs_20',
    # 'test_sokoban_lvls_14',
    # 'test_sokoban_lvls_21',
    # 'test_sokoban_rules_5',
    # 'test_sokoban_rules_10',

    # 'tiny treasure hunt',
    # 'Watch_Your_Step',
    # 'Indigestion',
    # 'castlemouse',
    # 'Sokodig',
    # 'nekopuzzle'

    'Multi-word_Dictionary_Game',
    'Take_Heart_Lass',
    'Travelling_salesman',
    'Zen_Puzzle_Garden',
    'atlas shrank',
    'blocks',
    'kettle',
    'limerick',
    'notsnake',
    'slidings',
    'sokoban_basic',
    'sokoban_match3',
    'constellationz',
]

GAMES_TO_SKIP = set({'easyenigma', 'A_Plaid_Puzzle'})
PLOTS_DIR = 'plots'
GAMES_N_RULES_SORTED_PATH = os.path.join('data', 'games_n_rules.json')
GAMES_TO_N_RULES_PATH = os.path.join('data', 'games_to_n_rules.json')
STANDALONE_NODEJS_RESULTS_PATH = os.path.join('data', 'standalone_nodejs_results.json')
GAMES_N_LEVELS_PATH = os.path.join('data', 'games_n_levels.json')
SOLUTION_REWARDS_PATH = os.path.join('data', 'solution_rewards.json')
JAX_PROFILING_RESULTS_DIR = os.path.join('data', 'jax_profiling_results')
JAX_VALIDATED_JS_SOLS_DIR = os.path.join('data', 'jax_validated_js_sols')
JS_TO_JAX_ACTIONS = [3, 0, 1, 2, 4]
JS_SOLS_DIR = os.path.join('data', 'js_sols')


