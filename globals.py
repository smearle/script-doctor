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

par_dir = os.path.dirname(os.path.abspath(__file__))

CUSTOM_GAMES_DIR = os.path.join(par_dir, 'custom_games')
DATA_DIR = os.path.join(par_dir, 'data')
GAMES_TO_SKIP = set({'easyenigma', 'A_Plaid_Puzzle'})
PLOTS_DIR = 'plots'
GAMES_N_RULES_SORTED_PATH = os.path.join(DATA_DIR, 'games_n_rules.json')
GAMES_TO_N_RULES_PATH = os.path.join(DATA_DIR, 'games_to_n_rules.json')
STANDALONE_NODEJS_RESULTS_PATH = os.path.join(DATA_DIR, 'standalone_nodejs_results.json')
GAMES_N_LEVELS_PATH = os.path.join(DATA_DIR, 'games_n_levels.json')
SOLUTION_REWARDS_PATH = os.path.join(DATA_DIR, 'solution_rewards.json')
JAX_PROFILING_RESULTS_DIR = os.path.join(DATA_DIR, 'jax_profiling_results')
JAX_VALIDATED_JS_SOLS_DIR = os.path.join(DATA_DIR, 'jax_validated_js_sols')
JS_TO_JAX_ACTIONS = [3, 0, 1, 2, 4]
JS_SOLS_DIR = os.path.join(DATA_DIR, 'js_sols')
LARK_SYNTAX_PATH = os.path.join(par_dir, 'syntax.lark')
GAMES_DIR = os.path.join(DATA_DIR, 'scraped_games')
MIN_GAMES_DIR = os.path.join(DATA_DIR, 'min_games')
SIMPLIFIED_GAMES_DIR = os.path.join(DATA_DIR, 'simplified_games')
TEST_GAMES = ["sokoban_basic", "blockfaker", "notsnake", "sokoban_match3"]
TREES_DIR = os.path.join(DATA_DIR, 'game_trees')
PRETTY_TREES_DIR = os.path.join(DATA_DIR, 'pretty_trees')
# parsed_games_filename = os.path.join(DATA_DIR, "parsed_games.txt")
