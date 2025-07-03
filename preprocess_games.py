import argparse
from enum import IntEnum
import glob
import json
import logging
import os
import pickle
import re
import shutil
import traceback
import signal  # Add this import
import contextlib
from typing import Optional

import hydra
import lark
from lark import Lark, Transformer, Tree, Token, Visitor
import numpy as np

from conf.config import PreprocessConfig
from env import PSEnv
from gen_tree import GenPSTree
from globals import GAMES_N_RULES_SORTED_PATH, GAMES_TO_N_RULES_PATH, GAMES_TO_SKIP, GAMES_N_LEVELS_PATH
from ps_game import PSGameTree

logger = logging.getLogger(__name__)

# TEST_GAMES = ['blockfaker', 'sokoban_match3', 'notsnake', 'sokoban_basic']
TEST_GAMES = []

DATA_DIR = 'data'
PS_LARK_GRAMMAR_PATH = "syntax.lark"
GAMES_DIR = os.path.join(DATA_DIR, 'scraped_games')
MIN_GAMES_DIR = os.path.join(DATA_DIR, 'min_games')
CUSTOM_GAMES_DIR = os.path.join('custom_games')
SIMPLIFIED_GAMES_DIR = os.path.join(DATA_DIR, 'simplified_games')
TREES_DIR = os.path.join(DATA_DIR, 'game_trees')
pretty_trees_dir = os.path.join(DATA_DIR, 'pretty_trees')
# parsed_games_filename = os.path.join(DATA_DIR, "parsed_games.txt")


@contextlib.contextmanager
def timeout_handler(seconds: int):
    """Context manager for handling timeouts using signals"""
    def signal_handler(signum, frame):
        raise TimeoutError("Parsing timed out")
        
    # Store previous handler
    previous_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)  # Disable alarm
        signal.signal(signal.SIGALRM, previous_handler)  # Restore previous handler



class StripPuzzleScript(Transformer):
    """
    Reduces the parse tree to a minimal functional version of the grammar.
    """
    def message(self, items):
        return None

    def strip_newlines_data(self, items, data_name):
        """Remove any instances of section data that are newlines/comments"""
        items = [item for item in items if not (isinstance(item, Tree) and item.data == "newlines_or_comments")]
        items = [item for item in items if not (isinstance(item, Token) and (item.type == "NEWLINES" or item.type == "NEWLINE"))]
        if len(items) > 0:
            return Tree(data_name, items)

    def strip_section_items(self, items, data_name):
        """Remove any empty section items (e.g. resulting from returning None above, when encountering a datum that is all newlines/comments)"""
        return [item for item in items if isinstance(item, Tree) and item.data == data_name]        

    def ps_game(self, items):
        items = [item for item in items if type(item) == Tree]
        return Tree('ps_game', items)

    def objects_section(self, items):
        return Tree('objects_section', self.strip_section_items(items, 'object_data'))

    def legend_section(self, items):
        return Tree('legend_section', self.strip_section_items(items, 'legend_data'))

    def levels_section(self, items):
        items = self.strip_section_items(items, 'level_data')
        items = [i for i in items if i]
        return Tree('levels_section', items)

    def winconditions_section(self, items):
        return Tree('winconditions_section', self.strip_section_items(items, 'condition_data'))
    
    def collisionlayers_section(self, items):
        return Tree('collisionlayers_section', self.strip_section_items(items, 'layer_data'))

    def rules_section(self, items):
        return Tree('rules_section', self.strip_section_items(items, 'rule_block'))

    def sounds_section(self, items):
        return

    def prelude_data(self, items):
        return self.strip_newlines_data(items, 'prelude_data')

    def object_data(self, items):
        return self.strip_newlines_data(items[0].children, 'object_data')

    def level_data(self, items):
        return self.strip_newlines_data(items, 'level_data')
        # Remove any Tokens

    def legend_data(self, items):
        return self.strip_newlines_data(items, 'legend_data')
    
    def rule_data(self, items):
        return self.strip_newlines_data(items, 'rule_data')

    def rule_block_once(self, items):
        items = [i for i in items if i]
        return self.strip_newlines_data(items, 'rule_block_once')

    def rule_block_loop(self, items):
        items = [i for i in items if i]
        return self.strip_newlines_data(items, 'rule_block_loop')
    
    def line_detector(self, items):
        return "..."

    # def rule_block(self, items):
    #     return items[0]
    
    def condition_data(self, items):
        return self.strip_newlines_data(items, 'condition_data')

    def layer_data(self, items):
        return self.strip_newlines_data(items, 'layer_data')

    def shape_2d(self, items):
        # Create a 2D array of the items
        grid = []
        row = []
        for s in items:
            # If we encounter a newline, start a new row
            if s == "\n":
                if len(row) > 0:
                    grid.append(row)
                    row = []
            else:
                row.append(s.value)
        row_lens = [len(r) for r in grid]
        if len(set(row_lens)) > 1:
            raise ValueError(f"Rows in grid have different lengths: {row_lens}")
        grid = np.array(grid)
        return grid

    def sprite(self, items):
        # Remote any item that is a message
        items = [i for i in items if not (isinstance(i, Token) and i.type == 'COMMENT')]
        return Tree('sprite', self.shape_2d(items))

    def levelline(self, items):
        line = [str(i) for i in items]
        assert line[-1] == "\n"
        return line[:-1]

    def levellines(self, items):
        grid = []
        level_lines = items
        grid = [line for line in level_lines[:-1]]
        # TODO: Thich of these does OG PS do? Does it do different things in different cases? :/
        # pad all rows with empty tiles
        max_len = max(len(row) for row in grid)
        for row in grid:
            row += row[-1] * (max_len - len(row))
        # Truncate all the rows to the same length
        # row_lens = [len(r) for r in grid]
        # if len(set(row_lens)) > 1:
        #     logger.warning(f"Rows in grid have different lengths: {row_lens}. Truncating to the shortest row.")
        #     min_len = min(row_lens)
        #     for i, row in enumerate(grid):
        #         if len(row) > min_len:
        #             grid[i] = row[:min_len]
        grid = np.array(grid)
        return grid


class RepairPuzzleScript(Transformer):
    def object_data(self, items):
        # If we're missing colors, add random ones
        child_trees = [i for i in items if isinstance(i, Tree)]
        child_tree_names = [i.data for i in child_trees]
        colors = [i for i in child_trees if i.data == 'color_line']
        n_colors = 0
        if len(colors) > 0:
            colors = colors[0]
            n_colors = len(colors.children)
        if 'sprite' in child_tree_names:
            sprite = [i for i in child_trees if i.data == 'sprite'][0]
            sprite_arr = np.array(sprite.children)
            if sprite_arr.shape != (5, 5):
                raise ValueError(f"Sprite shape is not 5x5: {sprite_arr.shape}")
            n_unique_pixels = np.vectorize(lambda x: int(x) if x != '.' else 0)(sprite_arr).max() + 1
            while n_unique_pixels > n_colors:
                colors.children.append(Token('COLOR', f'#{np.random.randint(0, 0xFFFFFF):06X}'))
                n_colors += 1
        return Tree('object_data', items)


def array_2d_to_str(arr):
    s = ""
    for row in arr:
        s += "".join(row) + "\n"
    return s


class PrintPuzzleScript(Transformer):
    def ps_game(self, items):
        return "\n\n".join(items)

    def prefix(self, items):
        return items[0].value

    def colors(self, items):
        return ' '.join(items)

    def color(self, items):
        return items[0].value

    def prelude(self, items):
        return '\n'.join(items)

    def prelude_data(self, items):
        return ' '.join(items)

    def rule_data(self, items):
        return ' '.join([item for item in items if item])

    def cell_border(self, items):
        return " | "

    def object_name(self, items):
        return items[0].value

    def command_keyword(self, items):
        return ' '.join(items)

    def sound(self, items):
        return ''

    def command(self, items):
        return ' '.join(items)

    def legend_data(self, items):
        # Omitting final NEWLINES
        return str(items[0])[:-1].strip() + ' = ' + ' '.join(items[1:])

    def legend_operation(self, items):
        if len(items) == 1:
            return items[0]
        else:
            return ' '.join(items)

    def legend_key(self, items):
        return items[0].value

    def object_name(self, items):
        return items[0].value

    def sprite(self, arr):
        return array_2d_to_str(arr)[:-1]

    def level_data(self, items):
        return '\n'.join(items)

    def levellines(self, arr):
        return array_2d_to_str(arr)
    
    def object_data(self, items):
        return '\n'.join(items) + '\n'

    def object_line(self, items):
        return ' '.join(items)

    def color_line(self, items):
        return ' '.join(items)
     
    def layer_data(self, items):
        return ', '.join(items)

    def rule_part(self, items):
        return '[ ' + ' '.join(items) + ' ]'

    def rule_block(self, items):
        return items[0]

    def rule_block_once(self, items):
        return '\n'.join([item for item in items if item])

    def rule_block_loop(self, items):
        return ''.join(items)

    def rule_content(self, items):
        return ' '.join(items)

    def line_detector(self, items):
        return '...'

    def rule_object(self, items):
        return ' '.join(items)
    
    def rule_object_with_modifier(self, items):
        return ' '.join(items)

    def levels_section(self, items):
        return 'LEVELS\n\n' + '\n'.join([i for i in items if i])

    def objects_section(self, items):
        return 'OBJECTS\n\n' + '\n'.join(items)
    
    def legend_section(self, items):
        return 'LEGEND\n\n' + '\n'.join(items)
    
    def rules_section(self, items):
        return 'RULES\n\n' + ''.join(items)

    def condition_id(self, items):
        return items[0].value

    def condition_data(self, items):
        return ' '.join(items)
    
    def winconditions_section(self, items):
        return 'WINCONDITIONS\n\n' + '\n'.join(items)
    
    def collisionlayers_section(self, items):
        return 'COLLISIONLAYERS\n\n' + '\n'.join(items)

    def level_data(self, arr):
        if arr[0] is not None:
            return array_2d_to_str(arr[0])


def add_empty_sounds_section(txt):
    ret = re.search(r'^SOUNDS\n', txt, re.MULTILINE)
    if ret is None:
        txt = re.sub(r'^COLLISIONLAYERS\n', 'SOUNDS\n\nCOLLISIONLAYERS\n', txt)
    return txt


def preprocess_rules(txt):
    # Replace any occurrence of `]...[` with `|...|`
    txt = re.sub(r'\]\s*\.\.\.\s*\[', ' | ... | ', txt)
    # Replace any occurrence of `] | [` with `] [`
    txt = re.sub(r'\]\s*\|\s*\[', '] [', txt)
    return txt

def preprocess_collisionlayers(txt):
    # Replace any pairs of commas, separated by whitespace, with a single comma
    txt = re.sub(r',\s*,', ',', txt)
    return txt

def preprocess_ps(txt):

    # Remove whitespace at end of any line
    txt = re.sub(r'[ \t]+$', '', txt, flags=re.MULTILINE)

    # Remove whitespace at start of any line
    txt = re.sub(r'^[ \t]+', '', txt, flags=re.MULTILINE)

    txt = add_empty_sounds_section(txt)

    # If the regular `LEGEND` header is not found, try the `LEGEND` header followed by some trailing characters
    if not re.search(r'^LEGEND\n', txt, flags=re.MULTILINE | re.IGNORECASE):
        txt = re.sub(r'^LEGEND\s*.*\n', 'LEGEND\n', txt, flags=re.MULTILINE | re.IGNORECASE)

    txt = txt.replace('\u00A0', ' ')
    # If the file does not end with 2 newlines, fix this
    for i in range(2):
        if not txt.endswith("\n\n"):
            txt += "\n"

    # Remove any lines beginning with "message" (case insensitive)
    txt = re.sub(r'^message.*\n', '\n', txt, flags=re.MULTILINE | re.IGNORECASE)

    # Truncate lines ending with "message"
    txt = re.sub(r'message.*\n', '\n', txt, flags=re.MULTILINE | re.IGNORECASE)

    ## Strip any comments
    txt = strip_comments(txt)

    # Remove any lines that are just r`=+` (or actually, at least 3 `=` followed by one accidental character;
    # a hack to get around some typos in the dataset)
    txt = re.sub(r'^===*.\n', '', txt, flags=re.MULTILINE)

    # Remove any lines that are just whitespace
    txt = re.sub(r'^\s*\n', '\n', txt, flags=re.MULTILINE)

    # any more-than-double newlines should be replaced by a double newline
    txt = re.sub(r'\n{3,}', '\n\n', txt)

    # Remove any lines that are just a single character. (Very niche patch, this one is. But we know such lines can 
    # never be anything useful, so this should be safe...)
    txt = re.sub(r'^[.]\n', '', txt, flags=re.MULTILINE)

    # Remove everything until "objects" (case insensitive)
    # txt = re.sub(r'^.*OBJECTS', 'OBJECTS', txt, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    sections_pattern = r"""
        ^OBJECTS\n|
        ^LEGEND\n|
        ^SOUNDS\n|
        ^COLLISIONLAYERS\n|
        ^RULES\n|
        ^WINCONDITIONS\n|
        ^LEVELS\n
    """

    sections = re.split(sections_pattern, txt, flags=re.MULTILINE | re.VERBOSE | re.IGNORECASE)
    prelude_section, objects_section, legend_section, sounds_section, collisionlayers_section, rules_section, \
        winconditions_section, levels_section = sections

    rules_section = preprocess_rules(rules_section)
    collisionlayers_section = preprocess_collisionlayers(collisionlayers_section)

    # Now put the sections back together
    txt = (f"{prelude_section}\n"
           f"OBJECTS\n{objects_section}"
           f"LEGEND\n{legend_section}"
           f"SOUNDS\n{sounds_section}"
           f"COLLISIONLAYERS\n{collisionlayers_section}"
           f"RULES\n{rules_section}"
           f"WINCONDITIONS\n{winconditions_section}"
           f"LEVELS\n{levels_section}")

    return txt.lstrip()


def strip_comments(text):
    new_text = ""
    n_open_brackets = 0
    # Move through the text, keeping track of how deep we are in brackets
    for i, c in enumerate(text):
        if c == "(":
            n_open_brackets += 1
        elif c == ")":
            # we ignore unmatched closing brackets if we are outside
            new_n_open_brackets = max(0, n_open_brackets - 1)
            if new_n_open_brackets == 0 and n_open_brackets == 1:
                # If the removed comment has left us with a double-newline (because there was a newline on either side 
                # of it), convert it to a single newline
                if new_text.endswith("\n") and text[i+1] == "\n":
                    new_text = new_text[:-1]
            n_open_brackets = new_n_open_brackets
        elif n_open_brackets == 0:
            new_text += c
    return new_text

def count_rules(tree: PSGameTree):
    n_rules = 0
    for rule_block in tree.rules:
        n_rules += len(rule_block[0].rules)
    return n_rules

class PSErrors(IntEnum):
    SUCCESS = 0
    PARSE_ERROR = 1
    TREE_ERROR = 2
    ENV_ERROR = 3
    TIMEOUT = 4
    SKIPPED = 5
    PREPROCESSING_ERROR = 6

def get_env_from_ps_file(parser, game, log_dir: str = None, overwrite: bool = True):
    tree, success, err_msg = get_tree_from_txt(parser, game, log_dir, overwrite, test_env_init=False)
    if success != PSErrors.SUCCESS:
        return None, tree, success, err_msg 
    try:
        env = PSEnv(tree)
        return env, tree, PSErrors.SUCCESS, ""
    except Exception as e:
        traceback.print_exc()
        print(f"Error initializing environment for {game}: {e}")
        return None, tree, PSErrors.ENV_ERROR, gen_error_str(e)

# Keeping this here only for backwards compatibility
def get_tree_from_txt(parser, game, log_dir: str = None, overwrite: bool = True, test_env_init: bool = True):
    filepath = os.path.join(CUSTOM_GAMES_DIR, game + '.txt')
    if not os.path.exists(filepath):
        filepath = os.path.join(GAMES_DIR, game + '.txt')
    print(f"Parsing {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        ps_text = f.read()
    simp_filename = game + '_simplified.txt' 
    # if game in parsed_games or os.path.basename(game) in games_to_skip:
    if os.path.basename(game) in GAMES_TO_SKIP:
        print(f"Skipping {filepath} because it has been marked for skippping in `GAMES_TO_SKIP`")
        return None, PSErrors.SKIPPED, "Game marked for skipping in GAMES_TO_SKIP"

    # print(f"Parsing game {filepath} ({i+1}/{len(game_files)})")
    simp_filepath = os.path.join(SIMPLIFIED_GAMES_DIR, simp_filename)
    os.makedirs(SIMPLIFIED_GAMES_DIR, exist_ok=True)
    os.makedirs(pretty_trees_dir, exist_ok=True)
    os.makedirs(MIN_GAMES_DIR, exist_ok=True)
    if overwrite or not os.path.exists(simp_filepath):
        # Now save the simplified version of the file
        try:
            content = preprocess_ps(ps_text)
        except ValueError as e:
            print(f"Error preprocessing {filepath}: {e}")
            return None, PSErrors.PREPROCESSING_ERROR, gen_error_str(e)
        with open(simp_filepath, "w", encoding='utf-8') as file:
            file.write(content)
    else:
        with open(simp_filepath, "r", encoding='utf-8') as file:
            content = file.read()
    # print(f"Parsing {simp_filepath}")
    
    log_filename = None
    if log_dir:
        log_filename = os.path.join(log_dir, game + '.log')

    # This timeout functionality only works on Unix
    print(f"Parsing {simp_filepath}")
    if os.name != 'nt':
        def parse_attempt_fn():
            with timeout_handler(10):
                return parser.parse(content)
    # FIXME: On windows, this will hang indefinitely on nasty games :(
    else:
        def parse_attempt_fn():
            return parser.parse(content)

    try:
        parse_tree = parse_attempt_fn()

    except TimeoutError:
        print(f"Timeout parsing {simp_filepath}")
        if log_filename:
            with open(log_filename, 'w') as file:
                file.write("timeout")
            print(f"Timeout parsing {simp_filepath}")
            # with open(parsed_games_filename, 'a') as file:
            #     file.write(game + "\n")
        return None, PSErrors.TIMEOUT, ""
    except Exception as e:
        print(traceback.format_exc())
        if log_filename:
            with open(log_filename, 'w') as file:
                traceback.print_exc(file=file)

            print(f"Error parsing {simp_filepath}:\n{e}")
            # with open(parsed_games_filename, 'a') as file:
            #     file.write(game + "\n")
        return None, PSErrors.PARSE_ERROR, gen_error_str(e)


    min_parse_tree = StripPuzzleScript().transform(parse_tree)
    min_tree_path = os.path.join(TREES_DIR, game + '.pkl')
    # print(f"Writing parse tree to {min_tree_path}")
    with open(min_tree_path, "wb") as f:
        pickle.dump(min_parse_tree, f)
    pretty_parse_tree_str = min_parse_tree.pretty()
    pretty_tree_filename = os.path.join(pretty_trees_dir, game)
    print(f"Writing pretty tree to {pretty_tree_filename}")
    with open(pretty_tree_filename, "w", encoding='utf-8') as file:
        file.write(pretty_parse_tree_str)
    # print(min_parse_tree.pretty())
    ps_str = PrintPuzzleScript().transform(min_parse_tree)
    ps_str = add_empty_sounds_section(ps_str)
    min_filename = os.path.join(MIN_GAMES_DIR, game + '.txt')
    # print(f"Writing minified game to {min_filename}")
    with open(min_filename, "w", encoding='utf-8') as file:
        file.write(ps_str)

    # with open(parsed_games_filename, 'a') as file:
    #     file.write(game + "\n")

    try:
        tree: PSGameTree = GenPSTree().transform(min_parse_tree)
    except Exception as e:
        traceback.print_exc()
        print(f"Error transforming tree: {game}")
        return None, PSErrors.TREE_ERROR, gen_error_str(e)
    if test_env_init:
        try:
            env = PSEnv(tree, level_i=0)
        except Exception as e:
            traceback.print_exc()
            print(f"Error initializing environment for {game}: {e}")
            return tree, PSErrors.ENV_ERROR, gen_error_str(e)

    print(f"Parsed {game} successfully")
    return tree, PSErrors.SUCCESS, ""

def gen_error_str(e):
    err_msg = f"{traceback.format_exc()}\n{type(e).__name__}: {e}"
    return err_msg


@hydra.main(version_base="1.3", config_path="conf", config_name="preprocess_config")
def main(cfg: PreprocessConfig):

    with open(PS_LARK_GRAMMAR_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    with open("syntax_generate.lark", "r", encoding='utf-8') as file:
        min_puzzlescript_grammar = file.read()

    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    # min_parser = Lark(min_puzzlescript_grammar, start="ps_game")

    # games_dir = os.path.join('script-doctor','games')

    os.makedirs(TREES_DIR, exist_ok=True)
    os.makedirs(pretty_trees_dir, exist_ok=True)
    os.makedirs(MIN_GAMES_DIR, exist_ok=True)
    parse_results = {
        'stats': {},
        'success': [],
        'preprocess_error': {},
        'parse_error': {},
        'tree_error': {},
        'env_error': {},
        'parse_timeout': [],
    }
    parse_results_path = os.path.join('data', 'parse_results.json')
    # Copy a backup of previous results if it exists
    if os.path.exists(parse_results_path):
        shutil.copyfile(parse_results_path, parse_results_path[:-5] + "_bkp" + ".json")

    # min_grammar = os.path.join('syntax_generate.lark')
    # if args.overwrite or not os.path.exists(parsed_games_filename):
    #     with open(parsed_games_filename, "w") as file:
    #         file.write("")
    # with open(parsed_games_filename, "r", encoding='utf-8') as file:
    #     # Get the set of all lines from this text file
    #     parsed_games = set(file.read().splitlines())
    # for i, filename in enumerate(['blank.txt'] + os.listdir(demo_games_dir)):
    if cfg.game is None:
        game_files = os.listdir(GAMES_DIR)
    else:
        game_files = [cfg.game + '.txt']
    # sort them alphabetically
    game_files.sort()
    test_game_files = [f"{test_game}.txt" for test_game in TEST_GAMES]
    game_files = test_game_files + game_files

    os.makedirs(SIMPLIFIED_GAMES_DIR, exist_ok=True)
    scrape_log_dir = 'scrape_logs'
    os.makedirs(scrape_log_dir, exist_ok=True)
    games_n_rules_sorted = []
    games_to_n_rules = {}
    if not cfg.overwrite:
        if os.path.exists(GAMES_N_RULES_SORTED_PATH):
            with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
                games_n_rules_sorted = json.load(f)
            games_to_n_rules = {game: (n_rules, has_randomness) for game, n_rules, has_randomness in games_n_rules_sorted}
            print(f"Loaded {len(games_n_rules_sorted)} games from {GAMES_N_RULES_SORTED_PATH}")
        if os.path.exists(parse_results_path):
            with open(parse_results_path, 'r') as f:
                parse_results = json.load(f)
            print(f"Loaded {len(parse_results['success'])} games from {parse_results_path}")

    for i, filename in enumerate(game_files):
        game_name = os.path.basename(filename)
        if not cfg.overwrite and game_name in games_to_n_rules:
            # We can assume we initialized the environment successfully then
            print(f"Skipping {game_name}: already in the list")
            continue

        og_game_path = os.path.join(GAMES_DIR, filename)
        print(f"Parsing {filename} ({i+1}/{len(game_files)})")
        env, ps_tree, success, err_msg = get_env_from_ps_file(parser, filename[:-4], log_dir=scrape_log_dir, overwrite=cfg.overwrite)

        if success == PSErrors.SUCCESS:
            parse_results['success'].append(game_name)
            n_rules = count_rules(ps_tree)
            has_randomness = env.has_randomness()
            games_n_rules_sorted.append((game_name, n_rules, has_randomness))
            games_to_n_rules[game_name] = (n_rules, has_randomness)
        elif success == PSErrors.PARSE_ERROR:
            if err_msg not in parse_results['parse_error']:
                parse_results['parse_error'][err_msg] = []
            parse_results['parse_error'][err_msg].append(game_name)
        elif success == PSErrors.TIMEOUT:
            parse_results['parse_timeout'].append(game_name)
        elif success == PSErrors.TREE_ERROR:
            if err_msg not in parse_results['tree_error']:
                parse_results['tree_error'][err_msg] = []
            parse_results['tree_error'][err_msg].append(game_name)
        elif success == PSErrors.ENV_ERROR:
            if err_msg not in parse_results['env_error']:
                parse_results['env_error'][err_msg] = []
            n_rules = count_rules(ps_tree)
            has_randomness = None
            parse_results['env_error'][err_msg].append((game_name, n_rules))
            games_n_rules_sorted.append((game_name, n_rules, has_randomness))
            games_to_n_rules[game_name] = (n_rules, has_randomness)
        elif success == PSErrors.SKIPPED:
            print(f"Skipping {game_name} because it has been marked for skipping in `GAMES_TO_SKIP`")
            continue
        elif success == PSErrors.PREPROCESSING_ERROR:
            if err_msg not in parse_results['parse_error']:
                parse_results['parse_error'][err_msg] = []
            parse_results['parse_error'][err_msg].append(game_name)
        else:
            raise Exception(f"Unknown error while parsing game: {success}")

        n_success = len(parse_results['success'])
        n_env_errors = np.sum([len(v) for k, v in parse_results['env_error'].items()]).item()
        n_tree_errors = np.sum([len(v) for k, v in parse_results['tree_error'].items()]).item()
        n_parse_errors = np.sum([len(v) for k, v in parse_results['parse_error'].items()]).item()
        n_timeouts = len(parse_results['parse_timeout'])
        parse_results['stats']['total'] = len(game_files)
        parse_results['stats']['success'] = n_success
        parse_results['stats']['env_error'] = n_env_errors
        parse_results['stats']['tree_error'] = n_tree_errors
        parse_results['stats']['parse_error'] = n_parse_errors
        parse_results['stats']['parse_timeout'] = n_timeouts

        with open(parse_results_path, "w", encoding='utf-8') as file:
            json.dump(parse_results, file, indent=4)

    # Save the sorted list to a json
    games_n_rules_sorted = sorted(games_n_rules_sorted, key=lambda x: x[1])
    with open(GAMES_N_RULES_SORTED_PATH, 'w', encoding='utf-8') as f:
        json.dump(games_n_rules_sorted, f, indent=4)
    with open(GAMES_TO_N_RULES_PATH, 'w', encoding='utf-8') as f:
        json.dump(games_to_n_rules, f, indent=4) 

    print(f"Attempted to parse and initialize {len(game_files)} games.")
    print(f"Initialized {len(parse_results['success'])} games successfully as jax envs")
    print(f"Env errors {n_env_errors}.")
    print(f"Tree transformation errors: {n_tree_errors}")
    print(f"Lark parse errors: {n_parse_errors}")
    print(f"Timeouts: {len(parse_results['parse_timeout'])}")



if __name__ == "__main__":
    main()