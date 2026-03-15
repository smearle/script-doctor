import json
import logging
import os
import shutil
from collections import OrderedDict
from typing import List, Optional

import hydra
import numpy as np
from lark import Lark
from tqdm import tqdm
import submitit

from puzzlescript_jax.preprocessing import count_rules, get_env_from_ps_file
from conf.config import PreprocessConfig
from puzzlescript_jax.detect_randomness import tree_has_randomness
from puzzlescript_jax.globals import (
    GAMES_N_RULES_SORTED_PATH, GAMES_TO_N_RULES_PATH, GAMES_METADATA_PATH,
    GAMES_TO_SKIP, LARK_SYNTAX_PATH, TEST_GAMES,
    TREES_DIR, SIMPLIFIED_GAMES_DIR, MIN_GAMES_DIR, PRETTY_TREES_DIR, CUSTOM_GAMES_DIR,
    GAMES_DIR,
)
from puzzlescript_jax.utils import get_list_of_games_for_testing, distribute_slurm_jobs
from puzzlescript_jax.preprocessing import PJParseErrors

logger = logging.getLogger(__name__)


def _dedupe_list(values):
    return list(OrderedDict.fromkeys(values))


def _dedupe_error_groups(error_groups):
    deduped = {}
    for err_msg, entries in error_groups.items():
        if not isinstance(entries, list):
            deduped[err_msg] = entries
            continue
        seen = OrderedDict()
        for entry in entries:
            key = tuple(entry) if isinstance(entry, list) else entry
            seen[key] = entry
        deduped[err_msg] = list(seen.values())
    return deduped


def _normalize_parse_results(parse_results):
    parse_results.setdefault('stats', {})
    parse_results.setdefault('success', [])
    parse_results.setdefault('preprocess_error', {})
    parse_results.setdefault('parse_error', {})
    parse_results.setdefault('tree_error', {})
    parse_results.setdefault('env_error', {})
    parse_results.setdefault('parse_timeout', [])

    parse_results['success'] = _dedupe_list(parse_results['success'])
    parse_results['parse_timeout'] = _dedupe_list(parse_results['parse_timeout'])
    for key in ('preprocess_error', 'parse_error', 'tree_error', 'env_error'):
        parse_results[key] = _dedupe_error_groups(parse_results[key])
    return parse_results


def _error_group_names(error_groups):
    names = set()
    for entries in error_groups.values():
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                names.add(entry[0])
            else:
                names.add(entry)
    return names


def _update_parse_stats(parse_results, total_games):
    parse_results['stats']['total'] = total_games
    parse_results['stats']['success'] = len(set(parse_results['success']))
    parse_results['stats']['preprocess_error'] = len(_error_group_names(parse_results['preprocess_error']))
    parse_results['stats']['parse_error'] = len(_error_group_names(parse_results['parse_error']))
    parse_results['stats']['tree_error'] = len(_error_group_names(parse_results['tree_error']))
    parse_results['stats']['env_error'] = len(_error_group_names(parse_results['env_error']))
    parse_results['stats']['parse_timeout'] = len(set(parse_results['parse_timeout']))


def _filter_error_groups(error_groups, allowed_games):
    filtered = {}
    for err_msg, entries in error_groups.items():
        kept_entries = []
        for entry in entries:
            game_name = entry[0] if isinstance(entry, (list, tuple)) else entry
            if game_name in allowed_games:
                kept_entries.append(entry)
        if kept_entries:
            filtered[err_msg] = kept_entries
    return filtered


def _filter_parse_results(parse_results, allowed_games):
    parse_results['success'] = [game for game in parse_results['success'] if game in allowed_games]
    parse_results['parse_timeout'] = [game for game in parse_results['parse_timeout'] if game in allowed_games]
    for key in ('preprocess_error', 'parse_error', 'tree_error', 'env_error'):
        parse_results[key] = _filter_error_groups(parse_results[key], allowed_games)
    return parse_results


def _save_all(parse_results, parse_results_path, games_n_rules_sorted, games_to_n_rules, total_games,
              games_metadata=None):
    """Save all progress to disk (parse_results + games_n_rules + metadata)."""
    _update_parse_stats(parse_results, total_games)
    with open(parse_results_path, "w", encoding='utf-8') as f:
        json.dump(parse_results, f, indent=4)
    sorted_rules = sorted(games_n_rules_sorted, key=lambda x: x[1])
    with open(GAMES_N_RULES_SORTED_PATH, 'w', encoding='utf-8') as f:
        json.dump(sorted_rules, f, indent=4)
    with open(GAMES_TO_N_RULES_PATH, 'w', encoding='utf-8') as f:
        json.dump(games_to_n_rules, f, indent=4)
    if games_metadata is not None:
        with open(GAMES_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(games_metadata, f, indent=4)


def extract_game_metadata(ps_tree) -> dict:
    """Extract rich metadata from a parsed PSGameTree."""
    n_rules = count_rules(ps_tree)
    has_randomness = tree_has_randomness(ps_tree)
    n_objects = len(ps_tree.objects)
    n_collision_layers = len(ps_tree.collision_layers)
    n_legend_entries = len(ps_tree.legend)
    n_win_conditions = len(ps_tree.win_conditions)
    n_levels = len(ps_tree.levels)

    level_dims = []
    for level in ps_tree.levels:
        if len(level) > 0 and len(level[0]) > 0:
            h = len(level)
            w = len(level[0])
            level_dims.append([w, h])

    mean_level_area = float(np.mean([w * h for w, h in level_dims])) if level_dims else 0.0
    max_level_area = max((w * h for w, h in level_dims), default=0)

    meta = {
        'n_rules': n_rules,
        'has_randomness': has_randomness,
        'n_objects': n_objects,
        'n_collision_layers': n_collision_layers,
        'n_legend_entries': n_legend_entries,
        'n_win_conditions': n_win_conditions,
        'n_levels': n_levels,
        'mean_level_area': mean_level_area,
        'max_level_area': max_level_area,
    }

    prelude = ps_tree.prelude
    if prelude.flickscreen is not None:
        meta['flickscreen'] = list(prelude.flickscreen)
    if prelude.zoomscreen is not None:
        meta['zoomscreen'] = list(prelude.zoomscreen)
    if prelude.run_rules_on_level_start:
        meta['run_rules_on_level_start'] = True
    if prelude.require_player_movement:
        meta['require_player_movement'] = True
    if prelude.noaction:
        meta['noaction'] = True

    return meta


@hydra.main(version_base="1.3", config_path="conf", config_name="preprocess_config")
def main_launch(cfg: PreprocessConfig):
    if cfg.slurm:
        if cfg.game is not None:
            games = [cfg.game]
        else:
            games = get_list_of_games_for_testing(dataset=cfg.dataset)

        game_sublists = distribute_slurm_jobs(games, cfg.n_games_per_job)
        n_jobs = len(game_sublists)
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "preprocess"))
        executor.update_parameters(
            slurm_job_name="preprocess_games",
            mem_gb=16,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=120,
            slurm_array_parallelism=n_jobs,
            slurm_account=os.environ.get("SLURM_ACCOUNT"),
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: PreprocessConfig, games: Optional[List[str]] = None):

    with open(LARK_SYNTAX_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()

    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)

    os.makedirs(TREES_DIR, exist_ok=True)
    os.makedirs(PRETTY_TREES_DIR, exist_ok=True)
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

    if games is not None:
        game_files = [f"{g}.txt" for g in games]
    elif cfg.game is not None:
        game_files = [cfg.game + '.txt']
    else:
        game_names = get_list_of_games_for_testing(dataset=cfg.dataset)
        game_files = [f"{g}.txt" for g in game_names]
    test_game_files = [f"{test_game}.txt" for test_game in TEST_GAMES]
    game_files = list(OrderedDict.fromkeys(test_game_files + game_files))
    current_game_set = set(game_files)

    os.makedirs(SIMPLIFIED_GAMES_DIR, exist_ok=True)
    scrape_log_dir = 'scrape_logs'
    os.makedirs(scrape_log_dir, exist_ok=True)
    games_n_rules_sorted = []
    games_to_n_rules = {}
    games_metadata = {}
    if not cfg.overwrite:
        if os.path.exists(GAMES_N_RULES_SORTED_PATH):
            with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
                games_n_rules_sorted = json.load(f)
            games_to_n_rules = {game: (n_rules, has_randomness) for game, n_rules, has_randomness in games_n_rules_sorted}
            print(f"Loaded {len(games_n_rules_sorted)} games from {GAMES_N_RULES_SORTED_PATH}")
        if os.path.exists(GAMES_METADATA_PATH):
            with open(GAMES_METADATA_PATH, 'r') as f:
                games_metadata = json.load(f)
            print(f"Loaded metadata for {len(games_metadata)} games from {GAMES_METADATA_PATH}")
        if os.path.exists(parse_results_path):
            with open(parse_results_path, 'r') as f:
                parse_results = json.load(f)
            parse_results = _normalize_parse_results(parse_results)
            parse_results = _filter_parse_results(parse_results, current_game_set)
            _update_parse_stats(parse_results, len(game_files))
            print(f"Loaded {len(parse_results['success'])} games from {parse_results_path}")

    success_games = set(parse_results['success'])
    parse_timeout_games = set(parse_results['parse_timeout'])
    parse_error_games = _error_group_names(parse_results['parse_error'])
    tree_error_games = _error_group_names(parse_results['tree_error'])
    env_error_games = _error_group_names(parse_results['env_error'])
    preprocess_error_games = _error_group_names(parse_results['preprocess_error'])

    n_skipped = 0
    n_processed = 0
    pbar = tqdm(game_files, desc="Preprocessing", unit="game")
    for filename in pbar:
        game_name = os.path.basename(filename)
        if not cfg.overwrite and game_name in games_to_n_rules and game_name in games_metadata:
            n_skipped += 1
            continue

        n_processed += 1
        pbar.set_postfix(game=game_name[:30], ok=len(success_games), skip=n_skipped)

        env, ps_tree, success, err_msg = get_env_from_ps_file(parser, filename[:-4], log_dir=scrape_log_dir, overwrite=cfg.overwrite)

        if success == PJParseErrors.SUCCESS:
            if game_name not in success_games:
                parse_results['success'].append(game_name)
                success_games.add(game_name)
            meta = extract_game_metadata(ps_tree)
            games_n_rules_sorted.append((game_name, meta['n_rules'], meta['has_randomness']))
            games_to_n_rules[game_name] = (meta['n_rules'], meta['has_randomness'])
            games_metadata[game_name] = meta
        elif success == PJParseErrors.PARSE_ERROR:
            if err_msg not in parse_results['parse_error']:
                parse_results['parse_error'][err_msg] = []
            if game_name not in parse_error_games:
                parse_results['parse_error'][err_msg].append(game_name)
                parse_error_games.add(game_name)
        elif success == PJParseErrors.TIMEOUT:
            if game_name not in parse_timeout_games:
                parse_results['parse_timeout'].append(game_name)
                parse_timeout_games.add(game_name)
        elif success == PJParseErrors.TREE_ERROR:
            if err_msg not in parse_results['tree_error']:
                parse_results['tree_error'][err_msg] = []
            if game_name not in tree_error_games:
                parse_results['tree_error'][err_msg].append(game_name)
                tree_error_games.add(game_name)
        elif success == PJParseErrors.ENV_ERROR:
            if err_msg not in parse_results['env_error']:
                parse_results['env_error'][err_msg] = []
            meta = extract_game_metadata(ps_tree)
            if game_name not in env_error_games:
                parse_results['env_error'][err_msg].append((game_name, meta['n_rules']))
                env_error_games.add(game_name)
            games_n_rules_sorted.append((game_name, meta['n_rules'], meta['has_randomness']))
            games_to_n_rules[game_name] = (meta['n_rules'], meta['has_randomness'])
            games_metadata[game_name] = meta
        elif success == PJParseErrors.SKIPPED:
            continue
        elif success == PJParseErrors.PREPROCESSING_ERROR:
            if err_msg not in parse_results['preprocess_error']:
                parse_results['preprocess_error'][err_msg] = []
            if game_name not in preprocess_error_games:
                parse_results['preprocess_error'][err_msg].append(game_name)
                preprocess_error_games.add(game_name)
        else:
            raise Exception(f"Unknown error while parsing game: {success}")

        # Save all progress after every game so nothing is lost on interruption
        _save_all(parse_results, parse_results_path, games_n_rules_sorted, games_to_n_rules, len(game_files),
                  games_metadata)

    # Final save
    _save_all(parse_results, parse_results_path, games_n_rules_sorted, games_to_n_rules, len(game_files),
              games_metadata)

    stats = parse_results['stats']
    print(f"\nPreprocessing complete ({len(game_files)} games, {n_skipped} skipped, {n_processed} processed):")
    print(f"  Success:      {stats.get('success', 0)}")
    print(f"  Env errors:   {stats.get('env_error', 0)}")
    print(f"  Tree errors:  {stats.get('tree_error', 0)}")
    print(f"  Parse errors: {stats.get('parse_error', 0)}")
    print(f"  Timeouts:     {stats.get('parse_timeout', 0)}")
    print(f"  Metadata:     {len(games_metadata)} games")


if __name__ == "__main__":
    main_launch()
