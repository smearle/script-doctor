"""Combined ExIt + search + RL + LLM win-rate heatmap.

Produces a single heatmap with one row per method (ExIt, search algos, RL
configs, LLM agents), games as columns, and cell values showing percent of
levels solved.  LLM win rates use JS-validated replay results when available.

Usage:
    python plot_aggregate_winrates.py                         # default dataset (priority)
    python plot_aggregate_winrates.py --dataset pedro
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from puzzlescript_jax.globals import GAMES_METADATA_PATH, PLOTS_DIR
from puzzlescript_jax.utils import game_names_remap, get_list_of_games_for_testing

# Local scripts (not packages), so Pylance may not resolve them
from scripts.plotting.plot_llm_results import collect_results  # type: ignore[import-not-found]
from scripts.plotting.plot_rl_results import collect_results_data as collect_rl_results  # type: ignore[import-not-found]

ALGO_NAMES = ['astar', 'gbfs', 'bfs', 'mcts']
HEATMAP_SEARCH_DEPTHS = [1_000_000]
ALL_RESULTS_PATH = os.path.join('data', 'all_search_results.json')
EXIT_TRAINING_DIR = os.path.join('data', 'exit_training')
JS_REPLAY_CSV_PATH = os.path.join('llm_agent_results', 'analysis_js_replay',
                                   'replay_results.csv')

LLM_NAME_MAP = {
    '4o-mini': 'GPT 4o-mini',
    'deepseek': 'Deepseek-chat',
    'qwen': 'Qwen-plus',
    'gemini': 'Gemini 2.0 flash exp',
    'gemini-2.5-pro': 'Gemini 2.5 Pro',
}
LLM_EXCLUDE = {'Qwen-plus', 'Gemini 2.0 flash exp', 'Deepseek-chat'}

ALGO_LABELS = {
    'astar': 'A*',
    'gbfs': 'GBFS',
    'bfs': 'BFS',
    'mcts': 'MCTS',
}


def _format_game_label(game: str) -> str:
    label = game_names_remap.get(game, game)
    label = label.replace('_', ' ')
    return label.title()


def _format_steps_label(n_steps: int) -> str:
    if n_steps >= 1_000_000 and n_steps % 1_000_000 == 0:
        return f'{n_steps // 1_000_000}M'
    if n_steps >= 1_000 and n_steps % 1_000 == 0:
        return f'{n_steps // 1_000}k'
    return f'{n_steps:,}'


# ── ExIt data ────────────────────────────────────────────────────────────────

# Readable names for ExIt hyperparameters that might vary across configs.
_EXIT_PARAM_LABELS = {
    'n_iterations': ('iters', str),
    'max_nodes': ('nodes', _format_steps_label),
    'batch_size': ('batch', _format_steps_label),
    'cost_weight': ('cw', str),
    'train_steps_per_iter': ('tsteps', str),
    'train_batch_size': ('tbatch', str),
    'lr': ('lr', str),
    'blend_alpha': ('alpha', str),
    'replay_max_size': ('replay', _format_steps_label),
    'initial_dim': ('dim', str),
    'hidden_dim': ('hdim', str),
    'res_n': ('res', str),
}


def _parse_exit_game_level(dirname: str):
    """Extract (game, level_i) from a directory name like 'blocks_level0'."""
    m = re.match(r'^(.*)_level(\d+)$', dirname)
    if m is None:
        return None, None
    return m.group(1), int(m.group(2))


def _load_exit_history(job_dir: str):
    """Load iteration history from checkpoint or standalone file."""
    for name in ('checkpoint.json', 'history.json'):
        path = os.path.join(job_dir, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            hist = data.get('history', data) if isinstance(data, dict) else data
            if isinstance(hist, list):
                return hist
        except Exception:
            pass
    return None


def _load_exit_run_config(job_dir: str) -> dict | None:
    """Load run_config.json, falling back to checkpoint metadata."""
    rc_path = os.path.join(job_dir, 'run_config.json')
    if os.path.exists(rc_path):
        try:
            with open(rc_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    ckpt_path = os.path.join(job_dir, 'checkpoint.json')
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, 'r') as f:
                return json.load(f).get('run_config')
        except Exception:
            pass
    return None


def _load_exit_rows(dataset: str) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Scan exit_training/ for per-config results.

    Returns (DataFrame[method, game, pct_solved], ordered method labels,
             {raw_game_name: n_levels_found}).
    """
    empty = pd.DataFrame(columns=['method', 'game', 'pct_solved']), [], {}
    if not os.path.isdir(EXIT_TRAINING_DIR):
        print(f'  WARNING: {EXIT_TRAINING_DIR} not found – skipping ExIt results.')
        return empty

    dataset_games = set(get_list_of_games_for_testing(dataset))

    # Discover all (game, level, config_dir) triples.
    # Structure: exit_training/{game}_level{i}/{config_subdir}/
    candidate_files = (
        glob.glob(os.path.join(EXIT_TRAINING_DIR, '**', 'checkpoint.json'), recursive=True) +
        glob.glob(os.path.join(EXIT_TRAINING_DIR, '**', 'history.json'), recursive=True)
    )
    candidate_dirs = sorted(set(os.path.dirname(p) for p in candidate_files))

    # Collect per-(config, game, level) solved status.
    # config_key → {game → {level → solved_bool}}
    config_game_levels: dict[str, dict[str, dict[int, bool]]] = {}
    # config_key → representative run_config dict (for label generation)
    config_params: dict[str, dict] = {}

    for job_dir in candidate_dirs:
        config_subdir = os.path.basename(job_dir)
        game_level_dir = os.path.basename(os.path.dirname(job_dir))
        game, level = _parse_exit_game_level(game_level_dir)
        if game is None or game not in dataset_games:
            continue

        history = _load_exit_history(job_dir)
        if not history:
            continue

        solved = any(
            bool(rec.get('solved', False))
            for rec in history if isinstance(rec, dict)
        )

        config_game_levels.setdefault(config_subdir, {}).setdefault(game, {})[level] = solved

        if config_subdir not in config_params:
            rc = _load_exit_run_config(job_dir)
            if rc is not None:
                config_params[config_subdir] = rc

    if not config_game_levels:
        return empty

    # Build coverage: union of levels seen across all configs per game.
    exit_coverage: dict[str, int] = {}
    for _cfg, game_levels in config_game_levels.items():
        for game, levels in game_levels.items():
            exit_coverage[game] = max(exit_coverage.get(game, 0), len(levels))

    # Detect which hyperparams vary across configs.
    all_rc = list(config_params.values())
    varying_keys: list[str] = []
    if len(all_rc) > 1:
        # Only consider keys present in _EXIT_PARAM_LABELS.
        candidate_keys = [k for k in _EXIT_PARAM_LABELS if any(k in rc for rc in all_rc)]
        for key in candidate_keys:
            vals = set()
            for rc in all_rc:
                vals.add(rc.get(key))
            if len(vals) > 1:
                varying_keys.append(key)

    # Build a readable label for each config.
    config_labels: dict[str, str] = {}
    for config_subdir in config_game_levels:
        rc = config_params.get(config_subdir, {})
        if not varying_keys:
            config_labels[config_subdir] = 'ExIt'
        else:
            parts = []
            for key in varying_keys:
                short, fmt = _EXIT_PARAM_LABELS.get(key, (key, str))
                val = rc.get(key)
                if val is not None:
                    parts.append(f'{fmt(val)} {short}')
            config_labels[config_subdir] = 'ExIt · ' + ', '.join(parts) if parts else 'ExIt'

    # Filter out configs with n_iterations == 500.
    excluded_configs = {
        csub for csub, rc in config_params.items()
        if rc.get('n_iterations') == 500
    }

    # Re-detect varying keys using only non-excluded configs, so that
    # parameters constant across included configs (e.g. n_iterations=200)
    # don't clutter the labels.
    included_rc = [config_params[csub] for csub in config_game_levels
                   if csub not in excluded_configs and csub in config_params]
    varying_keys = []
    if len(included_rc) > 1:
        candidate_keys = [k for k in _EXIT_PARAM_LABELS if any(k in rc for rc in included_rc)]
        for key in candidate_keys:
            vals = {rc.get(key) for rc in included_rc}
            if len(vals) > 1:
                varying_keys.append(key)

    # Rebuild labels with the corrected varying keys.
    config_labels = {}
    for config_subdir in config_game_levels:
        rc = config_params.get(config_subdir, {})
        if not varying_keys:
            config_labels[config_subdir] = 'ExIt'
        else:
            parts = []
            for key in varying_keys:
                short, fmt = _EXIT_PARAM_LABELS.get(key, (key, str))
                val = rc.get(key)
                if val is not None:
                    parts.append(f'{fmt(val)} {short}')
            config_labels[config_subdir] = 'ExIt · ' + ', '.join(parts) if parts else 'ExIt'

    # Build per-config, per-game pct_solved rows.
    rows = []
    for config_subdir, game_levels in config_game_levels.items():
        if config_subdir in excluded_configs:
            continue
        label = config_labels[config_subdir]
        for game, levels in game_levels.items():
            n = len(levels)
            solved = sum(1 for s in levels.values() if s)
            rows.append({
                'method': label,
                'game': _format_game_label(game),
                'pct_solved': solved / n if n > 0 else 0.0,
                'n_levels_tested': n,
            })

    df = pd.DataFrame(rows)
    # Sort ExIt rows descending by max_nodes.
    label_to_nodes: dict[str, int] = {}
    for csub in config_game_levels:
        if csub not in excluded_configs:
            rc = config_params.get(csub, {})
            label_to_nodes[config_labels[csub]] = rc.get('max_nodes', 0)
    method_order = sorted(
        df['method'].unique(),
        key=lambda m: -label_to_nodes.get(m, 0),
    )
    return df, method_order, exit_coverage


def method_perf_stub(method: str, df: pd.DataFrame) -> float:
    """Mean pct_solved for a method (helper for initial ordering)."""
    sub = df[df['method'] == method]
    return sub['pct_solved'].mean() if not sub.empty else 0.0


# ── Search data ──────────────────────────────────────────────────────────────

def _load_search_rows(dataset: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (DataFrame[method, game, pct_solved], {raw_game_name: n_levels})."""
    if not os.path.exists(ALL_RESULTS_PATH):
        print(f'  WARNING: {ALL_RESULTS_PATH} not found – skipping search results.')
        return pd.DataFrame(columns=['method', 'game', 'pct_solved']), {}

    with open(ALL_RESULTS_PATH, 'r') as f:
        all_results = json.load(f)

    dataset_games = set(get_list_of_games_for_testing(dataset))
    preferred_depths = set(HEATMAP_SEARCH_DEPTHS)

    rows = []
    search_coverage: dict[str, int] = {}
    for algo, depths in all_results.items():
        if algo not in ALGO_NAMES:
            continue
        # Pick only the preferred depths; fall back to largest if none match
        depth_keys = [d for d in depths if int(d) in preferred_depths]
        if not depth_keys:
            depth_keys = [max(depths.keys(), key=lambda d: int(d))]
        for depth_key in depth_keys:
            depth = int(depth_key)
            label = f'{ALGO_LABELS.get(algo, algo.upper())} · {_format_steps_label(depth)} nodes'
            for game, stats in depths[depth_key].items():
                if game not in dataset_games:
                    continue
                pct = stats.get('pct_solved')
                n_lvl = stats.get('n_levels')
                if pct is not None:
                    rows.append({
                        'method': label,
                        'game': _format_game_label(game),
                        'pct_solved': float(pct),
                        'n_levels_tested': int(n_lvl) if n_lvl is not None else None,
                    })
                if n_lvl is not None:
                    search_coverage[game] = max(
                        search_coverage.get(game, 0), int(n_lvl))

    return pd.DataFrame(rows), search_coverage


# ── RL data ──────────────────────────────────────────────────────────────────

RL_LOGS_DIR = 'rl_logs_jax'

def _load_rl_rows(dataset: str) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Return (DataFrame[method, game, pct_solved], ordered method labels,
             {raw_game_name: n_levels_found})."""
    rl_root = Path(RL_LOGS_DIR)
    if not rl_root.exists():
        print(f'  WARNING: {RL_LOGS_DIR} not found – skipping RL results.')
        return pd.DataFrame(columns=['method', 'game', 'pct_solved']), [], {}

    dataset_games = set(get_list_of_games_for_testing(dataset))
    summary_df, _level_df, _expected = collect_rl_results(rl_root)

    if summary_df.empty:
        return pd.DataFrame(columns=['method', 'game', 'pct_solved']), [], {}

    summary_df = summary_df[summary_df['game'].isin(dataset_games)]
    if summary_df.empty:
        return pd.DataFrame(columns=['method', 'game', 'pct_solved']), [], {}

    # Build coverage from summary_df
    rl_coverage: dict[str, int] = {}
    if 'n_levels' in summary_df.columns:
        for _, r in summary_df.iterrows():
            game = r['game']
            n = int(r['n_levels'])
            rl_coverage[game] = max(rl_coverage.get(game, 0), n)

    # Build PPO labels from the episode length config
    rows = []
    for _, r in summary_df.iterrows():
        ep_len = r.get('_cfg_ep_len', 100)
        label = f'PPO · {int(ep_len)} steps'
        n_lvl = int(r['n_levels']) if 'n_levels' in r and pd.notnull(r.get('n_levels')) else None
        rows.append({
            'method': label,
            'game': _format_game_label(r['game']),
            'pct_solved': float(r['avg_win_rate']),
            'n_levels_tested': n_lvl,
        })

    df = pd.DataFrame(rows)

    # Deterministic row order: sorted by episode length
    method_order = sorted(df['method'].unique(),
                          key=lambda m: int(m.split('·')[1].split()[0]))
    return df, method_order, rl_coverage


# ── LLM data ────────────────────────────────────────────────────────────────

def _load_llm_rows(dataset: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (DataFrame[method, game, pct_solved], {raw_game_name: n_levels}).

    Prefers JS-validated replay results (from replay_jax_results_in_js.py)
    when available, falling back to the original JAX results otherwise.
    """
    empty = pd.DataFrame(columns=['method', 'game', 'pct_solved']), {}

    # ── Try JS-validated replay CSV first ──
    if os.path.exists(JS_REPLAY_CSV_PATH):
        df = pd.read_csv(JS_REPLAY_CSV_PATH)
        if not df.empty and 'js_win' in df.columns:
            print(f'Using JS-validated LLM results from {JS_REPLAY_CSV_PATH}')
            df = df[~df['llm'].isin(['deepseek-r1', 'llama'])]
            df['game'] = df['game'].apply(
                lambda x: 'atlas_shrank' if 'atlas' in x and 'shrank' in x else x)

            # Build coverage before formatting game names
            dataset_games = set(get_list_of_games_for_testing(dataset))
            llm_coverage: dict[str, int] = {}
            if 'level' in df.columns:
                for game in df['game'].unique():
                    if game in dataset_games:
                        llm_coverage[game] = df[df['game'] == game]['level'].nunique()

            df['game'] = df['game'].apply(_format_game_label)
            df['llm'] = df['llm'].replace(LLM_NAME_MAP)
            df = df[~df['llm'].isin(LLM_EXCLUDE)]

            dataset_display = set(_format_game_label(g) for g in dataset_games)
            df = df[df['game'].isin(dataset_display)]

            if df.empty:
                return empty

            agg = df.groupby(['llm', 'game']).agg(
                pct_solved=('js_win', 'mean'),
                n_levels_tested=('level', 'nunique'),
            ).reset_index()
            agg.rename(columns={'llm': 'method'}, inplace=True)
            return agg, llm_coverage

    # ── Fallback: original JAX results ──
    print(f'  WARNING: {JS_REPLAY_CSV_PATH} not found – falling back to original JAX LLM results.')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'llm_agent_results')
    df = collect_results(results_dir)
    if df.empty:
        return empty

    df = df[~df['llm'].isin(['deepseek-r1', 'llama'])]
    if 'cot' in df.columns:
        df = df[df['cot'] == 0]
    df = df[~df['game'].str.lower().str.startswith('cot')]

    df['game'] = df['game'].apply(
        lambda x: 'atlas_shrank' if 'atlas' in x and 'shrank' in x else x)

    # Build coverage before formatting
    dataset_games = set(get_list_of_games_for_testing(dataset))
    llm_coverage = {}
    if 'level' in df.columns:
        for game in df['game'].unique():
            if game in dataset_games:
                llm_coverage[game] = df[df['game'] == game]['level'].nunique()

    df['game'] = df['game'].apply(_format_game_label)
    df['llm'] = df['llm'].replace(LLM_NAME_MAP)
    df = df[~df['llm'].isin(LLM_EXCLUDE)]

    dataset_display = set(_format_game_label(g) for g in dataset_games)
    df = df[df['game'].isin(dataset_display)]

    if df.empty:
        return empty

    agg = df.groupby(['llm', 'game']).agg(
        pct_solved=('win', 'mean'),
        n_levels_tested=('level', 'nunique'),
    ).reset_index()
    agg.rename(columns={'llm': 'method'}, inplace=True)
    return agg, llm_coverage


# ── Coverage summary ──────────────────────────────────────────────────────────

def _load_games_metadata() -> dict[str, dict]:
    """Load games_metadata.json, keyed by game name (without .txt)."""
    if not os.path.exists(GAMES_METADATA_PATH):
        print(f'  WARNING: {GAMES_METADATA_PATH} not found – cannot check level coverage.')
        return {}
    with open(GAMES_METADATA_PATH, 'r') as f:
        raw = json.load(f)
    # Strip .txt suffix from keys
    return {k[:-4] if k.endswith('.txt') else k: v for k, v in raw.items()}


def _print_coverage_summary(
    dataset: str,
    dataset_games: list[str],
    metadata: dict[str, dict],
    exit_coverage: dict[str, int],
    search_coverage: dict[str, int],
    rl_coverage: dict[str, int],
    llm_coverage: dict[str, int],
) -> None:
    """Print a summary of missing games and level-count mismatches."""
    sources = {
        'ExIt': exit_coverage,
        'Search': search_coverage,
        'RL': rl_coverage,
        'LLM': llm_coverage,
    }

    print('\n' + '=' * 72)
    print(f'COVERAGE SUMMARY  (dataset={dataset!r}, {len(dataset_games)} games)')
    print('=' * 72)

    any_issue = False
    for source_name, coverage in sources.items():
        missing_games = [g for g in dataset_games if g not in coverage]
        level_mismatches = []
        for g in dataset_games:
            if g not in coverage or g not in metadata:
                continue
            expected = metadata[g].get('n_levels')
            if expected is None:
                continue
            found = coverage[g]
            if found < expected:
                level_mismatches.append((g, found, expected))

        if not missing_games and not level_mismatches:
            continue

        any_issue = True
        print(f'\n  {source_name}:')
        if missing_games:
            print(f'    Missing games ({len(missing_games)}/{len(dataset_games)}):')
            for g in missing_games:
                expected = metadata.get(g, {}).get('n_levels', '?')
                print(f'      - {g}  (n_levels={expected})')
        if level_mismatches:
            print(f'    Incomplete levels ({len(level_mismatches)} games):')
            for g, found, expected in level_mismatches:
                print(f'      - {g}: {found}/{expected} levels')

    if not any_issue:
        print('\n  All sources have full coverage.')

    print('=' * 72 + '\n')


# ── Plotting ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='priority',
                        choices=['priority', 'gallery', 'pedro', 'increpare'])
    args = parser.parse_args()

    dataset_games = get_list_of_games_for_testing(args.dataset)
    metadata = _load_games_metadata()

    exit_df, exit_methods, exit_coverage = _load_exit_rows(args.dataset)
    search_df, search_coverage = _load_search_rows(args.dataset)
    rl_df, rl_methods, rl_coverage = _load_rl_rows(args.dataset)
    llm_df, llm_coverage = _load_llm_rows(args.dataset)

    all_dfs = [exit_df, search_df, rl_df, llm_df]
    if all(df.empty for df in all_dfs):
        print('No data to plot.')
        return

    # Adjust pct_solved: count missing levels as failures.
    # For each (method, game) pair, scale pct_solved by (n_tested / n_total).
    game_label_to_total = {}
    for g in dataset_games:
        meta = metadata.get(g, {})
        n_total = meta.get('n_levels')
        if n_total is not None:
            game_label_to_total[_format_game_label(g)] = int(n_total)

    for df in all_dfs:
        if df.empty or 'n_levels_tested' not in df.columns:
            continue
        for idx, row in df.iterrows():
            n_tested = row.get('n_levels_tested')
            n_total = game_label_to_total.get(row['game'])
            if n_tested is not None and n_total is not None and n_total > 0:
                n_solved = row['pct_solved'] * n_tested
                df.at[idx, 'pct_solved'] = n_solved / n_total

    # Compute mean performance per method for sorting within blocks
    combined_all = pd.concat(all_dfs, ignore_index=True)
    method_perf = combined_all.groupby('method')['pct_solved'].mean()

    def _sort_block_by_perf(methods: list[str]) -> list[str]:
        """Sort methods descending by mean pct_solved (best on top)."""
        return sorted(methods, key=lambda m: method_perf.get(m, 0), reverse=True)

    # exit_methods already sorted descending by max_nodes from _load_exit_rows

    # Collect search methods present in data
    search_methods = []
    for algo in ALGO_NAMES:
        for d in HEATMAP_SEARCH_DEPTHS:
            label = f'{ALGO_LABELS.get(algo, algo.upper())} · {_format_steps_label(d)} nodes'
            if not search_df.empty and label in search_df['method'].values:
                search_methods.append(label)
    search_methods = _sort_block_by_perf(search_methods)

    # RL: larger ep_len first, then by performance as tiebreaker
    rl_methods = sorted(
        rl_methods,
        key=lambda m: (-int(m.split('·')[1].split()[0]), -method_perf.get(m, 0)),
    ) if rl_methods else []

    llm_methods = _sort_block_by_perf(
        list(llm_df['method'].unique()) if not llm_df.empty else [])

    # Build block list — separators drawn between blocks
    # Order: ExIt, search, RL, LLM
    blocks = []
    if exit_methods:
        blocks.append(exit_methods)
    if search_methods:
        blocks.append(search_methods)
    if rl_methods:
        blocks.append(rl_methods)
    if llm_methods:
        blocks.append(llm_methods)

    all_methods = [m for block in blocks for m in block]

    # Determine game order: sort by mean pct_solved across all methods (descending)
    combined = pd.concat(all_dfs, ignore_index=True)
    game_mean = combined.groupby('game')['pct_solved'].mean()
    all_games = game_mean.sort_values(ascending=False).index.tolist()

    # Build the heatmap matrix
    heatmap = pd.DataFrame(np.nan, index=all_methods, columns=all_games)
    for _, row in combined.iterrows():
        if row['method'] in heatmap.index and row['game'] in heatmap.columns:
            heatmap.at[row['method'], row['game']] = row['pct_solved']

    # Annotation: integer percentages, blank for NaN
    annot = heatmap.map(
        lambda v: f'{v * 100:.0f}' if pd.notnull(v) else '')

    # Compute y-positions of block boundaries for separator lines
    sep_positions = []
    cum = 0
    for block in blocks[:-1]:
        cum += len(block)
        sep_positions.append(cum)

    # ── Figure sizing (single-column, two-column A4) ──
    COL_WIDTH = 3.5
    CELL_H = 0.28
    ANNOT_SIZE = 5.5
    TICK_SIZE = 6
    CBAR_LABEL_SIZE = 6
    CBAR_TICK_SIZE = 5

    num_rows = len(heatmap.index)
    fig_h = max(num_rows * CELL_H + 1.2, 1.8)
    fig_w = COL_WIDTH

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        heatmap,
        annot=annot,
        fmt='',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        annot_kws={'size': ANNOT_SIZE},
        linewidths=0.4,
        linecolor='white',
        ax=ax,
    )

    # Draw white gaps between blocks
    for sep_y in sep_positions:
        ax.hlines(sep_y, xmin=0, xmax=len(all_games),
                  colors='white', linewidth=4, clip_on=False)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.tick_params(axis='y', rotation=0, labelsize=TICK_SIZE)
    ax.tick_params(axis='x', rotation=45, labelsize=TICK_SIZE)
    for label in ax.get_xticklabels():
        label.set_ha('right')
        label.set_fontstyle('italic')

    fig.tight_layout()

    out_dir = os.path.join(PLOTS_DIR, 'search', args.dataset, 'heatmaps')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'aggregate_winrate_heatmap.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

    _print_coverage_summary(
        args.dataset, dataset_games, metadata,
        exit_coverage, search_coverage, rl_coverage, llm_coverage,
    )


if __name__ == '__main__':
    main()
