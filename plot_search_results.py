import glob
import json
import os
import re

import hydra
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from conf.config import PlotSearch
from puzzlescript_jax.globals import PLOTS_DIR, STANDALONE_NODEJS_RESULTS_PATH, JS_SOLS_DIR, CPP_SOLS_DIR, GAMES_TO_N_RULES_PATH, GAMES_METADATA_PATH
from search_nodejs import get_standalone_run_params_from_name
from puzzlescript_jax.utils import get_list_of_games_for_testing, game_names_remap


def _search_out_dir(dataset: str) -> str:
    return os.path.join(PLOTS_DIR, 'search', dataset)

def _heatmaps_dir(dataset: str) -> str:
    return os.path.join(_search_out_dir(dataset), 'heatmaps')

def _features_dir(dataset: str) -> str:
    return os.path.join(_search_out_dir(dataset), 'features')

def _correlations_dir(dataset: str) -> str:
    return os.path.join(_search_out_dir(dataset), 'correlations')
OOM_SENTINEL = -1.0
OOM_COLOR = '#FF8C00'  # dark orange
MAX_GAMES_FOR_HEATMAPS = 100  # skip per-game heatmaps when there are more games than this

BFS_RESULTS_PATH = os.path.join('data', 'bfs_results.json')
HEATMAP_SEARCH_DEPTHS = [1_000_000]
ALL_RESULTS_PATH = os.path.join('data', 'all_search_results.json')
EXIT_RESULTS_PATH = os.path.join('data', 'exit_results.json')
EXIT_TRAINING_DIR = os.path.join('data', 'exit_training')

def _per_level_results_path_for_algo(algo: str) -> str:
    return os.path.join('data', f'{algo}_per_level_results.json')


def _format_game_label(game: str) -> str:
    label = game_names_remap.get(game, game)
    label = label.replace('_', ' ')
    return label.title()


def _build_expanded_heatmap_columns(
    ordered_games: list[str], levels_by_game: dict[str, list[int]]
) -> tuple[list[tuple[str, int]], list[tuple[str, int, int]]]:
    columns = []
    game_spans = []
    for game in ordered_games:
        levels = levels_by_game.get(game, [])
        if not levels:
            continue
        start = len(columns)
        for level in levels:
            columns.append((game, level))
        end = len(columns)
        game_spans.append((game, start, end))
    return columns, game_spans


def _draw_game_dividers(ax, game_spans: list[tuple[str, int, int]], n_rows: int) -> None:
    for _, _, end in game_spans[:-1]:
        ax.vlines(end, ymin=0, ymax=n_rows, colors='black', linewidth=2.0)
    for game, start, end in game_spans:
        center = (start + end) / 2
        ax.text(
            center, 1.005,
            _format_game_label(game),
            ha='center', va='bottom', fontsize=9,
            transform=ax.get_xaxis_transform(),
        )


def _overlay_oom_cells(ax, data: pd.DataFrame, oom_mask: pd.DataFrame) -> None:
    """Paint orange rectangles over cells where oom_mask is True."""
    from matplotlib.patches import Rectangle
    for i in range(oom_mask.shape[0]):
        for j in range(oom_mask.shape[1]):
            if oom_mask.iloc[i, j]:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color=OOM_COLOR, zorder=2))
                ax.text(j + 0.5, i + 0.5, 'OOM', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white', zorder=3)


def _sort_games_by_mean_pct_solved(
    all_games: list[str], dfs: dict, column: str = 'pct_solved'
) -> list[str]:
    """Sort games by their mean pct_solved across all provided dataframes (descending)."""
    game_scores = {}
    for game in all_games:
        values = []
        for df in dfs.values():
            if game in df.index and column in df.columns:
                val = df.at[game, column]
                # df.at can return a Series if the index has duplicates
                if isinstance(val, pd.Series):
                    values.extend(v for v in val if pd.notnull(v))
                elif pd.notnull(val):
                    values.append(float(val))
        game_scores[game] = np.mean(values) if values else -1.0
    return sorted(all_games, key=lambda g: game_scores[g], reverse=True)


ALGO_NAMES = ['astar', 'gbfs', 'bfs', 'mcts']


def _results_path_for_algo(algo: str) -> str:
    if algo == 'exit':
        return EXIT_RESULTS_PATH
    return os.path.join('data', f'{algo}_results.json')


def _algo_label(algo: str) -> str:
    labels = {
        'bfs': 'BFS',
        'astar': 'A*',
        'gbfs': 'GBFS',
        'mcts': 'MCTS',
        'exit': 'ExIt',
    }
    return labels.get(algo, algo.upper())


def _parse_exit_job_dirname(dirname: str):
    match = re.match(r'^(.*)_level(\d+)$', dirname)
    if match is None:
        return None, None
    return match.group(1), int(match.group(2))


def _load_exit_history(job_dir: str):
    checkpoint_path = os.path.join(job_dir, 'checkpoint.json')
    history_path = os.path.join(job_dir, 'history.json')

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                ckpt = json.load(f)
            ckpt_history = ckpt.get('history')
            if isinstance(ckpt_history, list):
                return ckpt_history
        except Exception:
            pass

    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                hist = json.load(f)
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


# Readable short names for ExIt hyperparams.
_EXIT_PARAM_LABELS = {
    'n_iterations': ('iters', str),
    'max_nodes': ('nodes', lambda v: _format_nodes_label(int(v))),
    'batch_size': ('batch', lambda v: _format_nodes_label(int(v))),
    'cost_weight': ('cw', str),
    'train_steps_per_iter': ('tsteps', str),
    'train_batch_size': ('tbatch', str),
    'lr': ('lr', str),
    'blend_alpha': ('alpha', str),
    'replay_max_size': ('replay', lambda v: _format_nodes_label(int(v))),
    'initial_dim': ('dim', str),
    'hidden_dim': ('hdim', str),
    'res_n': ('res', str),
}


def _format_nodes_label(n: int) -> str:
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f'{n // 1_000_000}M'
    if n >= 1_000 and n % 1_000 == 0:
        return f'{n // 1_000}k'
    return f'{n:,}'


def _build_exit_config_label(rc: dict, varying_keys: list[str]) -> str:
    """Build a readable label for an ExIt config, showing only varying params."""
    if not varying_keys:
        return 'ExIt'
    parts = []
    for key in varying_keys:
        short, fmt = _EXIT_PARAM_LABELS.get(key, (key, str))
        val = rc.get(key)
        if val is not None:
            parts.append(f'{fmt(val)} {short}')
    return 'ExIt · ' + ', '.join(parts) if parts else 'ExIt'


def _collect_exit_results(games: list[str]) -> dict:
    """Collect ExIt results per config.

    Returns {config_label: {game: {pct_solved, n_levels, n_iters}}}.
    When only a single config exists, the label is just 'ExIt'.
    """
    if not os.path.exists(EXIT_TRAINING_DIR):
        return {}

    game_filter = set(games) if games is not None else None

    candidate_dirs = set()
    for path in (
        glob.glob(os.path.join(EXIT_TRAINING_DIR, '**', 'checkpoint.json'), recursive=True) +
        glob.glob(os.path.join(EXIT_TRAINING_DIR, '**', 'history.json'), recursive=True)
    ):
        candidate_dirs.add(os.path.dirname(path))

    # config_subdir → {game → [(level, solved, n_iters)]}
    config_game_data: dict[str, dict[str, list[tuple[int, bool, int]]]] = {}
    config_params: dict[str, dict] = {}

    for job_dir in sorted(candidate_dirs):
        config_subdir = os.path.basename(job_dir)
        game_level_dir = os.path.basename(os.path.dirname(job_dir))
        game, level = _parse_exit_job_dirname(game_level_dir)
        if game is None:
            continue
        if game_filter is not None and game not in game_filter:
            continue

        history = _load_exit_history(job_dir)
        if not history:
            continue

        solved = any(bool(rec.get('solved', False)) for rec in history if isinstance(rec, dict))
        config_game_data.setdefault(config_subdir, {}).setdefault(game, []).append(
            (level, solved, len(history))
        )

        if config_subdir not in config_params:
            rc = _load_exit_run_config(job_dir)
            if rc is not None:
                config_params[config_subdir] = rc

    if not config_game_data:
        return {}

    # Detect varying hyperparams.
    all_rc = list(config_params.values())
    varying_keys: list[str] = []
    if len(all_rc) > 1:
        for key in _EXIT_PARAM_LABELS:
            vals = set(rc.get(key) for rc in all_rc)
            if len(vals) > 1:
                varying_keys.append(key)

    # Build labelled results.
    results: dict[str, dict] = {}
    for config_subdir, game_data in config_game_data.items():
        rc = config_params.get(config_subdir, {})
        label = _build_exit_config_label(rc, varying_keys)

        per_game = {}
        for game, entries in game_data.items():
            n = len(entries)
            solved = sum(1 for _, s, _ in entries if s)
            avg_iters = float(np.mean([ni for _, _, ni in entries])) if entries else float('nan')
            per_game[game] = {
                'pct_solved': solved / n if n > 0 else 0.0,
                'n_levels': n,
                'n_iters': avg_iters,
            }

        results[label] = per_game

    # Summary.
    for label, per_game in results.items():
        total_levels = sum(r['n_levels'] for r in per_game.values())
        total_solved = sum(int(r['pct_solved'] * r['n_levels']) for r in per_game.values())
        print(
            f"  {label}: {len(per_game)} games, {total_levels} levels, "
            f"{total_solved} solved ({total_solved/total_levels:.0%})" if total_levels > 0
            else f"  {label}: {len(per_game)} games, 0 levels"
        )
        for game in sorted(per_game):
            r = per_game[game]
            solved = int(r['pct_solved'] * r['n_levels'])
            print(f"    {game}: {solved}/{r['n_levels']} levels solved, {r['n_iters']:.0f} avg iters")

    return results


def _parse_solver_run_name(filename: str):
    match = re.match(r'^(astar|bfs|gbfs|mcts)_(\d+)-steps_level-(\d+)\.json$', filename)
    if match is None:
        return None, None, None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _format_level_ranges(levels: list[int]) -> str:
    if not levels:
        return ''

    sorted_levels = sorted(set(levels))
    ranges = []
    start = sorted_levels[0]
    end = sorted_levels[0]

    for level in sorted_levels[1:]:
        if level == end + 1:
            end = level
            continue
        ranges.append(str(start) if start == end else f'{start}-{end}')
        start = level
        end = level

    ranges.append(str(start) if start == end else f'{start}-{end}')
    return ', '.join(ranges)


def _collect_results_for_algo(
    games: list[str], algo: str
) -> tuple[dict[int, dict], dict[int, dict[str, dict[int, float]]]]:
    results_by_depth = {}
    per_level_by_depth: dict[int, dict[str, dict[int, float]]] = {}
    for game in games:
        if game.startswith('test_'):
            continue
        game_dir = os.path.join(JS_SOLS_DIR, game)
        sol_jsons = glob.glob(f"{game_dir}/*.json")
        per_depth_stats = {}
        per_level_stats: dict[int, dict[int, dict]] = {}
        expected_levels = set()
        levels_seen_by_depth = {}
        for sol_json in sol_jsons:
            filename = os.path.basename(sol_json)
            sol_algo_name, n_steps, level_id = _parse_solver_run_name(filename)
            if sol_algo_name is None:
                continue

            expected_levels.add(level_id)

            if sol_algo_name != algo:
                continue

            if n_steps not in levels_seen_by_depth:
                levels_seen_by_depth[n_steps] = set()
            levels_seen_by_depth[n_steps].add(level_id)

            with open(sol_json, 'r') as f:
                sol_dict = json.load(f)
            if 'iterations' not in sol_dict or 'won' not in sol_dict:
                print(f"Skipping {sol_json} because it doesn't have 'iterations' or 'won'")
                continue

            error_type = sol_dict.get('error')
            is_oom = error_type == 'oom'
            is_timeout = error_type == 'timeout'

            if n_steps not in per_depth_stats:
                per_depth_stats[n_steps] = {
                    'n_levels': 0,
                    'n_solved': 0,
                    'n_oom': 0,
                    'n_timeout': 0,
                    'n_stepss': [],
                    'solution_lengths': [],
                    'solved_iterss': [],
                    'score_progresses': [],
                }

            stats = per_depth_stats[n_steps]
            solved = sol_dict['won']
            actions = sol_dict.get('actions')
            if n_steps not in per_level_stats:
                per_level_stats[n_steps] = {}
            if level_id not in per_level_stats[n_steps]:
                per_level_stats[n_steps][level_id] = {'n_solved': 0, 'n_runs': 0, 'n_oom': 0, 'n_timeout': 0}
            per_level_stats[n_steps][level_id]['n_runs'] += 1
            if is_oom:
                stats['n_oom'] += 1
                per_level_stats[n_steps][level_id]['n_oom'] += 1
            if is_timeout:
                stats['n_timeout'] += 1
                per_level_stats[n_steps][level_id]['n_timeout'] += 1
            if solved:
                stats['n_solved'] += 1
                stats['solution_lengths'].append(len(actions) if actions is not None else 0)
                stats['solved_iterss'].append(sol_dict['iterations'])
                per_level_stats[n_steps][level_id]['n_solved'] += 1
            stats['n_levels'] += 1
            clipped_steps = min(n_steps, sol_dict['iterations'])
            stats['n_stepss'].append(clipped_steps)
            raw_score = sol_dict.get('score')
            score_initial = sol_dict.get('score_initial')
            if raw_score is not None and score_initial is not None and not is_oom:
                try:
                    s = float(raw_score)
                    s0 = float(score_initial)
                    if s0 > 0:
                        # 1.0 = fully solved, 0.0 = no progress from initial state
                        stats['score_progresses'].append(1.0 - s / s0)
                    elif s == 0:
                        # initial score was 0 and final is 0 — already at goal
                        stats['score_progresses'].append(1.0)
                except (TypeError, ValueError):
                    pass

        for depth, seen_levels in sorted(levels_seen_by_depth.items()):
            missing_levels = sorted(expected_levels - seen_levels)
            if missing_levels:
                depth_label = _format_steps_label(depth)
                missing_levels_summary = _format_level_ranges(missing_levels)
                expected_levels_summary = _format_level_ranges(list(expected_levels))
                print(
                    f"Warning: missing search results for algorithm '{algo}', game '{game}', depth {depth_label} "
                    f"on levels {missing_levels_summary} (expected levels: {expected_levels_summary})."
                )

        for depth, stats in per_depth_stats.items():
            if stats['n_levels'] == 0:
                continue
            n_steps_mean = np.mean(stats['n_stepss'])
            pct_solved = stats['n_solved'] / stats['n_levels']
            mean_solution_length = float(np.mean(stats['solution_lengths'])) if stats['solution_lengths'] else float('nan')
            if depth not in results_by_depth:
                results_by_depth[depth] = {}
            mean_solved_iters = float(np.mean(stats['solved_iterss'])) if stats['solved_iterss'] else float('nan')
            mean_score_progress = float(np.mean(stats['score_progresses'])) if stats['score_progresses'] else float('nan')
            results_by_depth[depth][game] = {
                'pct_solved': pct_solved,
                'n_levels': stats['n_levels'],
                'n_iters': n_steps_mean,
                'mean_sol_len': mean_solution_length,
                'mean_solved_iters': mean_solved_iters,
                'has_oom': stats['n_oom'] > 0,
                'has_timeout': stats['n_timeout'] > 0,
                'mean_score_progress': mean_score_progress,
            }

        for depth, level_stats in per_level_stats.items():
            if depth not in per_level_by_depth:
                per_level_by_depth[depth] = {}
            per_level_by_depth[depth][game] = {
                level_id: OOM_SENTINEL if lvl['n_oom'] == lvl['n_runs'] else lvl['n_solved'] / lvl['n_runs']
                for level_id, lvl in level_stats.items()
            }

    for depth in sorted(results_by_depth.keys(), reverse=True):
        depth_results = results_by_depth[depth]
        n_games = len(depth_results)
        total_levels = sum(g.get('n_levels', 0) for g in depth_results.values())
        total_solved = sum(
            int(g.get('pct_solved', 0) * g.get('n_levels', 0))
            for g in depth_results.values()
        )
        n_oom = sum(1 for g in depth_results.values() if g.get('has_oom'))
        n_timeout = sum(1 for g in depth_results.values() if g.get('has_timeout'))
        suffix = ""
        if n_oom:
            suffix += f", {n_oom} games with OOM"
        if n_timeout:
            suffix += f", {n_timeout} games with timeout"
        print(
            f"  {algo} @ {_format_steps_label(depth)}: "
            f"{n_games} games, {total_levels} levels, "
            f"{total_solved} solved ({total_solved/total_levels:.0%})"
            + suffix
        )

    return results_by_depth, per_level_by_depth


def _depth_order_for_results(results_by_depth: dict[int, dict]) -> list[int]:
    preferred_depth_order = [depth for depth in HEATMAP_SEARCH_DEPTHS if depth in results_by_depth]
    fallback_depth_order = sorted(
        [depth for depth in results_by_depth.keys() if depth not in preferred_depth_order],
        reverse=True,
    )
    return preferred_depth_order + fallback_depth_order


@hydra.main(version_base="1.3", config_path="conf", config_name="plot_standalone_bfs_config")
def main(cfg: PlotSearch):
    if cfg.aggregate:
        aggregate_results(cfg)
    else:
        plot(cfg)
    

def aggregate_results(cfg: PlotSearch):
    if cfg.game is not None:
        games = [cfg.game]
    else:
        games = get_list_of_games_for_testing(cfg.dataset)
    print(games)
    if cfg.algo == 'all':
        algos = list(ALGO_NAMES)
    elif cfg.algo == 'exit':
        exit_results = _collect_exit_results(games)
        with open(EXIT_RESULTS_PATH, 'w') as f:
            json.dump(exit_results, f, indent=4)
        print(f'Saved aggregated ExIt results to {EXIT_RESULTS_PATH}')
        plot(cfg, exit_results)
        return
    elif cfg.algo in ALGO_NAMES:
        algos = [cfg.algo]
    else:
        raise ValueError(f'Unknown algo: {cfg.algo}')

    aggregated_results = {}
    aggregated_per_level = {}
    for algo in algos:
        results_by_depth, per_level_by_depth = _collect_results_for_algo(games, algo)
        results_path = _results_path_for_algo(algo)
        with open(results_path, 'w') as f:
            json.dump({str(k): v for k, v in results_by_depth.items()}, f, indent=4)
        print(f'Saved aggregated results to {results_path}')
        per_level_path = _per_level_results_path_for_algo(algo)
        with open(per_level_path, 'w') as f:
            json.dump(
                {str(depth): {game: {str(level): wr for level, wr in levels.items()}
                               for game, levels in game_data.items()}
                 for depth, game_data in per_level_by_depth.items()},
                f, indent=4,
            )
        print(f'Saved per-level results to {per_level_path}')
        aggregated_results[algo] = results_by_depth
        aggregated_per_level[algo] = per_level_by_depth

    # Heuristic quality analysis (needs per-level BFS+A* data)
    if cfg.algo == 'all' or cfg.algo in ('bfs', 'astar'):
        generate_heuristic_quality_report(games, cfg.dataset)

    if cfg.algo == 'all':
        with open(ALL_RESULTS_PATH, 'w') as f:
            json.dump({algo: {str(k): v for k, v in depths.items()} for algo, depths in aggregated_results.items()}, f, indent=4)
        print(f'Saved combined aggregated results to {ALL_RESULTS_PATH}')
        plot(cfg, aggregated_results, aggregated_per_level)
    else:
        plot(cfg, aggregated_results[cfg.algo], aggregated_per_level[cfg.algo])


def _format_steps_label(n_steps: int) -> str:
    if n_steps >= 1_000_000 and n_steps % 1_000_000 == 0:
        return f"{n_steps // 1_000_000}M steps"
    if n_steps >= 1_000 and n_steps % 1_000 == 0:
        return f"{n_steps // 1_000}k steps"
    return f"{n_steps:,} steps"


def _normalize_results_by_depth(results, default_depth: int):
    if not results:
        return {default_depth: {}}

    sample_value = next(iter(results.values()))
    if isinstance(sample_value, dict) and 'pct_solved' in sample_value:
        return {default_depth: results}

    normalized = {}
    for depth_key, depth_results in results.items():
        normalized[int(depth_key)] = depth_results
    return normalized


def _load_per_level_by_depth(algo: str) -> dict[int, dict[str, dict[int, float]]]:
    per_level_path = _per_level_results_path_for_algo(algo)
    if not os.path.exists(per_level_path):
        return {}
    with open(per_level_path, 'r') as f:
        raw = json.load(f)
    return {
        int(depth): {
            game: {int(level): wr for level, wr in levels.items()}
            for game, levels in game_data.items()
        }
        for depth, game_data in raw.items()
    }


def plot(cfg: PlotSearch, results=None, per_level_by_depth=None):
    M = 40  # max number of games per table

    if cfg.algo == 'all':
        plot_all_algos(cfg, results, per_level_by_depth)
        return

    if cfg.algo == 'exit':
        if results is None:
            if os.path.exists(EXIT_RESULTS_PATH):
                with open(EXIT_RESULTS_PATH, 'r') as f:
                    results = json.load(f)
            else:
                results = {}
        plot_exit_heatmap(results, cfg.dataset)
        return

    if results is None:
        results_path = _results_path_for_algo(cfg.algo)
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            with open(BFS_RESULTS_PATH, 'r') as f:
                results = json.load(f)

    if per_level_by_depth is None:
        per_level_by_depth = _load_per_level_by_depth(cfg.algo)

    results_by_depth = _normalize_results_by_depth(results, cfg.n_steps)
    depth_order = _depth_order_for_results(results_by_depth)
    if not depth_order:
        print('No search results found to plot.')
        return

    selected_depth = cfg.n_steps if cfg.n_steps in results_by_depth else depth_order[0]
    selected_results = results_by_depth[selected_depth]
    
    df = pd.DataFrame.from_dict(selected_results, orient='index')
    if 'sol_len' in df.columns and 'mean_sol_len' not in df.columns:
        df.rename(columns={'sol_len': 'mean_sol_len'}, inplace=True)
    # df = df.sort_values(by=['pct_solved', 'n_iters'], ascending=[False, True])

    out_dir = _search_out_dir(cfg.dataset)
    os.makedirs(out_dir, exist_ok=True)
    algo_slug = cfg.algo.lower()
    algo_label = _algo_label(cfg.algo)

    csv_file_path = os.path.join(out_dir, f'standalone_{algo_slug}_results.csv')
    df.to_csv(csv_file_path, index=True, float_format="%.2f")
    print(f'Saved results to {csv_file_path}')

    # Remap game names
    df.index = df.index.to_series().replace(game_names_remap)
    # Replace underscores and capitalize game names
    df.index = df.index.str.replace('_', ' ')
    df.index = df.index.str.title()

    heatmap_source_dfs = {}
    for depth in depth_order:
        depth_results = results_by_depth.get(depth, {})
        if not depth_results:
            continue
        depth_df = pd.DataFrame.from_dict(depth_results, orient='index')
        if 'sol_len' in depth_df.columns and 'mean_sol_len' not in depth_df.columns:
            depth_df.rename(columns={'sol_len': 'mean_sol_len'}, inplace=True)
        depth_df.index = depth_df.index.to_series().replace(game_names_remap)
        depth_df.index = depth_df.index.str.replace('_', ' ')
        depth_df.index = depth_df.index.str.title()
        heatmap_source_dfs[depth] = depth_df

    generate_heatmaps(heatmap_source_dfs, depth_order, algo_label, algo_slug, per_level_by_depth, cfg.dataset)

    latex_df = df.copy()

    col_renames = {
        'pct_solved': 'Solved Levels \\%',
        'n_levels': '\\# Total Levels',
        'n_iters': 'Mean Search Iterations',
        'mean_sol_len': 'Mean Solution Length',
    }
    latex_df.rename(columns=col_renames, inplace=True)

    def _format_with_commas(value, decimals):
        if pd.isna(value):
            return 'NaN'
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value
        return f"{numeric_value:,.{decimals}f}"

    # Clean and prepare index for LaTeX
    latex_df.index = latex_df.index.str.replace('_', ' ')
    latex_df.index = latex_df.index.str.replace('&', r'\&', regex=False)
    latex_df.index = latex_df.index.str.replace('^', r'\^', regex=False)
    latex_df.index = latex_df.index.to_series().apply(
        lambda name: f"\\parbox{{3.5cm}}{{\\strut {name[:50]}{'...' if len(name) > 50 else ''}}}"
    )
    latex_df.index.name = 'Game'

    # Modify % column to show, e.g. "0.75" as "75\%"
    latex_df['Solved Levels \\%'] = latex_df['Solved Levels \\%'].apply(lambda x: f"{x * 100:.0f}\\%")

    numeric_format_specs = {
        '\\# Total Levels': 0,
        'Mean Search Iterations': 2,
        'Mean Solution Length': 2,
    }
    for column, decimals in numeric_format_specs.items():
        if column in latex_df.columns:
            latex_df[column] = latex_df[column].apply(lambda value: _format_with_commas(value, decimals))

    latex_file_path = os.path.join(out_dir, f'{algo_slug}_results.tex')
    caption_steps = _format_steps_label(selected_depth)
    with open(latex_file_path, 'w') as f:
        f.write(latex_df.to_latex(index=True, float_format="%.2f", escape=False, caption=f"Results of {algo_label} on full dataset of games, with max {caption_steps} and a timeout of 1 minute.",
                            longtable=True, label="tab:bfs_results"))
    print(f'Saved latex table to {latex_file_path}')


def plot_exit_heatmap(results: dict, dataset: str = 'priority') -> None:
    """Plot ExIt heatmap.

    *results* is ``{config_label: {game: {pct_solved, ...}}}`` (multi-config)
    or the legacy ``{game: {pct_solved, ...}}`` (single flat dict).
    """
    if not results:
        print('No ExIt results found to plot.')
        return

    # Detect legacy (flat) format and wrap it.
    sample = next(iter(results.values()))
    if isinstance(sample, dict) and 'pct_solved' in sample:
        results = {'ExIt': results}

    heatmaps_dir = _heatmaps_dir(dataset)
    out_dir = _search_out_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)

    # Collect all games across configs and sort by mean pct_solved.
    all_games: set[str] = set()
    for per_game in results.values():
        all_games.update(per_game.keys())

    def _game_mean(game):
        vals = [pg[game]['pct_solved'] for pg in results.values() if game in pg]
        return np.mean(vals) if vals else 0.0

    game_order = sorted(all_games, key=_game_mean, reverse=True)
    pretty_games = [_format_game_label(g) for g in game_order]

    # Sort configs descending by node count (extracted from label).
    def _extract_nodes(label: str) -> int:
        m = re.search(r'(\d+(?:\.\d+)?)\s*([kKmM])?\s*nodes', label)
        if not m:
            return 0
        val = float(m.group(1))
        suffix = (m.group(2) or '').lower()
        if suffix == 'k':
            val *= 1_000
        elif suffix == 'm':
            val *= 1_000_000
        return int(val)

    config_order = sorted(results.keys(), key=_extract_nodes, reverse=True)

    # Build heatmap matrix.
    heatmap = pd.DataFrame(np.nan, index=config_order, columns=pretty_games)
    for label in config_order:
        for game, stats in results[label].items():
            heatmap.at[label, _format_game_label(game)] = stats['pct_solved']

    annot_data = [[f"{v:.0%}" if pd.notnull(v) else '' for v in row]
                  for row in heatmap.to_numpy(dtype=float)]

    target_cell_height = 1.0
    target_cell_width = 1.0
    min_total_figure_width = 8.0
    min_total_figure_height = 3.0

    num_cols = len(heatmap.columns)
    num_rows = len(heatmap.index)
    fig_h = max(num_rows * target_cell_height + 1.5, min_total_figure_height)
    fig_w = max(num_cols * target_cell_width + 3.0, min_total_figure_width)

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        heatmap,
        annot=annot_data,
        fmt='',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Solved Levels (%)', 'shrink': 0.8, 'pad': 0.01},
        annot_kws={'size': 9},
        linewidths=0.5,
        linecolor='white',
    )
    plt.title('ExIt Percent of Levels Solved per Game')
    plt.xlabel('Game', labelpad=10)
    plt.ylabel('Method', labelpad=10)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path = os.path.join(heatmaps_dir, 'exit_pct_solved_heatmap.png')
    try:
        plt.savefig(output_path, dpi=300)
        print(f'Saved heatmap to {output_path}')
    except Exception as e:
        print(f'Error saving ExIt heatmap: {e}')
    finally:
        plt.close()


def plot_all_algos(cfg: PlotSearch, results_by_algo=None, per_level_by_algo=None):
    if results_by_algo is None:
        results_by_algo = {}
        for algo in ALGO_NAMES:
            results_path = _results_path_for_algo(algo)
            if not os.path.exists(results_path):
                continue
            with open(results_path, 'r') as f:
                loaded = json.load(f)
            results_by_algo[algo] = _normalize_results_by_depth(loaded, cfg.n_steps)

    if per_level_by_algo is None:
        per_level_by_algo = {algo: _load_per_level_by_depth(algo) for algo in results_by_algo}

    normalized_by_algo = {}
    for algo, algo_results in results_by_algo.items():
        normalized_by_algo[algo] = _normalize_results_by_depth(algo_results, cfg.n_steps)

    if not normalized_by_algo:
        print('No search results found to plot.')
        return

    out_dir = _search_out_dir(cfg.dataset)
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    heatmap_source_dfs = {}
    per_level_by_algo_depth: dict[tuple[str, int], dict[str, dict[int, float]]] = {}
    for algo, depths in normalized_by_algo.items():
        depth_order = _depth_order_for_results(depths)
        algo_per_level = per_level_by_algo.get(algo, {})
        for depth in depth_order:
            depth_results = depths.get(depth, {})
            if not depth_results:
                continue

            depth_df = pd.DataFrame.from_dict(depth_results, orient='index')
            if depth_df.empty:
                continue
            if 'sol_len' in depth_df.columns and 'mean_sol_len' not in depth_df.columns:
                depth_df.rename(columns={'sol_len': 'mean_sol_len'}, inplace=True)

            for game, row in depth_df.iterrows():
                summary_rows.append({
                    'algo': algo,
                    'algo_label': _algo_label(algo),
                    'depth': depth,
                    'depth_label': _format_steps_label(depth),
                    'game': game,
                    'pct_solved': row.get('pct_solved', np.nan),
                    'n_levels': row.get('n_levels', np.nan),
                    'n_iters': row.get('n_iters', np.nan),
                    'mean_sol_len': row.get('mean_sol_len', np.nan),
                    'mean_solved_iters': row.get('mean_solved_iters', np.nan),
                    'mean_score_progress': row.get('mean_score_progress', np.nan),
                })

            remapped_df = depth_df.copy()
            remapped_df.index = remapped_df.index.to_series().replace(game_names_remap)
            remapped_df.index = remapped_df.index.str.replace('_', ' ')
            remapped_df.index = remapped_df.index.str.title()
            heatmap_source_dfs[(algo, depth)] = remapped_df

            if depth in algo_per_level:
                per_level_by_algo_depth[(algo, depth)] = algo_per_level[depth]

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.sort_values(by=['algo', 'depth', 'game'], ascending=[True, False, True], inplace=True)
        summary_csv_path = os.path.join(out_dir, 'standalone_all_search_results.csv')
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
        print(f'Saved results to {summary_csv_path}')

    generate_all_heatmaps(heatmap_source_dfs, per_level_by_algo_depth, cfg.dataset)

    if summary_rows:
        generate_rules_vs_difficulty(summary_df, cfg.dataset)
        generate_correlation_report(summary_df, cfg.dataset)


def _load_game_n_rules() -> dict[str, int]:
    """Load {game_name: n_rules} mapping, stripping .txt suffix."""
    with open(GAMES_TO_N_RULES_PATH, 'r') as f:
        raw = json.load(f)
    return {name.removesuffix('.txt'): meta[0] for name, meta in raw.items()}


def _normalize_game_name(name: str) -> str:
    return re.sub(r'[\s_]+', '', name).lower()


def _load_game_metadata() -> dict[str, dict]:
    """Load full game metadata, stripping .txt suffix from keys."""
    if not os.path.exists(GAMES_METADATA_PATH):
        return {}
    with open(GAMES_METADATA_PATH, 'r') as f:
        raw = json.load(f)
    return {name.removesuffix('.txt'): meta for name, meta in raw.items()}


# Features to analyze: (metadata_key, display_label, use_log_scale)
ANALYSIS_FEATURES = [
    ('n_rules', 'Number of Rules', True),
    ('n_objects', 'Number of Objects', True),
    ('n_collision_layers', 'Number of Collision Layers', False),
    ('n_win_conditions', 'Number of Win Conditions', False),
    ('n_levels', 'Number of Levels', True),
    ('mean_level_area', 'Mean Level Area (cells)', True),
    ('max_level_area', 'Max Level Area (cells)', True),
]


def _make_bin_edges(values: np.ndarray, use_log: bool, n_bins: int | None = None) -> np.ndarray:
    """Create bin edges for the given values, optionally log-spaced.

    When *n_bins* is None the number of bins scales with the data size so that
    each bin contains ~5 data points on average (clamped to [10, 80]).
    """
    min_val = max(1, int(values.min()))
    max_val = int(values.max())
    if min_val >= max_val:
        return np.array([min_val, max_val + 1])
    if n_bins is None:
        n_bins = int(np.clip(len(values) / 5, 10, 80))
    if use_log:
        return np.unique(np.geomspace(min_val, max_val + 1, num=n_bins).astype(int))
    else:
        return np.unique(np.linspace(min_val, max_val + 1, num=n_bins).astype(int))


def generate_rules_vs_difficulty(summary_df: pd.DataFrame, dataset: str = 'priority') -> None:
    """Generate feature-vs-difficulty plots from the summary DataFrame.

    Uses games_metadata.json when available (rich features), otherwise falls
    back to games_to_n_rules.json (n_rules only).
    """
    game_metadata = _load_game_metadata()
    if game_metadata:
        meta_lookup = {_normalize_game_name(g): meta for g, meta in game_metadata.items()}
        features_to_plot = ANALYSIS_FEATURES
    else:
        game_n_rules = _load_game_n_rules()
        meta_lookup = {_normalize_game_name(g): {'n_rules': n} for g, n in game_n_rules.items()}
        features_to_plot = [ANALYSIS_FEATURES[0]]  # n_rules only

    df = summary_df.copy()

    # Attach all metadata features to the dataframe
    for feat_key, _, _ in features_to_plot:
        df[feat_key] = df['game'].apply(
            lambda g: (meta_lookup.get(_normalize_game_name(g)) or {}).get(feat_key))

    df = df.dropna(subset=['n_rules', 'pct_solved'])
    if df.empty:
        print('No games matched for feature-vs-difficulty plots.')
        return

    features_dir = _features_dir(dataset)
    os.makedirs(features_dir, exist_ok=True)

    # Per-game max depth for each algo
    idx = df.groupby(['algo', 'game'])['depth'].idxmax()
    max_depth_df = df.loc[idx]

    algos = sorted(max_depth_df['algo'].unique())
    algo_colors = {a: f'C{i}' for i, a in enumerate(algos)}
    n_algos = len(algos)

    for feat_key, feat_label, use_log in features_to_plot:
        feat_df = df.dropna(subset=[feat_key]).copy()
        feat_df[feat_key] = feat_df[feat_key].astype(float)
        feat_max_depth = max_depth_df.dropna(subset=[feat_key]).copy()
        feat_max_depth[feat_key] = feat_max_depth[feat_key].astype(float)

        if feat_df.empty or feat_max_depth.empty:
            continue

        feat_slug = feat_key

        # --- 1. Per-algo scatterplots ---
        n_cols = min(n_algos, 3)
        n_rows_fig = (n_algos + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows_fig, n_cols, figsize=(n_cols * 5, n_rows_fig * 4), squeeze=False)

        for ax_i, algo in enumerate(algos):
            ax = axes[ax_i // n_cols][ax_i % n_cols]
            algo_sub = feat_max_depth[feat_max_depth['algo'] == algo]
            ax.scatter(
                algo_sub[feat_key], algo_sub['pct_solved'],
                alpha=0.5, s=20, color=algo_colors[algo], edgecolors='none',
            )
            ax.set_title(f'{_algo_label(algo)}')
            ax.set_xlabel(feat_label)
            ax.set_ylabel('% levels solved')
            ax.set_ylim(-0.05, 1.05)
            if use_log:
                ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[n_algos:]:
            ax.axis('off')

        fig.suptitle(f'Game Difficulty vs. {feat_label} (max search depth)', fontsize=14)
        fig.tight_layout()
        path = os.path.join(features_dir, f'{feat_slug}_vs_difficulty_scatter.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f'Saved scatter plot to {path}')

        # --- 2. Binned curve plot ---
        all_feat_vals = feat_df[feat_key].values
        bin_edges = _make_bin_edges(all_feat_vals, use_log)
        if len(bin_edges) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for algo in algos:
            algo_sub = feat_max_depth[feat_max_depth['algo'] == algo]
            bin_means = []
            bin_centers = []
            bin_counts = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (algo_sub[feat_key] >= lo) & (algo_sub[feat_key] < hi)
                bucket = algo_sub.loc[mask, 'pct_solved']
                if len(bucket) >= 1:
                    bin_means.append(bucket.mean())
                    bin_centers.append((lo + hi) / 2)
                    bin_counts.append(len(bucket))

            if bin_centers:
                ax.plot(
                    bin_centers, bin_means,
                    marker='o', markersize=5, label=_algo_label(algo),
                    color=algo_colors[algo], alpha=0.8,
                )
                for x, y, n in zip(bin_centers, bin_means, bin_counts):
                    ax.annotate(str(n), (x, y), textcoords='offset points',
                                xytext=(0, 6), ha='center', fontsize=6, color=algo_colors[algo])

        ax.set_xlabel(feat_label)
        ax.set_ylabel('Mean % levels solved')
        if use_log:
            ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'Mean Solve Rate vs. {feat_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(features_dir, f'{feat_slug}_vs_difficulty_curves.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f'Saved curve plot to {path}')

        # --- 3. Per-algo subplots: curves by search depth ---
        depths = sorted(feat_df['depth'].unique())
        depth_cmap = plt.cm.viridis
        depth_norm = plt.Normalize(vmin=0, vmax=max(len(depths) - 1, 1))

        n_cols = min(n_algos, 3)
        n_rows_fig = (n_algos + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows_fig, n_cols, figsize=(n_cols * 5.5, n_rows_fig * 4.5), squeeze=False)

        for ax_i, algo in enumerate(algos):
            ax = axes[ax_i // n_cols][ax_i % n_cols]
            algo_sub = feat_df[feat_df['algo'] == algo]
            algo_depths = sorted(algo_sub['depth'].unique())

            for di, depth in enumerate(algo_depths):
                sub = algo_sub[algo_sub['depth'] == depth]
                bin_means = []
                bin_centers = []
                for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                    mask = (sub[feat_key] >= lo) & (sub[feat_key] < hi)
                    bucket = sub.loc[mask, 'pct_solved']
                    if len(bucket) >= 1:
                        bin_means.append(bucket.mean())
                        bin_centers.append((lo + hi) / 2)

                if bin_centers:
                    color = depth_cmap(depth_norm(di))
                    ax.plot(
                        bin_centers, bin_means,
                        marker='o', markersize=4, alpha=0.8,
                        color=color, label=_format_steps_label(depth),
                    )

            ax.set_title(_algo_label(algo))
            ax.set_xlabel(feat_label)
            ax.set_ylabel('Mean % levels solved')
            if use_log:
                ax.set_xscale('log')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, title='Search depth', title_fontsize=7)

        for ax in axes.flat[n_algos:]:
            ax.axis('off')

        fig.suptitle(f'Mean Solve Rate vs. {feat_label} by Search Depth', fontsize=14)
        fig.tight_layout()
        path = os.path.join(features_dir, f'{feat_slug}_vs_difficulty_by_depth.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f'Saved depth curve plot to {path}')

    # --- 4. Search effort (iterations to solve) vs each feature ---
    if 'mean_solved_iters' not in df.columns:
        return

    solved_df = df[df['pct_solved'] > 0].dropna(subset=['mean_solved_iters'])
    if solved_df.empty:
        return

    # Keep deepest depth per (algo, game)
    idx = solved_df.groupby(['algo', 'game'])['depth'].idxmax()
    best_df = solved_df.loc[idx]

    for feat_key, feat_label, use_log in features_to_plot:
        effort_df = best_df.dropna(subset=[feat_key]).copy()
        effort_df[feat_key] = effort_df[feat_key].astype(float)
        if effort_df.empty:
            continue

        feat_slug = feat_key

        # Scatter: one subplot per algo
        n_cols = min(n_algos, 3)
        n_rows_fig = (n_algos + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows_fig, n_cols, figsize=(n_cols * 5, n_rows_fig * 4), squeeze=False)

        for ax_i, algo in enumerate(algos):
            ax = axes[ax_i // n_cols][ax_i % n_cols]
            sub = effort_df[effort_df['algo'] == algo]
            ax.scatter(
                sub[feat_key], sub['mean_solved_iters'],
                alpha=0.5, s=20, color=algo_colors[algo], edgecolors='none',
            )
            ax.set_title(_algo_label(algo))
            ax.set_xlabel(feat_label)
            ax.set_ylabel('Mean iterations to solve')
            if use_log:
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[n_algos:]:
            ax.axis('off')

        fig.suptitle(f'Search Effort to Solve vs. {feat_label}', fontsize=14)
        fig.tight_layout()
        path = os.path.join(features_dir, f'{feat_slug}_vs_search_effort_scatter.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f'Saved search effort scatter to {path}')

        # Binned curves
        all_feat_vals = effort_df[feat_key].values
        bin_edges = _make_bin_edges(all_feat_vals, use_log)
        if len(bin_edges) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for algo in algos:
            sub = effort_df[effort_df['algo'] == algo]
            bin_means = []
            bin_centers = []
            bin_counts = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (sub[feat_key] >= lo) & (sub[feat_key] < hi)
                bucket = sub.loc[mask, 'mean_solved_iters']
                if len(bucket) >= 1:
                    bin_means.append(bucket.mean())
                    bin_centers.append((lo + hi) / 2)
                    bin_counts.append(len(bucket))

            if bin_centers:
                ax.plot(
                    bin_centers, bin_means,
                    marker='o', markersize=5, label=_algo_label(algo),
                    color=algo_colors[algo], alpha=0.8,
                )
                for x, y, n in zip(bin_centers, bin_means, bin_counts):
                    ax.annotate(str(n), (x, y), textcoords='offset points',
                                xytext=(0, 6), ha='center', fontsize=6, color=algo_colors[algo])

        ax.set_xlabel(feat_label)
        ax.set_ylabel('Mean iterations to solve')
        if use_log:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'Search Effort to Solve vs. {feat_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(features_dir, f'{feat_slug}_vs_search_effort_curves.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f'Saved search effort curves to {path}')

    # --- 5. Heuristic score vs each feature ---
    if 'mean_score_progress' not in max_depth_df.columns:
        return
    score_df = max_depth_df.dropna(subset=['mean_score_progress']).copy()
    if score_df.empty:
        return

    for feat_key, feat_label, use_log in features_to_plot:
        feat_score_df = score_df.dropna(subset=[feat_key]).copy()
        feat_score_df[feat_key] = feat_score_df[feat_key].astype(float)
        if feat_score_df.empty:
            continue

        # Binned curves: feature vs mean_score_progress for each algo
        all_feat_vals = feat_score_df[feat_key].values
        bin_edges = _make_bin_edges(all_feat_vals, use_log)
        if len(bin_edges) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for algo in algos:
            sub = feat_score_df[feat_score_df['algo'] == algo]
            bin_means = []
            bin_centers = []
            bin_counts = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (sub[feat_key] >= lo) & (sub[feat_key] < hi)
                bucket = sub.loc[mask, 'mean_score_progress']
                if len(bucket) >= 1:
                    bin_means.append(bucket.mean())
                    bin_centers.append((lo + hi) / 2)
                    bin_counts.append(len(bucket))

            if bin_centers:
                ax.plot(
                    bin_centers, bin_means,
                    marker='o', markersize=5, label=_algo_label(algo),
                    color=algo_colors[algo], alpha=0.8,
                )
                for x, y, n in zip(bin_centers, bin_means, bin_counts):
                    ax.annotate(str(n), (x, y), textcoords='offset points',
                                xytext=(0, 6), ha='center', fontsize=6, color=algo_colors[algo])

        ax.set_xlabel(feat_label)
        ax.set_ylabel('Mean Score Progress')
        ax.set_ylim(-0.05, 1.05)
        if use_log:
            ax.set_xscale('log')
        ax.set_title(f'Mean Score Progress vs. {feat_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(features_dir, f'{feat_key}_vs_score_progress_curves.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f'Saved score progress curve plot to {path}')


def generate_correlation_report(summary_df: pd.DataFrame, dataset: str = 'priority') -> None:
    """Compute Spearman correlations between game metadata features and search outcomes.

    Prints a table and saves a CSV + heatmap of significant correlations.
    """
    from scipy import stats as scipy_stats

    game_metadata = _load_game_metadata()
    if not game_metadata:
        game_n_rules = _load_game_n_rules()
        meta_lookup = {_normalize_game_name(g): {'n_rules': n} for g, n in game_n_rules.items()}
        features_to_test = [ANALYSIS_FEATURES[0]]
    else:
        meta_lookup = {_normalize_game_name(g): meta for g, meta in game_metadata.items()}
        features_to_test = ANALYSIS_FEATURES

    df = summary_df.copy()
    for feat_key, _, _ in features_to_test:
        df[feat_key] = df['game'].apply(
            lambda g: (meta_lookup.get(_normalize_game_name(g)) or {}).get(feat_key))

    # Use max depth per (algo, game)
    idx = df.groupby(['algo', 'game'])['depth'].idxmax()
    df = df.loc[idx]

    outcome_metrics = [
        ('pct_solved', '% Solved'),
        ('mean_solved_iters', 'Iterations to Solve'),
        ('mean_sol_len', 'Solution Length'),
        ('mean_score_progress', 'Score Progress'),
    ]

    rows = []
    for feat_key, feat_label, _ in features_to_test:
        for outcome_key, outcome_label in outcome_metrics:
            for algo in sorted(df['algo'].unique()):
                sub = df[(df['algo'] == algo)].dropna(subset=[feat_key, outcome_key])
                if len(sub) < 5:
                    continue
                rho, p_value = scipy_stats.spearmanr(sub[feat_key], sub[outcome_key])
                rows.append({
                    'feature': feat_label,
                    'outcome': outcome_label,
                    'algo': _algo_label(algo),
                    'rho': rho,
                    'p_value': p_value,
                    'n': len(sub),
                    'significant': p_value < 0.05,
                })

    if not rows:
        print('No correlation data to report.')
        return

    corr_df = pd.DataFrame(rows)
    corr_df.sort_values('p_value', inplace=True)

    correlations_dir = _correlations_dir(dataset)
    os.makedirs(correlations_dir, exist_ok=True)
    csv_path = os.path.join(correlations_dir, 'feature_correlations.csv')
    corr_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f'Saved correlation report to {csv_path}')

    # Print significant correlations
    sig = corr_df[corr_df['significant']]
    if sig.empty:
        print('No statistically significant correlations found (p < 0.05).')
    else:
        print(f'\nSignificant correlations (p < 0.05):')
        for _, row in sig.iterrows():
            direction = 'positive' if row['rho'] > 0 else 'negative'
            print(
                f"  {row['algo']:5s} | {row['feature']:30s} vs {row['outcome']:20s} | "
                f"rho={row['rho']:+.3f} p={row['p_value']:.1e} n={row['n']:4d} ({direction})"
            )

    # Heatmap per outcome metric
    for outcome_key, outcome_label in outcome_metrics:
        outcome_corr = corr_df[corr_df['outcome'] == outcome_label]
        if outcome_corr.empty:
            continue

        pivot = outcome_corr.pivot(index='feature', columns='algo', values='rho')
        p_pivot = outcome_corr.pivot(index='feature', columns='algo', values='p_value')

        # Annotate with significance stars
        annot = pivot.copy().astype(str)
        for feat in pivot.index:
            for algo in pivot.columns:
                rho_val = pivot.at[feat, algo]
                p_val = p_pivot.at[feat, algo]
                if pd.isna(rho_val):
                    annot.at[feat, algo] = ''
                else:
                    stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                    annot.at[feat, algo] = f'{rho_val:.2f}{stars}'

        fig_h = max(len(pivot.index) * 0.6 + 2, 4)
        fig_w = max(len(pivot.columns) * 1.5 + 3, 6)
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            pivot.astype(float),
            annot=annot.values,
            fmt='',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Spearman rho'},
            linewidths=0.5,
            linecolor='white',
        )
        outcome_slug = outcome_key.replace(' ', '_').lower()
        plt.title(f'Feature vs. {outcome_label} Correlations (Spearman)')
        plt.xlabel('Algorithm')
        plt.ylabel('Game Feature')
        plt.tight_layout()
        path = os.path.join(correlations_dir, f'feature_correlation_{outcome_slug}_heatmap.png')
        plt.savefig(path, dpi=300)
        plt.close()
        print(f'Saved correlation heatmap to {path}')


def generate_heuristic_quality_report(games: list[str], dataset: str = 'priority') -> None:
    """Measure heuristic quality per game via A*/BFS iteration ratio and h₀–d* correlation.

    For each game+level solved by both A* and BFS at the same depth budget:
      - Iteration ratio: iterations_A* / iterations_BFS  (< 1 means heuristic helps)
      - h₀ vs d* correlation: Spearman ρ(initial_score, BFS_solution_length)

    Results are saved as a CSV and printed.
    """
    from scipy import stats as scipy_stats

    # Collect per-level results for BFS and A* from both JS and CPP sols
    per_level: dict[str, dict[str, dict[int, dict]]] = {}  # {game: {algo: {level: data}}}
    for game in games:
        if game.startswith('test_'):
            continue
        sol_jsons = []
        for sols_dir in (JS_SOLS_DIR, CPP_SOLS_DIR):
            game_dir = os.path.join(sols_dir, game)
            if os.path.isdir(game_dir):
                sol_jsons.extend(glob.glob(f"{game_dir}/*.json"))
        for sol_json in sol_jsons:
            filename = os.path.basename(sol_json)
            algo, n_steps, level_id = _parse_solver_run_name(filename)
            if algo not in ('bfs', 'astar'):
                continue
            with open(sol_json, 'r') as f:
                data = json.load(f)
            if data.get('error') or 'iterations' not in data or 'won' not in data:
                continue
            per_level.setdefault(game, {}).setdefault(algo, {})[level_id] = data

    rows = []
    for game, algo_data in per_level.items():
        bfs_levels = algo_data.get('bfs', {})
        astar_levels = algo_data.get('astar', {})

        # Shared levels solved by both
        shared_solved = []
        for lvl in set(bfs_levels) & set(astar_levels):
            b, a = bfs_levels[lvl], astar_levels[lvl]
            if b.get('won') and a.get('won') and b['iterations'] > 0:
                shared_solved.append((lvl, b, a))

        if not shared_solved:
            continue

        # Iteration ratio
        ratios = [a['iterations'] / b['iterations'] for _, b, a in shared_solved]
        median_ratio = float(np.median(ratios))

        # h₀ vs d* (BFS solution length) across all BFS-solved levels
        h0s, dstars = [], []
        for lvl, b_data in bfs_levels.items():
            if b_data.get('won') and b_data.get('score_initial') is not None:
                actions = b_data.get('actions', [])
                if actions:
                    h0s.append(float(b_data['score_initial']))
                    dstars.append(len(actions))

        h0_dstar_rho, h0_dstar_p = (float('nan'), float('nan'))
        if len(h0s) >= 5:
            h0_dstar_rho, h0_dstar_p = scipy_stats.spearmanr(h0s, dstars)

        rows.append({
            'game': game,
            'n_shared_solved': len(shared_solved),
            'median_iter_ratio': median_ratio,
            'n_bfs_solved': len(h0s),
            'h0_dstar_rho': h0_dstar_rho,
            'h0_dstar_p': h0_dstar_p,
        })

    if not rows:
        print('No overlapping BFS/A* results for heuristic quality analysis.')
        return

    hq_df = pd.DataFrame(rows).sort_values('median_iter_ratio')
    out_dir = _search_out_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'heuristic_quality.csv')
    hq_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f'\nSaved heuristic quality report to {csv_path}')

    valid = hq_df.dropna(subset=['median_iter_ratio'])
    print(f'\nHeuristic quality summary ({len(valid)} games with shared BFS/A* solves):')
    print(f'  Median A*/BFS iteration ratio: {valid["median_iter_ratio"].median():.3f}')
    print(f'  Mean A*/BFS iteration ratio:   {valid["median_iter_ratio"].mean():.3f}')

    h0_valid = hq_df.dropna(subset=['h0_dstar_rho'])
    if not h0_valid.empty:
        print(f'  Median h₀–d* ρ:               {h0_valid["h0_dstar_rho"].median():.3f}')
        n_sig = (h0_valid['h0_dstar_p'] < 0.05).sum()
        print(f'  Games with significant h₀–d*:  {n_sig}/{len(h0_valid)}')

    # Best and worst games by iteration ratio
    print(f'\n  Top 5 games (heuristic helps most):')
    for _, r in valid.head(5).iterrows():
        print(f'    {r["game"]:40s}  ratio={r["median_iter_ratio"]:.3f}  '
              f'h₀–d* ρ={r["h0_dstar_rho"]:+.3f}  (n={r["n_shared_solved"]})')
    print(f'\n  Bottom 5 games (heuristic helps least):')
    for _, r in valid.tail(5).iterrows():
        print(f'    {r["game"]:40s}  ratio={r["median_iter_ratio"]:.3f}  '
              f'h₀–d* ρ={r["h0_dstar_rho"]:+.3f}  (n={r["n_shared_solved"]})')

    # Correlate heuristic quality with game metadata features
    game_metadata = _load_game_metadata()
    if game_metadata:
        meta_lookup = {_normalize_game_name(g): meta for g, meta in game_metadata.items()}
        for feat_key, feat_label, _ in ANALYSIS_FEATURES:
            hq_df[feat_key] = hq_df['game'].apply(
                lambda g: (meta_lookup.get(_normalize_game_name(g)) or {}).get(feat_key))

        print(f'\n  Correlations of heuristic quality (iter ratio) with game features:')
        for feat_key, feat_label, _ in ANALYSIS_FEATURES:
            sub = hq_df.dropna(subset=[feat_key, 'median_iter_ratio'])
            if len(sub) < 5:
                continue
            rho, p = scipy_stats.spearmanr(sub[feat_key], sub['median_iter_ratio'])
            sig = '*' if p < 0.05 else ''
            print(f'    {feat_label:35s}  ρ={rho:+.3f}  p={p:.3e}{sig}')


def generate_all_heatmaps(
    dfs_by_algo_depth: dict,
    per_level_by_algo_depth: dict[tuple[str, int], dict[str, dict[int, float]]] | None = None,
    dataset: str = 'priority',
) -> None:
    if not dfs_by_algo_depth:
        print('No data available for all-algorithm heatmap generation.')
        return

    preferred_depths = set(HEATMAP_SEARCH_DEPTHS)
    # Default to only showing preferred depths (100k, 1M); fall back to all if none match
    heatmap_keys = [k for k in dfs_by_algo_depth if k[1] in preferred_depths]
    if not heatmap_keys:
        heatmap_keys = list(dfs_by_algo_depth.keys())
    ordered_keys = sorted(
        heatmap_keys,
        key=lambda key: (
            ALGO_NAMES.index(key[0]) if key[0] in ALGO_NAMES else 999,
            -key[1],
        ),
    )

    row_labels = [f"{_algo_label(algo)} · {_format_steps_label(depth)}" for algo, depth in ordered_keys]
    all_games = []
    for key in ordered_keys:
        for game in dfs_by_algo_depth[key].index:
            if game not in all_games:
                all_games.append(game)
    all_games = _sort_games_by_mean_pct_solved(all_games, dfs_by_algo_depth)

    if len(all_games) > MAX_GAMES_FOR_HEATMAPS:
        print(f'Skipping all-algo heatmaps: {len(all_games)} games exceeds limit of {MAX_GAMES_FOR_HEATMAPS}.')
        return

    # --- Build best-depth-per-algo data for solved-only metrics ---
    # For each algo, find the deepest depth available and take solved-only values.
    algos_seen = []
    for algo, _depth in ordered_keys:
        if algo not in algos_seen:
            algos_seen.append(algo)

    best_algo_dfs = {}
    best_algo_oom = {}
    for algo in algos_seen:
        # Keys for this algo sorted by depth descending
        algo_keys = sorted([k for k in ordered_keys if k[0] == algo], key=lambda k: -k[1])
        merged = pd.DataFrame(columns=['mean_solved_iters', 'mean_sol_len', 'has_oom', 'mean_score_progress'])
        for game in all_games:
            for key in algo_keys:
                key_df = dfs_by_algo_depth[key]
                if game not in key_df.index:
                    continue
                row = key_df.loc[game]
                pct = row.get('pct_solved', 0)
                if isinstance(pct, pd.Series):
                    pct = pct.iloc[0]
                if pct > 0:
                    merged.at[game, 'mean_solved_iters'] = row.get('mean_solved_iters', np.nan)
                    merged.at[game, 'mean_sol_len'] = row.get('mean_sol_len', np.nan)
                    merged.at[game, 'has_oom'] = bool(row.get('has_oom', False))
                    merged.at[game, 'mean_score_progress'] = row.get('mean_score_progress', np.nan)
                    break
        best_algo_dfs[algo] = merged
        algo_label_str = _algo_label(algo)
        best_algo_oom[algo_label_str] = merged.get('has_oom', pd.Series(dtype=bool))

    algo_row_labels = [_algo_label(a) for a in algos_seen]

    heatmap_configs = [
        {
            'column': 'pct_solved',
            'title': 'All Search Types: Percent of Levels Solved per Game',
            'cmap': 'RdYlGn',
            'vmin': 0.0,
            'vmax': 1.0,
            'colorbar_label': 'Average Win Rate',
            'formatter': lambda v: f"{v*100:.0f}",
            'output': 'all_search_pct_solved_heatmap.png',
            'per_depth': True,
        },
        {
            'column': 'mean_solved_iters',
            'title': 'All Search Types: Mean Iterations to Solve per Game (best depth, solved only)',
            'cmap': 'Blues',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Iterations to Solve',
            'formatter': lambda v: f"{v:.0f}",
            'output': 'all_search_mean_iterations_heatmap.png',
            'per_depth': False,
        },
        {
            'column': 'mean_sol_len',
            'title': 'All Search Types: Mean Solution Length per Game (best depth, solved only)',
            'cmap': 'Purples',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Sol. Length',
            'formatter': lambda v: f"{v:.0f}",
            'output': 'all_search_mean_solution_length_heatmap.png',
            'per_depth': False,
        },
        {
            'column': 'mean_score_progress',
            'title': 'All Search Types: Mean Score Progress per Game',
            'cmap': 'RdYlGn',
            'vmin': 0.0,
            'vmax': 1.0,
            'colorbar_label': 'Score Progress (0=none, 1=solved)',
            'formatter': lambda v: f"{v:.2f}",
            'output': 'all_search_score_progress_heatmap.png',
            'per_depth': True,
        },
    ]

    # Sizing for single-column (two-column A4 article, ~3.5 in column width)
    COL_WIDTH = 3.5  # inches
    CELL_H = 0.28    # per row
    ANNOT_SIZE = 5.5
    TICK_SIZE = 6
    CBAR_LABEL_SIZE = 6
    CBAR_TICK_SIZE = 5

    for config in heatmap_configs:
        column = config['column']
        per_depth = config['per_depth']

        if per_depth:
            # Multi-row: algo × depth
            if not any(column in df.columns for df in dfs_by_algo_depth.values()):
                continue

            cur_row_labels = row_labels
            heatmap_data = pd.DataFrame(index=cur_row_labels, columns=all_games, dtype=float)
            oom_mask = pd.DataFrame(False, index=cur_row_labels, columns=all_games)
            for key, row_label in zip(ordered_keys, cur_row_labels):
                key_df = dfs_by_algo_depth[key]
                if column not in key_df.columns:
                    continue
                for game, value in key_df[column].items():
                    heatmap_data.at[row_label, game] = value
                if 'has_oom' in key_df.columns:
                    for game, value in key_df['has_oom'].items():
                        if value:
                            oom_mask.at[row_label, game] = True
        else:
            # Single row per algo: best depth, solved only
            cur_row_labels = algo_row_labels
            heatmap_data = pd.DataFrame(index=cur_row_labels, columns=all_games, dtype=float)
            oom_mask = pd.DataFrame(False, index=cur_row_labels, columns=all_games)
            for algo, algo_lbl in zip(algos_seen, cur_row_labels):
                adf = best_algo_dfs[algo]
                if column not in adf.columns:
                    continue
                for game in all_games:
                    if game in adf.index and pd.notnull(adf.at[game, column]):
                        heatmap_data.at[algo_lbl, game] = float(adf.at[game, column])
                    if game in adf.index and adf.get('has_oom', pd.Series(dtype=bool)).get(game, False):
                        oom_mask.at[algo_lbl, game] = True

        stacked_values = heatmap_data.stack(future_stack=True).dropna()
        if stacked_values.empty:
            continue

        valid_values = stacked_values.astype(float)
        vmin = config.get('vmin')
        vmax = config.get('vmax')
        if vmin is None:
            vmin = float(valid_values.min())
        if vmax is None:
            vmax = float(valid_values.max())
        if np.isfinite(vmin) and np.isfinite(vmax) and np.isclose(vmin, vmax):
            vmax = vmin + (abs(vmin) * 0.05 + 1)

        annot_data = [
            [config['formatter'](val) if pd.notnull(val) else '' for val in row]
            for row in heatmap_data.values
        ]

        num_cols = len(heatmap_data.columns)
        num_rows = len(heatmap_data.index)
        fig_h = max(num_rows * CELL_H + 1.2, 1.5)  # +1.2 for x-tick labels
        fig_w = COL_WIDTH

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            heatmap_data,
            annot=annot_data,
            fmt="",
            cmap=config['cmap'],
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'shrink': 0.6, 'pad': 0.02, 'aspect': 15},
            annot_kws={"size": ANNOT_SIZE},
            linewidths=0.4,
            linecolor='white',
            ax=ax,
        )
        # Style the colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
        cbar.set_label(config['colorbar_label'], size=CBAR_LABEL_SIZE)

        if oom_mask.any().any():
            _overlay_oom_cells(ax, heatmap_data, oom_mask)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.tick_params(axis='y', rotation=0, labelsize=TICK_SIZE)
        ax.tick_params(axis='x', rotation=45, labelsize=TICK_SIZE)
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_fontstyle('italic')
        fig.tight_layout()

        heatmaps_dir = _heatmaps_dir(dataset)
        os.makedirs(heatmaps_dir, exist_ok=True)
        output_path = os.path.join(heatmaps_dir, config['output'])
        try:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to {output_path}")
        except Exception as e:
            print(f"Error saving heatmap {config['output']}: {e}")
        finally:
            plt.close(fig)

    if per_level_by_algo_depth:
        generate_all_expanded_heatmap(per_level_by_algo_depth, ordered_keys, row_labels, dataset)


def generate_all_expanded_heatmap(
    per_level_by_algo_depth: dict[tuple[str, int], dict[str, dict[int, float]]],
    ordered_keys: list[tuple[str, int]],
    row_labels: list[str],
    dataset: str = 'priority',
) -> None:
    levels_by_game: dict[str, set[int]] = {}
    for key in ordered_keys:
        for game, level_data in per_level_by_algo_depth.get(key, {}).items():
            levels_by_game.setdefault(game, set()).update(level_data.keys())

    # Build per-game solve-rate DFs to sort games by mean pct_solved
    _game_pct_dfs = {}
    for key in ordered_keys:
        data = per_level_by_algo_depth.get(key, {})
        if data:
            game_means = {game: np.mean(list(lvls.values())) for game, lvls in data.items() if lvls}
            _game_pct_dfs[key] = pd.DataFrame.from_dict(game_means, orient='index', columns=['pct_solved'])
    ordered_games = _sort_games_by_mean_pct_solved(sorted(levels_by_game.keys()), _game_pct_dfs)
    sorted_levels_by_game = {game: sorted(levels) for game, levels in levels_by_game.items()}
    columns, game_spans = _build_expanded_heatmap_columns(ordered_games, sorted_levels_by_game)
    if not columns:
        return

    column_keys = [f'{game}::level-{level}' for game, level in columns]
    heatmap_df = pd.DataFrame(index=row_labels, columns=column_keys, dtype=float)
    for key, row_label in zip(ordered_keys, row_labels):
        data = per_level_by_algo_depth.get(key, {})
        for game, level in columns:
            win_rate = data.get(game, {}).get(level)
            if win_rate is not None:
                heatmap_df.at[row_label, f'{game}::level-{level}'] = win_rate

    oom_mask = heatmap_df == OOM_SENTINEL
    heatmap_df = heatmap_df.where(~oom_mask, other=float('nan'))

    stacked = heatmap_df.stack(future_stack=True).dropna()
    if stacked.empty and not oom_mask.any().any():
        return

    annot_data = None
    if len(columns) <= 80:
        annot_data = [
            ['OOM' if oom_mask.iloc[i, j] else (f'{val:.0%}' if pd.notnull(val) else '')
             for j, val in enumerate(row)]
            for i, row in enumerate(heatmap_df.to_numpy(dtype=float))
        ]

    num_cols = len(columns)
    num_rows = len(row_labels)
    fig_w = max(num_cols * 0.35 + 3.0, 12.0)
    fig_h = max(num_rows * 0.8 + 2.5, 3.0)

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        heatmap_df,
        annot=annot_data if annot_data is not None else False,
        fmt='',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Win Rate', 'shrink': 0.8, 'pad': 0.01},
        annot_kws={'size': 7},
        linewidths=0.25,
        linecolor='white',
    )
    if oom_mask.any().any():
        _overlay_oom_cells(ax, heatmap_df, oom_mask)
    _draw_game_dividers(ax, game_spans, num_rows)
    ax.set_title('All Search Types: Win Rate per Level', pad=32)
    plt.xlabel('Level', labelpad=10)
    plt.ylabel('Algorithm · Search depth', labelpad=10)
    plt.yticks(rotation=0)
    ax.set_xticklabels([str(level) for _, level in columns], rotation=0, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    heatmaps_dir = _heatmaps_dir(dataset)
    os.makedirs(heatmaps_dir, exist_ok=True)
    output_path = os.path.join(heatmaps_dir, 'all_search_win_rate_expanded_heatmap.png')
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved expanded heatmap to {output_path}')
    except Exception as e:
        print(f'Error saving all-algos expanded heatmap: {e}')
    finally:
        plt.close()


def generate_heatmaps(
    dfs_by_depth: dict,
    depth_order: list[int],
    algo_label: str,
    algo_slug: str,
    per_level_by_depth: dict[int, dict[str, dict[int, float]]] | None = None,
    dataset: str = 'priority',
) -> None:
    if not dfs_by_depth:
        print('No data available for heatmap generation.')
        return

    # Default to preferred depths only; fall back to all if none match
    preferred = [d for d in HEATMAP_SEARCH_DEPTHS if d in dfs_by_depth]
    ordered_depths = preferred if preferred else [d for d in depth_order if d in dfs_by_depth]
    if not ordered_depths:
        print('No depth-specific data available for heatmap generation.')
        return

    all_games = []
    for depth in ordered_depths:
        for game in dfs_by_depth[depth].index:
            if game not in all_games:
                all_games.append(game)
    all_games = _sort_games_by_mean_pct_solved(all_games, dfs_by_depth)

    if len(all_games) > MAX_GAMES_FOR_HEATMAPS:
        print(f'Skipping {algo_label} heatmaps: {len(all_games)} games exceeds limit of {MAX_GAMES_FOR_HEATMAPS}.')
        return

    # Build best-depth data for solved-only metrics (single row)
    best_depth_df = pd.DataFrame(columns=['mean_solved_iters', 'mean_sol_len', 'has_oom'])
    for game in all_games:
        for depth in reversed(ordered_depths):
            depth_df = dfs_by_depth[depth]
            if game not in depth_df.index:
                continue
            row = depth_df.loc[game]
            pct = row.get('pct_solved', 0)
            if isinstance(pct, pd.Series):
                pct = pct.iloc[0]
            if pct > 0:
                best_depth_df.at[game, 'mean_solved_iters'] = row.get('mean_solved_iters', np.nan)
                best_depth_df.at[game, 'mean_sol_len'] = row.get('mean_sol_len', np.nan)
                best_depth_df.at[game, 'has_oom'] = bool(row.get('has_oom', False))
                break

    heatmap_configs = [
        {
            'column': 'pct_solved',
            'title': f'{algo_label} Percent of Levels Solved per Game',
            'cmap': 'RdYlGn',
            'vmin': 0.0,
            'vmax': 1.0,
            'colorbar_label': 'Average Win Rate',
            'formatter': lambda v: f"{v:.0%}",
            'output': f'{algo_slug}_pct_solved_heatmap.png',
            'per_depth': True,
        },
        {
            'column': 'mean_solved_iters',
            'title': f'{algo_label} Mean Iterations to Solve per Game (best depth, solved only)',
            'cmap': 'Blues',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Iterations to Solve',
            'formatter': lambda v: f"{v:.0f}",
            'output': f'{algo_slug}_mean_iterations_heatmap.png',
            'per_depth': False,
        },
        {
            'column': 'mean_sol_len',
            'title': f'{algo_label} Mean Solution Length per Game (best depth, solved only)',
            'cmap': 'Purples',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Sol. Length',
            'formatter': lambda v: f"{v:.0f}",
            'output': f'{algo_slug}_mean_solution_length_heatmap.png',
            'per_depth': False,
        },
    ]

    target_cell_height = 1.0
    target_cell_width = 1.0
    min_total_figure_width = 8.0
    min_total_figure_height = 3.0
    h_padding = 3.0
    v_padding = 1.5

    for config in heatmap_configs:
        column = config['column']
        per_depth = config['per_depth']

        if per_depth:
            if not any(column in df.columns for df in dfs_by_depth.values()):
                continue

            depth_labels = [_format_steps_label(depth) for depth in ordered_depths]
            heatmap_data = pd.DataFrame(index=depth_labels, columns=all_games, dtype=float)
            oom_mask = pd.DataFrame(False, index=depth_labels, columns=all_games)

            for depth, depth_label in zip(ordered_depths, depth_labels):
                depth_df = dfs_by_depth[depth]
                if column not in depth_df.columns:
                    continue
                for game, value in depth_df[column].items():
                    heatmap_data.at[depth_label, game] = value
                if 'has_oom' in depth_df.columns:
                    for game, value in depth_df['has_oom'].items():
                        if value:
                            oom_mask.at[depth_label, game] = True
            ylabel = 'Search depth'
        else:
            if column not in best_depth_df.columns:
                continue
            heatmap_data = pd.DataFrame(index=[algo_label], columns=all_games, dtype=float)
            oom_mask = pd.DataFrame(False, index=[algo_label], columns=all_games)
            for game in all_games:
                if game in best_depth_df.index and pd.notnull(best_depth_df.at[game, column]):
                    heatmap_data.at[algo_label, game] = float(best_depth_df.at[game, column])
                if game in best_depth_df.index and best_depth_df.get('has_oom', pd.Series(dtype=bool)).get(game, False):
                    oom_mask.at[algo_label, game] = True
            ylabel = 'Algorithm'

        stacked_values = heatmap_data.stack(future_stack=True).dropna()
        if stacked_values.empty:
            continue

        valid_values = stacked_values.astype(float)

        vmin = config.get('vmin')
        vmax = config.get('vmax')
        if vmin is None:
            vmin = float(valid_values.min())
        if vmax is None:
            vmax = float(valid_values.max())
        if np.isfinite(vmin) and np.isfinite(vmax) and np.isclose(vmin, vmax):
            vmax = vmin + (abs(vmin) * 0.05 + 1)

        annot_data = [
            [config['formatter'](val) if pd.notnull(val) else '' for val in row]
            for row in heatmap_data.values
        ]

        num_cols = len(heatmap_data.columns)
        num_rows = len(heatmap_data.index)
        fig_h = max(num_rows * target_cell_height + v_padding, min_total_figure_height)
        fig_w = max(num_cols * target_cell_width + h_padding, min_total_figure_width)

        plt.figure(figsize=(fig_w, fig_h))
        ax = sns.heatmap(
            heatmap_data,
            annot=annot_data,
            fmt="",
            cmap=config['cmap'],
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': config['colorbar_label'], 'shrink': 0.8, 'pad': 0.01},
            annot_kws={"size": 9},
            linewidths=0.5,
            linecolor='white'
        )
        if oom_mask.any().any():
            _overlay_oom_cells(ax, heatmap_data, oom_mask)
        plt.title(config['title'])
        plt.xlabel('Game', labelpad=10)
        plt.ylabel(ylabel, labelpad=10)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        heatmaps_dir = _heatmaps_dir(dataset)
        os.makedirs(heatmaps_dir, exist_ok=True)
        output_path = os.path.join(heatmaps_dir, config['output'])
        try:
            plt.savefig(output_path, dpi=300)
            print(f"Saved heatmap to {output_path}")
        except Exception as e:
            print(f"Error saving heatmap {config['output']}: {e}")
        finally:
            plt.close()

    if per_level_by_depth:
        generate_expanded_heatmap(per_level_by_depth, ordered_depths, algo_label, algo_slug, dataset)


def generate_expanded_heatmap(
    per_level_by_depth: dict[int, dict[str, dict[int, float]]],
    ordered_depths: list[int],
    algo_label: str,
    algo_slug: str,
    dataset: str = 'priority',
) -> None:
    depth_labels = [_format_steps_label(depth) for depth in ordered_depths]
    levels_by_game: dict[str, set[int]] = {}
    for depth in ordered_depths:
        for game, level_data in per_level_by_depth.get(depth, {}).items():
            levels_by_game.setdefault(game, set()).update(level_data.keys())

    _game_pct_dfs = {}
    for depth in ordered_depths:
        data = per_level_by_depth.get(depth, {})
        if data:
            game_means = {game: np.mean(list(lvls.values())) for game, lvls in data.items() if lvls}
            _game_pct_dfs[depth] = pd.DataFrame.from_dict(game_means, orient='index', columns=['pct_solved'])
    ordered_games = _sort_games_by_mean_pct_solved(sorted(levels_by_game.keys()), _game_pct_dfs)
    sorted_levels_by_game = {game: sorted(levels) for game, levels in levels_by_game.items()}
    columns, game_spans = _build_expanded_heatmap_columns(ordered_games, sorted_levels_by_game)
    if not columns:
        return

    column_keys = [f'{game}::level-{level}' for game, level in columns]
    heatmap_df = pd.DataFrame(index=depth_labels, columns=column_keys, dtype=float)
    for depth, depth_label in zip(ordered_depths, depth_labels):
        data = per_level_by_depth.get(depth, {})
        for game, level in columns:
            win_rate = data.get(game, {}).get(level)
            if win_rate is not None:
                heatmap_df.at[depth_label, f'{game}::level-{level}'] = win_rate

    oom_mask = heatmap_df == OOM_SENTINEL
    heatmap_df = heatmap_df.where(~oom_mask, other=float('nan'))

    stacked = heatmap_df.stack(future_stack=True).dropna()
    if stacked.empty and not oom_mask.any().any():
        return

    annot_data = None
    if len(columns) <= 80:
        annot_data = [
            ['OOM' if oom_mask.iloc[i, j] else (f'{val:.0%}' if pd.notnull(val) else '')
             for j, val in enumerate(row)]
            for i, row in enumerate(heatmap_df.to_numpy(dtype=float))
        ]

    num_cols = len(columns)
    num_rows = len(depth_labels)
    fig_w = max(num_cols * 0.35 + 3.0, 12.0)
    fig_h = max(num_rows * 0.8 + 2.5, 3.0)

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        heatmap_df,
        annot=annot_data if annot_data is not None else False,
        fmt='',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Win Rate', 'shrink': 0.8, 'pad': 0.01},
        annot_kws={'size': 7},
        linewidths=0.25,
        linecolor='white',
    )
    if oom_mask.any().any():
        _overlay_oom_cells(ax, heatmap_df, oom_mask)
    _draw_game_dividers(ax, game_spans, num_rows)
    ax.set_title(f'{algo_label} Win Rate per Level', pad=32)
    plt.xlabel('Level', labelpad=10)
    plt.ylabel('Search depth', labelpad=10)
    plt.yticks(rotation=0)
    ax.set_xticklabels([str(level) for _, level in columns], rotation=0, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    heatmaps_dir = _heatmaps_dir(dataset)
    os.makedirs(heatmaps_dir, exist_ok=True)
    output_path = os.path.join(heatmaps_dir, f'{algo_slug}_win_rate_expanded_heatmap.png')
    try:
        plt.savefig(output_path, dpi=300)
        print(f'Saved expanded heatmap to {output_path}')
    except Exception as e:
        print(f'Error saving expanded heatmap: {e}')
    finally:
        plt.close()


@hydra.main(version_base="1.3", config_path="conf", config_name="plot_standalone_bfs_config")
def old_plot(cfg: PlotSearch):
    with open(STANDALONE_NODEJS_RESULTS_PATH, 'r') as f:
        results = json.load(f)

    # Create a directory for the plots if necessary
    out_dir = _search_out_dir(cfg.dataset)
    os.makedirs(out_dir, exist_ok=True)

    sorted_games = get_list_of_games_for_testing(cfg.dataset)
    # Build a sorting key for each index based on the sorted_games list
    game_order = {game: i for i, game in enumerate(sorted_games)}


    for run_name in results.keys():
        run_results = results[run_name]
        if len(run_results) == 0:
            print(f'No results for {run_name}. Skipping.')
            continue

        print(f'Plotting results for {run_name}.')

        df_col_keys = [
            'solved',
            # 'FPS',
            'iterations',
            # 'score'
        ]
        df_row_headers = ['game', 'level']
        df_row_indices = []
        df_rows = []

        for game in run_results.keys():
            game_results = run_results[game]
            level_keys = list(game_results.keys())

            if len(game_results) == 0:
                print(f'No results for {game}. Skipping.')
                continue
                
            if game not in sorted_games:
                print(f'Game {game} not in sorted games. Skipping.')
                continue

            for level in level_keys:
                level_results = game_results[level]

                if len(level_results) == 0:
                    print(f'No results for {level}. Skipping.')
                    continue

                # Add this stuff to the dataframe
                df_row_indices.append((game, level))
                # df_rows.append([level_results[key] for key in df_col_keys])
                row_data = []
                for key in df_col_keys:
                    val = level_results[key]
                    if key == "solved":
                        val = r"\colorbox{green}{~}" if val else r"\colorbox{red}{~}"
                    row_data.append(val)
                df_rows.append(row_data)

        # Create a dataframe from the results
        df = pd.DataFrame(df_rows, columns=df_col_keys, index=pd.MultiIndex.from_tuples(df_row_indices, names=df_row_headers))

        # Sort games according to their ordering in `sorted_games`
        # Sort levels according to their integer value. Convert them to integers here!
        df = df.sort_values(
            by=['game', 'level'],
            key=lambda col: (
                col.map(game_order) if col.name == 'game'
                else col.astype(int)  # assumes levels like '0', '1', ...
            )
        )

        algo_name, n_steps, device_name = get_standalone_run_params_from_name(run_name)
        concise_run_name = f'{algo_name}_{n_steps}-steps'

        # Save the dataframe to a CSV file
        csv_file_name = f'{run_name}.csv'
        csv_file_path = os.path.join(out_dir, csv_file_name)

        # Remove underscores from game names
        df.index = df.index.set_levels([df.index.levels[0].str.replace('_', ' '), df.index.levels[1]])
        
        # Save to a latex table
        latex_file_name = f'{concise_run_name}.tex'
        latex_file_path = os.path.join(out_dir, latex_file_name)
        # df.to_latex(latex_file_path, index=True, float_format="%.2f", escape=False)
        # print(f'Saved latex table to {latex_file_path}')

        # Split the dataframe roughly in half
        split_index = len(df) // 2
        df1 = df.iloc[:split_index]
        df2 = df.iloc[split_index:]

        # Build LaTeX minipage output
        latex_output = r"""
\centering
\begin{subtable}[t]{0.48\linewidth}
\centering
""" + df1.to_latex(index=True, float_format="%.2f", escape=False, caption=False) + r"""
\end{subtable}%
\hfill
\begin{subtable}[t]{0.48\linewidth}
\centering
""" + df2.to_latex(index=True, float_format="%.2f", escape=False, caption=False) + r"""
\end{subtable}
        """

        latex_file_name = f'{concise_run_name}_split.tex'
        latex_file_path = os.path.join(out_dir, latex_file_name)
        with open(latex_file_path, 'w') as f:
            f.write(latex_output)

        print(f'Saved split latex table to {latex_file_path}')

    
if __name__ == "__main__":
    main()
