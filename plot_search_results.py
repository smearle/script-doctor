import glob
import json
import os
import re

import hydra
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from puzzlescript_jax.conf.config import PlotSearch
from puzzlescript_jax.globals import PLOTS_DIR, STANDALONE_NODEJS_RESULTS_PATH, JS_SOLS_DIR
from profile_nodejs import get_algo_name, get_standalone_run_params_from_name
from puzzlescript_jax.utils import get_list_of_games_for_testing, game_names_remap


BFS_RESULTS_PATH = os.path.join('data', 'bfs_results.json')
HEATMAP_SEARCH_DEPTHS = [1_000_000, 100_000]
ALL_RESULTS_PATH = os.path.join('data', 'all_search_results.json')
EXIT_RESULTS_PATH = os.path.join('data', 'exit_results.json')
EXIT_TRAINING_DIR = os.path.join('data', 'exit_training')

ALGO_SOLVER_NAMES = {
    'astar': 'solveAStar',
    'bfs': 'solveBFS',
    'gbfs': 'solveGBFS',
    'mcts': 'solveMCTS',
}


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


def _collect_exit_results(games: list[str]) -> dict:
    if not os.path.exists(EXIT_TRAINING_DIR):
        return {}

    game_filter = set(games) if games is not None else None

    candidate_dirs = set()
    checkpoint_paths = glob.glob(os.path.join(EXIT_TRAINING_DIR, '**', 'checkpoint.json'), recursive=True)
    history_paths = glob.glob(os.path.join(EXIT_TRAINING_DIR, '**', 'history.json'), recursive=True)
    for path in checkpoint_paths + history_paths:
        candidate_dirs.add(os.path.dirname(path))

    per_game = {}
    for job_dir in sorted(candidate_dirs):
        dirname = os.path.basename(job_dir)
        game, level = _parse_exit_job_dirname(dirname)
        if game is None:
            continue
        if game_filter is not None and game not in game_filter:
            continue

        history = _load_exit_history(job_dir)
        if not history:
            continue

        solved_any = any(bool(record.get('solved', False)) for record in history if isinstance(record, dict))

        if game not in per_game:
            per_game[game] = {
                'solved_levels': 0,
                'n_levels': 0,
                'iter_counts': [],
            }

        per_game[game]['n_levels'] += 1
        per_game[game]['solved_levels'] += int(solved_any)
        per_game[game]['iter_counts'].append(len(history))

    results = {}
    for game, stats in per_game.items():
        n_levels = stats['n_levels']
        if n_levels <= 0:
            continue
        results[game] = {
            'pct_solved': stats['solved_levels'] / n_levels,
            'n_levels': n_levels,
            'n_iters': float(np.mean(stats['iter_counts'])) if stats['iter_counts'] else float('nan'),
        }
    return results


def _parse_solver_run_name(filename: str):
    match = re.match(r'^(solve[A-Za-z0-9]+)_(\d+)-steps_level-(\d+)\.json$', filename)
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


def _collect_results_for_algo(games: list[str], solver_name: str) -> dict[int, dict]:
    results_by_depth = {}
    for game in games:
        if game.startswith('test_'):
            continue
        game_dir = os.path.join(JS_SOLS_DIR, game)
        sol_jsons = glob.glob(f"{game_dir}/*.json")
        per_depth_stats = {}
        expected_levels = set()
        levels_seen_by_depth = {}
        for sol_json in sol_jsons:
            filename = os.path.basename(sol_json)
            sol_algo_name, n_steps, level_id = _parse_solver_run_name(filename)
            if sol_algo_name is None:
                continue

            expected_levels.add(level_id)

            if sol_algo_name != solver_name:
                continue

            if n_steps not in levels_seen_by_depth:
                levels_seen_by_depth[n_steps] = set()
            levels_seen_by_depth[n_steps].add(level_id)

            with open(sol_json, 'r') as f:
                sol_dict = json.load(f)
            if 'iterations' not in sol_dict or 'won' not in sol_dict:
                print(f"Skipping {sol_json} because it doesn't have 'iterations' or 'won'")
                continue

            if n_steps not in per_depth_stats:
                per_depth_stats[n_steps] = {
                    'n_levels': 0,
                    'n_solved': 0,
                    'n_stepss': [],
                    'solution_lengths': [],
                }

            stats = per_depth_stats[n_steps]
            solved = sol_dict['won']
            actions = sol_dict.get('actions')
            if solved:
                stats['n_solved'] += 1
                stats['solution_lengths'].append(len(actions) if actions is not None else 0)
            stats['n_levels'] += 1
            clipped_steps = min(n_steps, sol_dict['iterations'])
            stats['n_stepss'].append(clipped_steps)

        for depth, seen_levels in sorted(levels_seen_by_depth.items()):
            missing_levels = sorted(expected_levels - seen_levels)
            if missing_levels:
                depth_label = _format_steps_label(depth)
                missing_levels_summary = _format_level_ranges(missing_levels)
                expected_levels_summary = _format_level_ranges(list(expected_levels))
                print(
                    f"Warning: missing search results for algorithm '{solver_name}', game '{game}', depth {depth_label} "
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
            results_by_depth[depth][game] = {
                'pct_solved': pct_solved,
                'n_levels': stats['n_levels'],
                'n_iters': n_steps_mean,
                'mean_sol_len': mean_solution_length
            }

    return results_by_depth


def _depth_order_for_results(results_by_depth: dict[int, dict]) -> list[int]:
    preferred_depth_order = [depth for depth in HEATMAP_SEARCH_DEPTHS if depth in results_by_depth]
    fallback_depth_order = sorted(
        [depth for depth in results_by_depth.keys() if depth not in preferred_depth_order],
        reverse=True,
    )
    return preferred_depth_order + fallback_depth_order


@hydra.main(version_base="1.3", config_path="puzzlejax/conf", config_name="plot_standalone_bfs_config")
def main(cfg: PlotSearch):
    if cfg.aggregate:
        aggregate_results(cfg)
    else:
        plot(cfg)
    

def aggregate_results(cfg: PlotSearch):
    if cfg.game is not None:
        games = [cfg.game]
    else:
        games = get_list_of_games_for_testing(cfg.all_games)
    print(games)
    if cfg.algo == 'all':
        algos = list(ALGO_SOLVER_NAMES.keys())
    elif cfg.algo == 'exit':
        exit_results = _collect_exit_results(games)
        with open(EXIT_RESULTS_PATH, 'w') as f:
            json.dump(exit_results, f, indent=4)
        print(f'Saved aggregated ExIt results to {EXIT_RESULTS_PATH}')
        plot(cfg, exit_results)
        return
    elif cfg.algo in ALGO_SOLVER_NAMES:
        algos = [cfg.algo]
    else:
        raise ValueError(f'Unknown algo: {cfg.algo}')

    aggregated_results = {}
    for algo in algos:
        solver_name = ALGO_SOLVER_NAMES[algo]
        results_by_depth = _collect_results_for_algo(games, solver_name)
        results_path = _results_path_for_algo(algo)
        with open(results_path, 'w') as f:
            json.dump({str(k): v for k, v in results_by_depth.items()}, f, indent=4)
        print(f'Saved aggregated results to {results_path}')
        aggregated_results[algo] = results_by_depth

    if cfg.algo == 'all':
        with open(ALL_RESULTS_PATH, 'w') as f:
            json.dump({algo: {str(k): v for k, v in depths.items()} for algo, depths in aggregated_results.items()}, f, indent=4)
        print(f'Saved combined aggregated results to {ALL_RESULTS_PATH}')
        plot(cfg, aggregated_results)
    else:
        plot(cfg, aggregated_results[cfg.algo])


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


def plot(cfg: PlotSearch, results=None):
    M = 40  # max number of games per table

    if cfg.algo == 'all':
        plot_all_algos(cfg, results)
        return

    if cfg.algo == 'exit':
        if results is None:
            if os.path.exists(EXIT_RESULTS_PATH):
                with open(EXIT_RESULTS_PATH, 'r') as f:
                    results = json.load(f)
            else:
                results = {}
        plot_exit_heatmap(results)
        return

    if results is None:
        results_path = _results_path_for_algo(cfg.algo)
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            with open(BFS_RESULTS_PATH, 'r') as f:
                results = json.load(f)

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

    os.makedirs(PLOTS_DIR, exist_ok=True)
    algo_slug = cfg.algo.lower()
    algo_label = _algo_label(cfg.algo)

    csv_file_path = os.path.join(PLOTS_DIR, f'standalone_{algo_slug}_results.csv')
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

    generate_heatmaps(heatmap_source_dfs, depth_order, algo_label, algo_slug)

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

    latex_file_path = os.path.join(PLOTS_DIR, f'{algo_slug}_results.tex')
    caption_steps = _format_steps_label(selected_depth)
    with open(latex_file_path, 'w') as f:
        f.write(latex_df.to_latex(index=True, float_format="%.2f", escape=False, caption=f"Results of {algo_label} on full dataset of games, with max {caption_steps} and a timeout of 1 minute.",
                            longtable=True, label="tab:bfs_results"))
    print(f'Saved latex table to {latex_file_path}')


def plot_exit_heatmap(results: dict) -> None:
    if not results:
        print('No ExIt results found to plot.')
        return

    df = pd.DataFrame.from_dict(results, orient='index')
    if df.empty or 'pct_solved' not in df.columns:
        print('No ExIt solve-rate data found to plot.')
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_file_path = os.path.join(PLOTS_DIR, 'exit_results.csv')
    df.to_csv(csv_file_path, index=True, float_format='%.4f')
    print(f'Saved results to {csv_file_path}')

    pretty_index = df.index.to_series().replace(game_names_remap)
    pretty_index = pretty_index.str.replace('_', ' ')
    pretty_index = pretty_index.str.title()

    solved_row = pd.DataFrame(
        [df['pct_solved'].to_numpy(dtype=float)],
        index=['ExIt (any solved during training)'],
        columns=pretty_index,
    )

    annot_data = [[f"{value:.0%}" if pd.notnull(value) else '' for value in solved_row.iloc[0]]]

    target_cell_height = 1.0
    target_cell_width = 1.0
    min_total_figure_width = 8.0
    min_total_figure_height = 3.0
    h_padding = 3.0
    v_padding = 1.5

    num_cols = len(solved_row.columns)
    num_rows = len(solved_row.index)
    fig_h = max(num_rows * target_cell_height + v_padding, min_total_figure_height)
    fig_w = max(num_cols * target_cell_width + h_padding, min_total_figure_width)

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        solved_row,
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

    output_path = os.path.join(PLOTS_DIR, 'exit_pct_solved_heatmap.png')
    try:
        plt.savefig(output_path, dpi=300)
        print(f'Saved heatmap to {output_path}')
    except Exception as e:
        print(f'Error saving ExIt heatmap: {e}')
    finally:
        plt.close()


def plot_all_algos(cfg: PlotSearch, results_by_algo=None):
    if results_by_algo is None:
        results_by_algo = {}
        for algo in ALGO_SOLVER_NAMES:
            results_path = _results_path_for_algo(algo)
            if not os.path.exists(results_path):
                continue
            with open(results_path, 'r') as f:
                loaded = json.load(f)
            results_by_algo[algo] = _normalize_results_by_depth(loaded, cfg.n_steps)

    normalized_by_algo = {}
    for algo, algo_results in results_by_algo.items():
        normalized_by_algo[algo] = _normalize_results_by_depth(algo_results, cfg.n_steps)

    if not normalized_by_algo:
        print('No search results found to plot.')
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    summary_rows = []
    heatmap_source_dfs = {}
    for algo, depths in normalized_by_algo.items():
        depth_order = _depth_order_for_results(depths)
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
                })

            remapped_df = depth_df.copy()
            remapped_df.index = remapped_df.index.to_series().replace(game_names_remap)
            remapped_df.index = remapped_df.index.str.replace('_', ' ')
            remapped_df.index = remapped_df.index.str.title()
            heatmap_source_dfs[(algo, depth)] = remapped_df

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.sort_values(by=['algo', 'depth', 'game'], ascending=[True, False, True], inplace=True)
        summary_csv_path = os.path.join(PLOTS_DIR, 'standalone_all_search_results.csv')
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
        print(f'Saved results to {summary_csv_path}')

    generate_all_heatmaps(heatmap_source_dfs)


def generate_all_heatmaps(dfs_by_algo_depth: dict) -> None:
    if not dfs_by_algo_depth:
        print('No data available for all-algorithm heatmap generation.')
        return

    preferred_depths = HEATMAP_SEARCH_DEPTHS
    ordered_keys = sorted(
        dfs_by_algo_depth.keys(),
        key=lambda key: (
            list(ALGO_SOLVER_NAMES.keys()).index(key[0]) if key[0] in ALGO_SOLVER_NAMES else 999,
            preferred_depths.index(key[1]) if key[1] in preferred_depths else len(preferred_depths),
            -key[1],
        ),
    )

    row_labels = [f"{_algo_label(algo)} · {_format_steps_label(depth)}" for algo, depth in ordered_keys]
    all_games = []
    for key in ordered_keys:
        for game in dfs_by_algo_depth[key].index:
            if game not in all_games:
                all_games.append(game)

    heatmap_configs = [
        {
            'column': 'pct_solved',
            'title': 'All Search Types: Percent of Levels Solved per Game',
            'cmap': 'RdYlGn',
            'vmin': 0.0,
            'vmax': 1.0,
            'colorbar_label': 'Average Win Rate',
            'formatter': lambda v: f"{v:.0%}",
            'output': 'all_search_pct_solved_heatmap.png',
        },
        {
            'column': 'n_iters',
            'title': 'All Search Types: Mean Search Iterations per Game',
            'cmap': 'Blues',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Search Iterations',
            'formatter': lambda v: f"{v:.0f}",
            'output': 'all_search_mean_iterations_heatmap.png',
        },
        {
            'column': 'mean_sol_len',
            'title': 'All Search Types: Mean Solution Length per Game',
            'cmap': 'Purples',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Sol. Length',
            'formatter': lambda v: f"{v:.0f}",
            'output': 'all_search_mean_solution_length_heatmap.png',
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
        if not any(column in df.columns for df in dfs_by_algo_depth.values()):
            continue

        heatmap_data = pd.DataFrame(index=row_labels, columns=all_games, dtype=float)
        for key, row_label in zip(ordered_keys, row_labels):
            key_df = dfs_by_algo_depth[key]
            if column not in key_df.columns:
                continue
            for game, value in key_df[column].items():
                heatmap_data.at[row_label, game] = value

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
        sns.heatmap(
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
        plt.title(config['title'])
        plt.xlabel('Game', labelpad=10)
        plt.ylabel('Algorithm · Search depth', labelpad=10)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = os.path.join(PLOTS_DIR, config['output'])
        try:
            plt.savefig(output_path, dpi=300)
            print(f"Saved heatmap to {output_path}")
        except Exception as e:
            print(f"Error saving heatmap {config['output']}: {e}")
        finally:
            plt.close()

def generate_heatmaps(dfs_by_depth: dict, depth_order: list[int], algo_label: str, algo_slug: str) -> None:
    if not dfs_by_depth:
        print('No data available for heatmap generation.')
        return

    ordered_depths = [depth for depth in depth_order if depth in dfs_by_depth]
    if not ordered_depths:
        print('No depth-specific data available for heatmap generation.')
        return

    all_games = []
    for depth in ordered_depths:
        for game in dfs_by_depth[depth].index:
            if game not in all_games:
                all_games.append(game)

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
        },
        {
            'column': 'n_iters',
            'title': f'{algo_label} Mean Search Iterations per Game',
            'cmap': 'Blues',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Search Iterations',
            'formatter': lambda v: f"{v:.0f}",
            'output': f'{algo_slug}_mean_iterations_heatmap.png',
        },
        {
            'column': 'mean_sol_len',
            'title': f'{algo_label} Mean Solution Length per Game',
            'cmap': 'Purples',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Sol. Length',
            'formatter': lambda v: f"{v:.0f}",
            'output': f'{algo_slug}_mean_solution_length_heatmap.png',
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
        if not any(column in df.columns for df in dfs_by_depth.values()):
            continue

        depth_labels = [_format_steps_label(depth) for depth in ordered_depths]
        heatmap_data = pd.DataFrame(index=depth_labels, columns=all_games, dtype=float)

        for depth, depth_label in zip(ordered_depths, depth_labels):
            depth_df = dfs_by_depth[depth]
            if column not in depth_df.columns:
                continue
            for game, value in depth_df[column].items():
                heatmap_data.at[depth_label, game] = value

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
        sns.heatmap(
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
        plt.title(config['title'])
        plt.xlabel('Game', labelpad=10)
        plt.ylabel('Search depth', labelpad=10)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha='right')
        # plt.tight_layout(pad=1.2, rect=[0, 0, 0.92, 1])
        plt.tight_layout()

        output_path = os.path.join(PLOTS_DIR, config['output'])
        try:
            plt.savefig(output_path, dpi=300)
            print(f"Saved heatmap to {output_path}")
        except Exception as e:
            print(f"Error saving heatmap {config['output']}: {e}")
        finally:
            plt.close()


@hydra.main(version_base="1.3", config_path="conf", config_name="plot_standalone_bfs_config")
def old_plot(cfg: PlotSearch):
    with open(STANDALONE_NODEJS_RESULTS_PATH, 'r') as f:
        results = json.load(f)

    # Create a directory for the plots if necessary
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sorted_games = get_list_of_games_for_testing(cfg.all_games) 
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
        csv_file_path = os.path.join(PLOTS_DIR, csv_file_name)

        # Remove underscores from game names
        df.index = df.index.set_levels([df.index.levels[0].str.replace('_', ' '), df.index.levels[1]])
        
        # Save to a latex table
        latex_file_name = f'{concise_run_name}.tex'
        latex_file_path = os.path.join(PLOTS_DIR, latex_file_name)
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
        latex_file_path = os.path.join(PLOTS_DIR, latex_file_name)
        with open(latex_file_path, 'w') as f:
            f.write(latex_output)

        print(f'Saved split latex table to {latex_file_path}')

    
if __name__ == "__main__":
    main()
