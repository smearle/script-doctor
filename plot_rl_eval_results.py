import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from puzzlescript_jax.globals import PLOTS_DIR
from puzzlescript_jax.utils import game_names_remap


RL_CONFIG_FIELD_ORDER = (
    'n_envs',
    'model',
    'ep_len',
)
RL_CONFIG_FIELD_LABELS = {
    'n_envs': 'Num envs',
    'model': 'Model',
    'ep_len': 'Episode length',
}
RL_DEFAULT_MAX_EPISODE_STEPS = 100


def _parse_scalar(value: str):
    if value.isdigit():
        return int(value)
    try:
        parsed = float(value)
    except ValueError:
        return value
    return int(parsed) if parsed.is_integer() else parsed


def _format_scalar(value) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        return f'{value:.12g}'
    return str(value)


def _parse_level_name(dirname: str):
    if not dirname.startswith('level-'):
        return None
    level_suffix = dirname[len('level-'):]
    return int(level_suffix) if level_suffix.isdigit() else None


def _parse_run_config(dirname: str) -> dict:
    config = {'ep_len': RL_DEFAULT_MAX_EPISODE_STEPS}
    for token in dirname.split('_'):
        if token.startswith('seed-'):
            continue
        if token.startswith('n-envs-'):
            config['n_envs'] = _parse_scalar(token[len('n-envs-'):])
            continue
        if token.startswith('ep-len-'):
            config['ep_len'] = _parse_scalar(token[len('ep-len-'):])
            continue
        config['model'] = token
    return config


def _iter_eval_runs(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"RL logs directory '{root}' does not exist.")

    for game_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for eval_path in sorted(game_dir.glob('**/eval_stats.json')):
            level_dir = eval_path.parent.parent
            run_dir = eval_path.parent
            level = _parse_level_name(level_dir.name)
            if level is None:
                continue
            yield {
                'game': game_dir.name,
                'level': level,
                'run_dir': run_dir,
                'config': _parse_run_config(run_dir.name),
                'eval_path': eval_path,
            }


def _load_eval_stats(eval_path: Path):
    try:
        with eval_path.open('r') as fh:
            stats = json.load(fh)
    except (OSError, json.JSONDecodeError):
        print(f"Skipping unreadable stats file: {eval_path}")
        return None

    n_eps = stats.get('n_eps') or stats.get('n_episodes')
    mean_wins = stats.get('mean_wins')
    n_wins = stats.get('n_wins')

    if isinstance(mean_wins, (int, float)):
        mean_wins = min(float(mean_wins), 1.0)

    eps_valid = isinstance(n_eps, (int, float)) and n_eps > 0
    wins_valid = isinstance(n_wins, (int, float))
    mean_valid = isinstance(mean_wins, (int, float))

    if not wins_valid and mean_valid and eps_valid:
        n_wins = mean_wins * n_eps
        wins_valid = True

    return {
        'n_eps': int(n_eps) if eps_valid else None,
        'n_wins': float(n_wins) if wins_valid else None,
        'mean_wins': float(mean_wins) if mean_valid else None,
    }


def _varying_config_fields(configs: list[dict]) -> list[str]:
    varying_fields = []
    for field in RL_CONFIG_FIELD_ORDER:
        values = {config.get(field) for config in configs}
        if len(values) > 1:
            varying_fields.append(field)
    return varying_fields


def _format_config_label(config: dict, varying_fields: list[str]) -> str:
    if not varying_fields:
        return 'Default config'

    parts = []
    for field in varying_fields:
        value = config.get(field)
        parts.append(f"{RL_CONFIG_FIELD_LABELS[field]}={_format_scalar(value)}")
    return ', '.join(parts)


def _format_game_label(game: str) -> str:
    label = game_names_remap.get(game, game)
    label = label.replace('_', ' ')
    return label.title()


def collect_eval_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    """Collect both aggregate and per-level evaluation win rates under ``root``."""
    run_infos = list(_iter_eval_runs(root))
    if not run_infos:
        return (
            pd.DataFrame(columns=['config_label', 'game', 'avg_win_rate', 'n_eval_files', 'total_eval_episodes', 'n_levels']),
            pd.DataFrame(columns=['config_label', 'game', 'level', 'win_rate']),
            {},
        )

    varying_fields = _varying_config_fields([run_info['config'] for run_info in run_infos])
    expected_levels_by_game = {}
    grouped_stats = {}

    for run_info in run_infos:
        expected_levels_by_game.setdefault(run_info['game'], set()).add(run_info['level'])
        stats = _load_eval_stats(run_info['eval_path'])
        if stats is None:
            continue

        config_label = _format_config_label(run_info['config'], varying_fields)
        key = (config_label, run_info['game'])
        if key not in grouped_stats:
            grouped_stats[key] = {
                'config_label': config_label,
                'game': run_info['game'],
                'level_win_rates': {},
                'n_eval_files': 0,
                'total_eval_episodes': 0,
            }

        grouped = grouped_stats[key]
        grouped['n_eval_files'] += 1
        if stats['n_eps'] is not None:
            grouped['total_eval_episodes'] += stats['n_eps']

        level_win_rate = None
        if stats['n_wins'] is not None and stats['n_eps'] is not None and stats['n_eps'] > 0:
            level_win_rate = stats['n_wins'] / stats['n_eps']
        elif stats['mean_wins'] is not None:
            level_win_rate = stats['mean_wins']

        if level_win_rate is None:
            continue

        grouped['level_win_rates'].setdefault(run_info['level'], []).append(level_win_rate)

    rows = []
    level_rows = []
    for grouped in grouped_stats.values():
        expected_levels = expected_levels_by_game.get(grouped['game'], set())
        if not expected_levels:
            continue

        observed_levels = set(grouped['level_win_rates'])
        missing_levels = sorted(expected_levels - observed_levels)
        if missing_levels:
            missing_levels_str = ', '.join(str(level) for level in missing_levels)
            print(
                f"Warning: missing RL eval results for config '{grouped['config_label']}', "
                f"game '{grouped['game']}' on levels {missing_levels_str}. Treating them as 0 win-rate."
            )

        per_level_win_rates = []
        for level in sorted(expected_levels):
            level_samples = grouped['level_win_rates'].get(level)
            if level_samples:
                level_win_rate = float(np.mean(level_samples))
            else:
                level_win_rate = 0.0
            per_level_win_rates.append(level_win_rate)
            level_rows.append(
                {
                    'config_label': grouped['config_label'],
                    'game': grouped['game'],
                    'level': level,
                    'win_rate': level_win_rate,
                }
            )

        avg_win_rate = float(np.mean(per_level_win_rates))

        rows.append(
            {
                'config_label': grouped['config_label'],
                'game': grouped['game'],
                'avg_win_rate': avg_win_rate,
                'n_eval_files': grouped['n_eval_files'],
                'total_eval_episodes': grouped['total_eval_episodes'],
                'n_levels': len(expected_levels),
            }
        )

    if not rows:
        return (
            pd.DataFrame(columns=['config_label', 'game', 'avg_win_rate', 'n_eval_files', 'total_eval_episodes', 'n_levels']),
            pd.DataFrame(columns=['config_label', 'game', 'level', 'win_rate']),
            {game: sorted(levels) for game, levels in expected_levels_by_game.items()},
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(level_rows),
        {game: sorted(levels) for game, levels in expected_levels_by_game.items()},
    )


def collect_eval_stats(root: Path) -> pd.DataFrame:
    summary_df, _, _ = collect_eval_data(root)
    return summary_df


def prettify_game_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    renamed = df.copy()
    if 'game' in renamed.columns:
        renamed['game'] = renamed['game'].replace(game_names_remap)
        renamed['game'] = renamed['game'].str.replace('_', ' ', regex=False)
        renamed['game'] = renamed['game'].str.title()
    else:
        renamed.index = renamed.index.to_series().replace(game_names_remap)
        renamed.index = renamed.index.str.replace('_', ' ', regex=False)
        renamed.index = renamed.index.str.title()
    return renamed


def plot_win_rate_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print('No data available for RL win-rate heatmap.')
        return

    data = df.copy()
    annot_data = [
        [f"{val:.0%}" if np.isfinite(val) else '' for val in row]
        for row in data.to_numpy(dtype=float)
    ]

    num_cols = len(data.columns)
    num_rows = len(data.index)
    target_cell_width = 1.0
    target_cell_height = 1.0
    min_total_width = 8.0
    min_total_height = 3.0
    h_padding = 3.0
    v_padding = 2.5

    fig_w = max(num_cols * target_cell_width + h_padding, min_total_width)
    fig_h = max(num_rows * target_cell_height + v_padding, min_total_height)

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        data,
        annot=annot_data,
        fmt='',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Average Win Rate', 'shrink': 0.8, 'pad': 0.01},
        annot_kws={'size': 9},
        linewidths=0.5,
        linecolor='white',
    )
    plt.title('RL Average Win Rate per Game and Hyperparameter Setting')
    plt.xlabel('Game', labelpad=10)
    plt.ylabel('RL Run Config', labelpad=10)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved RL win-rate heatmap to {output_path}")


def _build_expanded_heatmap_columns(
    ordered_games: list[str], expected_levels_by_game: dict[str, list[int]]
) -> tuple[list[tuple[str, int]], list[tuple[str, int, int]]]:
    columns = []
    game_spans = []
    for game in ordered_games:
        levels = expected_levels_by_game.get(game, [])
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
            center,
            1.005,
            _format_game_label(game),
            ha='center',
            va='bottom',
            fontsize=9,
            transform=ax.get_xaxis_transform(),
        )


def plot_expanded_win_rate_heatmap(
    level_df: pd.DataFrame,
    expected_levels_by_game: dict[str, list[int]],
    output_path: Path,
) -> None:
    if level_df.empty:
        print('No data available for expanded RL win-rate heatmap.')
        return

    ordered_games = [game for game in sorted(expected_levels_by_game) if expected_levels_by_game[game]]
    columns, game_spans = _build_expanded_heatmap_columns(ordered_games, expected_levels_by_game)
    if not columns:
        print('No level columns available for expanded RL win-rate heatmap.')
        return

    config_labels = list(dict.fromkeys(level_df['config_label']))
    column_keys = [f'{game}::level-{level}' for game, level in columns]
    heatmap_df = pd.DataFrame(index=config_labels, columns=column_keys, dtype=float)
    value_lookup = {
        (row.config_label, row.game, int(row.level)): float(row.win_rate)
        for row in level_df.itertuples(index=False)
    }

    for config_label in config_labels:
        for game, level in columns:
            value = value_lookup.get((config_label, game, level))
            if value is not None:
                heatmap_df.at[config_label, f'{game}::level-{level}'] = value

    annot_data = None
    if len(columns) <= 40:
        annot_data = [
            [f"{val:.0%}" if np.isfinite(val) else '' for val in row]
            for row in heatmap_df.to_numpy(dtype=float)
        ]

    num_cols = len(columns)
    num_rows = len(config_labels)
    target_cell_width = 0.35
    target_cell_height = 0.8
    min_total_width = 12.0
    min_total_height = 3.0
    h_padding = 3.0
    v_padding = 2.5

    fig_w = max(num_cols * target_cell_width + h_padding, min_total_width)
    fig_h = max(num_rows * target_cell_height + v_padding, min_total_height)

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
    _draw_game_dividers(ax, game_spans, num_rows)
    ax.set_title('RL Win Rate per Level and Hyperparameter Setting', pad=32)
    plt.xlabel('Level', labelpad=10)
    plt.ylabel('RL Run Config', labelpad=10)
    plt.yticks(rotation=0)
    ax.set_xticklabels([str(level) for _, level in columns], rotation=0, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved expanded RL win-rate heatmap to {output_path}")


def main(args: argparse.Namespace) -> None:
    rl_root = Path(args.rl_logs_dir)
    df, level_df, expected_levels_by_game = collect_eval_data(rl_root)

    if df.empty:
        print('No evaluation stats found; nothing to plot.')
        return

    display_df = prettify_game_index(df)

    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = plots_dir / 'rl_eval_results.csv'
    display_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved RL evaluation summary to {csv_path}")

    heatmap_df = display_df.pivot(index='config_label', columns='game', values='avg_win_rate')
    plot_win_rate_heatmap(heatmap_df, plots_dir / 'rl_avg_win_rate_heatmap.png')
    plot_expanded_win_rate_heatmap(level_df, expected_levels_by_game, plots_dir / 'rl_win_rate_expanded_heatmap.png')


def parse_args(argv: Iterable[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot RL evaluation win rates per game.')
    parser.add_argument(
        '--rl-logs-dir',
        default='rl_logs_jax',
        help='Path to the root directory containing RL experiment logs.',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
