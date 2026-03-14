"""Plot RL training results from progress.csv files.

A run is considered to have solved a level if it ever logged a win for that
level at any point during training.  For solution length, the shortest
(minimum) episode length among all won episodes across all timesteps and seeds
is reported.

See also: plot_rl_eval_results.py, which plots from separate eval_stats.json
files produced by a post-training evaluation pass.
"""
import argparse
import re
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


# ---------------------------------------------------------------------------
# Config parsing helpers (shared with plot_rl_eval_results.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _iter_training_runs(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"RL logs directory '{root}' does not exist.")
    for game_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for csv_path in sorted(game_dir.glob('**/progress.csv')):
            run_dir = csv_path.parent
            yield {
                'game': game_dir.name,
                'run_dir': run_dir,
                'config': _parse_run_config(run_dir.name),
                'csv_path': csv_path,
            }


def _load_training_stats(csv_path: Path) -> dict[int, dict]:
    """Return {level_i: {'ever_won': bool, 'min_sol_len': int|None}}.

    ``min_sol_len`` is the shortest winning episode length seen across all
    timesteps in this run; ``None`` if the level was never won.
    """
    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError):
        print(f"Skipping unreadable CSV: {csv_path}")
        return {}

    level_win_cols = [c for c in df.columns if re.match(r'^level-\d+-win$', c)]
    if not level_win_cols:
        return {}

    result = {}
    for col in level_win_cols:
        level_i = int(col.split('-')[1])
        ever_won = bool(df[col].max() > 0)
        min_sol_len = None
        if ever_won:
            sol_col = f'level-{level_i}-min_sol_len'
            if sol_col in df.columns:
                valid = df[sol_col].dropna()
                if len(valid) > 0:
                    min_sol_len = int(valid.min())
        result[level_i] = {'ever_won': ever_won, 'min_sol_len': min_sol_len}
    return result


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def collect_results_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    """Collect per-level win rates and best solution lengths under ``root``.

    Win rate for a (config, game, level) is the fraction of seeds that ever
    solved that level.  Solution length is the minimum across all seeds and
    timesteps (best solution found anywhere).
    """
    run_infos = list(_iter_training_runs(root))
    if not run_infos:
        return (
            pd.DataFrame(columns=['config_label', 'game', 'avg_win_rate', 'n_runs', 'n_levels']),
            pd.DataFrame(columns=['config_label', 'game', 'level', 'win_rate', 'min_sol_len']),
            {},
        )

    varying_fields = _varying_config_fields([r['config'] for r in run_infos])
    expected_levels_by_game: dict[str, set[int]] = {}
    grouped: dict[tuple, dict] = {}

    for run_info in run_infos:
        stats = _load_training_stats(run_info['csv_path'])
        if not stats:
            continue

        for level_i in stats:
            expected_levels_by_game.setdefault(run_info['game'], set()).add(level_i)

        config_label = _format_config_label(run_info['config'], varying_fields)
        key = (config_label, run_info['game'])
        if key not in grouped:
            grouped[key] = {
                'config_label': config_label,
                'game': run_info['game'],
                'level_wins': {},      # level_i -> list[float]
                'level_sol_lens': {},  # level_i -> list[int] (only from winning runs)
                'n_runs': 0,
            }
        g = grouped[key]
        g['n_runs'] += 1
        for level_i, s in stats.items():
            g['level_wins'].setdefault(level_i, []).append(1.0 if s['ever_won'] else 0.0)
            if s['min_sol_len'] is not None:
                g['level_sol_lens'].setdefault(level_i, []).append(s['min_sol_len'])

    rows = []
    level_rows = []
    for g in grouped.values():
        expected_levels = expected_levels_by_game.get(g['game'], set())
        if not expected_levels:
            continue

        missing = sorted(expected_levels - set(g['level_wins']))
        if missing:
            print(
                f"Warning: no training data for config '{g['config_label']}', "
                f"game '{g['game']}' on levels {', '.join(map(str, missing))}. "
                "Treating as 0 win-rate."
            )

        per_level_win_rates = []
        for level in sorted(expected_levels):
            win_samples = g['level_wins'].get(level, [])
            win_rate = float(np.mean(win_samples)) if win_samples else 0.0
            per_level_win_rates.append(win_rate)

            sol_samples = g['level_sol_lens'].get(level, [])
            min_sol_len = int(min(sol_samples)) if sol_samples else None

            level_rows.append({
                'config_label': g['config_label'],
                'game': g['game'],
                'level': level,
                'win_rate': win_rate,
                'min_sol_len': min_sol_len,
            })

        rows.append({
            'config_label': g['config_label'],
            'game': g['game'],
            'avg_win_rate': float(np.mean(per_level_win_rates)) if per_level_win_rates else 0.0,
            'n_runs': g['n_runs'],
            'n_levels': len(expected_levels),
        })

    if not rows:
        return (
            pd.DataFrame(columns=['config_label', 'game', 'avg_win_rate', 'n_runs', 'n_levels']),
            pd.DataFrame(columns=['config_label', 'game', 'level', 'win_rate', 'min_sol_len']),
            {game: sorted(levels) for game, levels in expected_levels_by_game.items()},
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(level_rows),
        {game: sorted(levels) for game, levels in expected_levels_by_game.items()},
    )


# ---------------------------------------------------------------------------
# Shared heatmap utilities
# ---------------------------------------------------------------------------

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
        game_spans.append((game, start, len(columns)))
    return columns, game_spans


def _draw_game_dividers(ax, game_spans: list[tuple[str, int, int]], n_rows: int) -> None:
    for _, _, end in game_spans[:-1]:
        ax.vlines(end, ymin=0, ymax=n_rows, colors='black', linewidth=2.0)
    for game, start, end in game_spans:
        center = (start + end) / 2
        ax.text(
            center, 1.005, _format_game_label(game),
            ha='center', va='bottom', fontsize=9,
            transform=ax.get_xaxis_transform(),
        )


def _heatmap_fig_size(num_cols: int, num_rows: int,
                      cell_w: float = 0.35, cell_h: float = 0.8) -> tuple[float, float]:
    return (
        max(num_cols * cell_w + 3.0, 12.0),
        max(num_rows * cell_h + 2.5, 3.0),
    )


# ---------------------------------------------------------------------------
# Win-rate heatmaps
# ---------------------------------------------------------------------------

def plot_expanded_win_rate_heatmap(
    level_df: pd.DataFrame,
    expected_levels_by_game: dict[str, list[int]],
    output_path: Path,
) -> None:
    if level_df.empty:
        print('No data available for win-rate heatmap.')
        return

    ordered_games = [g for g in sorted(expected_levels_by_game) if expected_levels_by_game[g]]
    columns, game_spans = _build_expanded_heatmap_columns(ordered_games, expected_levels_by_game)
    if not columns:
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

    num_cols, num_rows = len(columns), len(config_labels)
    fig_w, fig_h = _heatmap_fig_size(num_cols, num_rows)
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        heatmap_df,
        annot=annot_data if annot_data is not None else False,
        fmt='',
        cmap='RdYlGn',
        vmin=0.0, vmax=1.0,
        cbar_kws={'label': 'Win Rate', 'shrink': 0.8, 'pad': 0.01},
        annot_kws={'size': 7},
        linewidths=0.25, linecolor='white',
    )
    _draw_game_dividers(ax, game_spans, num_rows)
    ax.set_title('RL Win Rate per Level', pad=32)
    plt.xlabel('Level', labelpad=10)
    plt.ylabel('RL Run Config', labelpad=10)
    plt.yticks(rotation=0)
    ax.set_xticklabels([str(level) for _, level in columns], rotation=0, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved win-rate heatmap to {output_path}")


# ---------------------------------------------------------------------------
# Solution-length heatmap
# ---------------------------------------------------------------------------

def plot_expanded_sol_len_heatmap(
    level_df: pd.DataFrame,
    expected_levels_by_game: dict[str, list[int]],
    output_path: Path,
) -> None:
    """Heatmap of best (minimum) solution length per level.

    Only levels that were ever solved appear with a value; unsolved levels are
    shown as blank.  Shorter solutions are greener (better).
    """
    sol_df = level_df.dropna(subset=['min_sol_len'])
    if sol_df.empty:
        print('No solved levels found; skipping solution-length heatmap.')
        return

    ordered_games = [g for g in sorted(expected_levels_by_game) if expected_levels_by_game[g]]
    columns, game_spans = _build_expanded_heatmap_columns(ordered_games, expected_levels_by_game)
    if not columns:
        return

    config_labels = list(dict.fromkeys(level_df['config_label']))
    column_keys = [f'{game}::level-{level}' for game, level in columns]
    heatmap_df = pd.DataFrame(index=config_labels, columns=column_keys, dtype=float)
    value_lookup = {
        (row.config_label, row.game, int(row.level)): float(row.min_sol_len)
        for row in level_df.itertuples(index=False)
        if row.min_sol_len is not None and not (isinstance(row.min_sol_len, float) and np.isnan(row.min_sol_len))
    }
    for config_label in config_labels:
        for game, level in columns:
            value = value_lookup.get((config_label, game, level))
            if value is not None:
                heatmap_df.at[config_label, f'{game}::level-{level}'] = value

    data_vals = heatmap_df.to_numpy(dtype=float)
    finite_vals = data_vals[np.isfinite(data_vals)]
    if len(finite_vals) == 0:
        print('No finite solution lengths; skipping solution-length heatmap.')
        return

    vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
    if vmin == vmax:
        vmax = vmin + 1

    annot_data = None
    if len(columns) <= 40:
        annot_data = [
            [f"{int(val)}" if np.isfinite(val) else '' for val in row]
            for row in data_vals
        ]

    num_cols, num_rows = len(columns), len(config_labels)
    fig_w, fig_h = _heatmap_fig_size(num_cols, num_rows)
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        heatmap_df,
        annot=annot_data if annot_data is not None else False,
        fmt='',
        cmap='RdYlGn_r',   # shorter (lower) = greener = better
        vmin=vmin, vmax=vmax,
        cbar_kws={'label': 'Min solution length (steps)', 'shrink': 0.8, 'pad': 0.01},
        annot_kws={'size': 7},
        linewidths=0.25, linecolor='white',
    )
    _draw_game_dividers(ax, game_spans, num_rows)
    ax.set_title('RL Best Solution Length per Level (blank = unsolved)', pad=32)
    plt.xlabel('Level', labelpad=10)
    plt.ylabel('RL Run Config', labelpad=10)
    plt.yticks(rotation=0)
    ax.set_xticklabels([str(level) for _, level in columns], rotation=0, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved solution-length heatmap to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    rl_root = Path(args.rl_logs_dir)
    summary_df, level_df, expected_levels_by_game = collect_results_data(rl_root)

    if summary_df.empty:
        print('No training results found; nothing to plot.')
        return

    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = plots_dir / 'rl_results.csv'
    prettify_game_index(summary_df).to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved RL results summary to {csv_path}")

    plot_expanded_win_rate_heatmap(
        level_df, expected_levels_by_game,
        plots_dir / 'rl_win_rate_heatmap.png',
    )
    plot_expanded_sol_len_heatmap(
        level_df, expected_levels_by_game,
        plots_dir / 'rl_sol_len_heatmap.png',
    )


def parse_args(argv: Iterable[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot RL training win rates and solution lengths from progress.csv files.'
    )
    parser.add_argument(
        '--rl-logs-dir',
        default='rl_logs_jax',
        help='Path to the root directory containing RL experiment logs.',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
