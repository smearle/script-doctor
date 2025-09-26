import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from globals import PLOTS_DIR
from puzzlejax.utils import game_names_remap


def collect_eval_stats(root: Path) -> pd.DataFrame:
    """Aggregate evaluation win rates for each game under ``root``."""
    if not root.exists():
        raise FileNotFoundError(f"RL logs directory '{root}' does not exist.")

    rows = []
    for game_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        eval_paths = list(game_dir.glob('**/eval_stats.json'))
        if not eval_paths:
            continue

        total_wins = 0.0
        total_eps = 0
        mean_win_samples = []

        for eval_path in eval_paths:
            try:
                with eval_path.open('r') as fh:
                    stats = json.load(fh)
            except (OSError, json.JSONDecodeError):
                print(f"Skipping unreadable stats file: {eval_path}")
                continue

            n_eps = stats.get('n_eps') or stats.get('n_episodes')
            mean_wins = stats.get('mean_wins')
            mean_wins = min(mean_wins, 1.0)
            n_wins = stats.get('n_wins')
            print(stats)

            eps_valid = isinstance(n_eps, (int, float)) and n_eps > 0
            wins_valid = isinstance(n_wins, (int, float))
            mean_valid = isinstance(mean_wins, (int, float))

            if not wins_valid and mean_valid and eps_valid:
                n_wins = mean_wins * n_eps
                wins_valid = True

            if wins_valid and eps_valid:
                total_wins += float(n_wins)
                total_eps += int(n_eps)

            if mean_valid:
                mean_win_samples.append(float(mean_wins))

        if total_eps == 0 and not mean_win_samples:
            continue

        avg_win_rate = (
            total_wins / total_eps
            if total_eps > 0
            else float(np.mean(mean_win_samples))
        )

        rows.append(
            {
                'game': game_dir.name,
                'avg_win_rate': avg_win_rate,
                'n_eval_files': len(eval_paths),
                'total_eval_episodes': total_eps,
            }
        )

    if not rows:
        return pd.DataFrame(columns=['avg_win_rate', 'n_eval_files', 'total_eval_episodes'])

    # df = pd.DataFrame(rows).set_index('game').sort_index()
    df = pd.DataFrame(rows).set_index('game')
    return df


def prettify_game_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    renamed = df.copy()
    renamed.index = renamed.index.to_series().replace(game_names_remap)
    renamed.index = renamed.index.str.replace('_', ' ', regex=False)
    renamed.index = renamed.index.str.title()
    return renamed


def plot_win_rate_heatmap(series: pd.Series, output_path: Path) -> None:
    if series.empty:
        print('No data available for RL win-rate heatmap.')
        return

    display_names = [name.replace('_', ' ') for name in series.index]
    data = pd.DataFrame([series.values], columns=display_names, index=[''])
    annot_row = [f"{val:.0%}" if np.isfinite(val) else '' for val in series.values]

    num_cols = len(series)
    target_cell_width = 1.0
    target_cell_height = 1.0
    min_total_width = 8.0
    min_total_height = 3.0
    h_padding = 3.0
    v_padding = 1.5

    fig_w = max(num_cols * target_cell_width + h_padding, min_total_width)
    fig_h = max(target_cell_height + v_padding, min_total_height)

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        data,
        annot=[annot_row],
        fmt='',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Average Win Rate', 'shrink': 0.8, 'pad': 0.01},
        annot_kws={'size': 9},
        linewidths=0.5,
        linecolor='white',
    )
    plt.title('RL Average Win Rate per Game')
    plt.xlabel('Game', labelpad=10)
    plt.ylabel('')
    plt.yticks([])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved RL win-rate heatmap to {output_path}")


def main(args: argparse.Namespace) -> None:
    rl_root = Path(args.rl_logs_dir)
    df = collect_eval_stats(rl_root)

    if df.empty:
        print('No evaluation stats found; nothing to plot.')
        return

    # df.sort_values(by='avg_win_rate', ascending=False, inplace=True)

    display_df = prettify_game_index(df)

    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = plots_dir / 'rl_eval_results.csv'
    display_df.to_csv(csv_path, float_format='%.4f')
    print(f"Saved RL evaluation summary to {csv_path}")

    heatmap_series = display_df['avg_win_rate']
    # Cap average win rate at 1
    # heatmap_series = heatmap_series.clip(upper=1.0)
    plot_win_rate_heatmap(heatmap_series, plots_dir / 'rl_avg_win_rate_heatmap.png')


def parse_args(argv: Iterable[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot RL evaluation win rates per game.')
    parser.add_argument(
        '--rl-logs-dir',
        default='rl_logs',
        help='Path to the root directory containing RL experiment logs.',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
