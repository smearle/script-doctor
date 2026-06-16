"""Visualize validation results as a stacked horizontal bar chart.

Usage:
    python plot_validation_results.py                              # JAX, increpare
    python plot_validation_results.py --dataset pedro              # JAX, pedro
    python plot_validation_results.py --source cpp                 # C++, increpare
    python plot_validation_results.py --source cpp --dataset pedro # C++, pedro
"""
import argparse
import json
import os
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from puzzlescript_jax.globals import (
    GAMES_DIR, CUSTOM_GAMES_DIR, INCREPARE_GAMES_DIR, GAMES_METADATA_PATH,
)
from puzzlescript_jax.utils import get_list_of_games_for_testing

GAMES_DIRS = [GAMES_DIR, CUSTOM_GAMES_DIR, INCREPARE_GAMES_DIR]

OUT_DIR = os.path.join('plots', 'validation')


def count_levels_regex(filepath):
    """Estimate the number of levels in a PuzzleScript game file via regex."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    match = re.search(r'(?:^|\n)\s*=+\s*\n\s*LEVELS\s*\n\s*=+\s*\n', text, re.IGNORECASE)
    if not match:
        match = re.search(r'(?:^|\n)\s*LEVELS\s*\n', text, re.IGNORECASE)
    if not match:
        return 0
    levels_text = text[match.end():]
    n_levels = 0
    in_level = False
    for line in levels_text.split('\n'):
        stripped = line.strip()
        if stripped.lower().startswith('message'):
            in_level = False
            continue
        if stripped == '':
            if in_level:
                n_levels += 1
                in_level = False
            continue
        if re.match(r'^[a-zA-Z0-9.#@\$\*\+\-_=|><^v~!%&(){}\\[\]:;,?/ ]+$', stripped):
            in_level = True
        else:
            if in_level:
                n_levels += 1
            in_level = False
    if in_level:
        n_levels += 1
    return n_levels


def estimate_levels_for_games(game_names, metadata):
    """Get level count from metadata if available, else regex-count from file."""
    total = 0
    for g in game_names:
        key = g + '.txt'
        if key in metadata and metadata[key].get('n_levels', 0) > 0:
            total += metadata[key]['n_levels']
        else:
            for d in GAMES_DIRS:
                fp = os.path.join(d, g + '.txt')
                if os.path.exists(fp):
                    total += count_levels_regex(fp)
                    break
    return total


def scoped_per_level_count(error_dict, dataset_games):
    """Count total levels in a per-level error dict, scoped to dataset."""
    return sum(len(entries) for game, entries in error_dict.items() if game in dataset_games)


def scoped_per_level_games(error_dict, dataset_games):
    """Count unique games in a per-level error dict, scoped to dataset."""
    return {g for g in error_dict if g in dataset_games}


# --- Category definitions per source ---
# Each entry: (display_name, json_key, type)
#   type: 'per_level_dict' = dict of game -> [level entries]
#         'game_list'      = list of {game: ..., ...} (game-wide errors, estimate levels)
#         'timeout_list'   = list of {game: ..., level: ...}

JAX_CATEGORIES = [
    ('Successful',       'success',                'per_level_dict'),
    ('Solution error',   'solution_error',         'per_level_dict'),
    ('State error',      'state_error',            'per_level_dict'),
    ('Runtime error',    'runtime_error',           'per_level_dict'),
    ('Score error',      'score_error',             'per_level_dict'),
    ('Timeout',          'timeout',                 'timeout_list'),
    ('Random solution',  'random_solution_error',   'per_level_dict'),
    ('Random state',     'random_state_error',      'per_level_dict'),
    ('Compile error',    'compile_error',           'game_list'),
    ('Rigid body error', 'rigid_prefix_error',      'game_list'),
]

CPP_CATEGORIES = [
    ('Successful',       'success',       'per_level_dict'),
    ('State error',      'state_error',   'per_level_dict'),
    ('Win error',        'win_error',     'per_level_dict'),
    ('Runtime error',    'runtime_error', 'per_level_dict'),
    ('Timeout',          'timeout',       'timeout_list'),
    ('Compile error',    'compile_error', 'game_list'),
]

COLORS = {
    'Successful':       '#4CAF50',
    'Solution error':   '#F44336',
    'State error':      '#FF9800',
    'Runtime error':    '#9C27B0',
    'Random solution':  '#2196F3',
    'Random state':     '#03A9F4',
    'Score error':      '#FF5722',
    'Timeout':          '#795548',
    'Compile error':    '#607D8B',
    'Rigid body error': '#9E9E9E',
    'Win error':        '#E91E63',
}


def build_categories(vr, cat_defs, dataset_games, metadata):
    """Build ordered category counts and game counts from validation results."""
    categories = {}
    games_per_category = {}
    game_wide_game_counts = {}

    for display, key, typ in cat_defs:
        data = vr.get(key, [] if typ in ('game_list', 'timeout_list') else {})
        if typ == 'per_level_dict':
            categories[display] = scoped_per_level_count(data, dataset_games)
            games_per_category[display] = len(scoped_per_level_games(data, dataset_games))
        elif typ == 'timeout_list':
            scoped = [e for e in data if e.get('game', '') in dataset_games]
            categories[display] = len(scoped)
            games_per_category[display] = len({e.get('game', '') for e in scoped})
        elif typ == 'game_list':
            games = [e['game'] for e in data if e['game'] in dataset_games]
            categories[display] = estimate_levels_for_games(games, metadata)
            games_per_category[display] = len(games)
            game_wide_game_counts[display] = len(games)

    order = [display for display, _, _ in cat_defs]
    # Drop categories with 0 levels
    order = [k for k in order if categories[k] > 0]

    # Compute unique validated/error game counts
    all_games = set()
    error_games = set()
    for display, key, typ in cat_defs:
        data = vr.get(key, [] if typ in ('game_list', 'timeout_list') else {})
        if typ == 'per_level_dict':
            scoped = {g for g in data if g in dataset_games}
        elif typ == 'timeout_list':
            scoped = {e.get('game', '') for e in data if e.get('game', '') in dataset_games}
        elif typ == 'game_list':
            scoped = {e['game'] for e in data if e['game'] in dataset_games}
        all_games |= scoped
        if display != 'Successful':
            error_games |= scoped
    n_validated_games = len(all_games)
    n_error_games = len(error_games)

    return categories, games_per_category, order, n_validated_games, n_error_games


def plot_validation(categories, games_per_category, order, n_validated_games,
                    n_error_games, out_path):
    """Draw the two-bar stacked chart with magnification lines."""
    total_levels = sum(categories[k] for k in order)

    values = [categories[k] for k in order]
    color_list = [COLORS[k] for k in order]

    BAR_HEIGHT = 0.3
    LABEL_FONTSIZE = 7.5
    LEGEND_FONTSIZE = 8

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.1, 1.6), height_ratios=[1, 1],
                                    gridspec_kw={'hspace': 0.55})
    fig.subplots_adjust(right=0.99)

    def draw_bar(ax, names, values_subset, color_subset, total):
        left = 0
        for name, val, color in zip(names, values_subset, color_subset):
            ax.barh(0, val, left=left, color=color, edgecolor='white', linewidth=0.5, height=BAR_HEIGHT)
            left += val
        ax.set_xlim(0, total)
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    def add_bar_labels(ax, names, values_subset, total):
        renderer = fig.canvas.get_renderer()
        left = 0
        for name, val in zip(names, values_subset):
            mid = left + val / 2
            txt = ax.text(mid, 0, name, ha='center', va='center',
                          fontsize=LABEL_FONTSIZE, fontweight='bold', color='white')
            bb = txt.get_window_extent(renderer=renderer)
            x0_disp = ax.transData.transform((left, 0))[0]
            x1_disp = ax.transData.transform((left + val, 0))[0]
            band_width = x1_disp - x0_disp
            if bb.width > band_width * 0.85:
                txt.remove()
            left += val

    error_order = [k for k in order if k != 'Successful']

    draw_bar(ax1, order, values, color_list, total_levels)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.set_xlabel(f'All validated levels ({n_validated_games:,} games, {total_levels:,} levels)',
                   fontsize=11, fontweight='bold', loc='left')
    ax1.set_title('')

    error_values = [categories[k] for k in error_order]
    error_colors = [COLORS[k] for k in error_order]
    error_total = sum(error_values)

    draw_bar(ax2, error_order, error_values, error_colors, error_total)
    ax2.set_title('')
    ax2.set_xlabel(
        f'Errors only ({n_error_games:,} games, {error_total:,} levels, {100*error_total/total_levels:.1f}%)',
        fontsize=11, fontweight='bold', loc='left',
    )

    # Legend
    legend_labels = []
    for k in order:
        v = categories[k]
        pct = 100 * v / total_levels if total_levels > 0 else 0
        ng = games_per_category.get(k, 0)
        legend_labels.append(f'{k}: {v:,} ({pct:.1f}%) [{ng:,} games]')

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[k], edgecolor='white') for k in order]
    fig.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
               ncol=1, fontsize=LEGEND_FONTSIZE, frameon=False)

    # Magnification lines
    error_left = categories.get('Successful', 0)
    error_right = total_levels
    bar_half = BAR_HEIGHT / 2
    zoom_color = '#555555'
    zoom_lw = 0.8
    zoom_ls = (0, (4, 3))

    con_left = mpatches.ConnectionPatch(
        xyA=(error_left, -bar_half), coordsA=ax1.transData,
        xyB=(0, bar_half), coordsB=ax2.transData,
        color=zoom_color, linewidth=zoom_lw, linestyle=zoom_ls,
    )
    con_right = mpatches.ConnectionPatch(
        xyA=(error_right, -bar_half), coordsA=ax1.transData,
        xyB=(error_total, bar_half), coordsB=ax2.transData,
        color=zoom_color, linewidth=zoom_lw, linestyle=zoom_ls,
    )
    fig.add_artist(con_left)
    fig.add_artist(con_right)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    add_bar_labels(ax1, order, values, total_levels)
    add_bar_labels(ax2, error_order, error_values, error_total)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

    print(f'\nBreakdown ({total_levels:,} levels total):')
    for k in order:
        v = categories[k]
        pct = 100 * v / total_levels if total_levels > 0 else 0
        print(f'  {k:20s}: {v:6,} ({pct:5.1f}%)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='increpare',
                        choices=['priority', 'gallery', 'pedro', 'increpare'],
                        help='Dataset tier (default: increpare)')
    parser.add_argument('--source', default='all', choices=['jax', 'cpp', 'all'],
                        help='Which validation results to plot (default: all)')
    args = parser.parse_args()

    with open(GAMES_METADATA_PATH) as f:
        metadata = json.load(f)

    dataset_games = set(get_list_of_games_for_testing(dataset=args.dataset))

    sources = ['jax', 'cpp'] if args.source == 'all' else [args.source]

    for source in sources:
        if source == 'jax':
            val_path = os.path.join('data', 'validation_results.json')
            cat_defs = JAX_CATEGORIES
            label = 'jax'
        else:
            val_path = os.path.join('data', 'cpp_validation_results.json')
            cat_defs = CPP_CATEGORIES
            label = 'cpp'

        if not os.path.exists(val_path):
            print(f'Skipping {source}: {val_path} not found')
            continue

        with open(val_path) as f:
            vr = json.load(f)

        categories, games_per_cat, order, n_validated, n_error = build_categories(
            vr, cat_defs, dataset_games, metadata)

        out_path = os.path.join(OUT_DIR, f'validation_results_{label}_{args.dataset}.png')
        print(f'\n=== {source.upper()} ({args.dataset}) ===')
        plot_validation(categories, games_per_cat, order, n_validated, n_error, out_path)


if __name__ == '__main__':
    main()
