"""Plot the twelve-game paired evaluation of batched movement updates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from scripts.plotting.plot_movement_refinements import load, save

GAMES = ['sokoban_basic', 'blocks', 'Zen_Puzzle_Garden', 'notsnake', 'Slidings',
         'limerick', 'kettle', 'Take_Heart_Lass', 'atlas shrank', 'nekopuzzle',
         'sokoban_match3', 'Travelling_salesman']
TITLES = ['Sokoban', 'Blocks', 'Zen', 'Notsnake', 'Slidings', 'Lime Rick', 'Kettle',
          'Take Heart Lass', 'Atlas Shrank', 'Nekopuzzle', 'Sokoban Match 3',
          'Travelling Salesman']
BATCHES = [1, 256, 4096]
SEEDS = [42, 1042]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--large-results', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = {
        'inputs': {}, 'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matplotlib_version': matplotlib.__version__,
        'scope': 'Level zero; full-output 100-step rollouts; autoreset at 100; paired seeds 42/1042.',
        'aggregation': 'Geometric mean of seed median throughput ratios; whiskers span the two seeds, not a confidence interval.',
    }
    rows, device, trials = load(args.results, manifest)
    if device != 'H200':
        raise ValueError(f'Expected H200 measurements, got {device}')
    expected = {(g, b, s) for g in GAMES for b in BATCHES for s in SEEDS}
    actual = {(r['game'], r['batch'], r['seed']) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError('Expected complete twelve-game, three-batch, two-seed sweep')
    if any(r['candidate_name'] != 'production' or r['level'] != 0 or r['steps'] != 100 for r in rows):
        raise ValueError('Unexpected candidate, level or rollout length')
    sources = [json.loads(p.read_text()) for p in args.results]
    if len({d['benchmark_sha256'] for d in sources}) != 1:
        raise ValueError('Benchmark sources differ')
    if any(d['output_mode'] != 'full' or d['max_episode_steps'] != 100 for d in sources):
        raise ValueError('Unexpected output or reset configuration')
    lookup = {(r['game'], r['batch'], r['seed']): r for r in rows}
    large_rows = []
    if args.large_results:
        data = json.loads(args.large_results.read_text())
        for field in ('engine_sha256', 'benchmark_sha256', 'devices', 'jax_version',
                      'trials', 'seeds', 'output_mode', 'max_episode_steps'):
            if data[field] != sources[0][field]:
                raise ValueError(f'Large-batch configuration differs: {field}')
        large_rows = data['results']
        expected_large = {(g, 16384, s) for g in ('blocks', 'Slidings', 'nekopuzzle') for s in SEEDS}
        if len(large_rows) != 6 or {(r['game'], r['batch'], r['seed']) for r in large_rows} != expected_large:
            raise ValueError('Expected complete three-game large-batch check')
        if any(r['candidate_name'] != 'production' or r['level'] != 0 or r['steps'] != 100 for r in large_rows):
            raise ValueError('Unexpected large-batch workload')
        manifest['inputs'][str(args.large_results)] = {
            'sha256': hashlib.sha256(args.large_results.read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.1, 5.1))
    positions = np.arange(len(GAMES))
    for batch, offset, color in zip(BATCHES, [-0.23, 0, 0.23], ['#969696', '#377eb8', '#e68b33']):
        ratios = np.array([[lookup[g, batch, s]['speedup'] for s in SEEDS] for g in GAMES])
        midpoint = np.exp(np.mean(np.log(ratios), axis=1))
        ax.errorbar(100 * (midpoint - 1), positions + offset,
                    xerr=100 * np.maximum(0, np.stack([midpoint - ratios.min(1), ratios.max(1) - midpoint])),
                    fmt='o', markersize=4, capsize=2, color=color, label=f'Batch {batch:,}')
    ax.axvline(0, color='0.3', linestyle='--', linewidth=0.8)
    ax.set_yticks(positions, TITLES)
    ax.invert_yaxis()
    ax.set_xlabel('Throughput change from preceding engine (%)')
    ax.set_title(f'{device} · batched movement · {trials} alternating trials per seed')
    ax.grid(axis='x', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='best')
    fig.text(0.5, 0.01, 'Two seeds · points: geometric mean ratio · whiskers: seed range (not confidence intervals)',
             ha='center', fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, args.output_dir / 'movement_updates_speedup_h200')

    fig, axes = plt.subplots(3, 4, figsize=(8.5, 6.4))
    for ax, game, title in zip(axes.flat, GAMES, TITLES):
        for key, label, color, marker in [('baseline', 'Preceding engine', '#777777', 'o'),
                                           ('candidate', 'Batched movement', '#d62728', 'x')]:
            values = np.array([[lookup[game, b, s][key]['env_steps_per_s'] for s in SEEDS] for b in BATCHES])
            ax.plot(BATCHES, np.exp(np.mean(np.log(values), axis=1)), color=color,
                    marker=marker, markersize=4, linewidth=1.2, label=label)
            ax.fill_between(BATCHES, values.min(1), values.max(1), color=color, alpha=0.2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False)
    fig.text(0.5, 0.008, 'Lines: geometric mean of two seed medians · bands: seed range, not confidence intervals',
             ha='center', fontsize=7)
    fig.supxlabel('Environments per batch', y=0.10)
    fig.supylabel('Environment steps/s')
    fig.suptitle(f'{device} · 100-step full-output rollouts · level 0')
    fig.tight_layout(rect=(0.02, 0.12, 1, 0.97))
    save(fig, args.output_dir / 'movement_updates_throughput_h200')

    if large_rows:
        games = ['blocks', 'Slidings', 'nekopuzzle']
        large_lookup = {(r['game'], r['seed']): r for r in large_rows}
        fig, ax = plt.subplots(figsize=(5.5, 3.1))
        extrema = [0]
        for batch, offset, color in [(4096, -0.18, '#377eb8'), (16384, 0.18, '#e68b33')]:
            ratios = np.array([[lookup[g, batch, s]['speedup'] if batch == 4096
                                else large_lookup[g, s]['speedup'] for s in SEEDS] for g in games])
            mid = np.exp(np.mean(np.log(ratios), axis=1))
            x = np.arange(len(games)) + offset
            bars = ax.bar(x, 100 * (mid - 1), width=0.34, color=color, label=f'Batch {batch:,}')
            ax.errorbar(x, 100 * (mid - 1), fmt='none', ecolor='0.2', capsize=2,
                        yerr=100 * np.maximum(0, np.stack([mid - ratios.min(1), ratios.max(1) - mid])))
            ax.bar_label(bars, fmt='%.1f%%', fontsize=8, padding=3)
            extrema.extend((100 * (ratios - 1)).ravel())
        ax.axhline(0, color='0.3', linewidth=0.8)
        ax.set_xticks(np.arange(3), ['Blocks', 'Slidings', 'Nekopuzzle'])
        ax.set_ylim(min(extrema) - 5, max(extrema) + 10)
        ax.set_ylabel('Throughput change (%)')
        ax.set_title(f'{device} · full-output rollouts · {trials} trials per seed')
        ax.legend(frameon=False, loc='upper right')
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
        fig.text(0.5, 0.01, 'Geometric mean of two seed ratios · whiskers show seed range', ha='center', fontsize=7)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save(fig, args.output_dir / 'movement_updates_large_batch_h200')

    with (args.output_dir / 'movement_updates_h200.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['game', 'batch', 'seed', 'baseline_steps_per_s', 'current_steps_per_s',
                         'speedup', 'baseline_temporary_bytes', 'current_temporary_bytes'])
        exported = [lookup[g, b, s] for g in GAMES for b in BATCHES for s in SEEDS] + large_rows
        for r in exported:
            writer.writerow([r['game'], r['batch'], r['seed'], r['baseline']['env_steps_per_s'],
                             r['candidate']['env_steps_per_s'], r['speedup'],
                             r['baseline']['temporary_bytes'], r['candidate']['temporary_bytes']])
    (args.output_dir / 'movement_updates_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
