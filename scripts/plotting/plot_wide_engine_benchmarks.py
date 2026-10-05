"""Show the broader H200 validation of the retained movement optimizations."""
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

GAMES = ['notsnake', 'Slidings', 'limerick', 'kettle', 'Take_Heart_Lass',
         'atlas shrank', 'nekopuzzle', 'sokoban_match3', 'Travelling_salesman']
TITLES = ['Notsnake', 'Slidings', 'Lime Rick', 'Kettle', 'Take Heart Lass',
          'Atlas Shrank', 'Nekopuzzle', 'Sokoban Match 3', 'Travelling Salesman']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', nargs='+', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    manifest = {'inputs': {}, 'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'matplotlib_version': matplotlib.__version__,
                'scope': 'Level zero, full-output paired A/B, 100 steps, fixed seed 42; not final-carry paper FPS.'}
    rows, device, trials = load(args.results, manifest)
    if any(json.loads(path.read_text()).get('seed') != 42 for path in args.results):
        raise ValueError('The wider sweep expects seed 42')
    expected = {(game, batch) for game in GAMES for batch in (256, 4096)}
    if len(rows) != len(expected) or {(r['game'], r['batch']) for r in rows} != expected:
        raise ValueError('The nine-game, two-batch sweep must be complete')
    if any(r['candidate'] != 'all' or r['steps'] != 100 for r in rows):
        raise ValueError('Unexpected candidate or rollout length')
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    y = np.arange(len(GAMES))
    for batch, offset, color in ((256, -0.18, '#377eb8'), (4096, 0.18, '#e68b33')):
        points = [next(r for r in rows if r['game'] == game and r['batch'] == batch) for game in GAMES]
        values = np.array([r['speedup'] for r in points])
        low = np.array([r['baseline']['min_s'] / r['all']['max_s'] for r in points])
        high = np.array([r['baseline']['max_s'] / r['all']['min_s'] for r in points])
        ax.barh(y + offset, values, height=0.34, label=f'Batch {batch:,}', color=color)
        ax.errorbar(values, y + offset, xerr=np.stack([values - low, high - values]),
                    fmt='none', ecolor='0.2', linewidth=0.7, capsize=2)
        for value, upper, pos in zip(values, high, y + offset):
            ax.text(upper + 0.015, pos, f'{value:.3f}×', va='center', fontsize=8)
    ax.axvline(1, color='0.25', linestyle='--', linewidth=1)
    ax.set_yticks(y, TITLES)
    ax.invert_yaxis()
    ax.set_xlabel('Throughput / preceding engine throughput (×)')
    ax.set_title(f'{device} · 100-step full rollouts · {trials} alternating trials')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc='lower center',
               bbox_to_anchor=(0.5, 0.055))
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(r['baseline']['max_s'] / r['all']['min_s'] for r in rows) + 0.18)
    fig.text(0.5, 0.01, 'Level 0 · seed 42 · whiskers span min/max timing ratios.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(fig, args.output_dir / 'wide_movement_speedup_h200')
    with (args.output_dir / 'wide_movement_throughput_h200.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['game', 'batch', 'baseline_steps_per_s', 'current_steps_per_s',
                         'speedup', 'baseline_temporary_bytes', 'current_temporary_bytes'])
        for game in GAMES:
            for batch in (256, 4096):
                row = next(r for r in rows if r['game'] == game and r['batch'] == batch)
                writer.writerow([game, batch, row['baseline']['env_steps_per_s'],
                                 row['all']['env_steps_per_s'], row['speedup'],
                                 row['baseline']['temporary_bytes'], row['all']['temporary_bytes']])
    (args.output_dir / 'wide_benchmark_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
