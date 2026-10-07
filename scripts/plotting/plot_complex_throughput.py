"""Compare expanded C++/H200 sweeps, retaining missing and censored measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.stats import spearmanr

from scripts.benchmarks.benchmark_paper_throughput import BENCHMARK_GAMES
from scripts.benchmarks.complex_games import COMPLEX_GAMES

BATCHES = [16, 256, 1024, 4096]
SATURATED = {'plateau', 'regression'}


def read_sweeps(directories, games, inputs):
    found = {}
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f'Sweep directory does not exist: {directory}')
        for game in games:
            path = directory / f'{game}.json'
            if not path.exists():
                continue
            if game in found:
                raise ValueError(f'Duplicate sweep for {game}')
            data = json.loads(path.read_text())
            if data['game'] != game:
                raise ValueError(f'Mislabeled sweep: {path}')
            batches = [r['batch'] for r in data['results']]
            if len(set(batches)) != len(batches):
                raise ValueError(f'Duplicate batch: {path}')
            for row in data['results']:
                if row['median_fps'] <= 0 or not np.isfinite(row['median_fps']):
                    raise ValueError(f'Invalid throughput: {path}')
            found[game] = data
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def peak(data):
    return max(data.get('results', []), key=lambda p: p['median_fps'], default=None)


def check_configuration(sweeps, fields, label):
    for key in fields:
        if len({json.dumps(v[key], sort_keys=True) for v in sweeps.values()}) > 1:
            raise ValueError(f'Mixed {label} configuration: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jax-results', type=Path, nargs='+', required=True)
    parser.add_argument('--cpp-results', type=Path, nargs='+', required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--prepared-results', type=Path, nargs='+', default=[])
    parser.add_argument('--prepared-validation', type=Path)
    parser.add_argument('--job-status', type=Path,
                        help='Optional mapping of cpp/jax/prepared -> game -> terminal job status.')
    parser.add_argument('--rule-work', type=Path, required=True)
    parser.add_argument('--reachable-result', type=Path,
                        help='Optional fixed-level experimental pilot, plotted separately from main comparisons.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    games = BENCHMARK_GAMES + COMPLEX_GAMES
    inputs = {}
    jax = read_sweeps(args.jax_results, games, inputs)
    ordinary_jax = dict(jax)
    prepared = read_sweeps(args.prepared_results, games, inputs)
    cpp = read_sweeps(args.cpp_results, games, inputs)
    jax_fields = ('devices', 'jax_version', 'trials', 'base_steps', 'min_steps', 'seed', 'output_mode',
                  'max_episode_steps', 'action_generation', 'level')
    check_configuration(jax, ('engine_sha256', *jax_fields), 'ordinary JAX')
    check_configuration(prepared, ('engine_sha256', *jax_fields), 'prepared JAX')
    check_configuration({**{f'base:{k}': v for k, v in jax.items()}, **prepared}, jax_fields, 'JAX workload')
    check_configuration(cpp, ('library_sha256', 'max_threads', 'cpu_affinity', 'trials', 'base_steps',
                              'min_steps', 'seed'), 'C++')
    if prepared and args.prepared_validation is None:
        parser.error('--prepared-validation is required with prepared sweeps')
    if any(v.get('prepare_params', False) for v in jax.values()):
        raise ValueError('Prepared results supplied as ordinary JAX')
    if any(not v.get('prepare_params') or (v['results'] and not v.get('reset_cache_used'))
           for v in prepared.values()):
        raise ValueError('Prepared sweep did not use the reset cache')
    jax.update(prepared)
    jobs = {}
    if args.job_status:
        jobs = json.loads(args.job_status.read_text())
        inputs[str(args.job_status)] = hashlib.sha256(args.job_status.read_bytes()).hexdigest()

    def status(variant, game, data):
        return data.get('stop_reason') or jobs.get(variant, {}).get(game, 'incomplete')

    profiles = json.loads(args.rule_work.read_text())
    inputs[str(args.rule_work)] = hashlib.sha256(args.rule_work.read_bytes()).hexdigest()
    table, summaries = [], []
    matrix = np.full((len(games), len(BATCHES) + 1), np.nan)
    for i, game in enumerate(games):
        meta_path = args.metadata / f'{game}.json'
        metadata = json.loads(meta_path.read_text())
        inputs[str(meta_path)] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
        cpu, gpu = cpp.get(game, {}), jax.get(game, {})
        validation_status = 'prior_suite'
        if game in COMPLEX_GAMES or game in prepared:
            validation_dir = args.prepared_validation if game in prepared else args.validation
            validation_path = validation_dir / f'{game}.json'
            validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
            if validation:
                inputs[str(validation_path)] = hashlib.sha256(validation_path.read_bytes()).hexdigest()
            validation_status = 'passed' if validation.get('passed') else ('failed' if validation else 'incomplete')
            if gpu and validation_status != 'passed':
                raise ValueError(f'Unvalidated JAX throughput: {game}')
            if gpu and validation.get('source_hashes', {}).get('puzzlescript_jax/env.py') != gpu['engine_sha256']:
                raise ValueError(f'Validation and timing use different JAX engines: {game}')
        # Ordinary curves remain visible when a prepared curve is selected for
        # the comparison matrix; validate those plotted inputs independently.
        if game in COMPLEX_GAMES and game in prepared and game in ordinary_jax:
            path = args.validation / f'{game}.json'
            ordinary_validation = json.loads(path.read_text())
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if not ordinary_validation.get('passed') or ordinary_validation['source_hashes']['puzzlescript_jax/env.py'] != ordinary_jax[game]['engine_sha256']:
                raise ValueError(f'Unvalidated ordinary JAX overlay: {game}')
        if cpu and cpu['compiled_sha256'] != metadata['compiled_sha256']:
            raise ValueError(f'C++ metadata uses different compiled rules: {game}')
        cp, gp = peak(cpu), peak(gpu)
        variant = 'prepared' if game in prepared else 'jax'
        row = {**metadata, 'validation': validation_status,
               'jax_variant': 'prepared' if game in prepared else 'ordinary',
               'cpp_status': status('cpp', game, cpu), 'jax_status': status(variant, game, gpu),
               'cpp_best_fps': cp['median_fps'] if cp else None,
               'jax_best_fps': gp['median_fps'] if gp else None,
               'cpp_best_batch': cp['batch'] if cp else None,
               'jax_best_batch': gp['batch'] if gp else None}
        row['both_saturated'] = row['cpp_status'] in SATURATED and row['jax_status'] in SATURATED
        row['dense_rule_state_proxy'] = metadata['jax_rule_functions'] * metadata['board_cells'] * (metadata['objects'] + metadata['force_channels'])
        work = [p for p in profiles['results'] if p['game'] == game]
        if work:
            if any(p['compiled_sha256'] != metadata['compiled_sha256'] for p in work):
                raise ValueError(f'Rule-work profile uses different compiled rules: {game}')
            fractions = [p['board_pruned_fraction'] for p in work if p['board_pruned_fraction'] is not None]
            row['cpp_board_pruned_fraction'] = float(np.mean(fractions)) if fractions else None
            row['cpp_cell_checks_per_step'] = float(np.mean([p['cell_start_checks'] / profiles['steps'] for p in work]))
            row['cpp_group_passes_per_step'] = float(np.mean([p['group_passes'] / profiles['steps'] for p in work]))
        curves = [('cpp_cpu', cpu, 'cpp'), ('jax_h200_ordinary', ordinary_jax.get(game, {}), 'jax')]
        if game in prepared:
            curves.append(('jax_h200_prepared', prepared[game], 'prepared'))
        for label, data, curve_variant in curves:
            for p in data.get('results', []):
                table.append({'game': game, 'engine': label, 'batch': p['batch'], 'median_fps': p['median_fps'],
                              'q25_fps': p['q25_fps'], 'q75_fps': p['q75_fps'],
                              'compile_s': p.get('compile_s'), 'temporary_bytes': p.get('temporary_bytes'),
                              'stop_reason': status(curve_variant, game, data)})
        for j, batch in enumerate(BATCHES):
            c = next((p for p in cpu.get('results', []) if p['batch'] == batch), None)
            g = next((p for p in gpu.get('results', []) if p['batch'] == batch), None)
            row[f'cpp_over_jax_b{batch}'] = c['median_fps'] / g['median_fps'] if c and g else None
            row[f'jax_temporary_bytes_b{batch}'] = g.get('temporary_bytes') if g else None
            row[f'jax_compile_s_b{batch}'] = g.get('compile_s') if g else None
            if c and g:
                matrix[i, j] = np.log2(row[f'cpp_over_jax_b{batch}'])
        row['cpp_over_jax_best_measured'] = cp['median_fps'] / gp['median_fps'] if cp and gp else None
        if cp and gp and row['both_saturated']:
            matrix[i, -1] = np.log2(row['cpp_over_jax_best_measured'])
        summaries.append(row)
    status_by_game = {r['game']: r for r in summaries}
    features = ['run_rules_on_level_start', 'source_rules', 'compiled_rules', 'jax_rule_functions',
                'jax_rule_groups', 'objects', 'layers', 'board_cells', 'multirow_rules', 'max_pattern_rows',
                'ellipsis_rules', 'dense_rule_state_proxy', 'cpp_board_pruned_fraction',
                'cpp_cell_checks_per_step', 'cpp_group_passes_per_step']
    correlations = {}
    for target in [*[f'cpp_over_jax_b{b}' for b in BATCHES], 'cpp_over_jax_best_measured']:
        sample = [r for r in summaries if r[target] is not None
                  and (target != 'cpp_over_jax_best_measured' or r['both_saturated'])]
        correlations[target] = {}
        for feature in features:
            subset = [r for r in sample if r.get(feature) is not None]
            if len(subset) >= 3:
                rho = float(spearmanr([r[feature] for r in subset], [r[target] for r in subset]).statistic)
                correlations[target][feature] = {'n': len(subset), 'spearman_rho': rho if np.isfinite(rho) else None}
        if target.startswith('cpp_over_jax_b') and target != 'cpp_over_jax_best_measured':
            batch = target.removeprefix('cpp_over_jax_b')
            for feature in (f'jax_temporary_bytes_b{batch}', f'jax_compile_s_b{batch}'):
                subset = [r for r in sample if r.get(feature) is not None]
                if len(subset) >= 3:
                    rho = float(spearmanr([r[feature] for r in subset], [r[target] for r in subset]).statistic)
                    correlations[target][feature] = {'n': len(subset), 'spearman_rho': rho if np.isfinite(rho) else None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(10, 12))
    cmap = plt.get_cmap('RdBu').copy(); cmap.set_bad('#e5e5e5')
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10))
    ax.set_yticks(range(len(games)), [g.replace('_', ' ').replace('Heroes of Sokoban III  The Bard and The Druid', 'Heroes of Sokoban III') + (' *' if g in prepared else '') for g in games], fontsize=8)
    ax.set_xticks(range(len(BATCHES) + 1), [f'Batch {b:,}' for b in BATCHES] + ['Best saturated\n(each engine)'], fontsize=9)
    ax.xaxis.tick_top()
    for i in range(len(games)):
        for j in range(len(BATCHES) + 1):
            value = matrix[i, j]
            if np.isfinite(value):
                ratio = 2 ** value
                text = f'C++ {ratio:.1f}×' if ratio >= 1 else f'JAX {1 / ratio:.1f}×'
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color='white' if abs(value) > 5.5 else 'black')
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='#666666')
    ax.axhline(len(BENCHMARK_GAMES) - .5, color='black', linewidth=1)
    fig.colorbar(im, ax=ax, label='log₂(C++ CPU throughput / JAX H200 throughput)', fraction=.03, pad=.025)
    fig.text(.5, .035, 'Gray: not measured, unfinished, or resource-limited; no extrapolation. Median of five warmed calls.', ha='center', fontsize=8)
    fig.text(.5, .018, 'C++: full RL outputs, reset trials, ≤32 CPU threads. JAX: final carry, continuing rollouts.', ha='center', fontsize=8)
    if prepared:
        fig.text(.5, .003, '* JAX uses prepared reset parameters; all other rows use ordinary parameters.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ('pdf', 'png'):
        fig.savefig(args.output_dir / f'engine_crossover.{suffix}', dpi=200)
    plt.close(fig)
    def plot_curves(panel_games, stem):
        fig, axes = plt.subplots(4, 4, figsize=(13, 10))
        for ax, game in zip(axes.flat, panel_games):
            curves = [(cpp.get(game, {}), '#1f77b4', 'C++ CPU', 'cpp'),
                      (ordinary_jax.get(game, {}), '#d62728', 'JAX H200', 'jax')]
            if game in prepared:
                curves.append((prepared[game], '#218c45', 'JAX prepared', 'prepared'))
            for line_i, (data, color, label, variant) in enumerate(curves):
                points = sorted(data.get('results', []), key=lambda p: p['batch'])
                if not points:
                    note = f"{label}: {status(variant, game, data)}, no timing"
                    if variant == 'jax' and game not in ordinary_jax and status_by_game[game]['validation'] == 'failed':
                        note = 'JAX validation: failed'
                    ax.text(.03, .96 - .10 * line_i, note, color=color, transform=ax.transAxes, va='top', fontsize=7)
                    continue
                x = [p['batch'] for p in points]
                ax.plot(x, [p['median_fps'] for p in points], color=color, marker='.', markersize=3, label=label)
                ax.fill_between(x, [p['q25_fps'] for p in points], [p['q75_fps'] for p in points], color=color, alpha=.15)
                ax.text(.03, .96 - .10 * line_i, f"{label}: {status(variant, game, data)}", color=color, transform=ax.transAxes, va='top', fontsize=7)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.tick_params(labelsize=7)
            title = game.replace('_', ' ').replace('Heroes of Sokoban III  The Bard and The Druid', 'Heroes of Sokoban III')
            ax.set_title(title, fontsize=9)
            lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * (hi / lo) ** .2)
        fig.supxlabel('Batch size', y=.06); fig.supylabel('Environment steps/s')
        fig.text(.5, .022, 'Median / IQR, five warmed calls. C++: full RL outputs, reset trials. JAX: final carry, continuing rollouts.', ha='center', fontsize=8)
        fig.tight_layout(rect=(.02, .08, 1, 1))
        for suffix in ('pdf', 'png'):
            fig.savefig(args.output_dir / f'{stem}.{suffix}', dpi=200)
        plt.close(fig)

    plot_curves(COMPLEX_GAMES, 'complex_game_throughput')
    plot_curves(BENCHMARK_GAMES, 'original_game_throughput')
    target = 'cpp_over_jax_b4096'
    sample = [r for r in summaries if r[target] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    annotations = {'atlas shrank': 'Atlas', 'Caramelban': 'Caramelban',
                   'SwapBot': 'SwapBot', 'Sokoboros': 'Sokoboros',
                   'sokoban_basic': 'Sokoban'}
    for ax, feature, label in zip(axes, ['source_rules', 'board_cells'],
                                  ['Source rule count', 'Board cells']):
        for variant, color, marker in [('ordinary', '#d62728', 'o'), ('prepared', '#218c45', '^')]:
            rows = [r for r in sample if r['jax_variant'] == variant]
            ax.scatter([r[feature] for r in rows], [r[target] for r in rows],
                       color=color, marker=marker, label=f'JAX {variant}', s=35, alpha=.8)
        for row in sample:
            if row['game'] in annotations:
                ax.annotate(annotations[row['game']], (row[feature], row[target]),
                            xytext=(4, 4), textcoords='offset points', fontsize=7)
        ax.axhline(1, color='#777777', linewidth=.8, linestyle='--')
        if feature == 'source_rules':
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlim(-.1, max(r[feature] for r in sample) * 2)
        else:
            ax.set_xscale('log')
            ax.set_xlim(min(r[feature] for r in sample) * .7, max(r[feature] for r in sample) * 2)
        ax.set_yscale('log')
        ax.set_ylim(min(r[target] for r in sample) * .7, max(r[target] for r in sample) * 1.7)
        ax.set_xlabel(label); ax.grid(alpha=.15)
        correlation = correlations[target].get(feature, {})
        if correlation.get('spearman_rho') is not None:
            ax.set_title(f"Spearman ρ = {correlation['spearman_rho']:.2f}, n = {correlation['n']}")
    axes[0].set_ylabel('C++ CPU / JAX H200 throughput, batch 4,096')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.text(.5, .025, 'Selected games; correlated features, not causal attribution. Above 1: C++ faster; below 1: JAX faster.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for suffix in ('pdf', 'png'):
        fig.savefig(args.output_dir / f'complexity_factors.{suffix}', dpi=200)
    plt.close(fig)
    if args.reachable_result:
        path = args.reachable_result
        pilot = json.loads(path.read_text())
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        game = pilot['game']
        if not pilot.get('validation', {}).get('passed'):
            raise ValueError('Reachability pilot did not pass its JS correctness gate')
        if pilot['compiled_sha256'] != status_by_game[game]['compiled_sha256']:
            raise ValueError('Reachability pilot uses different compiled input')
        if not pilot['results']:
            raise ValueError('Reachability pilot has no timings')
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for data, color, label in [(cpp[game], '#1f77b4', 'C++ CPU'),
                                    (pilot, '#9467bd', 'JAX H200, fixed-level pruning prototype')]:
            points = data['results']
            x = [p['batch'] for p in points]
            ax.plot(x, [p['median_fps'] for p in points], '.-', color=color, label=label)
            ax.fill_between(x, [p['q25_fps'] for p in points], [p['q75_fps'] for p in points], color=color, alpha=.15)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Batch size'); ax.set_ylabel('Environment steps/s')
        ax.set_title(game.replace('_', ' ') + ': experimental fixed-level specialization')
        ax.legend(fontsize=8); ax.grid(alpha=.15)
        fig.text(.5, .035, 'Ordinary JAX timed out before validation/timing. Prototype: sampled JS trace passes; no saturation claim.',
                 ha='center', fontsize=7)
        fig.text(.5, .01, 'C++ tail is resource-limited while attempting batch 2,048; its higher batches are unmeasured.',
                 ha='center', fontsize=7)
        fig.tight_layout(rect=(0, .09, 1, 1))
        for suffix in ('pdf', 'png'):
            fig.savefig(args.output_dir / f'experimental_reachability.{suffix}', dpi=200)
        plt.close(fig)
        with (args.output_dir / 'experimental_reachability.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['batch', 'median_fps', 'q25_fps', 'q75_fps', 'compile_s'],
                                    extrasaction='ignore', lineterminator='\n')
            writer.writeheader(); writer.writerows(pilot['results'])
    for name, rows in [('throughput', table), ('game_comparison', summaries)]:
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with (args.output_dir / f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    result = {'games': summaries, 'exploratory_correlations': correlations,
              'correlation_caveat': 'Small selected sample with correlated features, not causal attribution; best-throughput correlations exclude unsaturated sweeps.',
              'inputs': inputs, 'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
