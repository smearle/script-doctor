"""Archive this experiment's checkpoints; no timings or statuses are inferred."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BASE_GAMES = [
    'Beam_Islands', 'Boxes_&_Balloons', 'Caramelban', 'Crate_Assembler',
    'Heroes_of_Sokoban_III__The_Bard_and_The_Druid', 'IceCrates', 'Indigestion',
    'Memories_Of_Castlemouse', 'ParaLands', 'Sokoboros', 'SwapBot', 'Symbolism',
    'Transition', 'Unconventional_Guns', 'Vacuum', 'castlecloset',
]
PREPARED_GAMES = ['atlas shrank', 'Magnet_Jack', 'Beam_Islands', 'Caramelban', 'IceCrates',
                  'Heroes_of_Sokoban_III__The_Bard_and_The_Druid', 'ParaLands', 'SwapBot',
                  'Symbolism', 'Transition', 'Indigestion']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    src, dst = args.source, args.destination
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('cpp-throughput', 'torch', 'torch-wide', 'torch-terminal', 'torch-prepared', 'torch-reachable',
                 'torch-reachable-v2', 'cached-reset-h200', 'rtx4090-safe-large'):
        if (src / name).exists():
            shutil.copytree(src / name, dst / name, dirs_exist_ok=True)
    for name in ('slurm-status.txt', 'guarded-and-curves.log', 'guarded-blocks-recheck.json', 'guarded-blocks-recheck.log',
                 'guarded-notsnake-recheck.json', 'guarded-notsnake-recheck.log',
                 'cached-reset-final-tests.log', 'final-cpu-checks.log', 'resume-cpu-suite.log', 'reporting-tests.log',
                 'extend-kettle.log', 'batch-rule-work.json', 'batch-rule-work.log',
                 'reachable-structure-v2.json', 'reachable-structure-v2.log',
                 'cached-reset-b4096-rtx4090.json', 'cached-reset-b4096-rtx4090.log'):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    effective = dst / 'validation'
    effective.mkdir(exist_ok=True)
    origins = {}
    for name in ('torch', 'torch-wide', 'torch-terminal'):
        for path in sorted((dst / name / 'validation').glob('*.json')):
            shutil.copy2(path, effective / path.name)
            origins[path.stem] = {'path': str(path.relative_to(dst)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (effective / 'manifest.json').write_text(json.dumps(origins, indent=2) + '\n')
    statuses = {'cpp': {}, 'jax': {}, 'prepared': {}}
    cpu = json.loads((dst / 'cpp-throughput/launch-status.json').read_text())
    statuses['cpp'] = {r['game']: r['status'] for r in cpu['games']}
    extension = dst / 'cpp-throughput/extension-status.json'
    if extension.exists():
        statuses['cpp'].update({r['game']: r['status'] for r in json.loads(extension.read_text())})
    accounting = {}
    for line in (dst / 'slurm-status.txt').read_text().splitlines():
        fields = line.split('|')
        if len(fields) >= 4 and '.' not in fields[0]:
            accounting[fields[0]] = fields[1].lower().split()[0]
    jobs = [('jax', '19323485', dict(enumerate(BASE_GAMES))),
            ('jax', '19325515', {i: BASE_GAMES[i] for i in (8, 9, 11, 13, 14)}),
            ('prepared', '19328743', dict(enumerate(PREPARED_GAMES))),
            ('jax', '19328876', {0: 'Beam_Islands', 1: 'IceCrates'})]
    for variant, job, games in jobs:
        for index, game in games.items():
            key = f'{job}_{index}'
            if key in accounting:
                statuses[variant][game] = accounting[key]
    if '19328507' in accounting:
        statuses['jax']['Indigestion'] = accounting['19328507']
    (dst / 'job-status.json').write_text(json.dumps(statuses, indent=2) + '\n')


if __name__ == '__main__':
    main()
