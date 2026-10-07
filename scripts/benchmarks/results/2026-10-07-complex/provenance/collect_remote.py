"""Download checkpoints from the explicitly authorized, frozen benchmark roots."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PREFIX = '/scratch/se2161/puzzlejax-perf-20261002/'
SNAPSHOTS = {
    'complex-20261007': 'torch',
    'complex-wide-20261007': 'torch-wide',
    'complex-terminal-20261007': 'torch-terminal',
    'complex-prepared-20261007': 'torch-prepared',
    'complex-reachable-20261007': 'torch-reachable',
    'complex-reachable-v2-20261007': 'torch-reachable-v2',
}


def collect(item):
    snapshot, destination = item
    target = ROOT / 'results' / destination
    target.mkdir(parents=True, exist_ok=True)
    remote = 'torch:' + PREFIX + snapshot + '/'
    subprocess.run(['rsync', '-a', remote + 'results/', str(target) + '/'], check=True)
    subprocess.run(['rsync', '-a', '--include=/*.log', '--exclude=*', remote, str(target) + '/'], check=True)
    return destination


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        for destination in pool.map(collect, SNAPSHOTS.items()):
            print(destination, flush=True)
