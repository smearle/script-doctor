"""Recreate ordinary and prepared-reset figures from this archived round."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[5]
results = Path('scripts/benchmarks/results/2026-10-07-complex')
output = Path('paper/figures/complex_throughput_20261007')
common = [
    sys.executable, '-m', 'scripts.plotting.plot_complex_throughput',
    '--jax-results', 'scripts/benchmarks/results/2026-10-05-adaptive/adaptive-throughput-h200',
    *[str(results / name / 'complex-throughput-h200') for name in ('torch', 'torch-wide', 'torch-terminal')],
    '--cpp-results', str(results / 'cpp-throughput'),
    '--metadata', str(results / 'metadata-original'),
    '--validation', str(results / 'validation'),
    '--rule-work', str(results / 'cpp-rule-work-original.json'),
    '--job-status', str(results / 'job-status.json'),
]
subprocess.run([*common, '--output-dir', str(output / 'ordinary')], cwd=root, check=True)
subprocess.run([
    *common, '--prepared-results', str(results / 'torch-prepared/prepared-throughput-h200'),
    '--prepared-validation', str(results / 'torch-prepared/validation'),
    '--reachable-result', str(results / 'torch-reachable-v2/reachable-crate.json'),
    '--output-dir', str(output),
], cwd=root, check=True)
