"""Check final plotted medians against raw timings, and summarize coverage."""
import json
from pathlib import Path
import hashlib
from collections import Counter

import numpy as np

root = Path(__file__).resolve().parents[5]
archive = root / 'scripts/benchmarks/results/2026-10-07-complex'
figure_dir = root / 'paper/figures/complex_throughput_20261007'
comparison = json.loads((figure_dir / 'comparison.json').read_text())
curves = []
for name, expected_hash in comparison['inputs'].items():
    path = root / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, name
    if path.suffix != '.json':
        continue
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or 'game' not in data or 'results' not in data:
        continue
    batches = []
    for point in data['results']:
        times = np.asarray(point['samples_s'])
        assert len(times) == data['trials'] == 5 and np.all(times > 0), name
        fps = point['batch'] * point['steps'] / times
        np.testing.assert_allclose(fps, point['fps_samples'], rtol=1e-12)
        np.testing.assert_allclose(np.percentile(fps, [25, 50, 75]),
                                   [point['q25_fps'], point['median_fps'], point['q75_fps']], rtol=1e-12)
        batches.append(point['batch'])
    assert batches == sorted(set(batches)), name
    curves.append({'path': name, 'points': len(batches), 'timed_calls': len(batches) * data['trials']})
result = {
    'checks': 'All input hashes match; positive timings, five calls per point, exact median/IQR recomputation, unique increasing batches.',
    'curves': curves,
    'total_points': sum(x['points'] for x in curves),
    'total_timed_calls': sum(x['timed_calls'] for x in curves),
    'selected_cpp_statuses': dict(Counter(x['cpp_status'] for x in comparison['games'])),
    'selected_jax_statuses': dict(Counter(x['jax_status'] for x in comparison['games'])),
    'selected_jax_games_with_timing': sum(x['jax_best_fps'] is not None for x in comparison['games']),
    'selected_both_saturated': sum(x['both_saturated'] for x in comparison['games']),
}
(archive / 'data-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'curves'}, indent=2))
