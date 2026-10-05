"""Timing must execute and synchronize every claimed repetition."""

import pytest

from scripts.benchmarks import benchmark_env_subsystems as benchmark


def test_each_repetition_executes_and_synchronizes_full_output(monkeypatch):
    calls, synchronized = [], []
    output = {"state": object(), "reward": object()}

    def compiled(*args):
        calls.append(args)
        return output

    clock = iter([0.0, 0.006, 1.0, 1.006])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(benchmark.jax, "block_until_ready", synchronized.append)
    stats = benchmark.benchmark_runner(compiled, ("input",), 4, 3, 2)
    assert calls == [("input",)] * 6
    assert synchronized == [output] * 6
    assert stats["median_us"] == pytest.approx(500)


@pytest.mark.parametrize("counts", [(0, 3, 2), (4, 0, 2), (4, 3, 0)])
def test_reject_empty_timing_run(counts):
    with pytest.raises(ValueError, match="positive"):
        benchmark.benchmark_runner(None, (), *counts)
