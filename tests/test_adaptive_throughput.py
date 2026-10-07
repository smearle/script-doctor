"""Stopping decisions must not turn noisy/unfinished sweeps into measured peaks."""
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.benchmarks import benchmark_paper_throughput as benchmark
from scripts.benchmarks.benchmark_paper_throughput import scaling_stop


def rows(values):
    return [{"batch": 4096 * 2**i, "median_fps": value} for i, value in enumerate(values)]


def test_growth_and_isolated_noise_continue():
    assert scaling_stop(rows([100, 200, 300])) is None
    assert scaling_stop(rows([100, 98])) is None
    assert scaling_stop(rows([100, 98, 120])) is None


def test_plateau_and_clear_regression_stop():
    assert scaling_stop(rows([100, 101, 102])) == "plateau"
    assert scaling_stop(rows([100, 85])) == "regression"
    assert scaling_stop(rows([100, 98, 97])) == "plateau"


def test_low_batch_noise_is_not_a_plateau():
    points = [{"batch": b, "median_fps": 100} for b in [1, 16, 256, 1024, 4096]]
    assert scaling_stop(points) is None


def test_resume_preserves_samples_and_extends_until_plateau(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "init_ps_env", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark.jax, "devices", lambda: [SimpleNamespace(device_kind="test-device")])
    monkeypatch.setattr(benchmark.jax, "clear_caches", lambda: None)
    calls = []

    def measure(env, batch, steps, trials, seed, *, params=None):
        assert params is None
        calls.append(batch)
        return {"batch": batch, "steps": steps, "median_fps": 100.0, "samples_s": [1.0]}

    monkeypatch.setattr(benchmark, "benchmark", measure)
    command = ["benchmark", "--game", "test", "--batches", "4096", "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", command)
    benchmark.main()
    original = json.loads((tmp_path / "test.json").read_text())["results"][0]
    monkeypatch.setattr(sys, "argv", command + ["--resume", "--adaptive"])
    benchmark.main()
    result = json.loads((tmp_path / "test.json").read_text())
    assert calls == [4096, 8192, 16384]
    assert result["stop_reason"] == "plateau"
    for key, value in original.items():
        assert result["results"][0][key] == value
    assert "pending_batch" not in result
    monkeypatch.setattr(sys, "argv", command + ["--resume", "--seed", "43"])
    with pytest.raises(SystemExit, match="2"):
        benchmark.main()
    assert json.loads((tmp_path / "test.json").read_text()) == result
    monkeypatch.setattr(sys, "argv", command + ["--resume", "--prepare-params"])
    with pytest.raises(SystemExit, match="2"):
        benchmark.main()
    assert 'cannot resume with changed reset preparation' in capsys.readouterr().err
    assert json.loads((tmp_path / "test.json").read_text()) == result
