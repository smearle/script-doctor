"""Do not publish unfinished or incompatible C++ throughput curves."""
import json
from copy import deepcopy

import pytest

from scripts.benchmarks.benchmark_cpp_throughput import stop_reason, validate_resume
from scripts.plotting.plot_engine_throughput import load_cpp_results


def test_scaling_ignores_small_batch_noise_and_requires_sustained_plateau():
    rows = [{"batch": b, "median_fps": f} for b, f in [(1, 100), (2, 50), (64, 500), (128, 200), (1024, 200), (2048, 201)]]
    assert stop_reason(rows) is None
    assert stop_reason(rows + [{"batch": 4096, "median_fps": 202}]) == "plateau"
    assert stop_reason(rows + [{"batch": 4096, "median_fps": 250}]) is None
    assert stop_reason(rows + [{"batch": 4096, "median_fps": 150}]) == "regression"


def resume_config():
    return {"game": "test", "library_sha256": "library", "seed": 42,
            "source_hashes": {"scripts/benchmarks/benchmark_cpp_throughput.py": "driver",
                              "scripts/benchmarks/benchmark_cpp_scoring.py": "worker"},
            "adaptive": {"min_batch": 64, "max_batch": 8192, "min_gain": 0.03},
            "results": [{"batch": 1, "median_fps": 100}]}


def test_extension_changes_only_driver_and_widens_stopping_limits():
    previous = resume_config()
    current = deepcopy(previous)
    current["source_hashes"]["scripts/benchmarks/benchmark_cpp_throughput.py"] = "new-driver"
    current["adaptive"].update(min_batch=1024, max_batch=16384)
    current["results"] = []
    with pytest.raises(ValueError):
        validate_resume(previous, current)
    validate_resume(previous, current, extend=True)
    assert previous == resume_config()


@pytest.mark.parametrize("fault", ["library", "worker", "seed", "lower_limit", "threshold"])
def test_extension_rejects_changed_workload_or_narrower_limits(fault):
    previous = resume_config()
    current = deepcopy(previous)
    if fault == "library":
        current["library_sha256"] = "changed"
    elif fault == "worker":
        current["source_hashes"]["scripts/benchmarks/benchmark_cpp_scoring.py"] = "changed"
    elif fault == "seed":
        current["seed"] += 1
    elif fault == "lower_limit":
        current["adaptive"]["max_batch"] //= 2
    else:
        current["adaptive"]["min_gain"] = 0.1
    with pytest.raises(ValueError):
        validate_resume(previous, current, extend=True)


def complete_result(game):
    return {"game": game, "level": 0, "stop_reason": "plateau", "max_threads": 2, "trials": 3,
            "library_sha256": "library", "source_hashes": {}, "cpu_model": "cpu", "cpu_affinity": [0, 1],
            "host": "host", "seed": 42, "base_steps": 5000, "min_steps": 100,
            "timing": "test", "thread_environment": {}, "adaptive": {"min_batch": 1024},
            "results": [{"batch": 2**i, "threads": min(2**i, 2), "samples_s": [1., 1., 1.]} for i in range(13)]}


@pytest.mark.parametrize("fault", ["pending", "missing_batch", "short_trial", "wrong_threads"])
def test_incomplete_curves_are_rejected(tmp_path, fault):
    data = complete_result("test")
    if fault == "pending":
        data["pending_batch"] = 8
    elif fault == "missing_batch":
        data["results"].pop(1)
    elif fault == "short_trial":
        data["results"][0]["samples_s"].pop()
    else:
        data["results"][-1]["threads"] = 1
    (tmp_path / "test.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_cpp_results(tmp_path, ["test"])


def test_mixed_libraries_are_rejected(tmp_path):
    for game in ("a", "b"):
        (tmp_path / f"{game}.json").write_text(json.dumps(complete_result(game)))
    assert len(load_cpp_results(tmp_path, ["a", "b"])) == 2
    data = complete_result("b")
    data["library_sha256"] = "another-library"
    (tmp_path / "b.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="library_sha256"):
        load_cpp_results(tmp_path, ["a", "b"])
