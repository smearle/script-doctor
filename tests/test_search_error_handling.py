"""End-to-end tests for search error classification, logging, and aggregation.

Covers:
  1. _classify_error correctly buckets OOM / timeout / unknown exceptions
  2. write_level_error_log produces valid, parseable JSON with required fields
  3. The search loop (search_nodejs.main) catches injected OOM and timeout
     errors from a mock backend and writes proper error logs
  4. The aggregation path (plot_search_results._collect_results_for_algo)
     picks up OOM and timeout flags from those logs
  5. SLURM timeout derivation logic
  6. Simulated SLURM kill (SIGTERM mid-search) — verifies that results written
     before the kill survive on disk
"""

import json
import os
import signal
import multiprocessing
import tempfile
from dataclasses import dataclass
from typing import Optional, List
from unittest.mock import patch, MagicMock

import pytest

from backends.base import SearchResult
from search_nodejs import (
    _classify_error,
    write_level_error_log,
    TIMEOUT_ERROR_PATTERNS,
    OOM_ERROR_PATTERNS,
)


# ---------------------------------------------------------------------------
# 1. _classify_error unit tests
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_memory_error(self):
        assert _classify_error(MemoryError()) == "oom"

    def test_timeout_error(self):
        assert _classify_error(TimeoutError()) == "timeout"

    def test_oom_pattern_in_message(self):
        for pattern in OOM_ERROR_PATTERNS:
            exc = RuntimeError(f"Node.js {pattern} during BFS")
            assert _classify_error(exc) == "oom", f"Failed for pattern: {pattern}"

    def test_timeout_pattern_in_message(self):
        for pattern in TIMEOUT_ERROR_PATTERNS:
            exc = RuntimeError(f"Search {pattern} after 60s")
            assert _classify_error(exc) == "timeout", f"Failed for pattern: {pattern}"

    def test_unknown_error(self):
        assert _classify_error(ValueError("something else")) == "unknown"

    def test_oom_takes_priority_over_timeout_in_message(self):
        # If both patterns appear, OOM is checked first
        exc = RuntimeError("out of memory after timeout")
        assert _classify_error(exc) == "oom"


# ---------------------------------------------------------------------------
# 2. write_level_error_log unit tests
# ---------------------------------------------------------------------------


class TestWriteLevelErrorLog:
    def test_writes_valid_json(self, tmp_path):
        path = str(tmp_path / "error_log.json")
        result = write_level_error_log(path, "oom", "heap out of memory")

        assert os.path.isfile(path)
        with open(path, "r") as f:
            loaded = json.load(f)

        assert loaded["won"] is False
        assert loaded["error"] == "oom"
        assert loaded["error_message"] == "heap out of memory"
        assert loaded["iterations"] == 0
        assert loaded["actions"] == []
        assert result == loaded

    def test_timeout_error_log(self, tmp_path):
        path = str(tmp_path / "timeout_log.json")
        write_level_error_log(path, "timeout", "search timed out after 60s")

        with open(path, "r") as f:
            loaded = json.load(f)

        assert loaded["error"] == "timeout"
        assert loaded["won"] is False

    def test_overwrites_existing(self, tmp_path):
        path = str(tmp_path / "result.json")
        # Write a "good" result first
        with open(path, "w") as f:
            json.dump({"won": True, "iterations": 500}, f)

        write_level_error_log(path, "oom", "crash")
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded["error"] == "oom"
        assert loaded["won"] is False


# ---------------------------------------------------------------------------
# 3. End-to-end: mock backend raises, search loop writes error logs
# ---------------------------------------------------------------------------


@dataclass
class FakeCfg:
    algo: str = "bfs"
    n_steps: int = 1000
    timeout: int = 60
    game: Optional[str] = "sokoban_basic"
    dataset: str = "priority"
    include_randomness: bool = False
    random_order: bool = False
    overwrite: bool = True
    render: bool = False
    slurm: bool = False
    slurm_timeout_min: int = 180
    level: Optional[int] = None
    # PSConfig fields that may be accessed
    map_width: int = 10
    max_board_scans: float = 1.0
    randomize_map_shape: bool = False


def _make_mock_cfg(**kwargs):
    """Build a minimal config-like object for search_nodejs.main."""
    return FakeCfg(**kwargs)


class _MockBackend:
    """A fake backend that raises on specific levels."""

    def __init__(self, error_schedule: dict[int, Exception], n_levels: int | None = None):
        """error_schedule: {level_i: exception_to_raise}"""
        self._error_schedule = error_schedule
        if n_levels is not None:
            self._n_levels = n_levels
        elif error_schedule:
            self._n_levels = max(error_schedule.keys()) + 2  # room for a success level
        else:
            self._n_levels = 3

    def compile_game(self, parser, game):
        return "fake_game_text"

    def unload_game(self):
        pass

    def get_num_levels(self):
        return self._n_levels

    def run_search(self, algo, *, game_text, level_i, n_steps, timeout_ms, warmup=False):
        if level_i in self._error_schedule:
            raise self._error_schedule[level_i]
        return SearchResult(
            solved=True,
            actions=(0, 1, 2),
            iterations=42,
            time=0.1,
            score=1,
            state=[],
            timeout=False,
            objs=[],
        )


class TestSearchLoopErrorHandling:
    """Inject OOM and timeout errors into the search loop via a mock backend."""

    def test_oom_and_timeout_are_logged(self, tmp_path):
        error_schedule = {
            0: MemoryError("heap out of memory"),        # level 0: OOM
            1: TimeoutError("search timed out"),          # level 1: timeout
            # level 2: succeeds normally
        }
        mock_backend = _MockBackend(error_schedule)
        sols_dir = str(tmp_path / "js_sols")

        cfg = _make_mock_cfg()

        with (
            patch("search_nodejs.NodeJSPuzzleScriptBackend", return_value=mock_backend),
            patch("search_nodejs.init_ps_lark_parser", return_value=None),
            patch("search_nodejs.JS_SOLS_DIR", sols_dir),
            patch("search_nodejs.STANDALONE_NODEJS_RESULTS_PATH",
                  str(tmp_path / "results.json")),
        ):
            from search_nodejs import main
            main(cfg)

        game_dir = os.path.join(sols_dir, "sokoban_basic")

        # Check OOM log
        oom_path = os.path.join(game_dir, "bfs_1000-steps_level-0.json")
        assert os.path.isfile(oom_path), f"OOM log not found at {oom_path}"
        with open(oom_path) as f:
            oom_result = json.load(f)
        assert oom_result["error"] == "oom"
        assert oom_result["won"] is False

        # Check timeout log
        timeout_path = os.path.join(game_dir, "bfs_1000-steps_level-1.json")
        assert os.path.isfile(timeout_path), f"Timeout log not found at {timeout_path}"
        with open(timeout_path) as f:
            timeout_result = json.load(f)
        assert timeout_result["error"] == "timeout"
        assert timeout_result["won"] is False

        # Check success log
        success_path = os.path.join(game_dir, "bfs_1000-steps_level-2.json")
        assert os.path.isfile(success_path), f"Success log not found at {success_path}"
        with open(success_path) as f:
            success_result = json.load(f)
        assert success_result["won"] is True
        assert "error" not in success_result

    def test_unknown_error_still_raises(self, tmp_path):
        error_schedule = {0: ValueError("something unexpected")}
        mock_backend = _MockBackend(error_schedule)
        sols_dir = str(tmp_path / "js_sols")
        cfg = _make_mock_cfg()

        with (
            patch("search_nodejs.NodeJSPuzzleScriptBackend", return_value=mock_backend),
            patch("search_nodejs.init_ps_lark_parser", return_value=None),
            patch("search_nodejs.JS_SOLS_DIR", sols_dir),
            patch("search_nodejs.STANDALONE_NODEJS_RESULTS_PATH",
                  str(tmp_path / "results.json")),
        ):
            from search_nodejs import main
            with pytest.raises(ValueError, match="something unexpected"):
                main(cfg)


# ---------------------------------------------------------------------------
# 4. Aggregation picks up error types
# ---------------------------------------------------------------------------


class TestAggregationErrorPickup:
    """Write synthetic result JSONs and verify _collect_results_for_algo reads them."""

    def _write_result(self, game_dir, algo, n_steps, level, result_dict):
        os.makedirs(game_dir, exist_ok=True)
        path = os.path.join(game_dir, f"{algo}_{n_steps}-steps_level-{level}.json")
        with open(path, "w") as f:
            json.dump(result_dict, f)

    def test_oom_and_timeout_flags(self, tmp_path):
        sols_dir = str(tmp_path / "js_sols")
        game = "mock_game"
        game_dir = os.path.join(sols_dir, game)
        algo = "bfs"
        n_steps = 1000

        # Level 0: success
        self._write_result(game_dir, algo, n_steps, 0, {
            "won": True, "actions": [0, 1], "iterations": 42,
            "score": 1, "timeout": False, "time": 0.1,
        })
        # Level 1: OOM
        self._write_result(game_dir, algo, n_steps, 1, {
            "won": False, "actions": [], "iterations": 0,
            "score": None, "timeout": False, "time": 0,
            "error": "oom", "error_message": "heap out of memory",
        })
        # Level 2: timeout
        self._write_result(game_dir, algo, n_steps, 2, {
            "won": False, "actions": [], "iterations": 0,
            "score": None, "timeout": False, "time": 0,
            "error": "timeout", "error_message": "search timed out",
        })
        # Level 3: normal failure (exhausted search space)
        self._write_result(game_dir, algo, n_steps, 3, {
            "won": False, "actions": [], "iterations": 1000,
            "score": 0, "timeout": False, "time": 1.0,
        })

        with patch("plot_search_results.JS_SOLS_DIR", sols_dir):
            from plot_search_results import _collect_results_for_algo
            results_by_depth, per_level_by_depth = _collect_results_for_algo([game], algo)

        assert n_steps in results_by_depth
        game_result = results_by_depth[n_steps][game]
        assert game_result["has_oom"] is True
        assert game_result["has_timeout"] is True
        assert game_result["n_levels"] == 4
        # Only level 0 solved
        assert game_result["pct_solved"] == pytest.approx(0.25)

    def test_no_errors_means_no_flags(self, tmp_path):
        sols_dir = str(tmp_path / "js_sols")
        game = "clean_game"
        game_dir = os.path.join(sols_dir, game)
        algo = "astar"
        n_steps = 500

        for level in range(3):
            self._write_result(game_dir, algo, n_steps, level, {
                "won": True, "actions": [0], "iterations": 10,
                "score": 1, "timeout": False, "time": 0.05,
            })

        with patch("plot_search_results.JS_SOLS_DIR", sols_dir):
            from plot_search_results import _collect_results_for_algo
            results_by_depth, _ = _collect_results_for_algo([game], algo)

        game_result = results_by_depth[n_steps][game]
        assert game_result["has_oom"] is False
        assert game_result["has_timeout"] is False
        assert game_result["pct_solved"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. SLURM timeout derivation
# ---------------------------------------------------------------------------


class TestSlurmTimeoutDerivation:
    def test_explicit_timeout_takes_priority(self):
        cfg = _make_mock_cfg(timeout=30, slurm=True, slurm_timeout_min=180)
        # When timeout > 0, it should be used directly
        timeout_ms = cfg.timeout * 1_000 if cfg.timeout > 0 else -1
        assert timeout_ms == 30_000

    def test_slurm_derives_timeout_when_none_set(self):
        cfg = _make_mock_cfg(timeout=-1, slurm=True, slurm_timeout_min=180)
        # Replicate the derivation logic from search_nodejs.main
        safe_seconds = max(int((cfg.slurm_timeout_min - 2) * 60 * 0.9), 60)
        timeout_ms = safe_seconds * 1_000
        # 178 min * 60 * 0.9 = 9612s
        assert safe_seconds == 9612
        assert timeout_ms == 9_612_000
        # Must be less than SLURM wall-time
        assert timeout_ms < cfg.slurm_timeout_min * 60 * 1_000

    def test_short_slurm_timeout_floors_at_60s(self):
        cfg = _make_mock_cfg(timeout=-1, slurm=True, slurm_timeout_min=3)
        safe_seconds = max(int((cfg.slurm_timeout_min - 2) * 60 * 0.9), 60)
        assert safe_seconds == 60

    def test_no_slurm_no_timeout(self):
        cfg = _make_mock_cfg(timeout=-1, slurm=False)
        if cfg.timeout > 0:
            timeout_ms = cfg.timeout * 1_000
        elif cfg.slurm:
            safe_seconds = max(int((cfg.slurm_timeout_min - 2) * 60 * 0.9), 60)
            timeout_ms = safe_seconds * 1_000
        else:
            timeout_ms = -1
        assert timeout_ms == -1


# ---------------------------------------------------------------------------
# 6. Simulated SLURM kill (SIGTERM) — partial results survive
# ---------------------------------------------------------------------------


def _worker_that_writes_then_hangs(sols_dir: str, ready_event, result_dict: dict):
    """Simulate a search job: write one result, then block until killed."""
    game_dir = os.path.join(sols_dir, "killed_game")
    os.makedirs(game_dir, exist_ok=True)
    path = os.path.join(game_dir, "bfs_1000-steps_level-0.json")
    with open(path, "w") as f:
        json.dump(result_dict, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    # Signal that the file has been written
    ready_event.set()
    # Block forever (simulating long-running search on next level)
    signal.pause()


class TestSlurmKillSurvival:
    def test_written_results_survive_sigterm(self, tmp_path):
        """Simulate SLURM killing a job with SIGTERM after one level is written."""
        sols_dir = str(tmp_path / "js_sols")
        result_dict = {
            "won": True, "actions": [0, 1, 2], "iterations": 100,
            "score": 1, "timeout": False, "time": 0.5,
            "FPS": 200, "objs": [], "state": [],
        }

        ready_event = multiprocessing.Event()
        worker = multiprocessing.Process(
            target=_worker_that_writes_then_hangs,
            args=(sols_dir, ready_event, result_dict),
        )
        worker.start()
        # Wait for the worker to write its result
        ready_event.wait(timeout=10)
        assert ready_event.is_set(), "Worker didn't signal readiness"

        # Simulate SLURM sending SIGTERM
        os.kill(worker.pid, signal.SIGTERM)
        worker.join(timeout=5)
        assert not worker.is_alive(), "Worker didn't terminate after SIGTERM"

        # Verify the written result survived
        result_path = os.path.join(sols_dir, "killed_game", "bfs_1000-steps_level-0.json")
        assert os.path.isfile(result_path), "Result file missing after SIGTERM"
        with open(result_path) as f:
            loaded = json.load(f)
        assert loaded["won"] is True
        assert loaded["iterations"] == 100


# ---------------------------------------------------------------------------
# 7. Error log compatibility with should_skip_existing_level_result
# ---------------------------------------------------------------------------


class TestSkipExistingWithErrors:
    """Error logs should be treated as existing results (skip on re-run)."""

    def test_error_log_is_skipped_on_rerun(self, tmp_path):
        from search_nodejs import should_skip_existing_level_result

        path = str(tmp_path / "error_result.json")
        write_level_error_log(path, "oom", "crash")
        assert should_skip_existing_level_result(path) is True

    def test_missing_file_is_not_skipped(self, tmp_path):
        from search_nodejs import should_skip_existing_level_result

        path = str(tmp_path / "nonexistent.json")
        assert should_skip_existing_level_result(path) is False

    def test_corrupt_json_is_skipped(self, tmp_path):
        """Corrupt files are skipped (not re-attempted) to avoid infinite loops."""
        from search_nodejs import should_skip_existing_level_result

        path = str(tmp_path / "corrupt.json")
        with open(path, "w") as f:
            f.write("{truncated")
        assert should_skip_existing_level_result(path) is True
