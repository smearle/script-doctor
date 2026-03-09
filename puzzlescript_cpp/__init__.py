"""Python wrapper around the C++ PuzzleScript engine.

This module provides:
  - CppPuzzleScriptEngine: low-level engine API
  - CppPuzzleScriptBackend: high-level backend matching NodeJSPuzzleScriptBackend
  - CppPuzzleScriptEnv: single-env gym-style interface
  - CppBatchedPuzzleScriptEnv: batched/vectorized gym-style interface
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from puzzlejax.backends.base import SearchResult
from puzzlescript_cpp._puzzlescript_cpp import Engine as _CppEngine, LevelBackup
from puzzlescript_cpp._puzzlescript_cpp import BatchedEngine as _BatchedEngine
from puzzlescript_cpp._puzzlescript_cpp import Renderer
from puzzlescript_cpp._puzzlescript_cpp import (
    MCTSOptions,
    RandomRolloutResult,
    SolverResult,
    random_rollout_raw as _random_rollout_raw,
    solve_astar as _solve_astar,
    solve_bfs as _solve_bfs,
    solve_gbfs as _solve_gbfs,
    solve_mcts as _solve_mcts,
    solve_random as _solve_random,
)


class CppPuzzleScriptEngine:
    """Low-level wrapper around the C++ PuzzleScript engine."""

    def __init__(self):
        self._engine = _CppEngine()
        self._json_state = None

    def load_from_json_file(self, path: str) -> None:
        with open(path, 'r') as f:
            json_str = f.read()
        self.load_from_json(json_str)

    def load_from_json(self, json_str: str) -> None:
        if not self._engine.load_from_json(json_str):
            raise RuntimeError("Failed to load compiled state from JSON")
        self._json_state = json.loads(json_str)

    def load_level(self, level_index: int) -> None:
        self._engine.load_level(level_index)

    def process_input(self, direction: int) -> bool:
        """Process input. dir: 0=up, 1=left, 2=down, 3=right, 4=action, -1=tick."""
        return self._engine.process_input(direction)

    def check_win(self) -> bool:
        return self._engine.check_win()

    def get_score(self) -> float:
        return float(self._engine.get_score())

    def get_score_normalized(self) -> float:
        return float(self._engine.get_score_normalized())

    @property
    def winning(self) -> bool:
        return self._engine.is_winning()

    @property
    def againing(self) -> bool:
        return self._engine.is_againing()

    def get_objects(self) -> np.ndarray:
        return np.array(self._engine.get_objects(), dtype=np.int32)

    def get_objects_2d(self) -> np.ndarray:
        return np.array(self._engine.get_objects_2d(), dtype=np.int32)

    @property
    def width(self) -> int:
        return self._engine.get_width()

    @property
    def height(self) -> int:
        return self._engine.get_height()

    @property
    def object_count(self) -> int:
        return self._engine.get_object_count()

    @property
    def num_levels(self) -> int:
        return self._engine.get_num_levels()

    @property
    def id_dict(self) -> list[str]:
        return self._engine.get_id_dict()

    def restart(self) -> None:
        self._engine.restart()

    def backup_level(self) -> LevelBackup:
        return self._engine.backup_level()

    def restore_level(self, backup: LevelBackup) -> None:
        self._engine.restore_level(backup)

    def random_rollout_raw(self, max_iters: int = 100_000, timeout_ms: int = -1) -> RandomRolloutResult:
        return _random_rollout_raw(self._engine, max_iters, timeout_ms)

    def solve_random(
        self,
        max_length: int = 100,
        max_iters: int = 100_000,
        timeout_ms: int = 60_000,
    ) -> SolverResult:
        return _solve_random(self._engine, max_length, max_iters, timeout_ms)

    def solve_bfs(self, max_iters: int = 100_000, timeout_ms: int = -1) -> SolverResult:
        return _solve_bfs(self._engine, max_iters, timeout_ms)

    def solve_astar(self, max_iters: int = 100_000, timeout_ms: int = -1) -> SolverResult:
        return _solve_astar(self._engine, max_iters, timeout_ms)

    def solve_gbfs(self, max_iters: int = 100_000, timeout_ms: int = -1) -> SolverResult:
        return _solve_gbfs(self._engine, max_iters, timeout_ms)

    def solve_mcts(self, options: MCTSOptions | None = None) -> SolverResult:
        if options is None:
            options = MCTSOptions()
        return _solve_mcts(self._engine, options)


class CppPuzzleScriptBackend:
    """High-level backend for the C++ PuzzleScript engine.
    
    Uses the JS compiler (via Node.js) to compile games, then serializes
    the compiled state to JSON and loads it into the C++ engine.
    """
    _ROOT_DIR = Path(__file__).resolve().parents[1]
    _ENGINE_JS_PATH = str(_ROOT_DIR / "puzzlescript_nodejs" / "puzzlescript" / "engine.js")
    SEARCH_ALGOS = {
        "bfs": "solve_bfs",
        "astar": "solve_astar",
        "gbfs": "solve_gbfs",
        "mcts": "solve_mcts",
        "random": "solve_random",
    }

    def __init__(self):
        self.cpp_engine = CppPuzzleScriptEngine()
        self._js_engine = None

    def _ensure_js_engine(self):
        if self._js_engine is None:
            from javascript import require
            self._js_engine = require(self._ENGINE_JS_PATH)

    def compile_game(self, parser: Any, game: str) -> str:
        """Compile a game using the JS compiler and load into C++ engine."""
        self._ensure_js_engine()
        from puzzlescript_nodejs.utils import compile_game
        game_text = compile_game(parser, self._js_engine, game, 0)
        # Serialize the compiled state from JS and load into C++
        json_str = str(self._js_engine.serializeCompiledStateJSON())
        self.cpp_engine.load_from_json(json_str)
        return game_text

    def compile_and_serialize(self, parser: Any, game: str) -> str:
        """Compile a game and return the serialized JSON state."""
        self._ensure_js_engine()
        from puzzlescript_nodejs.utils import compile_game
        compile_game(parser, self._js_engine, game, 0)
        return str(self._js_engine.serializeCompiledStateJSON())

    def load_from_json(self, json_str: str) -> None:
        """Load a pre-compiled game state from JSON."""
        self.cpp_engine.load_from_json(json_str)

    def load_level(self, game_text: str, level_i: int) -> None:
        """Load a level. Note: game must already be compiled."""
        self.cpp_engine.load_level(level_i)

    def unload_game(self) -> None:
        self.cpp_engine = CppPuzzleScriptEngine()

    def get_num_levels(self) -> int:
        return self.cpp_engine.num_levels

    def process_input(self, direction: int) -> bool:
        return self.cpp_engine.process_input(direction)

    @property
    def winning(self) -> bool:
        return self.cpp_engine.winning

    @property
    def againing(self) -> bool:
        return self.cpp_engine.againing

    def get_state(self) -> np.ndarray:
        return self.cpp_engine.get_objects()

    def get_id_dict(self) -> list[str]:
        return self.cpp_engine.id_dict

    def run_search(
        self,
        algo: str,
        *,
        game_text: str,
        level_i: int,
        n_steps: int,
        timeout_ms: int,
        warmup: bool = False,
    ) -> SearchResult:
        self.load_level(game_text, level_i)
        loops = 3 if warmup else 1
        method_name = self.SEARCH_ALGOS[algo]
        method = getattr(self.cpp_engine, method_name)
        raw_result = None
        for _ in range(loops):
            if algo == "random":
                raw_result = method(max_iters=n_steps, timeout_ms=timeout_ms)
            elif algo == "mcts":
                options = MCTSOptions()
                options.max_iterations = n_steps
                raw_result = method(options=options)
            else:
                raw_result = method(max_iters=n_steps, timeout_ms=timeout_ms)
        return self._normalize_result(raw_result)

    def run_random_rollout_raw(
        self,
        *,
        game_text: str,
        level_i: int,
        n_steps: int,
        timeout_ms: int,
    ) -> dict[str, float | int | bool]:
        self.load_level(game_text, level_i)
        result = self.cpp_engine.random_rollout_raw(max_iters=n_steps, timeout_ms=timeout_ms)
        return {
            "iterations": result.iterations,
            "time": result.time,
            "timeout": result.timeout,
        }

    @staticmethod
    def _normalize_result(raw_result: SolverResult) -> SearchResult:
        n_objs = len(raw_result.id_dict)
        stride_obj = (n_objs + 31) // 32
        objects = np.asarray(raw_result.state.dat, dtype=np.int32)
        state = objects.reshape((raw_result.state.width, raw_result.state.height, stride_obj)).tolist()
        return SearchResult(
            solved=bool(raw_result.won),
            actions=tuple(raw_result.actions),
            iterations=int(raw_result.iterations),
            time=float(raw_result.time),
            score=float(raw_result.score),
            state=state,
            timeout=bool(raw_result.timeout),
            objs=list(raw_result.id_dict),
        )


# ---------------------------------------------------------------------------
# Gym-style environments
# ---------------------------------------------------------------------------

MAX_AGAIN = 50


class CppPuzzleScriptEnv:
    """Single-environment gym-style wrapper around the C++ engine.

    Matches the interface of ``NodeJSPuzzleEnv`` / ``PuzzleJaxEnv``:
      - ``reset()  -> (obs, info)``
      - ``step(action) -> (obs, reward, done, truncated, info)``

    Observations are multihot uint8 arrays of shape ``(n_objs, height, width)``.
    Actions: 0=up, 1=left, 2=down, 3=right, 4=action.
    """

    def __init__(self, json_str: str, level_i: int = 0,
                 max_episode_steps: int = 200):
        self._engine = _CppEngine()
        if not self._engine.load_from_json(json_str):
            raise RuntimeError("Failed to load compiled game JSON")
        self._json_str = json_str
        self._level_i = level_i
        self._max_steps = max_episode_steps
        self._steps = 0

        # Load level once to get geometry
        self._engine.load_level(level_i)
        self._n_objs = self._engine.get_object_count()
        self._width = self._engine.get_width()
        self._height = self._engine.get_height()
        self._stride_obj = (self._n_objs + 31) // 32
        self._prev_score = float(self._engine.get_score())

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self._n_objs, self._height, self._width)

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def num_levels(self) -> int:
        return self._engine.get_num_levels()

    def set_level(self, level_i: int) -> None:
        self._level_i = level_i

    def reset(self) -> tuple[np.ndarray, dict]:
        self._engine.load_level(self._level_i)
        self._steps = 0
        self._prev_score = float(self._engine.get_score())
        obs = self._get_obs()
        return obs, {"won": False, "steps": 0, "score": self._prev_score, "score_delta": 0.0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._engine.process_input(action)
        ag = 0
        while self._engine.is_againing() and ag < MAX_AGAIN:
            self._engine.process_input(-1)
            ag += 1

        self._steps += 1
        won = self._engine.is_winning()
        truncated = self._steps >= self._max_steps
        done = won
        score = float(self._engine.get_score())
        score_delta = self._prev_score - score
        reward = score_delta + (1.0 if won else 0.0) - 0.01
        self._prev_score = score
        obs = self._get_obs()
        info = {"won": won, "steps": self._steps, "score": score, "score_delta": score_delta}
        return obs, reward, done, truncated, info

    def get_objects(self) -> np.ndarray:
        return np.array(self._engine.get_objects(), dtype=np.int32)

    def _get_obs(self) -> np.ndarray:
        objects = self._engine.get_objects()
        obs = np.zeros((self._n_objs, self._height, self._width), dtype=np.uint8)
        for x in range(self._width):
            for y in range(self._height):
                flat_idx = (x * self._height + y) * self._stride_obj
                for obj in range(self._n_objs):
                    word = obj // 32
                    bit = obj % 32
                    if word < self._stride_obj and (int(objects[flat_idx + word]) & (1 << bit)):
                        obs[obj, y, x] = 1
        return obs


class CppBatchedPuzzleScriptEnv:
    """Batched/vectorized gym-style wrapper around the C++ BatchedEngine.

    Manages ``batch_size`` independent environments running in parallel.

    Interface:
      - ``reset()         -> obs``         (reset all)
      - ``reset(indices)  -> obs``         (reset specific envs)
      - ``step(actions)   -> (obs, rewards, dones, truncated, infos)``

    Observations: uint8 ``(batch, n_objs, height, width)``
    Actions: int32 ``(batch,)``, values 0-4.
    Auto-resets environments that reach ``done``.
    """

    def __init__(self, json_str: str, batch_size: int,
                 level_indices: list[int] | None = None,
                 max_episode_steps: int = 200):
        self._be = _BatchedEngine(batch_size)
        if not self._be.load_from_json(json_str):
            raise RuntimeError("Failed to load compiled game JSON")
        self._json_str = json_str
        self._max_steps = max_episode_steps

        if level_indices is None:
            level_indices = [0] * batch_size
        self._be.set_levels(level_indices)

        # Do an initial reset to populate geometry caches
        self._be.reset_all()

        self._steps = np.zeros(batch_size, dtype=np.int32)

    @property
    def batch_size(self) -> int:
        return self._be.batch_size

    @property
    def num_objects(self) -> int:
        return self._be.num_objects

    @property
    def observation_shape(self) -> tuple[int, int, int, int]:
        """(batch, n_objs, height, width)"""
        return tuple(self._be.get_obs_shape())

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def num_levels(self) -> int:
        return self._be.num_levels

    def set_levels(self, level_indices: list[int]) -> None:
        self._be.set_levels(level_indices)

    def reset(self, env_indices: list[int] | None = None) -> np.ndarray:
        if env_indices is None:
            self._be.reset_all()
            self._steps[:] = 0
        else:
            self._be.reset(env_indices)
            for i in env_indices:
                self._steps[i] = 0
        return np.array(self._be.get_obs())

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        actions = np.asarray(actions, dtype=np.int32)
        self._be.step(actions)
        self._steps += 1

        obs = np.array(self._be.get_obs())
        rewards = np.array(self._be.get_rewards())
        scores = np.array(self._be.get_scores())
        score_deltas = np.array(self._be.get_score_deltas())
        dones = np.array(self._be.get_dones())
        wins = np.array(self._be.get_wins())
        truncated = self._steps >= self._max_steps

        # Auto-reset truncated envs (won envs already auto-reset in C++)
        trunc_only = truncated & ~dones
        trunc_indices = np.where(trunc_only)[0].tolist()
        if trunc_indices:
            self._be.reset(trunc_indices)
            # Re-read obs for the reset envs
            obs = np.array(self._be.get_obs())
            for i in trunc_indices:
                self._steps[i] = 0

        # Also reset step counters for done (won) envs
        done_indices = np.where(dones)[0]
        self._steps[done_indices] = 0

        infos = {
            "won": wins,
            "steps": self._steps.copy(),
            "score": scores,
            "score_delta": score_deltas,
        }
        return obs, rewards, dones | truncated, truncated, infos

    def get_objects(self, env_idx: int) -> np.ndarray:
        return np.array(self._be.get_objects(env_idx))
