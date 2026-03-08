"""Python wrapper around the C++ PuzzleScript engine.

This module provides:
  - CppPuzzleScriptEngine: low-level engine API
  - CppPuzzleScriptBackend: high-level backend matching NodeJSPuzzleScriptBackend
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from puzzlescript_cpp._puzzlescript_cpp import Engine as _CppEngine, LevelBackup


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


class CppPuzzleScriptBackend:
    """High-level backend for the C++ PuzzleScript engine.
    
    Uses the JS compiler (via Node.js) to compile games, then serializes
    the compiled state to JSON and loads it into the C++ engine.
    """
    _ROOT_DIR = Path(__file__).resolve().parents[1]
    _ENGINE_JS_PATH = str(_ROOT_DIR / "puzzlescript_nodejs" / "puzzlescript" / "engine.js")

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
        json_str = str(self._js_engine.serializeCompiledState())
        self.cpp_engine.load_from_json(json_str)
        return game_text

    def compile_and_serialize(self, parser: Any, game: str) -> str:
        """Compile a game and return the serialized JSON state."""
        self._ensure_js_engine()
        from puzzlescript_nodejs.utils import compile_game
        compile_game(parser, self._js_engine, game, 0)
        return str(self._js_engine.serializeCompiledState())

    def load_from_json(self, json_str: str) -> None:
        """Load a pre-compiled game state from JSON."""
        self.cpp_engine.load_from_json(json_str)

    def load_level(self, game_text: str, level_i: int) -> None:
        """Load a level. Note: game must already be compiled."""
        self.cpp_engine.load_level(level_i)

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
