from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    actions: tuple[int, ...]
    iterations: int
    time: float
    score: float | int
    state: list[list[int]]
    timeout: bool
    objs: list[str]

    @property
    def fps(self) -> float:
        return self.iterations / (self.time if self.time > 0 else 1.0e-4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solved": self.solved,
            "actions": self.actions,
            "iterations": self.iterations,
            "time": self.time,
            "FPS": self.fps,
            "score": self.score,
            "state": self.state,
            "timeout": self.timeout,
            "objs": self.objs,
        }


class PuzzleScriptSearchBackend(ABC):
    @abstractmethod
    def compile_game(self, parser: Any, game: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_num_levels(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def load_level(self, game_text: str, level_i: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def unload_game(self) -> None:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
