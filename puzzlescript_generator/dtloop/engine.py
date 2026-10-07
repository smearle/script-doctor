"""Host side of the PuzzleScript DT loop: script-doctor's C++ engine, without JAX (the Torch
generator process uses it too). See ps_env.py for the design.

- export_games: compile standard PuzzleScript with the reference JS engine (ps_export.js);
- admission: deterministic games without message or checkpoint;
- GamePool: one C++ Engine per game, shared by many environments through restore_level;
- rollout_changes: the seeded 200-move random rollout of ps_check.js, in the C++ engine;
- replay_consistent: the games that reproduce an episode (version-space information).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPORT_JS = HERE.parent / "ps_export.js"
AGAIN_CAP = 50
HORIZON = 128
ACTIONS = 5   # up, left, down, right, action (engine directions 0..4)


def load_cpp(so_path):
    """Import the compiled extension from an explicit path (the package __init__ needs extras)."""
    name = "puzzlescript_cpp._puzzlescript_cpp"
    spec = importlib.util.spec_from_file_location(name, str(so_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def export_games(items, engine_dir, workers=8, stall_s=60.0):
    """{id: json string or None} for items [{id, text}], via ps_export.js in parallel Node workers
    (check_games.check_texts: a game that stalls a worker for stall_s seconds is refused)."""
    sys.path.insert(0, str(HERE.parent))
    from check_games import check_texts
    rows = check_texts([{"id": it["id"], "text": it["text"]} for it in items], engine_dir, workers=workers,
                       script=EXPORT_JS.name, stall_s=stall_s)
    return {r["id"]: (r["json"] if r.get("ok") else None) for r in rows}


def admission(json_str):
    """(admitted, reason): deterministic games without message or checkpoint commands."""
    d = json.loads(json_str)
    commands, stochastic = set(), False
    for key in ("rules", "lateRules"):
        for group in d[key]:
            for rule in group:
                commands |= {c[0] for c in rule["commands"] if c}
                stochastic |= bool(rule["isRandom"])
                for row in rule["patterns"]:
                    for cell in row:
                        if isinstance(cell, dict) and cell.get("replacement"):
                            rep = cell["replacement"]
                            stochastic |= any(rep["randomEntityMask"]) or any(rep["randomDirMask"])
    if stochastic:
        return False, "random"
    for c in ("message", "checkpoint"):
        if c in commands:
            return False, c
    return True, None


class GamePool:
    """Compiled games of one level, each in its own C++ Engine, keyed by integer game id."""

    def __init__(self, cpp, noaction=False):
        self.cpp, self.noaction = cpp, noaction
        self.engines, self.start, self.meta = {}, {}, None

    def add(self, gid, json_str):
        eng = self.cpp.Engine()
        if not eng.load_from_json(json_str):
            raise ValueError(f"engine refused game {gid}")
        eng.load_level(0)
        words = np.array(eng.get_objects(), np.int32)
        meta = (tuple(eng.get_id_dict()), eng.get_width(), eng.get_height(), words.size)
        if self.meta is None:
            self.meta = meta
        elif meta != self.meta:
            raise ValueError(f"game {gid} does not share the pool's objects or level size")
        self.engines[int(gid)], self.start[int(gid)] = eng, words

    def drop(self, gid):
        self.engines.pop(int(gid), None)
        self.start.pop(int(gid), None)

    @property
    def width(self):
        return self.meta[1]

    @property
    def height(self):
        return self.meta[2]

    @property
    def n_words(self):
        return self.meta[3]

    @property
    def n_objects(self):
        return len(self.meta[0])

    def starts(self, game):
        game = np.asarray(game, np.int64).reshape(-1)
        return np.stack([self.start[int(g)] for g in game]) if len(game) else np.zeros((0, self.n_words), np.int32)

    def step(self, words, game, action):
        """Batched turn: words [B, N] int32, game [B], action [B] -> (words, won, capped)."""
        words = np.asarray(words, np.int32).reshape(-1, self.n_words)
        game = np.asarray(game).reshape(-1)
        action = np.asarray(action).reshape(-1)
        out, won, capped = words.copy(), np.zeros(len(words), bool), np.zeros(len(words), bool)
        backup = self.cpp.LevelBackup
        for b in range(len(words)):
            a = int(action[b])
            if self.noaction and a == 4:
                continue
            eng = self.engines[int(game[b])]
            eng.restore_level(backup(words[b].tolist(), self.width, self.height))
            eng.process_input(a)
            ticks = 0
            while eng.is_againing() and ticks < AGAIN_CAP:
                eng.process_input(-1)
                ticks += 1
            out[b] = eng.get_objects()
            won[b], capped[b] = eng.is_winning(), eng.is_againing()
        return out, won, capped


def mulberry32(seed):
    """ps_check.js's action RNG."""
    s = seed & 0xFFFFFFFF

    def rand():
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((s ^ (s >> 15)) * (1 | s)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rand


def rollout_changes(pool: GamePool, gid, steps=200, seed=12345):
    """ps_check.js's dynamics rollout in the C++ engine: (moves that changed the level, won)."""
    rand = mulberry32(seed)
    n_actions = 4 if pool.noaction else 5
    words, changed = pool.start[int(gid)], 0
    for _ in range(steps):
        a = int(rand() * n_actions)
        out, won, _ = pool.step(words[None], np.array([gid]), np.array([a]))
        changed += int(not np.array_equal(out[0], words))
        words = out[0]
        if won[0]:
            return changed, True
    return changed, False


def replay_consistent(pool: GamePool, games, actions, observed_words, observed_won, observed_capped):
    """The games among `games` whose dynamics reproduce an episode: its start words, and after each
    action the words, the win flag and the capped-again flag (observed_words has one more row than
    actions). All candidates step together, one engine call per action."""
    actions = np.asarray(actions)
    games = np.asarray([int(g) for g in games if np.array_equal(pool.start[int(g)], observed_words[0])], np.int64)
    words = pool.starts(games)
    for t in range(len(actions)):
        if not len(games):
            break
        out, won, capped = pool.step(words, games, np.full(len(games), actions[t]))
        keep = ((out == observed_words[t + 1][None]).all(1) & (won == bool(observed_won[t]))
                & (capped == bool(observed_capped[t])))
        games, words = games[keep], out[keep]
    return [int(g) for g in games]
