"""Multi-game padded grid data for the NCA-belief model.

Loads the compiled gist subset (multigame_build.py output), rolls haoo' episodes
from a random game, and pads each frame to fixed (Cmax,Hmax,Wmax) with validity
masks. Channels >=2 are permuted per-trajectory (background=0, player=1 pinned by
the engine id_dict convention) so the model cannot memorize slot->meaning and must
infer each game's object dynamics in-context.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W   # for step_engine / read helpers

_DIR = Path(__file__).resolve().parent / "_multigame"


@dataclass
class Game:
    gist: str
    json_str: str
    n_obj: int
    H: int
    W: int
    n_levels: int


class MultiGameSet:
    def __init__(self, cmax=24, hmax=16, wmax=20, split="train", holdout=30, seed=0):
        rows = [json.loads(l) for l in (_DIR / "manifest.jsonl").read_text().splitlines()]
        rows.sort(key=lambda r: r["gist"])
        random.Random(seed).shuffle(rows)
        rows = rows[holdout:] if split == "train" else rows[:holdout]
        self.cmax, self.hmax, self.wmax = cmax, hmax, wmax
        self.games = []
        for r in rows:
            js = (_DIR / "json" / f"{r['gist']}.json").read_text()
            self.games.append(Game(r["gist"], js, r["n_obj"], r["H"], r["W"], r["n_levels"]))

    def __len__(self):
        return len(self.games)


def _engine(json_str, level_i):
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    e = Engine(); e.load_from_json(json_str); e.load_level(level_i)
    return e


def _read_padded(engine, n_obj, perm, cmax, hmax, wmax):
    """(Cmax,Hmax,Wmax) multihot; channels in id_dict order (background=0)."""
    a = np.asarray(engine.get_objects_2d())            # (W,H,stride)
    Wd, H = a.shape[0], a.shape[1]
    cell = a[:, :, 0].T.astype(np.int64)               # (H,Wd) bitmask; int64 avoids bit-31 sign issues
    g = np.zeros((cmax, hmax, wmax), dtype=np.float32)
    hh, ww = min(H, hmax), min(Wd, wmax)
    for c in range(min(n_obj, cmax)):
        g[perm[c], :hh, :ww] = ((cell >> c) & 1).astype(np.float32)[:hh, :ww]
    return g


def _masks(n_obj, H, W, perm, cmax, hmax, wmax):
    cell = np.zeros((hmax, wmax), dtype=np.float32); cell[:H, :W] = 1.0
    chan = np.zeros((cmax,), dtype=np.float32)
    for c in range(min(n_obj, cmax)):
        chan[perm[c]] = 1.0
    return cell, chan


def _perm(n_obj, cmax, rng):
    """Identity channel map (in-distribution objective: no slot permutation)."""
    return list(range(cmax))


def sample_trajectory(mgset: MultiGameSet, n_steps, rng, policy=None):
    g = rng.choice(mgset.games)
    perm = _perm(g.n_obj, mgset.cmax, rng)
    eng = _engine(g.json_str, 0)   # level 0 (matches manifest H,W)
    Cm, Hm, Wm = mgset.cmax, mgset.hmax, mgset.wmax

    def rd():
        return _read_padded(eng, g.n_obj, perm, Cm, Hm, Wm)

    grids = [rd()]; acts = []; resamps = []
    for _ in range(n_steps):
        ai = rng.randrange(len(V.ACTIONS)) if policy is None else policy(eng)
        a = V.ACTIONS[ai]
        s1, s2 = str(rng.getrandbits(40)), str(rng.getrandbits(40))
        bak = eng.backup_level()
        W.step_engine(eng, a, seed=s1); o1 = rd()
        eng.restore_level(bak); W.step_engine(eng, a, seed=s2); o2 = rd()
        eng.restore_level(bak); W.step_engine(eng, a, seed=s1)
        grids.append(o1); acts.append(ai); resamps.append(o2)
    cell, chan = _masks(g.n_obj, g.H, g.W, perm, Cm, Hm, Wm)
    return (np.stack(grids), np.asarray(acts, np.int64), np.stack(resamps), cell, chan)


def batch(mgset, n, n_steps, rng):
    O, A, R, CM, CH = [], [], [], [], []
    for _ in range(n):
        o, a, r, cell, chan = sample_trajectory(mgset, n_steps, rng)
        O.append(o); A.append(a); R.append(r); CM.append(cell); CH.append(chan)
    t = torch.from_numpy
    return (t(np.stack(O)), t(np.stack(A)).long(), t(np.stack(R)),
            t(np.stack(CM)), t(np.stack(CH)))
