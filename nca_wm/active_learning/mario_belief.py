"""Two-world Mario definitions + shared world utilities.

Worlds: custom_games/autumn/mario.txt (base) and mario_breakable.txt (variant).
They share level/objects/obs0 and differ in EXACTLY one transition: rising into a
Step from below leaves it intact (base) vs. removes that cell (variant). So the
world is hidden and the sole disambiguating action is a jump into a platform from
underneath — the basis of the information-gain experiment.

This module is now ONLY world building + engine-read helpers. All world-model
TRAINING uses the mature offline pipeline (``train_recurrent``'s A*-transition
predecessor-chain sampler), shared by both belief backbones:
  - Transformer belief: ``mario_transformer_baseline.py``  -> ckpts/mario2_transformer
  - Recurrent-NCA belief: ``mario_nca_belief.py``          -> ckpts/mario2_nca_belief
The old bespoke Mario collector (BFS start-state pool + heuristic/random/novelty
rollouts) was removed: data collection is now either the mature offline pipeline
or a deliberate online IG-driven scheme (see ``mario_explore.py``), never an
ad-hoc per-game heuristic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nca_wm.active_learning.multigame_data import Game, _engine

ROOT = Path(__file__).resolve().parents[2]
_CACHE = Path(__file__).resolve().parent / "_mario_worlds"
CMAX, HMAX, WMAX = 24, 18, 16
WORLDS = [("mario", "custom_games/autumn/mario.txt"),
          ("mario_breakable", "custom_games/autumn/mario_breakable.txt")]

# Mario is realtime: the action index == the engine input id, and id 5 is the
# no-op REALTIME TICK on which the world's autonomous dynamics advance (gravity
# pulls the player down; bullets rise). Enemy patrol fires on every input, but
# gravity ONLY on the tick, so the world only evolves correctly if ticks are
# issued. (Matches _enabled_actions() / num_actions for realtime_interval games.)
ACTIONS = ["UP", "LEFT", "DOWN", "RIGHT", "ACTION", "TICK"]
N_ACT = len(ACTIONS)
TICK = 5


def _step(eng, a, seed=None):
    """One engine frame: action index a is the engine input id (5 = realtime tick)."""
    if seed is not None:
        eng.seed_rng(seed)
    eng.process_input(a)
    k = 0
    while eng.is_againing() and k < 50:
        eng.process_input(-1); k += 1


def _compile(name, txt_rel):
    from nca_wm import game_curriculum as gc
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser
    _CACHE.mkdir(exist_ok=True)
    mat = _CACHE / "_scratch"; mat.mkdir(exist_ok=True)
    gc._set_materialize_dir(mat)
    parser = init_ps_lark_parser()
    code = (ROOT / txt_rel).read_text()
    gc._materialize_game(name, code)
    return CppPuzzleScriptBackend().compile_and_serialize(parser, name)


def build_worlds(force=False):
    games = []
    for name, rel in WORLDS:
        jf = _CACHE / f"{name}.json"
        if force or not jf.exists():
            _CACHE.mkdir(exist_ok=True)
            jf.write_text(_compile(name, rel))
        js = jf.read_text()
        e = _engine(js, 0)
        C, H, Wd = e.get_object_count(), e.get_height(), e.get_width()
        games.append(Game(name, js, C, H, Wd, e.get_num_levels()))
    return games


def _bit(eng, name):
    low = [str(x).lower() for x in eng.get_id_dict()]
    return low.index(name.lower())


def _grid(eng):
    a = np.asarray(eng.get_objects_2d())          # (W,H,stride)
    return a[:, :, 0].T.astype(np.int64)          # (H,W)
