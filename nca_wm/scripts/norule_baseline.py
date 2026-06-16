#!/usr/bin/env python3
"""No-rule baseline for the n_per_rule scaling figure.

The identity baseline predicts that the next state equals the current state.
The *no-rule* baseline is a stronger, mechanics-aware reference: it predicts
the state the game would reach if its RULES section were empty but its
collision layers and movement resolution were left intact. Concretely, any
"player" object moves one cell in the direction of the applied force unless
the move would collide (out of bounds, or into another object sharing the
player's collision layer); nothing else moves. This isolates how much of a
game's dynamics is plain force-driven movement versus rule-driven behavior.

We realize this baseline directly from the C++ engine by stripping the
compiled game's rules and stepping it: PuzzleScript's movement-resolution
phase (force application + collision) runs regardless of the RULES section, so
an empty-rules engine *is* the no-rule dynamics. Because the baseline is
teacher-forced (predict s_{t+1} from the real s_t), we inject each real
ground-truth state into the stripped engine via the engine's level
backup/restore API before taking the single step.

Both ``compute_n_per_rule_id_norule.py`` (in-distribution) and
``compute_n_per_rule_ood_norule.py`` (Heldout-26) build on the helpers here so
the two panels share identical baseline mechanics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "nca_wm"))

from train import _enabled_action_count            # noqa: E402
from puzzlescript_cpp import CppPuzzleScriptEnv  # noqa: E402


def strip_rules(json_str: str) -> str:
    """Return a compiled-game JSON with the RULES section emptied.

    Clears ``rules`` and ``lateRules`` plus the loop-point tables that index
    into them, leaving collision layers, win conditions, and movement intact.
    """
    c = json.loads(json_str)
    c["rules"] = []
    c["lateRules"] = []
    c["loopPoint"] = {}
    c["lateLoopPoint"] = {}
    return json.dumps(c)


def compute_cell_err_baselines(
    json_str: str, level_i: int, *,
    n_episodes: int, max_steps: int, base_seed: int = 0,
    n_act: int | None = None,
) -> dict[str, float] | None:
    """Per-cell identity and no-rule TF error for one level.

    Rolls out the real (full-rules) engine under random actions; at each step
    the no-rule prediction is obtained by restoring the real current state into
    a rules-stripped engine and stepping it once. Both baselines are scored as
    the fraction of grid cells mispredicted relative to the real next state,
    averaged over steps (mean over episodes per step, then mean over steps),
    matching the identity aggregation used elsewhere.

    Seeds and aggregation mirror the identity baseline: episode ``ep`` uses
    ``base_seed + 1000 * level_i + ep``. ``n_act`` is the number of actions to
    sample from; pass it explicitly to match a specific evaluator's action set
    (the OOD heldout eval sampled all 5 actions including the no-op action key),
    or leave it ``None`` to use the game's enabled-action count. Returns
    ``None`` if the level cannot be rolled out.
    """
    if n_act is None:
        try:
            n_act = _enabled_action_count(json_str)
        except Exception:
            return None
    norule_json = strip_rules(json_str)

    per_ep_id: list[list[float]] = []
    per_ep_nr: list[list[float]] = []
    max_len = 0
    for ep in range(n_episodes):
        rng = np.random.RandomState(base_seed + 1000 * level_i + ep)
        try:
            real = CppPuzzleScriptEnv(json_str, level_i=level_i,
                                      max_episode_steps=max_steps)
            norule = CppPuzzleScriptEnv(norule_json, level_i=level_i,
                                        max_episode_steps=max_steps)
            real_obs, _ = real.reset()
            norule.reset()
        except Exception:
            return None
        _, H, W = real_obs.shape
        total_cells = H * W
        prev = real_obs.copy()
        id_errs: list[float] = []
        nr_errs: list[float] = []
        for _t in range(max_steps):
            action = int(rng.randint(n_act))
            # Teacher-force the no-rule engine to the real current state, then
            # take its single force-driven step.
            bak = real._engine.backup_level()
            norule._engine.restore_level(bak)
            norule._engine.process_input(action)
            ag = 0
            while norule._engine.is_againing() and ag < 50:
                norule._engine.process_input(-1)
                ag += 1
            nr_pred = norule._get_obs()

            real_next, _, done, truncated, _ = real.step(action)
            id_errs.append(float((prev != real_next).any(axis=0).sum()) / total_cells)
            nr_errs.append(float((nr_pred != real_next).any(axis=0).sum()) / total_cells)
            prev = real_next
            if done or truncated:
                break
        if id_errs:
            per_ep_id.append(id_errs)
            per_ep_nr.append(nr_errs)
            max_len = max(max_len, len(id_errs))
    if not per_ep_id:
        return None

    def _agg(per_ep: list[list[float]]) -> float:
        padded = np.full((len(per_ep), max_len), np.nan)
        for i, e in enumerate(per_ep):
            padded[i, : len(e)] = e
        return float(np.nanmean(np.nanmean(padded, axis=0)))

    return {"identity": _agg(per_ep_id), "norule": _agg(per_ep_nr)}
