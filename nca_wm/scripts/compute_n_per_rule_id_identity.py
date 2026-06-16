#!/usr/bin/env python3
"""Compute the in-distribution identity baseline for the n_per_rule sweep.

The in-distribution evaluator (eval_multigame.npz) only stores *model*
rollout error, never the identity (copy-last-state) baseline that the OOD
heldout evaluator records. Identity error is model-independent -- it is just
the per-step state churn: the fraction of grid cells that change between
consecutive ground-truth states under a random-action rollout. So we
recompute it directly from the C++ engine on each training game, matching the
eval settings (n_random_episodes=10, max_steps=50) and the OOD aggregation
(per-step mean over episodes, then mean over steps).

Identity is a per-game property, so we compute it once per unique game across
the whole sweep and cache it; the plot averages over each scale's own
training-game set.

Writes:
  nca_wm/paper/figures/n_per_rule_scaling/id_identity.json
    {game_name: per_cell_identity_error_rate}   (in [0,1])

Usage:
    .venv/bin/python3 nca_wm/scripts/compute_n_per_rule_id_identity.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "nca_wm"))

from train import N_ACTIONS, _enabled_action_count  # noqa: E402
from puzzlescript_cpp import CppPuzzleScriptEnv       # noqa: E402

LOG_DIR = REPO_ROOT / "nca_wm" / "logs"
OUT     = REPO_ROOT / "nca_wm" / "paper" / "figures" / "n_per_rule_scaling" / "id_identity.json"

N_EPISODES = 10
MAX_STEPS  = 50

# cond runs cover the widest game set (n up to 400); load every available
# sweep run so we cache identity for every training game that appears.
RUN_GLOBS = ["n_per_rule_*_cond_dp_val0.10_s0", "n_per_rule_*_uncond_h288_dp_val0.10_s0"]


def _collect_games() -> dict[str, dict]:
    """name -> game info dict (json_str, n_levels) over the whole sweep."""
    games: dict[str, dict] = {}
    for glob_pat in RUN_GLOBS:
        for run in sorted(LOG_DIR.glob(glob_pat)):
            pkl = run / "game_infos.pkl"
            if not pkl.exists():
                continue
            for info in pickle.loads(pkl.read_bytes()):
                games.setdefault(info["name"], info)
    return games


def _identity_cell_err(json_str: str, level_i: int, n_objs: int) -> float | None:
    """Per-cell identity error for one level: mean over steps of the
    per-step (mean-over-episodes) fraction of changed cells. None if the
    level cannot be rolled out."""
    try:
        n_act = _enabled_action_count(json_str)
    except Exception:
        return None
    per_ep = []
    max_len = 0
    for ep in range(N_EPISODES):
        rng = np.random.RandomState(1000 * level_i + ep)
        try:
            env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=MAX_STEPS)
            real_obs, _ = env.reset()
        except Exception:
            return None
        _, H, W = real_obs.shape
        total_cells = H * W
        prev = real_obs.copy()
        errs = []
        for _t in range(MAX_STEPS):
            action = int(rng.randint(n_act))
            real_next, _, done, truncated, _ = env.step(action)
            changed = (prev != real_next).any(axis=0).sum()
            errs.append(changed / total_cells)
            prev = real_next
            if done or truncated:
                break
        if errs:
            per_ep.append(errs)
            max_len = max(max_len, len(errs))
    if not per_ep:
        return None
    padded = np.full((len(per_ep), max_len), np.nan)
    for i, e in enumerate(per_ep):
        padded[i, : len(e)] = e
    per_step = np.nanmean(padded, axis=0)
    return float(np.nanmean(per_step))


def main() -> None:
    games = _collect_games()
    print(f"computing identity churn for {len(games)} unique training games")
    out: dict[str, float] = {}
    for i, (name, info) in enumerate(sorted(games.items())):
        n_levels = int(info.get("n_levels", 1))
        n_objs = int(info["n_objs"])
        lvl_vals = []
        for li in range(n_levels):
            v = _identity_cell_err(info["json_str"], li, n_objs)
            if v is not None:
                lvl_vals.append(v)
        if lvl_vals:
            out[name] = float(np.mean(lvl_vals))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(games)} done")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    vals = np.array(list(out.values())) * 100
    print(f"wrote {len(out)} games -> {OUT}")
    print(f"identity per-cell error over all games: "
          f"mean={vals.mean():.2f}%  median={np.median(vals):.2f}%")


if __name__ == "__main__":
    main()
