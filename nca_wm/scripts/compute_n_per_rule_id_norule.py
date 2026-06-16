#!/usr/bin/env python3
"""Compute the in-distribution no-rule baseline for the n_per_rule sweep.

The no-rule baseline predicts the next state by stepping the game with its
RULES section emptied (collision layers and force-driven movement intact); see
``norule_baseline.py`` for the mechanics. Like the identity baseline
(``compute_n_per_rule_id_identity.py``), it is a model-independent, per-game
property, so we compute it once per unique training game across the whole
sweep and cache it; the plot averages over each corpus's own training games.

We reuse the identity baseline's eval settings (n_episodes=10, max_steps=50)
and aggregation so the two baselines sit on identical trajectories.

Writes:
  nca_wm/paper/figures/n_per_rule_scaling/id_norule.json
    {game_name: per_cell_norule_error_rate}   (in [0,1])

Usage:
    .venv/bin/python3 nca_wm/scripts/compute_n_per_rule_id_norule.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "nca_wm" / "scripts"))

from norule_baseline import compute_cell_err_baselines  # noqa: E402

LOG_DIR = REPO_ROOT / "nca_wm" / "logs"
OUT     = REPO_ROOT / "nca_wm" / "paper" / "figures" / "n_per_rule_scaling" / "id_norule.json"

N_EPISODES = 10
MAX_STEPS  = 50

# Mirror compute_n_per_rule_id_identity.py: cond runs cover the widest game set
# (n up to 400); load every available sweep run so we cache every training game.
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


def main() -> None:
    games = _collect_games()
    print(f"computing no-rule churn for {len(games)} unique training games")
    out: dict[str, float] = {}
    for i, (name, info) in enumerate(sorted(games.items())):
        n_levels = int(info.get("n_levels", 1))
        lvl_vals = []
        for li in range(n_levels):
            r = compute_cell_err_baselines(
                info["json_str"], li,
                n_episodes=N_EPISODES, max_steps=MAX_STEPS,
            )
            if r is not None:
                lvl_vals.append(r["norule"])
        if lvl_vals:
            out[name] = float(np.mean(lvl_vals))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(games)} done")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    vals = np.array(list(out.values())) * 100
    print(f"wrote {len(out)} games -> {OUT}")
    print(f"no-rule per-cell error over all games: "
          f"mean={vals.mean():.2f}%  median={np.median(vals):.2f}%")


if __name__ == "__main__":
    main()
