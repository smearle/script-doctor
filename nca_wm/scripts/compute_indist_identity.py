#!/usr/bin/env python3
r"""Identity baseline for the in-distribution intersection figure.

The trained-model 1-step (TF) error in
`nca_wm/paper/figures/indist_intersection/intersection_slope_id.pdf` lacks
the identity-predictor reference line that the OOD panel already shows.
Identity error is purely a property of the env (`#cells changed by one
step under a uniform random policy`), so no checkpoint is needed: we
just run the C++ engine for the same 14 intersection games at the same
eval settings (`n_random_episodes=10`, `max_steps=50`) used by
`train.evaluate_multigame`, and average per-step `(s_t != s_{t+1})`
counts across episodes / levels / games.

Writes:
  nca_wm/paper/figures/indist_intersection/identity.json
    {
      "games": ["..."],
      "n_random_episodes": 10, "max_steps": 50, "seed": 0,
      "per_game": {game: {"mean_per_step_cell_err": float, ...}},
      "aggregate": {"mean": float, "median": float}
    }

Run:
    .venv/bin/python3 nca_wm/scripts/compute_indist_identity.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv  # noqa: E402
from puzzlescript_jax.utils import init_ps_lark_parser  # noqa: E402

OUT_DIR = REPO_ROOT / "nca_wm" / "paper" / "figures" / "indist_intersection"
SUMMARY_JSON = OUT_DIR / "summary.json"
N_ACTIONS = 5  # mirrors nca_wm.train.N_ACTIONS


def _identity_for_level(json_str: str, level_i: int, *,
                        n_episodes: int, max_steps: int,
                        seed: int) -> np.ndarray:
    """Return per-step identity cell-error rate, shape (max_steps,).

    Mirrors the random-rollout setup in `train.evaluate_multigame`:
    uniform-random actions, `max_episode_steps=max_steps`, mean over
    `n_episodes`. Identity prediction $\\hat s_{t+1}{=}s_t$ has wrong-cell
    count == #{cells that changed in the env at step t}.
    """
    env = CppPuzzleScriptEnv(json_str, level_i=level_i,
                             max_episode_steps=max_steps)
    cells_per_ep = np.full((n_episodes, max_steps), np.nan)
    total_cells = None
    for ep in range(n_episodes):
        obs, _ = env.reset()
        _, H, W = obs.shape
        if total_cells is None:
            total_cells = H * W
        rng = np.random.RandomState(seed * 10_000 + level_i * 1_000 + ep)
        prev = obs.copy()
        for t in range(max_steps):
            a = int(rng.randint(N_ACTIONS))
            nxt, _, done, trunc, _ = env.step(a)
            cells_per_ep[ep, t] = int((prev != nxt).any(axis=0).sum())
            prev = nxt
            if done or trunc:
                break
    mean_cells = np.nanmean(cells_per_ep, axis=0)
    return mean_cells / total_cells


def _games_from_summary() -> list[str]:
    """Read the 14 intersection games from the existing summary.json."""
    if not SUMMARY_JSON.exists():
        raise SystemExit(
            f"missing {SUMMARY_JSON}; run collate_indist_intersection.py first"
        )
    return list(json.loads(SUMMARY_JSON.read_text())["intersection_games"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_random_episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    games = _games_from_summary()
    print(f"Computing identity baselines for {len(games)} intersection games "
          f"({args.n_random_episodes} eps x {args.max_steps} steps)...")

    ps = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()

    per_game: dict[str, dict] = {}
    per_game_means: list[float] = []
    t_start = time.time()
    for gi, game in enumerate(games):
        t0 = time.time()
        json_str = backend.compile_and_serialize(ps, game)
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=1)
        n_levels = env0.num_levels
        per_step_levels = []
        for li in range(n_levels):
            cer = _identity_for_level(
                json_str, li,
                n_episodes=args.n_random_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
            )
            per_step_levels.append(cer)
        per_step_arr = np.stack(per_step_levels, axis=0)  # (n_levels, max_steps)
        # Per-game scalar: mean over levels of mean over timesteps
        # (matches plot_indist_intersection's per-game aggregation).
        per_level_mean = np.nanmean(per_step_arr, axis=1)  # (n_levels,)
        game_mean = float(np.mean(per_level_mean))
        per_game[game] = {
            "n_levels": n_levels,
            "mean_per_step_cell_err": game_mean,
            "per_level_mean_cell_err": [float(x) for x in per_level_mean],
        }
        per_game_means.append(game_mean)
        print(f"  [{gi+1:>2d}/{len(games)}] {game:<40s} "
              f"levels={n_levels:>2d}  mean={100*game_mean:6.3f}%  "
              f"({time.time()-t0:.1f}s)")

    agg = {
        "mean":   float(np.mean(per_game_means)),
        "median": float(np.median(per_game_means)),
    }
    out = {
        "games": games,
        "n_random_episodes": args.n_random_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "per_game": per_game,
        "aggregate": agg,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "identity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nDone in {time.time()-t_start:.1f}s. "
          f"Aggregate mean={100*agg['mean']:.3f}%  median={100*agg['median']:.3f}%")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
