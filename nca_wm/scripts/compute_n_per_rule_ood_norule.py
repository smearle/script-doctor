#!/usr/bin/env python3
"""Compute the out-of-distribution no-rule baseline for the n_per_rule sweep.

The OOD heldout evaluator records the identity (copy-last-state) baseline per
(game, level) in ``heldout_v4_n30/results.json`` but not the no-rule baseline.
Both baselines are model-independent -- they depend only on the game, level,
and the (seeded) random action sequence -- so we recompute the no-rule
baseline once over the Heldout-26 (game, level) pairs, matching the heldout
evaluator's settings (n_random_episodes=3, max_steps=30, seed=0) and its
per-episode seeding (``1000 * level_i + ep``) so the new baseline sits on the
exact trajectories that produced the recorded identity numbers.

We also recompute identity here as a cross-check against the recorded value.

Writes:
  nca_wm/paper/figures/n_per_rule_scaling/ood_norule.json
    {"<game>|<level>": {"norule": float, "identity": float}}   (rates in [0,1])

Usage:
    .venv/bin/python3 nca_wm/scripts/compute_n_per_rule_ood_norule.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "nca_wm"))
sys.path.insert(0, str(REPO_ROOT / "nca_wm" / "scripts"))

from norule_baseline import compute_cell_err_baselines  # noqa: E402
from heldout_eval import _build_heldout_game_info        # noqa: E402
from puzzlescript_jax.utils import init_ps_lark_parser   # noqa: E402

LOG_DIR  = REPO_ROOT / "nca_wm" / "logs"
OUT_DIR  = REPO_ROOT / "nca_wm" / "paper" / "figures" / "n_per_rule_scaling"
OUT      = OUT_DIR / "ood_norule.json"

# Match nca_wm/scripts/run_heldout_rollout_eval.sh. The recorded heldout eval
# sampled the full action set (5: four directions + the no-op action key), so
# we fix n_act=5 here; the per-pair drift check below confirms our recomputed
# identity reproduces the recorded identity exactly under this action set.
N_EPISODES = 3
MAX_STEPS  = 30
SEED       = 0
N_ACT      = 5

RUN_GLOB = "n_per_rule_*/heldout_v4_n30/results.json"


def _collect_pairs() -> dict[str, dict[str, float]]:
    """{game: {level_str: recorded_identity_mean}} over every sweep run.

    Unioned across runs so we cover every (game, level) the plot might
    intersect down to Heldout-26.
    """
    pairs: dict[str, dict[str, float]] = {}
    for res in sorted(LOG_DIR.glob(RUN_GLOB)):
        try:
            r = json.loads(res.read_text())
        except Exception:
            continue
        for game, lvls in r.get("heldout", {}).items():
            for lvl, rollouts in lvls.items():
                tf = rollouts.get("random_tf")
                if tf is None:
                    continue
                pairs.setdefault(game, {})[str(lvl)] = float(tf["identity_cell_err_mean"])
    return pairs


def main() -> None:
    pairs = _collect_pairs()
    n_pairs = sum(len(v) for v in pairs.values())
    print(f"computing no-rule baseline for {n_pairs} (game, level) pairs "
          f"across {len(pairs)} heldout games")
    ps_parser = init_ps_lark_parser()

    out: dict[str, dict[str, float]] = {}
    max_id_drift = 0.0
    for gi, (game, lvls) in enumerate(sorted(pairs.items())):
        info = _build_heldout_game_info(
            game, ps_parser, encode_sprites=False, kernel_sep=False,
        )
        if info is None:
            print(f"  SKIP {game}: build failed")
            continue
        for lvl in sorted(lvls, key=int):
            li = int(lvl)
            r = compute_cell_err_baselines(
                info["json_str"], li,
                n_episodes=N_EPISODES, max_steps=MAX_STEPS, base_seed=SEED,
                n_act=N_ACT,
            )
            if r is None:
                print(f"  SKIP {game} L{li}: rollout failed")
                continue
            out[f"{game}|{lvl}"] = {"norule": r["norule"], "identity": r["identity"]}
            drift = abs(r["identity"] - lvls[lvl])
            max_id_drift = max(max_id_drift, drift)
        print(f"  [{gi + 1}/{len(pairs)}] {game}: {len(lvls)} level(s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    nr = np.array([v["norule"] for v in out.values()]) * 100
    idv = np.array([v["identity"] for v in out.values()]) * 100
    print(f"\nwrote {len(out)} pairs -> {OUT}")
    print(f"no-rule  per-cell error over all pairs: mean={nr.mean():.2f}%  median={np.median(nr):.2f}%")
    print(f"identity per-cell error over all pairs: mean={idv.mean():.2f}%  median={np.median(idv):.2f}%")
    print(f"max identity drift vs recorded results.json: {max_id_drift * 100:.4f}% "
          f"(should be ~0 -- confirms matching trajectories)")


if __name__ == "__main__":
    main()
