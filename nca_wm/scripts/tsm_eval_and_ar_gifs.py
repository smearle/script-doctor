"""Recompute TSM eval on the shared AR/TF random rollout and render AR GIFs.

For each tsm_pool_diag config:
  1. Reload the model from its config.json (via serve_wm._build_wm — no CLI
     flag footgun) and refresh ``eval_multigame.npz`` through
     ``evaluate_multigame`` (now AR + TF share one pre-rolled rollout).
  2. Per level, call ``_run_eval_rollouts_jax(return_both=True)`` to get the
     raw AR/TF grids on the SAME action sequence, print a per-level
     AR-vs-TF table, and assert the same-rollout invariant: any episode that
     is teacher-forced-perfect (all steps wrong_cells==0) is also AR-perfect.
  3. Render AR-rollout GIFs (real | soft pred | hard pred) for the requested
     levels using the EXACT eval episode-0 action row, so each GIF is the
     rollout behind its reported AR error.

Usage (run from repo root, GPU 1 to respect the box-sharing rule):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 nca_wm/scripts/tsm_eval_and_ar_gifs.py \
        --config pool_off_skip_on --levels all
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from puzzlescript_jax.utils import init_ps_lark_parser  # noqa: E402
from nca_wm.serve_wm import _build_wm, _unwrap_wm  # noqa: E402
from nca_wm.train import (  # noqa: E402
    N_ACTIONS, make_apply_fn, evaluate_multigame,
    _run_eval_rollouts_jax, _render_training_gif, _enabled_action_count,
)
from puzzlescript_cpp import CppPuzzleScriptBackend  # noqa: E402

LOGDIR = os.path.join(REPO, "nca_wm", "logs", "tsm_pool_diag")
OUTDIR = os.path.join(REPO, "nca_wm", "figures", "tsm_pool_diag", "ar_gifs")
ALL_CONFIGS = ["pool_off_skip_on", "pool_on_skip_on",
               "pool_off_skip_off", "pool_on_skip_off"]

# Match evaluate_multigame's random-rollout settings exactly so the GIF actions
# and the reported metrics come from the identical rollout.
N_EPS = 10
MAX_STEPS = 50
EVAL_SEED = 0


def _eval_action_grid(json_str):
    """The (N_EPS, MAX_STEPS) action grid eval draws (default_rng seed 0),
    restricted to the game's ENABLED actions (respects `noaction`) — matching
    the fixed eval sampling so the GIF rollout reproduces the reported numbers."""
    return np.random.default_rng(EVAL_SEED).integers(
        0, _enabled_action_count(json_str), size=(N_EPS, MAX_STEPS), dtype=np.int32)


def _token_kwargs(info, max_tok_len, conditional):
    if not conditional:
        return {}
    tids = info.get("token_ids", [])
    padded = np.zeros(max_tok_len, dtype=np.int32)
    mask = np.zeros(max_tok_len, dtype=np.bool_)
    padded[: len(tids)] = tids
    mask[: len(tids)] = True
    return {"game_tokens": padded, "game_mask": mask}


def _load(config):
    cfg_dir = os.path.join(LOGDIR, config)
    with open(os.path.join(cfg_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(cfg_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(cfg_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    model, max_tok_len = _build_wm(cfg, game_infos)
    return cfg_dir, cfg, _unwrap_wm(params), game_infos, model, max_tok_len


def per_level_table(model, params, game_infos, max_tok_len, conditional):
    """Per-level AR vs TF on the shared rollout + invariant check."""
    info = game_infos[0]
    name, json_str, n_objs = info["name"], info["json_str"], info["n_objs"]
    n_levels = info["n_levels"]
    max_C = model.n_out
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    ck = _token_kwargs(info, max_tok_len, conditional)

    rows = []
    violations = 0
    for lvl in range(n_levels):
        r = _run_eval_rollouts_jax(
            model, params, json_str, lvl, n_objs, max_C, max_H, max_W,
            n_episodes=N_EPS, max_steps=MAX_STEPS, return_both=True, **ck,
        )
        ar_c = r["ar_wrong_cells_grid"]   # (n_eps, T) NaN past termination
        tf_c = r["tf_wrong_cells_grid"]
        # Per episode: TF-perfect (nan-safe all-zero) must imply AR-perfect.
        for i in range(ar_c.shape[0]):
            tf_perfect = np.nan_to_num(tf_c[i], nan=0.0).max() == 0
            ar_perfect = np.nan_to_num(ar_c[i], nan=0.0).max() == 0
            if tf_perfect and not ar_perfect:
                violations += 1
        ar_final = float(np.nanmean(ar_c[:, -1])) if ar_c.shape[1] else 0.0
        tf_max = float(np.nanmax(tf_c)) if tf_c.size else 0.0
        ar_max = float(np.nanmax(ar_c)) if ar_c.size else 0.0
        fd = np.array([ar_c.shape[1] if x < 0 else x for x in r["ar_first_div"]])
        rows.append((lvl, max_W, max_H, tf_max, ar_max, ar_final,
                     float(fd.mean())))

    print(f"\n=== {name} per-level AR vs TF (shared rollout, "
          f"{N_EPS} eps x {MAX_STEPS} steps) ===")
    print(f"{'lvl':>3} {'tf_max':>7} {'ar_max':>7} "
          f"{'ar_mean_final':>13} {'ar_first_div':>13}")
    for lvl, _w, _h, tf_max, ar_max, ar_final, fd in rows:
        print(f"{lvl:>3} {tf_max:>7.0f} {ar_max:>7.0f} "
              f"{ar_final:>13.2f} {fd:>13.1f}")
    inv = "OK (TF-perfect ⟹ AR-perfect, 0 violations)" if violations == 0 \
        else f"VIOLATED ({violations} episodes)"
    print(f"same-rollout invariant: {inv}")
    return rows, violations


def render_gifs(model, params, game_infos, max_tok_len, conditional,
                cfg, config, levels, n_steps, ps_parser):
    info = game_infos[0]
    name, n_objs, json_str = info["name"], info["n_objs"], info["json_str"]
    max_C = model.n_out
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    apply_fn = make_apply_fn(model)
    ck = _token_kwargs(info, max_tok_len, conditional)
    grid = _eval_action_grid(json_str)  # exactly what eval draws (respects noaction)

    backend_render = CppPuzzleScriptBackend()
    backend_render.compile_game(ps_parser, name)

    out_dir = os.path.join(OUTDIR, config)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for lvl in levels:
        # Render the WORST-diverging episode for this level (the eval averages
        # over N_EPS; episode 0 may be a clean rollout). On clean levels every
        # episode is perfect, so this still shows perfect tracking.
        r = _run_eval_rollouts_jax(
            model, params, json_str, lvl, n_objs, max_C, max_H, max_W,
            n_episodes=N_EPS, max_steps=MAX_STEPS, actions_2d=grid,
            return_both=True, **ck,
        )
        finals = np.nan_to_num(r["ar_wrong_cells_grid"][:, -1])
        ep = int(np.argmax(finals))
        actions = grid[ep, :n_steps].tolist()
        save_path = os.path.join(out_dir, f"L{lvl:02d}_ar.gif")
        _render_training_gif(
            apply_fn, params, info,
            max_C=max_C, max_H=max_H, max_W=max_W,
            save_path=save_path, backend_render=backend_render,
            n_steps=n_steps, conditional=conditional,
            banner_text=f"{config} L{lvl} ep{ep}",
            level_i=lvl, actions=actions,
            **{k: ck[k] for k in ("game_tokens", "game_mask") if k in ck},
        )
        print(f"  wrote {save_path}  (ep{ep}, final_wrong={int(finals[ep])})")
        paths.append(save_path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pool_off_skip_on",
                    help="config under logs/tsm_pool_diag, or 'all' for the table/eval refresh")
    ap.add_argument("--levels", default="all",
                    help="'all' or comma-separated level indices to render GIFs for")
    ap.add_argument("--n_steps", type=int, default=MAX_STEPS)
    ap.add_argument("--no_eval", action="store_true",
                    help="skip evaluate_multigame npz refresh")
    ap.add_argument("--no_gif", action="store_true", help="skip GIF rendering")
    args = ap.parse_args()

    ps_parser = init_ps_lark_parser()
    configs = ALL_CONFIGS if args.config == "all" else [args.config]

    for config in configs:
        print(f"\n########## {config} ##########")
        cfg_dir, cfg, params, game_infos, model, max_tok_len = _load(config)
        conditional = cfg.get("conditional", True)

        if not args.no_eval:
            evaluate_multigame(
                model, params, game_infos, ps_parser=None,
                n_random_episodes=N_EPS, max_steps=MAX_STEPS,
                save_dir=cfg_dir,
            )

        per_level_table(model, params, game_infos, max_tok_len, conditional)

        if not args.no_gif and args.config != "all":
            n_levels = game_infos[0]["n_levels"]
            levels = (list(range(n_levels)) if args.levels == "all"
                      else [int(x) for x in args.levels.split(",")])
            render_gifs(model, params, game_infos, max_tok_len, conditional,
                        cfg, config, levels, args.n_steps, ps_parser)


if __name__ == "__main__":
    main()
