"""Dump real-vs-predicted rollout frames from a trained NCA WM on OOD games.

Loads a trained rule_attn checkpoint (e.g. Train-199 cond), runs the
model autoregressively for N steps on each requested OOD game, and
saves per-step frames as PNGs:

    <out_dir>/<game>/real_t{i}.png
    <out_dir>/<game>/pred_t{i}.png

These are the slightly-wrong ŝ_t images consumed by the teaser figure.

Usage (Train-199 cond, 3 OOD games, 4 steps):

    .venv/bin/python3 -m nca_wm.scripts.dump_ood_rollout_frames \\
        --run_dir nca_wm/logs/multi_scaling_gallery_v4_cond_match_s0 \\
        --games link_by_jeffjeff123456,in_the_way_by_giovanni_mota,Hamiltwo \\
        --n_steps 4 \\
        --out_dir nca_wm/paper/figures/id_ood_teaser/rollouts
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Force CPU jax for this small workload (GPUs are busy).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import imageio
import jax
import jax.numpy as jnp

from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv

from nca_wm.train import (
    N_ACTIONS,
    _multihot_to_objects,
    _pad_state_for_model,
    _wm_p,
    make_apply_fn,
)
from nca_wm.heldout_eval import _build_heldout_game_info, _build_model, _load_run


def _bfs_actions(name: str, level: int, n_steps: int) -> list[int]:
    """Look up cached BFS actions for an OOD game/level."""
    cands = [
        _REPO_ROOT / "nca_wm" / "data_cache" / "heldout_search"
            / f"{name}_L{level}_bfs_100000_60000.npz",
        _REPO_ROOT / "rollout_data" / name / f"level_{level}"
            / "search_bfs_100000_60000.npz",
    ]
    for p in cands:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if "actions" in d.files and len(d["actions"]) > 0:
                return [int(a) for a in d["actions"][:n_steps]]
    rng = np.random.default_rng(0)
    return [int(rng.integers(N_ACTIONS)) for _ in range(n_steps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--games", required=True,
                    help="comma-separated game names")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--n_steps", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading run {args.run_dir}")
    cfg, params, game_infos = _load_run(args.run_dir)
    params = _wm_p(params)

    # Build model and apply fn using the train-time game_infos (needed to
    # match max_C / max_seq_len).
    model = _build_model(cfg, game_infos)
    apply_fn = jax.jit(make_apply_fn(model))

    max_C = max(g["n_objs"] for g in game_infos)
    train_max_tok = max(len(g.get("token_ids", [])) for g in game_infos)
    print(f"  model max_C={max_C}, max_seq_len={train_max_tok}")

    parser = init_ps_lark_parser()

    for name in games:
        print(f"\n=== {name} L{args.level} ===")
        info = _build_heldout_game_info(
            name, parser,
            encode_sprites=cfg.get("encode_sprites", False),
            kernel_sep=cfg.get("kernel_sep", False),
        )
        if info is None:
            print(f"  build failed; skipping")
            continue

        # Token padding for the model.
        tids = info["token_ids"][:train_max_tok]
        pad_tok = np.zeros(train_max_tok, dtype=np.int32)
        pad_mask = np.zeros(train_max_tok, dtype=np.bool_)
        pad_tok[:len(tids)] = tids
        pad_mask[:len(tids)] = True
        gt = jnp.array(pad_tok[None])
        gm = jnp.array(pad_mask[None])

        # Get authored actions.
        actions = _bfs_actions(name, args.level, args.n_steps)
        print(f"  actions: {actions}")

        # Render side: separate backend for sprite rendering of the original
        # PuzzleScript (compile via parser, not from JSON, so sprites map).
        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(parser, name)

        # Engine env to get the real obs sequence + spatial dims.
        env = CppPuzzleScriptEnv(
            info["json_str"], level_i=args.level,
            max_episode_steps=args.n_steps + 4,
        )
        real_obs, _ = env.reset()
        n_objs, H, W = real_obs.shape

        # Pad to next-pow2 for model input (matches heldout_eval).
        def _next_pow2(x: int, mn: int = 8) -> int:
            v = max(mn, int(x))
            p = 1
            while p < v:
                p <<= 1
            return p
        H_eval = max(_next_pow2(info["H"]), _next_pow2(H))
        W_eval = max(_next_pow2(info["W"]), _next_pow2(W))

        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_state = _pad_state_for_model(real_obs, max_C, H_eval, W_eval)

        # Save initial real frame (which doubles as predicted s_0).
        f0 = backend_render.render_frame_from_objects(
            _multihot_to_objects(real_obs), W, H,
        )
        imageio.imwrite(str(out_dir / "real_t0.png"), f0)
        imageio.imwrite(str(out_dir / "pred_t0.png"), f0)
        diffs = [{"t": 0, "wrong_cells": 0, "total_cells": int(H * W)}]

        for t, a in enumerate(actions):
            a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a][None])
            logits, _wl, _sl = apply_fn(params, pred_state, a_oh, gt, gm)
            pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

            real_next, _, done, trunc, _ = env.step(a)

            pred_obs = np.array(pred_next[0, :n_objs, :H, :W])
            pred_frame = backend_render.render_frame_from_objects(
                _multihot_to_objects(pred_obs), W, H,
            )
            real_frame = backend_render.render_frame_from_objects(
                _multihot_to_objects(real_next), W, H,
            )
            imageio.imwrite(str(out_dir / f"real_t{t+1}.png"), real_frame)
            imageio.imwrite(str(out_dir / f"pred_t{t+1}.png"), pred_frame)

            # Per-step error (cell-level).
            wrong = (pred_obs > 0.5).astype(np.uint8) != real_next
            wc = int(wrong.any(axis=0).sum())
            diffs.append({"t": t + 1, "wrong_cells": wc,
                          "total_cells": int(H * W)})
            print(f"    t={t+1}: {wc}/{H*W} cells differ")

            # AR feedback: same masking as heldout_eval (zero outside extent).
            clean = jnp.zeros_like(pred_next)
            clean = clean.at[:, :n_objs, :H, :W].set(
                pred_next[:, :n_objs, :H, :W]
            )
            pred_state = clean
            real_obs = real_next
            if done or trunc:
                print(f"    env done at t={t+1}")
                break

        with open(out_dir / "diffs.json", "w") as f:
            json.dump({"name": name, "level": args.level, "diffs": diffs}, f)
        print(f"  saved frames to {out_dir}")


if __name__ == "__main__":
    main()
