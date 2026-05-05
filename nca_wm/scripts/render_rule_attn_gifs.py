"""Render real-vs-predicted rollout GIFs for a rule_attn NCA world model.

The shared train.py renderer (`render_multigame_gifs`) only supports
ConditionalNCAWorldModel/NCAWorldModel — it constructs a `*_intermediates`
variant the rule_attn model doesn't expose. This script is a focused
fallback: load a rule_attn checkpoint and emit per-level side-by-side GIFs
(real | predicted) for BFS-solution rollouts on every authored level.

Usage:
    .venv/bin/python3 nca_wm/scripts/render_rule_attn_gifs.py \\
        --run_dir nca_wm/logs_neko_arch/neko_d16_pool_perstep_tpe_n256_5sizes_s0 \\
        --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio
import jax
import jax.numpy as jnp
import numpy as np

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from puzzlescript_jax.utils import init_ps_lark_parser

from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
from nca_wm.train import (
    N_ACTIONS,
    _multihot_to_objects,
    _pad_state_for_model,
    _unpad_pred,
    _wm_p,
    make_apply_fn,
)


def _compose_label_frame(real: np.ndarray, pred: np.ndarray, label: str) -> np.ndarray:
    """Stack real | pred horizontally with a text banner above."""
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont

    H = max(real.shape[0], pred.shape[0])
    real_p = np.zeros((H, real.shape[1], 3), dtype=np.uint8)
    pred_p = np.zeros((H, pred.shape[1], 3), dtype=np.uint8)
    real_p[: real.shape[0]] = real
    pred_p[: pred.shape[0]] = pred
    row = np.concatenate([real_p, pred_p], axis=1)

    banner_h = 24
    banner = np.full((banner_h, row.shape[1], 3), 255, dtype=np.uint8)
    img = PIL.Image.fromarray(banner)
    draw = PIL.ImageDraw.Draw(img)
    try:
        font = PIL.ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = PIL.ImageFont.load_default()
    draw.text((4, 4), label, fill=(0, 0, 0), font=font)
    banner = np.array(img)

    return np.concatenate([banner, row], axis=0)


def _render_level_frames(
    model: RuleAttnNCAWorldModel,
    params,
    apply_fn,
    json_str: str,
    backend_render: CppPuzzleScriptBackend,
    level_i: int,
    n_objs: int,
    actions: list[int],
    label_prefix: str,
    game_tokens: np.ndarray,
    game_mask: np.ndarray,
    max_H: int,
    max_W: int,
    action_names: dict[int, str],
):
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=len(actions) + 4)
    real_obs, _ = env.reset()
    _, grid_h, grid_w = env.observation_shape

    pred_state = _pad_state_for_model(real_obs, model.n_out, max_H, max_W)

    gt = jnp.array(game_tokens[None])
    gm = jnp.array(game_mask[None])

    frames = []
    durations = []

    for t, action in enumerate(actions):
        real_frame = backend_render.render_frame_from_objects(
            _multihot_to_objects(real_obs), grid_w, grid_h
        )
        pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
        pred_frame = backend_render.render_frame_from_objects(
            _multihot_to_objects(pred_obs), grid_w, grid_h
        )
        frames.append(_compose_label_frame(
            real_frame, pred_frame,
            f"{label_prefix}  t={t}  action={action_names.get(action, str(action))}"
        ))
        durations.append(0.4)

        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])
        logits, _win, _spr = apply_fn(params, pred_state, a_oh, gt, gm)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        real_obs, _, done, truncated, _info = env.step(action)
        if done or truncated:
            break

    real_frame = backend_render.render_frame_from_objects(
        _multihot_to_objects(real_obs), grid_w, grid_h
    )
    pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
    pred_frame = backend_render.render_frame_from_objects(
        _multihot_to_objects(pred_obs), grid_w, grid_h
    )
    frames.append(_compose_label_frame(
        real_frame, pred_frame, f"{label_prefix}  t={len(actions)} (final)"
    ))
    durations.append(1.0)

    return frames, durations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_subdir", default="gifs")
    ap.add_argument("--max_steps", type=int, default=30)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    print(f"Loading {run_dir.name} (arch={cfg['architecture']}, "
          f"d={cfg['n_nca_steps']}, h={cfg['n_hid']})")

    with open(run_dir / "game_infos.pkl", "rb") as f:
        game_infos = pickle.load(f)
    print(f"  {len(game_infos)} game(s): {[g['name'] for g in game_infos]}")

    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    max_C = max(g["n_objs"] for g in game_infos)
    max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
    vocab_size = cfg["vocab_size"]

    model = RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
        vocab_size=vocab_size + 1,
        enc_d_model=cfg["d_model"], enc_n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"], n_app_slots=cfg["n_app_slots"],
        d_slot=cfg["d_slot"],
        n_attn_heads=cfg["n_heads"],
        max_seq_len=max_tok_len + 1,
        axis_pool=cfg["axis_pool"],
        axis_cummax=cfg["axis_cummax"],
        global_pool=cfg["global_pool"],
        use_vq=cfg["vq_codebook"],
        vq_codebook_size=cfg["vq_codebook_size"],
        vq_commitment_weight=cfg["vq_commitment_weight"],
        use_layernorm=cfg["use_layernorm"],
        input_skip=cfg["input_skip"],
        n_repeats=cfg["n_nca_repeats"],
        adaptive_halt=cfg["adaptive_halt"],
        mask_hidden=cfg["mask_hidden"],
    )

    ckpt_path = run_dir / "params_best.pkl"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "params.pkl"
    print(f"  loading params from {ckpt_path.name}")
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)
    params = _wm_p(params)

    apply_fn = make_apply_fn(model)

    # Lark parser for tokenizer (game_infos already has tokens, so this
    # is only needed for the cpp backend compile).
    ps_parser = init_ps_lark_parser()

    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    action_names = {0: "left", 1: "up", 2: "right", 3: "down", 4: "action"}

    for info in game_infos:
        name = info["name"]
        json_str = info["json_str"]
        n_objs = info["n_objs"]
        n_levels = info["n_levels"]
        token_ids = info.get("token_ids", [])

        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[: len(token_ids)] = token_ids
        mask[: len(token_ids)] = True

        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(ps_parser, name)

        backend_search = CppPuzzleScriptBackend()
        backend_search.load_from_json(json_str)

        all_frames, all_durs = [], []
        for level_i in range(n_levels):
            try:
                backend_search.load_level("", level_i)
                result = backend_search.run_search(
                    "bfs", game_text="", level_i=level_i,
                    n_steps=100_000, timeout_ms=-1,
                )
            except Exception as e:
                print(f"    {name} L{level_i} BFS failed: {e}")
                continue
            if not result.actions:
                print(f"    {name} L{level_i} BFS no solution; skipping")
                continue
            actions = list(result.actions)[: args.max_steps]
            label = (f"{name} L{level_i} BFS "
                     f"({'win' if result.solved else 'no win'}, {len(actions)} steps)")
            print(f"  {label}")

            frames, durs = _render_level_frames(
                model, params, apply_fn, json_str, backend_render,
                level_i, n_objs, actions, f"{name} L{level_i}",
                padded, mask, max_H, max_W, action_names,
            )
            all_frames.extend(frames)
            all_durs.extend(durs)

        if not all_frames:
            print(f"  {name}: no frames produced")
            continue

        max_fh = max(f.shape[0] for f in all_frames)
        max_fw = max(f.shape[1] for f in all_frames)
        padded_frames = []
        for f in all_frames:
            pf = np.zeros((max_fh, max_fw, 3), dtype=np.uint8)
            pf[: f.shape[0], : f.shape[1]] = f
            padded_frames.append(pf)

        safe_name = name.replace(" ", "_")
        gif_path = out_dir / f"rollout_{safe_name}_bfs.gif"
        imageio.mimsave(str(gif_path), padded_frames, duration=all_durs, loop=0)
        print(f"  saved {gif_path} ({len(padded_frames)} frames)")


if __name__ == "__main__":
    main()
