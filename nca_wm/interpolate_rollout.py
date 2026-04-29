"""Roll out the conditional NCA world model under interpolated game latents.

Loads a trained ConditionalNCAWorldModel checkpoint, picks two endpoint
games from its training set, encodes each to a latent z via the trained
encoder, then for each interpolation step t ∈ [0, 1] rolls out the WM under
z_t = (1-t)*z_A + t*z_B starting from each endpoint's level 0 with a fixed
action sequence. Output: one PNG panel per donor level (rows = t, cols =
rollout frame).

Relies on `ConditionalNCAWorldModel.__call__`'s `z_override` kwarg to bypass
the encoder for non-endpoint t's. Endpoint t's still go through the encoder
(via the same kwarg = the encoded z) so reconstruction matches training.

Usage:
    python nca_wm/interpolate_rollout.py \
        --load nca_wm/logs/multi_scaling_2_cond_level-None_nca-4_hid-128_lr-0.001_s-0 \
        --game_a sokoban_basic --game_b nekopuzzle \
        --n_interp 5 --n_steps 12 --action_seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_wm.train import (
    ConditionalNCAWorldModel,
    N_ACTIONS,
    _multihot_to_objects,
    _pad_state_for_model,
    _unpad_pred,
)
from nca_wm.tokenize_game import VOCAB_SIZE_BASE, VOCAB_SIZE_EXT
from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from puzzlescript_jax.utils import init_ps_lark_parser


# ----------------------------------------------------------------------
# Loading & model rebuild
# ----------------------------------------------------------------------

def load_run(save_dir: str):
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(save_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(save_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    return cfg, params, game_infos


def build_model(cfg, game_infos):
    if not cfg.get("conditional", False):
        raise ValueError("interpolate_rollout requires a conditional checkpoint")
    if cfg.get("architecture", "film") != "film":
        raise NotImplementedError(
            f"Only --architecture film is supported here; got {cfg.get('architecture')!r}"
        )
    max_C = max(g["n_objs"] for g in game_infos)
    max_tok_len = max(max((len(g.get("token_ids", [])) for g in game_infos), default=1), 1)
    vocab_size = VOCAB_SIZE_EXT if cfg.get("encode_sprites", False) else VOCAB_SIZE_BASE
    return ConditionalNCAWorldModel(
        n_hid=cfg["n_hid"],
        n_steps=cfg["n_nca_steps"],
        n_out=max_C,
        vocab_size=vocab_size + 1,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_enc_layers=cfg["n_enc_layers"],
        d_z=cfg["d_z"],
        max_seq_len=max_tok_len + 1,
        axis_pool=cfg.get("axis_pool", False),
        axis_cummax=cfg.get("axis_cummax", False),
        global_pool=cfg.get("global_pool", False),
        use_layernorm=cfg.get("use_layernorm", False),
        sprite_decoder=cfg.get("sprite_loss_weight", 0.0) > 0.0,
    )


def pad_tokens(token_ids, max_seq_len):
    pad = np.zeros(max_seq_len, dtype=np.int32)
    mask = np.zeros(max_seq_len, dtype=np.bool_)
    L = min(len(token_ids), max_seq_len)
    pad[:L] = token_ids[:L]
    mask[:L] = True
    return pad, mask


def find_game(game_infos, name):
    for i, info in enumerate(game_infos):
        if info["name"] == name:
            return i, info
    names = [g["name"] for g in game_infos]
    raise ValueError(f"Game {name!r} not in checkpoint's training set; have {names}")


# ----------------------------------------------------------------------
# Latent encoding & rollout
# ----------------------------------------------------------------------

def encode_z(model, params, info, max_seq_len):
    tids = info.get("token_ids", [])
    pad, mask = pad_tokens(tids, max_seq_len)
    # Reach into the GameSpecEncoder submodule directly via its bound
    # subtree of params. We can't call ConditionalNCAWorldModel without
    # also providing a state, so use the encoder module's apply.
    from nca_wm.train import GameSpecEncoder
    enc = GameSpecEncoder(
        vocab_size=model.vocab_size,
        d_model=model.d_model,
        n_heads=model.n_heads,
        n_layers=model.n_enc_layers,
        d_z=model.d_z,
        max_seq_len=model.max_seq_len,
    )
    enc_params = {"params": params["params"]["game_encoder"]}
    z = enc.apply(enc_params, jnp.array(pad[None]), jnp.array(mask[None]))
    return np.array(z[0]), pad, mask


def make_rollout_fn(model):
    """jit'd one-step apply that uses z_override (skips encoder forward)."""
    def step(params, state, action_oh, dummy_tokens, dummy_mask, z):
        logits, _, _ = model.apply(
            params, state, action_oh, dummy_tokens, dummy_mask, z_override=z
        )
        return (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    return jax.jit(step)


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------

def make_renderer(ps_parser, name):
    backend = CppPuzzleScriptBackend()
    backend.compile_game(ps_parser, name)
    return backend


def render_state(backend, state_chw, n_objs, grid_h, grid_w):
    """state_chw is the unpadded (n_objs, H, W) multihot prediction."""
    return backend.render_frame_from_objects(
        _multihot_to_objects(state_chw), grid_w, grid_h
    )


def label_strip(images, label_text, font):
    """Stack a banner (label) above a horizontal row of images."""
    if not images:
        return np.zeros((20, 1, 3), dtype=np.uint8)
    row_h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] < row_h:
            p = np.zeros((row_h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
            im = np.concatenate([im, p], axis=0)
        padded.append(im)
    row = np.concatenate(padded, axis=1)
    bbox = font.getbbox(label_text)
    bw = max(row.shape[1], bbox[2] - bbox[0] + 8)
    bh = 18
    banner = PIL.Image.new("RGB", (bw, bh), (0, 0, 0))
    draw = PIL.ImageDraw.Draw(banner)
    draw.text((4, 2), label_text, fill=(255, 255, 255), font=font)
    if row.shape[1] < bw:
        pad = np.zeros((row_h, bw - row.shape[1], 3), dtype=np.uint8)
        row = np.concatenate([row, pad], axis=1)
    return np.concatenate([np.array(banner), row], axis=0)


def stack_panels(panel_imgs):
    """Vertically stack (variable-width) images, padding to common width."""
    max_w = max(p.shape[1] for p in panel_imgs)
    out = []
    for p in panel_imgs:
        if p.shape[1] < max_w:
            pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
            p = np.concatenate([p, pad], axis=1)
        out.append(p)
    return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", required=True,
                   help="Trained ConditionalNCAWorldModel run dir.")
    p.add_argument("--game_a", required=True)
    p.add_argument("--game_b", required=True)
    p.add_argument("--n_interp", type=int, default=5,
                   help="Number of t values across [0, 1] inclusive.")
    p.add_argument("--n_steps", type=int, default=12,
                   help="Rollout length per t.")
    p.add_argument("--action_seed", type=int, default=0,
                   help="RNG seed for the (shared) random action sequence.")
    p.add_argument("--save_dir", default=None,
                   help="Output dir (default: <load>/interp).")
    args = p.parse_args()

    cfg, params, game_infos = load_run(args.load)
    model = build_model(cfg, game_infos)
    save_dir = args.save_dir or os.path.join(args.load, "interp")
    os.makedirs(save_dir, exist_ok=True)

    max_C = max(g["n_objs"] for g in game_infos)
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    print(f"Padded shape (C, H, W) = ({max_C}, {max_H}, {max_W})")

    # Encode endpoint games
    _, info_a = find_game(game_infos, args.game_a)
    _, info_b = find_game(game_infos, args.game_b)
    z_a, pad_a, mask_a = encode_z(model, params, info_a, model.max_seq_len - 1)
    z_b, pad_b, mask_b = encode_z(model, params, info_b, model.max_seq_len - 1)
    print(f"|z_a|={np.linalg.norm(z_a):.3f}, |z_b|={np.linalg.norm(z_b):.3f}, "
          f"cos(z_a, z_b)={float(np.dot(z_a, z_b) / (np.linalg.norm(z_a) * np.linalg.norm(z_b) + 1e-8)):.3f}")

    ts = np.linspace(0.0, 1.0, args.n_interp)
    z_path = (1 - ts[:, None]) * z_a[None, :] + ts[:, None] * z_b[None, :]

    # Shared random action sequence
    rng = np.random.RandomState(args.action_seed)
    actions = rng.randint(0, N_ACTIONS, size=args.n_steps)
    print(f"Action sequence: {actions.tolist()}")

    rollout_step = make_rollout_fn(model)
    ps_parser = init_ps_lark_parser()

    try:
        font = PIL.ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
    except OSError:
        font = PIL.ImageFont.load_default()

    panels = []
    for donor_info, donor_label in [(info_a, args.game_a), (info_b, args.game_b)]:
        backend = make_renderer(ps_parser, donor_info["name"])
        env = CppPuzzleScriptEnv(donor_info["json_str"], level_i=0,
                                 max_episode_steps=args.n_steps + 1)
        obs0, _ = env.reset()
        n_objs, gh, gw = obs0.shape

        # Use a dummy token sequence to satisfy the encoder forward — its
        # output is overridden by z. The donor's tokens make a sensible
        # default but anything of the right shape would do.
        dummy_tokens = jnp.array(pad_a[None] if donor_info is info_a else pad_b[None])
        dummy_mask = jnp.array(mask_a[None] if donor_info is info_a else mask_b[None])

        # Per-t rollout
        rows = []
        for t_idx, t in enumerate(ts):
            z = jnp.array(z_path[t_idx][None].astype(np.float32))
            state = jnp.array(_pad_state_for_model(obs0, max_C, max_H, max_W))

            row_imgs = []
            # Frame 0 = initial state
            init_img = render_state(backend, obs0, n_objs, gh, gw)
            row_imgs.append(init_img)

            for k, a in enumerate(actions):
                a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a][None])
                state = rollout_step(params, state, a_oh, dummy_tokens, dummy_mask, z)
                pred_obs = _unpad_pred(state, n_objs, gh, gw)
                # _unpad_pred returns a (n_objs, gh, gw) jnp array; cast to np
                pred_obs = np.array(pred_obs)
                row_imgs.append(render_state(backend, pred_obs, n_objs, gh, gw))

            t_label = f"t={t:.2f} (z = (1-t)·{args.game_a} + t·{args.game_b})"
            rows.append(label_strip(row_imgs, t_label, font))
        panel = stack_panels(rows)
        # Top banner per panel: which level we're rolling out on.
        title = f"Donor level: {donor_label}  |  actions={actions.tolist()}"
        bbox = font.getbbox(title)
        bh = 22
        bw = max(panel.shape[1], bbox[2] - bbox[0] + 8)
        banner = PIL.Image.new("RGB", (bw, bh), (40, 40, 40))
        draw = PIL.ImageDraw.Draw(banner)
        draw.text((4, 4), title, fill=(255, 255, 255), font=font)
        if panel.shape[1] < bw:
            pad = np.zeros((panel.shape[0], bw - panel.shape[1], 3), dtype=np.uint8)
            panel = np.concatenate([panel, pad], axis=1)
        panel = np.concatenate([np.array(banner), panel], axis=0)
        panels.append(panel)

        out_path = os.path.join(save_dir, f"interp_on_{donor_label}.png")
        PIL.Image.fromarray(panel).save(out_path)
        print(f"Saved {out_path}  (shape={panel.shape})")

    # Combined panel: stack both donor panels vertically with a separator.
    if len(panels) == 2:
        sep = np.full((6, max(p.shape[1] for p in panels), 3), 60, dtype=np.uint8)
        # pad panels to common width
        mw = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            if p.shape[1] < mw:
                pp = np.zeros((p.shape[0], mw - p.shape[1], 3), dtype=np.uint8)
                p = np.concatenate([p, pp], axis=1)
            padded.append(p)
        combined = np.concatenate([padded[0], sep, padded[1]], axis=0)
        out_path = os.path.join(save_dir, "interp_combined.png")
        PIL.Image.fromarray(combined).save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
