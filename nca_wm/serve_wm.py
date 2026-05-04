"""Serve a trained NCA world model from a saved run dir.

Usage:
    python nca_wm/serve_wm.py --load nca_wm/logs/scaling_4_long_v1

Reads config.json, params.pkl, and game_infos.pkl from the run dir, rebuilds the
model at the trained recipe, and launches the interactive web app from
nca_wm.serve.serve_world_model.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzlescript_jax.utils import init_ps_lark_parser
from nca_wm.serve import serve_world_model


def _unwrap_wm(params):
    """Joint checkpoints are {'wm': ..., 'dec': ...}; WM-only is the param dict."""
    if isinstance(params, dict) and {"wm", "dec"} <= set(params.keys()):
        return params["wm"]
    return params


def _build_wm(cfg, game_infos):
    """Reconstruct just the world-model module (we don't need the decoder for serving)."""
    arch = cfg.get("architecture", "rule_attn")
    max_C = max(g["n_objs"] for g in game_infos)
    max_tok_len = max(max((len(g.get("token_ids", [])) for g in game_infos), default=1), 1)
    vocab_size = int(cfg["vocab_size"])
    pool_kwargs = dict(
        axis_pool=cfg.get("axis_pool", False),
        axis_cummax=cfg.get("axis_cummax", False),
        global_pool=cfg.get("global_pool", False),
    )

    if arch == "rule_attn":
        from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
        return RuleAttnNCAWorldModel(
            n_hid=cfg["n_hid"],
            n_steps=cfg["n_nca_steps"],
            n_out=max_C,
            vocab_size=vocab_size + 1,
            enc_d_model=cfg["d_model"],
            enc_n_self_layers=cfg["n_enc_layers"],
            n_slots=cfg["n_slots"],
            n_app_slots=cfg.get("n_app_slots", 0),
            d_slot=cfg["d_slot"],
            n_attn_heads=cfg["n_heads"],
            max_seq_len=max_tok_len + 1,
            n_repeats=cfg.get("n_nca_repeats", 1),
            mask_hidden=cfg.get("mask_hidden", False),
            use_layernorm=cfg.get("use_layernorm", False),
            input_skip=cfg.get("input_skip", False),
            adaptive_halt=cfg.get("adaptive_halt", False),
            use_vq=cfg.get("vq_codebook", False),
            vq_codebook_size=cfg.get("vq_codebook_size", 512),
            vq_commitment_weight=cfg.get("vq_commitment_weight", 0.25),
            **pool_kwargs,
        ), max_tok_len
    if arch == "film":
        from nca_wm.train import ConditionalNCAWorldModel
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
            sprite_decoder=False,
            use_layernorm=cfg.get("use_layernorm", False),
            **pool_kwargs,
        ), max_tok_len
    raise NotImplementedError(
        f"serve_wm only supports rule_attn / film conditional architectures; "
        f"got {arch!r}. (Single-game unconditional checkpoints have a different "
        f"shape — serve them via train.py --serve for now.)"
    )


def main():
    p = argparse.ArgumentParser(description="Serve a trained NCA world model.")
    p.add_argument("--load", required=True, metavar="DIR",
                   help="Run directory containing config.json, params.pkl, "
                        "and game_infos.pkl.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--game", default=None,
                   help="Initial game name (default: first game in checkpoint).")
    p.add_argument("--level", type=int, default=0)
    args = p.parse_args()

    with open(os.path.join(args.load, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(args.load, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(args.load, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)

    wm_params = _unwrap_wm(params)
    model, max_tok_len = _build_wm(cfg, game_infos)

    initial_game_id = 0
    if args.game is not None:
        for i, info in enumerate(game_infos):
            if info["name"] == args.game:
                initial_game_id = i
                break
        else:
            names = [g["name"] for g in game_infos]
            raise ValueError(f"Game {args.game!r} not in checkpoint; available: {names}")

    max_C = max(g["n_objs"] for g in game_infos)
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)

    print(f"Serving {os.path.basename(args.load.rstrip('/'))}: "
          f"{len(game_infos)} games, arch={cfg.get('architecture','rule_attn')}, "
          f"vocab={cfg['vocab_size']}, max_pad=({max_C},{max_H},{max_W})")

    serve_world_model(
        model, wm_params, game_infos, init_ps_lark_parser(),
        initial_game_id=initial_game_id,
        level_i=args.level,
        port=args.port,
        host=args.host,
        conditional=cfg.get("conditional", True),
        max_pad=(max_C, max_H, max_W),
        max_tok_len=max_tok_len,
    )


if __name__ == "__main__":
    main()
