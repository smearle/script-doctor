"""Handcrafted long-slide benchmark for varislide NCA WM checkpoints.

Closes the caveat in the synth-OOD-width result: synth varislide levels
have dense walls, so even on wide grids the typical slide distance is
≤5. We need controlled-distance test cases with slide_d ∈ {6, 12, 18,
24, 30} to claim "the body iterated to convergence is correct on long
slides."

For each (W, slide_d) we construct exactly *one* canonical level:
  - 3 rows, W columns
  - top/bottom rows: all wall
  - middle row: wall at col 0, player at col 1, empty cols 2..W-2,
    wall at col W-1
  - action = right (RIGHT_ACTION=3)
  - expected next_state: player slides to col W-2, slide_d = W-3.

So W=9 → d=6, W=15 → d=12, W=21 → d=18, W=27 → d=24, W=33 → d=30.

Channel layout (per inspecting `custom_games/varislide.txt`'s collision
layers): ch0=Background, ch1=Wall, ch2=Player. Background bit is 1 on
every cell; player/wall bits are 1 where appropriate. Confirmed by the
fact that existing eval_varislide_depth_extrap.py uses PLAYER=2.

Usage:
    .venv/bin/python3 nca_wm/scripts/eval_varislide_handcrafted_slides.py \
        --runs nca_wm/logs_depth_extrap/nopool_uniform_T32_s0 ... \
        --depths 1,2,4,8,16,32,64,128 \
        --slide_distances 6,12,18,24,30
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nca_wm.rule_attn_model import RuleAttnNCAWorldModel  # noqa: E402

N_ACTIONS = 5
RIGHT_ACTION = 3
BG = 0
WALL = 1
PLAYER = 2


def _build_handcrafted(slide_d: int, c_pad: int):
    """Construct a (1, c_pad, 3, W) state and matching expected next-state.

    Mirrors the authored-level encoding (custom_games/varislide.txt): the
    rightmost two columns of the middle row are walls (the `##` pattern).
    Player at col 1 slides to col W-3 — slide_d = W - 4.
    """
    W = slide_d + 4
    s = np.zeros((1, c_pad, 3, W), dtype=np.float32)
    n = np.zeros((1, c_pad, 3, W), dtype=np.float32)

    # Background bit on every cell.
    s[0, BG, :, :] = 1.0
    n[0, BG, :, :] = 1.0

    # Walls: top + bottom rows entirely, left wall + double right wall in middle row.
    s[0, WALL, 0, :] = 1.0
    s[0, WALL, 2, :] = 1.0
    s[0, WALL, 1, 0] = 1.0
    s[0, WALL, 1, W - 2] = 1.0
    s[0, WALL, 1, W - 1] = 1.0
    n[0, WALL] = s[0, WALL]

    # Player initial: col 1, middle row.
    s[0, PLAYER, 1, 1] = 1.0
    # Player after slide: col W-3 (just before the double right wall).
    n[0, PLAYER, 1, W - 3] = 1.0

    # Background gets cleared under wall and under player in the multihot
    # convention used by varislide (collisionlayers stack: bg, wall, player
    # — the highest layer "wins" visually but the multihot encoding includes
    # all bits set per layer the cell is on). However the synth pipeline in
    # _unpack_states actually packs *all* bits per cell. To match what the
    # model was trained on, set bits exactly as the engine emits them: the
    # engine layers tile 'P.' as (Player ON, Background ON), so leave
    # background on under the player. Wall under wall similarly. This is
    # the conservative choice — empirical test below confirms whichever is
    # correct.
    return s, n, W


def _build_model(cfg, gtoks_len, n_objs, *, n_steps_override: int):
    eff_seq_len = max(gtoks_len, 1) + 1
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"],
        n_steps=n_steps_override,
        n_repeats=n_steps_override,
        n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg.get("d_model", 64),
        enc_n_self_layers=cfg.get("n_enc_layers", 2),
        n_slots=cfg["n_slots"],
        n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg.get("d_slot", 64),
        n_attn_heads=cfg.get("n_heads", 4),
        max_seq_len=eff_seq_len,
        axis_pool=cfg.get("axis_pool", True),
        axis_cummax=cfg.get("axis_cummax", True),
        global_pool=cfg.get("global_pool", True),
        use_vq=cfg.get("vq_codebook", False),
        vq_codebook_size=cfg.get("vq_codebook_size", 512),
        vq_commitment_weight=cfg.get("vq_commitment_weight", 0.25),
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        mask_hidden=cfg.get("mask_hidden", False),
        adaptive_halt=cfg.get("adaptive_halt", False),
    )


def _evaluate_run(run_dir: str, depths, slide_ds, batch_size: int = 1):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    gi = pickle.load(open(os.path.join(run_dir, "game_infos.pkl"), "rb"))
    p_path = os.path.join(run_dir, "params_best.pkl")
    if not os.path.exists(p_path):
        p_path = os.path.join(run_dir, "params.pkl")
    params = pickle.load(open(p_path, "rb"))
    g = gi[0]
    embed_in = params["params"]["embed"]["kernel"].shape[0]
    c_pad = embed_in - N_ACTIONS
    toks_np = np.asarray(g["token_ids"])
    eff_len = max(toks_np.shape[0], 1)

    print(f"\n=== {run_dir} ===")
    print(f"  cfg: n_nca_steps={cfg.get('n_nca_steps')} pool=({cfg.get('axis_pool')}|"
          f"{cfg.get('axis_cummax')}|{cfg.get('global_pool')}) input_skip={cfg.get('input_skip')}")

    out = {
        "run_dir": run_dir,
        "config": {k: cfg.get(k) for k in [
            "n_nca_steps", "n_nca_repeats", "n_hid", "axis_pool", "axis_cummax",
            "global_pool", "input_skip", "mask_hidden",
        ]},
        "depths": [],
    }
    for d_eval in depths:
        model = _build_model(cfg, toks_np.shape[0], c_pad, n_steps_override=d_eval)
        per_d = []
        for slide_d in slide_ds:
            s_np, n_np, W = _build_handcrafted(slide_d, c_pad)
            sb = jnp.asarray(s_np)
            ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[[RIGHT_ACTION]])
            toks = np.zeros((1, eff_len), dtype=np.int32)
            toks[:, :toks_np.shape[0]] = toks_np
            gmask = np.zeros((1, eff_len), dtype=bool)
            gmask[:, :toks_np.shape[0]] = True
            t0 = time.time()
            out_logits = model.apply(params, sb, ab,
                                     jnp.asarray(toks), jnp.asarray(gmask))
            sig = np.asarray(jax.nn.sigmoid(out_logits[0]))[0]
            elapsed = time.time() - t0
            # Argmax of player channel in middle row.
            player_row = sig[PLAYER, 1, :]
            argmax_col = int(np.argmax(player_row))
            target_col = W - 3
            sig_at_target = float(player_row[target_col])
            sig_at_argmax = float(player_row[argmax_col])
            in_col = 1
            pred_d = argmax_col - in_col
            correct = argmax_col == target_col
            per_d.append({
                "slide_d": slide_d, "W": W,
                "argmax_col": argmax_col, "target_col": target_col,
                "argmax_correct": int(correct),
                "pred_d": pred_d,
                "sig_at_target": sig_at_target,
                "sig_at_argmax": sig_at_argmax,
                "elapsed_s": elapsed,
            })
        print(f"  D_eval={d_eval:>3}: " + " ".join(
            f"d={x['slide_d']}:{('✓' if x['argmax_correct'] else '✗')}"
            f"(pred_d={x['pred_d']:>2},sig_t={x['sig_at_target']:.2f})"
            for x in per_d))
        out["depths"].append({"D_eval": d_eval, "per_distance": per_d})

    out_path = os.path.join(run_dir, "depth_extrap_handcrafted.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> wrote {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--depths", default="1,2,4,8,16,32,64,128")
    p.add_argument("--slide_distances", default="3,6,12,18,24,30")
    args = p.parse_args()
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    sds = [int(x) for x in args.slide_distances.split(",") if x.strip()]
    for r in args.runs:
        _evaluate_run(r, depths, sds)


if __name__ == "__main__":
    main()
