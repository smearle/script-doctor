#!/usr/bin/env python3
"""Montage GIFs accompanying the n_per_rule scaling figure (OOD panel).

For a held-out (OOD) game, roll the rule-conditioned NCA-WM autoregressively
under a fixed action sequence at several training-corpus sizes and lay the
frames out side by side:

    [ ground truth | cond n=1 | cond n=20 | cond n=200 ]

over time, so the prediction visibly sharpens from garbage (tiny corpus) to a
faithful rollout (large corpus) -- the OOD-generalization-improves-with-scale
story behind the right panel of figures/n_per_rule_scaling/n_per_rule_slope.

Outputs one GIF per game:
    nca_wm/paper/figures/n_per_rule_scaling/gifs/<game>_L<lvl>_scaling.gif

Usage:
    .venv/bin/python3 nca_wm/scripts/n_per_rule_scaling_gifs.py \\
        --games cyberpunk_2020_by_gzhao-jpg,monophobic_multiban_by_increpare,block_game_by_emily_dresden \\
        --n_steps 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Small workload; keep off the shared GPUs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw

from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from nca_wm.train import (
    N_ACTIONS, _enabled_action_count, _multihot_to_objects,
    _pad_state_for_model, _wm_p, make_apply_fn,
)
from nca_wm.heldout_eval import _build_heldout_game_info, _build_model, _load_run

OUT_DIR = _REPO_ROOT / "nca_wm" / "paper" / "figures" / "n_per_rule_scaling" / "gifs"

# Column presets. Each column is (header_label, run_dir); the montage rolls
# every column's model on the SAME game + action sequence next to the ground
# truth. `games` is the default Heldout-26 (OOD) or training-set (in-dist)
# game list for that preset; `suffix` keeps output files from colliding.
COND   = "n_per_rule_{n}_cond_dp_val0.10_s0"
UNCOND = "n_per_rule_{n}_uncond_h288_dp_val0.10_s0"
PRESETS = {
    # OOD generalization sharpening with corpus size (existing 3-scale GIFs).
    # ref_col=None -> deterministic BFS/seed-0 actions (n=1 diverges on any
    # rollout, so no worst-episode search needed).
    "ood_scaling": {
        "columns": [("n=1", COND.format(n=1)), ("n=10", COND.format(n=10)),
                    ("n=200", COND.format(n=200))],
        "games": ["cyberpunk_2020_by_gzhao-jpg",
                  "monophobic_multiban_by_increpare",
                  "block_game_by_emily_dresden"],
        "suffix": "scaling", "ref_col": None,
    },
    # Unconditional model, OOD generalization vs corpus size (companion to
    # ood_scaling). Same games for direct cond-vs-uncond comparison; uncond
    # tops out at n=200.
    "ood_scaling_uncond": {
        "columns": [("uncond n=1", UNCOND.format(n=1)),
                    ("uncond n=10", UNCOND.format(n=10)),
                    ("uncond n=200", UNCOND.format(n=200))],
        "games": ["cyberpunk_2020_by_gzhao-jpg",
                  "monophobic_multiban_by_increpare",
                  "block_game_by_emily_dresden"],
        "suffix": "scaling_uncond", "ref_col": None,
    },
    # Same, extended to cond's largest corpus (n=400) -- "higher n".
    "ood_scaling_n400": {
        "columns": [("n=1", COND.format(n=1)), ("n=10", COND.format(n=10)),
                    ("n=200", COND.format(n=200)), ("n=400", COND.format(n=400))],
        "games": ["cyberpunk_2020_by_gzhao-jpg",
                  "monophobic_multiban_by_increpare",
                  "block_game_by_emily_dresden"],
        "suffix": "scaling_n400", "ref_col": None,
    },
    # In-distribution conditioning benefit: cond fits, uncond dilutes (n=200).
    # Games chosen for the largest uncond-minus-cond ID gap. ID dilution is
    # sparse, so search random episodes for the worst uncond (last col)
    # divergence.
    "indist_cond_vs_uncond": {
        "columns": [("cond n=200", COND.format(n=200)),
                    ("uncond n=200", UNCOND.format(n=200))],
        "games": ["gdd301_prototype1_by_gamesatqu",
                  "Swap_Sokoban",
                  "p_vs_z_by_tdsakajasonhan"],
        "suffix": "indist_cond_vs_uncond", "ref_col": 1,
    },
    # In-distribution dilution across corpus size on a game the model fits at
    # small n but degrades on as the corpus grows (fixed capacity). Columns
    # start at n=50 so every chosen game is in-distribution (trained) at every
    # column; games all enter the corpus by n=50. Search for the worst
    # large-corpus (last col) divergence.
    "indist_dilution": {
        "columns": [("cond n=50", COND.format(n=50)),
                    ("cond n=200", COND.format(n=200)),
                    ("cond n=400", COND.format(n=400))],
        "games": ["Bad_Example", "bit_treat_by_barefootengineer1"],
        "suffix": "indist_dilution", "ref_col": 2,
    },
}

# Random episodes searched for the worst reference-column divergence when a
# preset sets ref_col.
N_SEARCH_EPISODES = 16

UPSCALE   = 6     # nearest-neighbour zoom so small grids are legible
HDR_H     = 22    # column-header strip height (px, pre-upscale-independent)
FTR_H     = 20    # per-step footer strip height
SEP       = 4     # white separator between columns
PAD_SECS  = 1.2   # hold first/last frame this long
FPS       = 2.0


def _bfs_actions(name: str, level: int, n_steps: int) -> list[int]:
    """Cached BFS/A* action sequence for the OOD game, else fixed-seed random."""
    cands = [
        _REPO_ROOT / "nca_wm" / "data_cache" / "heldout_search"
            / f"{name}_L{level}_bfs_100000_60000.npz",
        _REPO_ROOT / "rollout_data" / name / f"level_{level}"
            / "search_bfs_100000_60000.npz",
        _REPO_ROOT / "rollout_data" / name / f"level_{level}"
            / "search_astar_100000_60000.npz",
    ]
    for p in cands:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if "actions" in d.files and len(d["actions"]) > 0:
                acts = [int(a) for a in d["actions"][:n_steps]]
                if len(acts) >= n_steps:
                    return acts
    rng = np.random.default_rng(0)
    return [int(rng.integers(N_ACTIONS)) for _ in range(n_steps)]


def _next_pow2(x: int, mn: int = 8) -> int:
    v = max(mn, int(x))
    p = 1
    while p < v:
        p <<= 1
    return p


def _load_model(run_dir: str):
    """Return a column-model dict for a run (handles cond + uncond)."""
    cfg, params, game_infos = _load_run(run_dir)
    params = _wm_p(params)
    model = _build_model(cfg, game_infos)
    apply_fn = make_apply_fn(model)
    return {
        "cfg": cfg,
        "apply_fn": apply_fn,
        "params": params,
        "conditional": cfg.get("conditional", True),
        "max_C": max(g["n_objs"] for g in game_infos),
        "max_tok": max(len(g.get("token_ids", [])) for g in game_infos),
    }


def _tokens(info, max_tok):
    tids = info["token_ids"][:max_tok]
    pad_tok = np.zeros(max_tok, dtype=np.int32)
    pad_mask = np.zeros(max_tok, dtype=np.bool_)
    pad_tok[: len(tids)] = tids
    pad_mask[: len(tids)] = True
    return jnp.array(pad_tok[None]), jnp.array(pad_mask[None])


def _ar_rollout(mdl, info, real0, actions, real_seq, gt, gm):
    """Autoregressive rollout for one column-model. Returns
    (pred_obs_list, wrong_cells_list) aligned with real_seq[0..T] (entry 0 is
    the shared ground-truth s0). Handles conditional and unconditional models
    (the latter take no game tokens)."""
    apply_fn, params, max_C = mdl["apply_fn"], mdl["params"], mdl["max_C"]
    conditional = mdl["conditional"]
    n_objs, H, W = real0.shape
    H_eval = max(_next_pow2(info["H"]), _next_pow2(H))
    W_eval = max(_next_pow2(info["W"]), _next_pow2(W))
    pred_state = _pad_state_for_model(real0, max_C, H_eval, W_eval)
    preds = [real0.copy()]
    wrong = [0]
    for t, a in enumerate(actions):
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a][None])
        if conditional:
            logits, _wl, _sl = apply_fn(params, pred_state, a_oh, gt, gm)
        else:
            logits, _wl, _sl = apply_fn(params, pred_state, a_oh)
        pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        pred_obs = np.array(pred_next[0, :n_objs, :H, :W])
        preds.append((pred_obs > 0.5).astype(np.uint8))
        real_next = real_seq[t + 1]
        wc = int(((pred_obs > 0.5).astype(np.uint8) != real_next).any(axis=0).sum())
        wrong.append(wc)
        clean = jnp.zeros_like(pred_next)
        clean = clean.at[:, :n_objs, :H, :W].set(pred_next[:, :n_objs, :H, :W])
        pred_state = clean
    return preds, wrong


def _zoom(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame)
    return np.asarray(img.resize((img.width * UPSCALE, img.height * UPSCALE),
                                 Image.NEAREST))


def _compose(gt_frame, pred_frames, labels, wrongs, total_cells, step_text):
    """Stack [GT | pred_0 | pred_1 | ...] with header labels and a footer."""
    tiles = [_zoom(gt_frame)] + [_zoom(p) for p in pred_frames]
    h = tiles[0].shape[0]
    w = tiles[0].shape[1]
    n = len(tiles)
    canvas_w = n * w + (n - 1) * SEP
    canvas = np.full((HDR_H + h + FTR_H, canvas_w, 3), 255, dtype=np.uint8)
    xs = []
    x = 0
    for tile in tiles:
        canvas[HDR_H:HDR_H + h, x:x + w] = tile
        xs.append(x)
        x += w + SEP
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    col_labels = ["ground truth"] + labels
    for xi, lab, wc in zip(xs, col_labels, [None] + wrongs):
        draw.text((xi + 3, 5), lab, fill=(0, 0, 0))
        if wc is not None:
            pct = 100.0 * wc / max(1, total_cells)
            color = (0, 130, 0) if wc == 0 else (200, 0, 0)
            draw.text((xi + 3, HDR_H + h + 4),
                      f"{pct:4.1f}% off", fill=color)
    draw.text((xs[0] + 3, HDR_H + h + 4), step_text, fill=(0, 0, 0))
    return np.asarray(img)


def _env_rollout(json_str, level, actions, n_steps):
    """Step the engine along `actions`; return (real_seq, used_actions)."""
    env = CppPuzzleScriptEnv(json_str, level_i=level,
                             max_episode_steps=n_steps + 4)
    real0, _ = env.reset()
    real_seq = [real0.copy()]
    used = []
    for a in actions:
        real_next, _, done, trunc, _ = env.step(a)
        real_seq.append(real_next.copy())
        used.append(a)
        if done or trunc:
            break
    return real_seq, used


def _select_actions(name, level, n_steps, info, models, columns, ref_col):
    """Choose the action sequence to render. If ref_col is None, use cached
    BFS / seed-0 random. Otherwise sample N_SEARCH_EPISODES random sequences
    and keep the one that maximizes the reference column's AR divergence, so
    sparse in-distribution dilution is actually exercised in the GIF."""
    json_str = info["json_str"]
    n_act = _enabled_action_count(json_str)
    if ref_col is None:
        actions = _bfs_actions(name, level, n_steps)
        real_seq, used = _env_rollout(json_str, level, actions, n_steps)
        return real_seq, used
    ref_mdl = models[ref_col]
    gt, gm = _tokens(info, ref_mdl["max_tok"])
    best = None
    for seed in range(N_SEARCH_EPISODES):
        rng = np.random.default_rng(seed)
        cand = [int(rng.integers(n_act)) for _ in range(n_steps)]
        real_seq, used = _env_rollout(json_str, level, cand, n_steps)
        _preds, wrong = _ar_rollout(ref_mdl, info, real_seq[0], used, real_seq,
                                    gt, gm)
        score = max(wrong)
        if best is None or score > best[0]:
            best = (score, real_seq, used)
    return best[1], best[2]


def render_game(name, level, n_steps, models, columns, suffix, ref_col, parser):
    print(f"\n=== {name} L{level} ===")
    # Build game info once (works for OOD and in-dist games alike; token
    # encoding from the first model's cfg -- all sweep cfgs share defaults).
    cfg0 = models[0]["cfg"]
    info = _build_heldout_game_info(
        name, parser,
        encode_sprites=cfg0.get("encode_sprites", False),
        kernel_sep=cfg0.get("kernel_sep", False),
    )
    if info is None:
        print("  build failed; skipping")
        return None

    backend = CppPuzzleScriptBackend()
    backend.compile_game(parser, name)

    real_seq, used_actions = _select_actions(
        name, level, n_steps, info, models, columns, ref_col)
    real0 = real_seq[0]
    T = len(used_actions)
    n_objs, H, W = real0.shape
    total_cells = H * W

    # Per-model AR rollouts on the identical action sequence.
    all_preds, all_wrong, labels = [], [], []
    for mdl, (lab, _rd) in zip(models, columns):
        gt, gm = _tokens(info, mdl["max_tok"])
        preds, wrong = _ar_rollout(mdl, info, real0, used_actions, real_seq,
                                   gt, gm)
        all_preds.append(preds)
        all_wrong.append(wrong)
        labels.append(lab)
        print(f"  {lab:>14}: final {100*wrong[-1]/total_cells:5.1f}% off "
              f"(max {100*max(wrong)/total_cells:5.1f}%)")

    # Compose frames.
    frames = []
    for t in range(T + 1):
        gt_frame = backend.render_frame_from_objects(
            _multihot_to_objects(real_seq[t]), W, H)
        pred_frames = [backend.render_frame_from_objects(
            _multihot_to_objects(all_preds[mi][t]), W, H)
            for mi in range(len(models))]
        wrongs_t = [all_wrong[mi][t] for mi in range(len(models))]
        frames.append(_compose(gt_frame, pred_frames, labels, wrongs_t,
                               total_cells, f"t={t}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}_L{level}_{suffix}.gif"
    durations = [PAD_SECS if (i == 0 or i == len(frames) - 1) else 1.0 / FPS
                 for i in range(len(frames))]
    imageio.mimsave(str(out_path), frames, duration=durations, loop=0)
    print(f"  wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="ood_scaling", choices=list(PRESETS),
                    help="column set + default game list (see PRESETS)")
    ap.add_argument("--games", default=None,
                    help="comma-separated game override (else preset default)")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--n_steps", type=int, default=10)
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    columns = preset["columns"]
    suffix = preset["suffix"]
    ref_col = preset["ref_col"]
    games = ([g.strip() for g in args.games.split(",") if g.strip()]
             if args.games else preset["games"])

    parser = init_ps_lark_parser()
    print(f"Preset '{args.preset}': loading {len(columns)} column models...")
    models = []
    for lab, run_dir in columns:
        full = str(_REPO_ROOT / "nca_wm" / "logs" / run_dir)
        print(f"  {lab}: {run_dir}")
        models.append(_load_model(full))

    written = []
    for name in games:
        p = render_game(name, args.level, args.n_steps, models, columns,
                        suffix, ref_col, parser)
        if p:
            written.append(p)
    print("\nGIFs written:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
