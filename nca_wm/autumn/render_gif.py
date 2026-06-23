#!/usr/bin/env python3
"""Render a side-by-side GIF: real Autumn engine vs. NCA world model.

Both start from the same board; the engine gives ground truth while the WM
predicts autoregressively (feeding its own output back). A third panel tints
cells where the two disagree.

Usage:
    python -m nca_wm.autumn.render_gif --save_dir nca_wm/autumn/runs/gameOfLife \
        --demo glider --steps 14 --out nca_wm/autumn/figures/gameOfLife_glider.gif
"""
import argparse
import os

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nca_wm.autumn import infer


def demo_sequence(demo, cfg):
    """Returns (setup_actions, test_actions). Clicks are ('click',x,y)."""
    if demo.startswith("mario") or cfg["game"] == "mario":
        if demo == "mario_play":
            test = [("right",), ("right",), ("up",), ("noop",), ("left",), ("noop",),
                    ("up",), ("right",), ("noop",), ("click", 0, 0), ("noop",), ("noop",)]
        else:  # mario_patrol: noops let the enemy patrol — exposes direction blindspot
            test = [("noop",)] * 18
        return [], test
    if cfg["game"] == "paint":
        # paint a rainbow: cycle color with UP, then click — tests hidden currColor
        g = cfg["grid_size"]; test = []
        for i, x in enumerate(range(2, g - 2, 2)):
            test += [("up",), ("click", x, 4 + i)]
        return [], test
    btn = cfg["buttons"]
    nxt = tuple(btn["buttonNext"]) if "buttonNext" in btn else None
    rst = tuple(btn["buttonReset"]) if "buttonReset" in btn else None
    if nxt is None:
        # generic button-less game (e.g. sand): drop a few cells, watch evolution
        gs = cfg["grid_size"]
        c = gs // 2
        test = []
        for (x, y) in [(c, 1), (c - 1, 1), (c + 1, 1), (c, 2)]:
            test += [("click", x, y), ("noop",)]
        test += [("noop",)] * 8
        return [], test
    setup = []
    if rst:
        setup.append(("click", rst[0], rst[1]))  # clear board
    if demo == "glider":
        for (x, y) in [(2, 1), (3, 2), (1, 3), (2, 3), (3, 3)]:
            setup.append(("click", x, y))
        test = [("click", *nxt) for _ in range(14)]
    elif demo == "blinker":
        for (x, y) in [(7, 7), (8, 7), (9, 7)]:
            setup.append(("click", x, y))
        test = [("click", *nxt) for _ in range(8)]
    elif demo == "rpento":
        for (x, y) in [(7, 6), (8, 6), (6, 7), (7, 7), (7, 8)]:
            setup.append(("click", x, y))
        test = [("click", *nxt) for _ in range(16)]
    else:  # mixed: place some, step, place, step
        g = cfg["grid_size"]
        test = [("click", 5, 5), ("click", 6, 5), ("click", 5, 6), ("click", *nxt),
                ("click", *nxt), ("noop",), ("click", g - 6, g - 6), ("click", *nxt)]
    return setup, test


def _obj_palette(cfg):
    return list(dict.fromkeys([cfg["bg"]] + list(cfg["display_colors"])))


def _obj_to_idx(board, cfg, pal):
    names = infer.object_display(board, cfg)
    idx = {n: i for i, n in enumerate(pal)}
    H = len(names); W = len(names[0])
    out = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            out[y, x] = idx.get(names[y][x], 0)
    return out


def rollout_objects(game, model, cfg, setup, test, seed=0):
    C = cfg["n_colors"]; vocab = list(cfg["vocab"]); pal = _obj_palette(cfg)
    env = infer.make_engine(game, seed=seed)
    for a in setup:
        infer.engine_apply(env, a)
    b0 = infer.object_engine_board(env, C, vocab)
    eng_frames = [_obj_to_idx(b0, cfg, pal)]
    wm_frames = [_obj_to_idx(b0, cfg, pal)]
    acts = [None]
    prev, cur, h = b0.copy(), b0.copy(), None
    for a in test:
        infer.engine_apply(env, a)
        eng_frames.append(_obj_to_idx(infer.object_engine_board(env, C, vocab), cfg, pal))
        nxt, h = infer.object_step(model, cfg, cur, a, prev, h)
        prev, cur = cur, nxt
        wm_frames.append(_obj_to_idx(cur, cfg, pal))
        acts.append(a)
    return eng_frames, wm_frames, acts, pal


def rollout(game, model, cfg, setup, test, seed=0):
    pal_index = {name: i for i, name in enumerate(cfg["palette"])}
    env = infer.make_engine(game, seed=seed)
    for a in setup:
        infer.engine_apply(env, a)
    b0 = infer.engine_board(env, pal_index)
    eng_frames, wm_frames, acts = [b0.copy()], [b0.copy()], [None]
    wm_board, wm_prev, h = b0.copy(), b0.copy(), None
    recurrent = cfg.get("recurrent")
    for a in test:
        infer.engine_apply(env, a)
        eng_frames.append(infer.engine_board(env, pal_index))
        if recurrent:
            nxt, h = infer.recurrent_step(model, cfg, wm_board, a, h)
        else:
            nxt = infer.wm_step(model, cfg, wm_board, a, prev_board=wm_prev)
        wm_prev, wm_board = wm_board, nxt
        wm_frames.append(wm_board.copy())
        acts.append(a)
    return eng_frames, wm_frames, acts


def act_str(a):
    if a is None:
        return "initial"
    if a[0] == "click":
        return f"click({a[1]},{a[2]})"
    return a[0]  # noop / up / down / left / right


def render(eng, wm, acts, palette, out, fps=2):
    rgb = infer.palette_rgb(palette)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames = []
    for t, (e, w, a) in enumerate(zip(eng, wm, acts)):
        fig, axs = plt.subplots(1, 3, figsize=(11, 4.2))
        diff = (e != w)
        overlay = infer.board_to_rgb(w, rgb).copy()
        overlay[diff] = np.array([1.0, 0.0, 0.0])
        for ax, img, title in zip(
            axs, [infer.board_to_rgb(e, rgb), infer.board_to_rgb(w, rgb), overlay],
            ["Engine (truth)", "World Model", f"Disagree: {int(diff.sum())} cells"],
        ):
            ax.imshow(img, interpolation="nearest")
            ax.set_title(title, fontsize=15)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"frame {t}  |  action: {act_str(a)}", fontsize=17)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    total = sum(int((e != w).any()) for e, w in zip(eng, wm))
    print(f"saved -> {out}  ({len(frames)} frames; {total}/{len(frames)} frames with any disagreement)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--game", default=None)
    ap.add_argument("--demo", default="glider",
                    choices=["glider", "blinker", "rpento", "mixed",
                             "mario_patrol", "mario_play", "generic"])
    ap.add_argument("--ckpt", default="model_best.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=2)
    args = ap.parse_args()

    model, cfg = infer.load_run(args.save_dir, ckpt=args.ckpt, device=args.device)
    game = args.game or cfg["game"]
    setup, test = demo_sequence(args.demo, cfg)
    out = args.out or f"nca_wm/autumn/figures/{game}_{args.demo}.gif"
    if cfg.get("multihot"):
        eng, wm, acts, pal = rollout_objects(game, model, cfg, setup, test)
        render(eng, wm, acts, pal, out, fps=args.fps)
    else:
        eng, wm, acts = rollout(game, model, cfg, setup, test)
        render(eng, wm, acts, cfg["palette"], out, fps=args.fps)


if __name__ == "__main__":
    main()
