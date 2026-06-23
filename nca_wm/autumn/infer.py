"""Shared inference helpers: load a trained Autumn WM, step it, map colors,
and drive the real engine in lock-step. Used by render_gif.py and serve_compare.py.
"""
import json
import os

import numpy as np
import torch
import matplotlib.colors as mcolors

from nca_wm.autumn.model import (AutumnNCA, RecurrentAutumnNCA, encode_batch,
                                 action_to_fields)
from nca_wm.autumn.model import _onehot, N_ATYPES, ATYPE_IDX
import torch.nn.functional as F
from nca_wm.autumn.collect import AutumnGame


def load_run(save_dir, ckpt="model_best.pt", device="cuda:0"):
    cfg = json.load(open(os.path.join(save_dir, "config.json")))
    dev = device if torch.cuda.is_available() else "cpu"
    if cfg.get("recurrent"):
        model = RecurrentAutumnNCA(cfg["n_colors"], n_hid=cfg["n_hid"],
                                   n_micro=cfg["n_micro"], pool=cfg.get("pool")).to(dev)
    else:
        model = AutumnNCA(cfg["n_colors"], n_hid=cfg["n_hid"], n_steps=cfg["n_steps"],
                          global_pool=cfg["global_pool"], history=cfg.get("history", 0)).to(dev)
    model.load_state_dict(torch.load(os.path.join(save_dir, ckpt), map_location=dev))
    model.eval()
    cfg["_device"] = dev
    if cfg.get("multihot"):
        attach_object_display(cfg, cfg["game"])
    return model, cfg


def attach_object_display(cfg, game):
    """For multi-hot object models: per-channel display color + background."""
    vocab = list(cfg["vocab"])
    env = AutumnGame(game)
    d = json.loads(env.itp.render_all())
    colors = []
    fallback = ["red", "gold", "blue", "mediumpurple", "green", "orange", "pink"]
    for i, k in enumerate(vocab):
        if k in d and d[k]:
            colors.append(d[k][0].get("color", fallback[i % len(fallback)]))
        else:
            colors.append({"bullets": "mediumpurple"}.get(k, fallback[i % len(fallback)]))
    cfg["display_colors"] = colors
    cfg["bg"] = env.bg


def object_engine_board(env, n_ch, vocab):
    """Engine state -> multi-hot (n_ch,H,W) over object-type channels."""
    from nca_wm.autumn.objects import render_objects
    vd = {k: i for i, k in enumerate(vocab)}
    g = render_objects(env, vd)
    if g.shape[0] < n_ch:
        out = np.zeros((n_ch, g.shape[1], g.shape[2]), dtype=np.uint8)
        out[:g.shape[0]] = g
        return out
    return g[:n_ch]


def object_display(board, cfg):
    """Multi-hot (C,H,W) -> (H,W) display color names (lowest active channel wins)."""
    C, H, W = board.shape
    cols = cfg["display_colors"]; bg = cfg["bg"]
    out = [[bg] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            active = np.nonzero(board[:, y, x])[0]
            if len(active):
                out[y][x] = cols[int(active[0])]
    return out


@torch.no_grad()
def object_step(model, cfg, board, action, prev_board, h, sample=False):
    """One step for a multi-hot object model. Returns (next_board (C,H,W), h).
    sample=True draws each channel Bernoulli(p) instead of thresholding at 0.5."""
    from nca_wm.autumn.objects import encode
    dev = cfg["_device"]
    ai, cx, cy = action_to_fields(action)
    if cfg.get("recurrent"):
        if h is None:
            h = model.init_hidden(1, board.shape[1], board.shape[2], dev)
        st, ao, cm, _ = encode(board[None], [ai], [cx], [cy], dev)
        logits, h = model.step(st, ao, cm, h)
    else:
        hist = prev_board[None] if cfg.get("history", 0) else None
        st, ao, cm, ht = encode(board[None], [ai], [cx], [cy], dev, hist=hist)
        logits = model(st, ao, cm, ht)
    p = torch.sigmoid(logits)[0]
    out = torch.bernoulli(p) if sample else (p > 0.5).float()
    return out.cpu().numpy().astype(np.uint8), h


@torch.no_grad()
def recurrent_step(model, cfg, board, action, h):
    """Stateful step for recurrent models. board (H,W); h (1,n_hid,H,W) or None.
    Returns (next_board (H,W) uint8, new_h)."""
    dev = cfg["_device"]
    H, W = board.shape
    if h is None:
        h = model.init_hidden(1, H, W, dev)
    ai, cx, cy = action_to_fields(action)
    onehot = _onehot(board[None], cfg["n_colors"], dev)
    at = torch.tensor([ai], device=dev)
    atype_oh = F.one_hot(at, N_ATYPES).float()
    click_map = torch.zeros(1, 1, H, W, device=dev)
    if ai == ATYPE_IDX["click"]:
        click_map[0, 0, cy, cx] = 1.0
    logits, h = model.step(onehot, atype_oh, click_map, h)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8), h


@torch.no_grad()
def wm_step(model, cfg, board, action, prev_board=None, sample=False):
    """board: (H,W) uint8 indices. action: ('noop',)/('click',x,y)/('up',)...
    prev_board: previous frame for history models (defaults to board = stationary).
    sample=True draws each cell from its predicted color distribution (shows
    aleatoric outcomes, e.g. random food spawns) instead of taking the argmax/mode."""
    dev = cfg["_device"]
    ai, cx, cy = action_to_fields(action)
    hist = None
    if cfg.get("history", 0):
        p = board if prev_board is None else prev_board
        hist = [p[None]]
    onehot, atype_oh, click_map, hist_oh = encode_batch(
        board[None], [ai], [cx], [cy], cfg["n_colors"], dev, hist_states=hist)
    logits = model(onehot, atype_oh, click_map, hist_oh)[0]   # (C,H,W)
    if sample:
        C, H, W = logits.shape
        probs = torch.softmax(logits, 0).permute(1, 2, 0).reshape(-1, C)
        out = torch.multinomial(probs, 1).reshape(H, W)
        return out.cpu().numpy().astype(np.uint8)
    return logits.argmax(0).cpu().numpy().astype(np.uint8)


def palette_rgb(palette):
    """(C,3) float RGB for a palette of CSS color names ('black','lightpink',...)."""
    rgb = []
    for name in palette:
        if name == "transparent":
            rgb.append((0, 0, 0))
        else:
            rgb.append(mcolors.to_rgb(name))
    return np.array(rgb, dtype=np.float32)


def board_to_rgb(board, rgb):
    return rgb[board]  # (H,W,3)


# --- real engine driver (ground truth) ---
def engine_board(env, palette_index):
    g = env.color_grid()
    gs = len(g)
    out = np.zeros((gs, gs), dtype=np.uint8)
    for y in range(gs):
        for x in range(gs):
            out[y, x] = palette_index.get(g[y][x], 0)
    return out


def make_engine(game, seed=0):
    return AutumnGame(game, seed=seed)


def engine_apply(env, action):
    env.apply(action)
