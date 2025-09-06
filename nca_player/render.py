# NCA action-field renderer for grid games (PuzzleScript-like)
# Generates both "vector-field hue+arrows" and "gamepad glyph" visualizations,
# with optional targets overlay, and can export PNG/GIF frames.
#
# Usage example at bottom creates demo PNGs and GIFs in /mnt/data.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from PIL import Image, ImageSequence
import imageio.v2 as imageio
import os
from typing import Optional, Tuple

# ---------------------------------
# Helpers
# ---------------------------------

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _soft_dir_vector(probs_4: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    probs_4: (..., 4) ordered as [down, left, right, up]
    Returns: hue (0..1), magnitude (0..1), vector field v (...,2)
    """
    dirs = np.array([[0,1], [-1,0], [1,0], [0,-1]], dtype=np.float32)  # down,left,right,up
    v = probs_4 @ dirs  # (...,2)
    mag = np.linalg.norm(v, axis=-1, keepdims=False)
    ang = np.arctan2(-v[...,1], v[...,0])  # [-pi,pi], negative y=up
    hue = (np.degrees(ang) % 360) / 360.0
    return hue, mag, v

def _entropy(p, eps=1e-9):
    p = np.clip(p, eps, 1.0)
    H = -np.sum(p * np.log(p), axis=-1)
    H /= np.log(p.shape[-1])  # normalize to 0..1
    return H

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """ hsv in [0,1], returns rgb in [0,1] """
    return matplotlib.colors.hsv_to_rgb(hsv)

# ---------------------------------
# Vector-field (Hue) renderer
# ---------------------------------

def render_vector_field(
    probs: np.ndarray,
    targets: np.ndarray = None,
    grid_overlay: np.ndarray = None,
    cell_px: int = 32,
    save_path: str = None,
    show: bool = False,
):
    H, W, C = probs.shape
    moves = probs[..., :4]
    action_p = probs[..., 4]

    hue, mag, v = _soft_dir_vector(moves)
    maxp = probs.max(axis=-1)
    conf = maxp * (1.0 - _entropy(probs))

    Hh = hue
    Ss = np.clip(mag, 0, 1) * 0.9
    Vv = 0.35 + 0.65 * conf
    Ss = np.where(action_p > 0.5, 0.0, Ss)
    hsv = np.stack([Hh, Ss, Vv], axis=-1)
    rgb = hsv_to_rgb(hsv)

    fig_w = W * cell_px / 100.0
    fig_h = H * cell_px / 100.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = plt.gca()
    ax.set_axis_off()

    # Fill entire axes with data (no margins)
    ax.set_position([0, 0, 1, 1])

    if grid_overlay is not None:
        palette = {
            0: np.array([1.0, 1.0, 1.0]),
            1: np.array([0.8, 0.8, 0.85]),
            2: np.array([0.85, 0.8, 0.8]),
            3: np.array([0.8, 0.85, 0.8]),
        }
        bg = np.zeros((H, W, 3), dtype=np.float32)
        for k, col in palette.items():
            mask = (grid_overlay == k)[..., None]
            bg = np.where(mask, col, bg)
        ax.imshow(bg, interpolation="nearest", origin="upper", extent=(0, W, H, 0))

    ax.imshow(rgb, interpolation="nearest", origin="upper", extent=(0, W, H, 0))

    # draw glyphs
    arrow_scale = 0.35
    for y in range(H):
        for x in range(W):
            vx, vy = v[y, x]
            ap = action_p[y, x]
            fx = x + 0.5
            fy = y + 0.5
            if ap > 0.5:
                circ = Circle((fx, fy), 0.18, edgecolor=None, facecolor="white", alpha=0.9)
                ax.add_patch(circ)
                circ2 = Circle((fx, fy), 0.10, edgecolor=None, facecolor="black", alpha=0.9)
                ax.add_patch(circ2)
            else:
                if np.hypot(vx, vy) > 1e-3:
                    tx = fx + vx * arrow_scale
                    ty = fy - vy * arrow_scale
                    arr = FancyArrow(fx, fy, tx - fx, ty - fy, width=0.03,
                                     length_includes_head=True, head_width=0.25, head_length=0.25,
                                     color="black", alpha=0.9)
                    ax.add_patch(arr)

    if targets is not None:
        tm = targets[..., :4]
        ta = targets[..., 4]
        _, _, v_t = _soft_dir_vector(tm)
        for y in range(H):
            for x in range(W):
                fx = x + 0.5
                fy = y + 0.5
                if ta[y, x] > 0.5:
                    ring = Circle((fx, fy), 0.22, edgecolor="black", facecolor="none", linewidth=1.2, alpha=0.9)
                    ax.add_patch(ring)
                else:
                    tx, ty = v_t[y, x]
                    if np.hypot(tx, ty) > 1e-3:
                        ex = fx + tx * arrow_scale * 0.9
                        ey = fy - ty * arrow_scale * 0.9
                        arr = FancyArrow(fx, fy, ex - fx, ey - fy, width=0.01,
                                         length_includes_head=True, head_width=0.18, head_length=0.18,
                                         color="white", alpha=0.9)
                        ax.add_patch(arr)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)

# ---------------------------------
# Gamepad-glyph renderer (sub-cells)
# ---------------------------------

def render_gamepad_glyphs(
    probs: np.ndarray,
    targets: Optional[np.ndarray] = None,
    grid_overlay: Optional[np.ndarray] = None,
    cell_px: int = 32,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Each cell is divided into 5 sub-cells laid out like a D-pad:
        [ ][U][ ]
        [L][A][R]
        [ ][D][ ]
    Fill each with a fixed palette scaled by probability.
    """
    H, W, C = probs.shape
    assert C == 5
    # Palette (colorblind-friendly-ish). These are for rendering (not "charts")
    # You can customize later.
    palette = np.array([
        [0.85, 0.30, 0.30],  # Down
        [0.35, 0.60, 0.35],  # Left
        [0.95, 0.70, 0.25],  # Right
        [0.30, 0.45, 0.85],  # Up
        [0.50, 0.50, 0.50],  # Action (center)
    ], dtype=np.float32)

    fig_w = W * cell_px / 100.0
    fig_h = H * cell_px / 100.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = plt.gca()
    ax.set_axis_off()

    # Optional faint background
    if grid_overlay is not None:
        bg_palette = {
            0: np.array([1.0, 1.0, 1.0]),
            1: np.array([0.92, 0.92, 0.95]),
            2: np.array([0.95, 0.92, 0.92]),
            3: np.array([0.92, 0.95, 0.92]),
        }
        bg = np.zeros((H, W, 3), dtype=np.float32)
        for k, col in bg_palette.items():
            mask = (grid_overlay == k)[..., None]
            bg = np.where(mask, col, bg)
        ax.imshow(bg, interpolation="nearest", origin="upper")

    # Draw sub-cells
    for y in range(H):
        for x in range(W):
            p = probs[y, x]  # [D,L,R,U,A]
            # Positions in cell (3x3 grid)
            # subcell size
            s = 1.0 / 3.0
            x0 = x
            y0 = y
            # mapping: Up=(1,0), Down=(1,2), Left=(0,1), Right=(2,1), Action=(1,1)
            subcells = {
                3: (1, 0),  # Up index 3
                0: (1, 2),  # Down index 0
                1: (0, 1),  # Left index 1
                2: (2, 1),  # Right index 2
                4: (1, 1),  # Action index 4
            }
            for idx, (sx, sy) in subcells.items():
                col = palette[idx] * (0.25 + 0.75 * p[idx])  # scale brightness by prob
                rect = Rectangle((x0 + sx * s, y0 + sy * s), s, s, facecolor=col, edgecolor=None)
                ax.add_patch(rect)

            # optional thin grid for subcells
            # for i in range(4):
            #     ax.plot([x0 + i*s, x0 + i*s], [y0, y0+1], color=(0,0,0,0.04), linewidth=0.5)
            #     ax.plot([x0, x0+1], [y0 + i*s, y0 + i*s], color=(0,0,0,0.04), linewidth=0.5)

    # Overlay targets as black outlines of the same sub-cells
    if targets is not None:
        for y in range(H):
            for x in range(W):
                t = targets[y, x]
                subcells = {
                    3: (1, 0),
                    0: (1, 2),
                    1: (0, 1),
                    2: (2, 1),
                    4: (1, 1),
                }
                # Outline only the argmax target (or all >0.5 if soft)
                if np.max(t) > 0.5:
                    idxs = [int(np.argmax(t))]
                else:
                    idxs = [i for i in range(5) if t[i] > 0.5]
                s = 1.0 / 3.0
                for idx in idxs:
                    sx, sy = subcells[idx]
                    rect = Rectangle((x + sx * s, y + sy * s), s, s, fill=False, edgecolor="black", linewidth=1.2)
                    ax.add_patch(rect)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)

# ---------------------------------
# Animation helpers
# ---------------------------------

def animate_sequence(
    probs_seq: np.ndarray,            # (T,H,W,5)
    targets_seq: Optional[np.ndarray] = None,  # (T,H,W,5) or None
    mode: str = "vector",             # "vector" or "glyph"
    out_path: str = "nca_player/demo.gif",
    grid_overlay: Optional[np.ndarray] = None,
    fps: int = 8,
):
    _ensure_dir(out_path)
    T, H, W, C = probs_seq.shape
    frames = []
    tmp_png = "nca_player/__tmp_frame.png"
    for t in range(T):
        probs = probs_seq[t]
        targets = None if targets_seq is None else targets_seq[t]
        if mode == "vector":
            render_vector_field(probs, targets, grid_overlay=grid_overlay, save_path=tmp_png, show=False)
        else:
            render_gamepad_glyphs(probs, targets, grid_overlay=grid_overlay, save_path=tmp_png, show=False)
        frames.append(imageio.imread(tmp_png))
    imageio.mimsave(out_path, frames, fps=fps)
    return out_path

# ---------------------------------
# Demo: generate random-but-structured fields
# ---------------------------------

def _demo_data(T=16, H=10, W=14, seed=0):
    rng = np.random.default_rng(seed)
    probs_seq = []
    targets_seq = []
    # Create a player trajectory to bias actions near player
    px, py = 1, 1
    for t in range(T):
        logits = rng.normal(0, 0.5, size=(H, W, 5))
        # bias a "wind" direction slowly changing over time
        theta = 2*np.pi * (t / T)
        wind = np.array([
            0.4 + 0.4*np.cos(theta + np.pi/2),  # down
            0.4 + 0.4*np.cos(theta + np.pi),    # left
            0.4 + 0.4*np.cos(theta + 0),        # right
            0.4 + 0.4*np.cos(theta + 3*np.pi/2),# up
            0.0                                 # action
        ])
        logits[..., :5] += wind

        # Add a "player" that prefers moving toward (W-2,H-2)
        for y in range(H):
            for x in range(W):
                dx = np.sign((W-2) - x)
                dy = np.sign((H-2) - y)
                # map dx,dy to action logits
                if dy > 0: logits[y,x,0] += 0.6  # down
                if dx < 0: logits[y,x,1] += 0.6  # left
                if dx > 0: logits[y,x,2] += 0.6  # right
                if dy < 0: logits[y,x,3] += 0.6  # up

        # occasional "action" spikes
        mask = rng.random((H, W)) < 0.05
        logits[mask, 4] += 2.0

        # softmax
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)

        # targets = argmax one-hot
        targ = np.zeros_like(probs)
        argm = probs.argmax(axis=-1)
        for i in range(5):
            targ[..., i] = (argm == i).astype(np.float32)

        probs_seq.append(probs.astype(np.float32))
        targets_seq.append(targ.astype(np.float32))

    return np.stack(probs_seq), np.stack(targets_seq)

# Create demo outputs
probs_seq, targets_seq = _demo_data(T=16, H=10, W=14, seed=42)

# Optional simple background: walls around border
H, W = probs_seq.shape[1:3]
grid_overlay = np.zeros((H, W), dtype=np.uint8)
grid_overlay[0, :] = 1
grid_overlay[-1, :] = 1
grid_overlay[:, 0] = 1
grid_overlay[:, -1] = 1

# Single-frame renders
vector_png = "nca_player/demo_vector_frame.png"
glyph_png = "nca_player/demo_glyph_frame.png"
render_vector_field(probs_seq[0], targets_seq[0], grid_overlay=grid_overlay, save_path=vector_png)
render_gamepad_glyphs(probs_seq[0], targets_seq[0], grid_overlay=grid_overlay, save_path=glyph_png)

# GIF animations
vector_gif = "nca_player/demo_vector.gif"
glyph_gif = "nca_player/demo_glyph.gif"
animate_sequence(probs_seq, targets_seq, mode="vector", out_path=vector_gif, grid_overlay=grid_overlay, fps=6)
animate_sequence(probs_seq, targets_seq, mode="glyph", out_path=glyph_gif, grid_overlay=grid_overlay, fps=6)

(vector_png, glyph_png, vector_gif, glyph_gif)
