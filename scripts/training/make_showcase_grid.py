"""Tile per-game winning GIFs of the recurrent agent into a single grid
animation (and a first-frame contact sheet PNG), for a quick "best agents
across multiple games" overview.

Usage:
    .venv/bin/python3 scripts/training/make_showcase_grid.py \
        --showcase_dir rl_logs_jax/_recurrent_showcase
"""
import argparse
import glob
import os
from collections import OrderedDict

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageSequence


def load_gif(path):
    # Coalesce frames to full canvas (GIFs store optimized delta frames of
    # differing sizes; PIL's convert("RGB") per seeked frame gives full frames).
    im = Image.open(path)
    frames = [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(im)]
    return np.stack(frames)  # (T, H, W, 3)


def pad_to(frame, H, W):
    h, w = frame.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    y, x = (H - h) // 2, (W - w) // 2
    out[y:y + h, x:x + w] = frame
    return out


def label_strip(W, text, height=16):
    """A black strip with the game name (rendered as simple pixel blocks would
    need a font; instead we just return a black bar — name lives in the PNG
    filename / README). Kept minimal to avoid font deps."""
    return np.zeros((height, W, 3), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--showcase_dir", default="rl_logs_jax/_recurrent_showcase")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--cell", type=int, default=160, help="cell size in px (square)")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    # One representative winning gif per game (prefer greedy, else sample).
    gifs = sorted(glob.glob(os.path.join(args.showcase_dir, "*WIN*.gif")))
    by_game = OrderedDict()
    for g in gifs:
        base = os.path.basename(g)
        game = base.split("_greedy_")[0].split("_sample_")[0]
        is_greedy = "_greedy_" in base
        # Prefer greedy; first one wins otherwise.
        if game not in by_game or (is_greedy and "_sample_" in os.path.basename(by_game[game])):
            by_game[game] = g

    if not by_game:
        print(f"No *WIN*.gif found in {args.showcase_dir}")
        return

    print(f"Found winning gifs for {len(by_game)} games:")
    for game, path in by_game.items():
        print(f"  {game}: {os.path.basename(path)}")

    C = args.cell
    clips = OrderedDict()
    for game, path in by_game.items():
        frames = load_gif(path)
        # Resize-by-padding each frame to CxC (nearest upscale to fill, keep aspect).
        resized = []
        for f in frames:
            h, w = f.shape[:2]
            scale = min(C // max(h, 1), C // max(w, 1))
            scale = max(scale, 1)
            up = np.kron(f, np.ones((scale, scale, 1), dtype=np.uint8))
            resized.append(pad_to(up[:C, :C], C, C))
        clips[game] = np.stack(resized)

    max_T = max(c.shape[0] for c in clips.values())
    # Freeze each clip on its last frame to the common length.
    for game in clips:
        c = clips[game]
        if c.shape[0] < max_T:
            pad = np.repeat(c[-1:], max_T - c.shape[0], axis=0)
            clips[game] = np.concatenate([c, pad], axis=0)

    games = list(clips.keys())
    cols = min(args.cols, len(games))
    rows = (len(games) + cols - 1) // cols

    grid_frames = []
    for t in range(max_T):
        row_imgs = []
        for r in range(rows):
            cell_imgs = []
            for cc in range(cols):
                idx = r * cols + cc
                if idx < len(games):
                    cell_imgs.append(clips[games[idx]][t])
                else:
                    cell_imgs.append(np.zeros((C, C, 3), dtype=np.uint8))
            row_imgs.append(np.concatenate(cell_imgs, axis=1))
        grid_frames.append(np.concatenate(row_imgs, axis=0))
    grid = np.stack(grid_frames)

    out_gif = os.path.join(args.showcase_dir, "showcase_grid.gif")
    iio.imwrite(out_gif, grid, duration=1.0 / args.fps, loop=0)
    print(f"\nWrote grid animation: {out_gif}  (shape {grid.shape})")

    # Contact sheet: mid-frame of each clip.
    sheet_rows = []
    for r in range(rows):
        cell_imgs = []
        for cc in range(cols):
            idx = r * cols + cc
            if idx < len(games):
                clip = clips[games[idx]]
                cell_imgs.append(clip[min(clip.shape[0] - 1, clip.shape[0] // 2)])
            else:
                cell_imgs.append(np.zeros((C, C, 3), dtype=np.uint8))
        sheet_rows.append(np.concatenate(cell_imgs, axis=1))
    sheet = np.concatenate(sheet_rows, axis=0)
    out_png = os.path.join(args.showcase_dir, "showcase_contact_sheet.png")
    iio.imwrite(out_png, sheet)
    print(f"Wrote contact sheet: {out_png}")
    print("Game order (row-major): " + ", ".join(games))


if __name__ == "__main__":
    main()
