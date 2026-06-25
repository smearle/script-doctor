"""Render GIFs of the active learner exploring an unknown world.

Each frame shows (left) the true grid and (right) the planner's predicted value
(expected information gain) per action, with the chosen action highlighted. The
agent navigates a 1-D corridor toward a seed it has not yet investigated, acts,
and — once the hidden mechanism is revealed — its predicted information gain
collapses. This is the animated version of the note's TV-world trace.

    .venv/bin/python -m nca_wm.active_learning.render_explore
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from nca_wm.active_learning import collect as C
from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.compare_collection import train_on_dataset
from nca_wm.active_learning.planner import PlannerConfig, plan_action

# 1-D corridor: only horizontal motion + ACTION matter (cheap expectimax).
PLAN_ACTIONS = [V.LEFT, V.RIGHT, V.ACTION]
GRID_H, GRID_W = 1, 6
# Player starts 2 cells from a 2-seed cluster so depth-3 expectimax can plan to it.
CUSTOM_LEVELS = [
    ".P.AA.\n",
    ".P.AA.\n",
    "P..AA.\n",
    ".P.AA.\n",
]

_COLORS = {  # canonical bit -> RGB
    "background": (0.96, 0.96, 0.96),
    "wall": (0.30, 0.30, 0.30),
    "player": (0.15, 0.45, 0.95),
    "seed": (0.20, 0.70, 0.25),
    "sprout": (0.98, 0.65, 0.10),
}


def _grid_rgb(masks):
    H, Wd = V.GRID_H, V.GRID_W
    img = np.ones((H, Wd, 3))
    order = ["background", "wall", "seed", "sprout", "player"]  # player on top
    for i, m in enumerate(masks):
        y, x = i // Wd, i % Wd
        for name in order:
            if m & (1 << V.NAME_TO_BIT[name]):
                img[y, x] = _COLORS[name]
    return img


def _frame(masks, values, chosen, step, mech, fig):
    import matplotlib.pyplot as plt
    fig.clf()
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    axg = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])
    axg.imshow(_grid_rgb(masks), interpolation="nearest", aspect="equal")
    axg.set_xticks([]); axg.set_yticks([])
    axg.set_title(f"step {step}   (hidden mechanism: {mech})", fontsize=12)
    acts = list(values.keys())
    vals = [values[a] for a in acts]
    colors = ["#d62728" if a == chosen else "#9aa0a6" for a in acts]
    axb.barh(range(len(acts)), vals, color=colors)
    axb.set_yticks(range(len(acts))); axb.set_yticklabels(acts, fontsize=10)
    axb.invert_yaxis()
    axb.set_xlabel("predicted info gain", fontsize=10)
    axb.set_title(f"chosen: {chosen}", fontsize=11)
    axb.axvline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()


def rollout_frames(model, family, mech, li, device, cfg, steps, fig, seed=0):
    rng = random.Random(seed)
    eng = W._new_engine(family.jsons[mech], li)
    id2b = W._engine_id_to_canon_bit(eng)
    obs = W.read_obs(eng, id2b)
    hist = V.encode([V.BOS] + V.serialize_obs(obs))
    frames = []
    for t in range(steps):
        a, values = plan_action(model, hist, cfg, device, actions=PLAN_ACTIONS)
        print(f"  {mech} step{t}: chose {a:6s}  vals="
              + " ".join(f"{k}:{v:+.2f}" for k, v in values.items()), flush=True)
        frames.append(_frame(obs, values, a, t, mech, fig))
        W.step_engine(eng, a, seed=str(rng.getrandbits(40)))
        obs = W.read_obs(eng, id2b)
        hist += V.encode(V.serialize_action(a) + V.serialize_obs(obs))
    frames.append(_frame(obs, values, a, steps, mech, fig))
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-updates", type=int, default=2500)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n-chance", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    family = W.build_family(n_seeds=2, seed=args.seed, form="adjacency",
                            grid_h=GRID_H, grid_w=GRID_W, style="far_cluster",
                            custom_levels=CUSTOM_LEVELS, tag_suffix="_gif")
    family.activate_geometry()
    print(f"family {family.mechanisms} geom {GRID_H}x{GRID_W} | training reference...",
          flush=True)
    data = (C.build_dataset(family, C.random_policy, 2500, seed=args.seed)
            + C.build_dataset(family, C.navigate_policy, 2500, seed=args.seed + 1))
    random.Random(args.seed).shuffle(data)
    model = train_on_dataset(data, device, args.ref_updates, seed=args.seed)
    model.eval()

    cfg = PlannerConfig(depth=args.depth, n_chance=args.n_chance)
    fig = plt.figure(figsize=(8, 2.6))
    outdir = "nca_wm/figures"
    for mech in family.mechanisms:
        frames = rollout_frames(model, family, mech, 0, device, cfg, args.steps,
                                fig, seed=args.seed)
        path = f"{outdir}/active_explore_{mech}.gif"
        imageio.mimsave(path, frames, duration=0.9, loop=0)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
