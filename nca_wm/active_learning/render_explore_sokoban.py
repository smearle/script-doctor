"""Render GIFs of the NCA-belief active agent exploring the sokoban variants.

The agent = trained belief world model + expectimax IG planner. It is curiosity-
driven (maximizes information gain), so it navigates to the unknown box, pushes to
see what happens, and loses interest (IG -> 0) once it has identified the mechanic.

    .venv/bin/python -u -m nca_wm.active_learning.render_explore_sokoban
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from nca_wm.active_learning import belief_planner as BP
from nca_wm.active_learning import grid_data as G
from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.nca_belief_model import BeliefConfig, NCABeliefModel

_COLORS = {"background": (0.97, 0.97, 0.97), "wall": (0.30, 0.30, 0.30),
           "player": (0.15, 0.45, 0.95), "seed": (0.60, 0.40, 0.20),   # box = brown
           "sprout": (0.98, 0.65, 0.10)}


def _grid_rgb(masks):
    H, Wd = V.GRID_H, V.GRID_W
    img = np.ones((H, Wd, 3))
    for i, m in enumerate(masks):
        y, x = i // Wd, i % Wd
        for name in ["background", "wall", "seed", "sprout", "player"]:
            if m & (1 << V.NAME_TO_BIT[name]):
                img[y, x] = _COLORS[name]
    return img


def _frame(masks, values, chosen, step, mech, fig):
    import matplotlib.pyplot as plt
    fig.clf()
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0])
    axg, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    axg.imshow(_grid_rgb(masks), interpolation="nearest")
    axg.set_xticks([]); axg.set_yticks([])
    axg.set_title(f"step {step}   (hidden mechanic: {mech})", fontsize=12)
    acts = [V.ACTIONS[i] for i in sorted(values)]
    vals = [values[i] for i in sorted(values)]
    colors = ["#d62728" if V.ACTIONS[i] == chosen else "#9aa0a6" for i in sorted(values)]
    axb.barh(range(len(acts)), vals, color=colors)
    axb.set_yticks(range(len(acts))); axb.set_yticklabels(acts, fontsize=10)
    axb.invert_yaxis(); axb.axvline(0, color="k", lw=0.6)
    axb.set_xlabel("planner value (info gain)", fontsize=10)
    axb.set_title(f"chosen: {chosen}", fontsize=11)
    fig.tight_layout(); fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="nca_wm/active_learning/ckpts/nca_belief_sokoban.pt")
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n-chance", type=int, default=2)
    p.add_argument("--n-ig", type=int, default=8)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    fam = W.build_family(n_layouts=64, seed=0, form="sokoban", grid_h=7, grid_w=8,
                         style="box_pushable")
    fam.activate_geometry()
    ck = torch.load(args.ckpt, map_location=device)
    model = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()

    fig = plt.figure(figsize=(8, 3.0))
    rng = random.Random(args.seed)
    for mech in fam.mechanisms:
        li = rng.randrange(fam.n_layouts)
        eng = W._new_engine(fam.jsons[mech], li); id2b = W._engine_id_to_canon_bit(eng)
        obs = W.read_obs(eng, id2b)
        B = model.init_belief(torch.from_numpy(G.obs_to_grid(obs))[None].to(device))
        frames = []
        for t in range(args.steps):
            a, vals = BP.plan_action(model, B, device, depth=args.depth,
                                     n_chance=args.n_chance, n_ig=args.n_ig)
            print(f"  {mech} step{t}: {a}", flush=True)
            frames.append(_frame(obs, vals, a, t, mech, fig))
            W.step_engine(eng, a, seed=str(rng.getrandbits(40)))
            obs = W.read_obs(eng, id2b)
            B = model.update_belief(B, torch.from_numpy(G.obs_to_grid(obs))[None].to(device),
                                    torch.tensor([V.ACTIONS.index(a)], device=device))
        frames.append(_frame(obs, vals, a, args.steps, mech, fig))
        path = f"nca_wm/figures/sokoban_explore_{mech}.gif"
        imageio.mimsave(path, frames, duration=0.9, loop=0)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
