"""Diagnostic: do the IG-agent and random fire the SAME or DIFFERENT mechanics?

Aggregate rule-coverage count hides composition. Here we record the SET of rules
each agent fires per game (paired: same game, same seed) and report overlap:
shared, IG-only, random-only, union, Jaccard. If IG-only ~ 0 the on-policy data
is largely the same as random (explains a null bootstrap); if IG-only > 0 the
agent reaches mechanics random misses (the union is what helps the WM).

    .venv/bin/python -u -m nca_wm.active_learning.rule_overlap --ckpt <path>
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.multigame_data import (MultiGameSet, _engine,
                                                   _masks, _perm, _read_padded)
from nca_wm.active_learning.nca_belief_model import BeliefConfig, NCABeliefModel

NA = len(V.ACTIONS)


@torch.no_grad()
def fired_set(model, game, mg, device, steps, mode, n_ig=4, eps=0.15, seed=0):
    rng = random.Random(seed)
    cm, hm, wm = mg.cmax, mg.hmax, mg.wmax
    perm = _perm(game.n_obj, cm, rng)
    cell, chan = _masks(game.n_obj, game.H, game.W, perm, cm, hm, wm)
    cellT = torch.from_numpy(cell)[None].to(device)
    vmask = torch.from_numpy(chan[:, None, None] * cell[None]).to(device)[None]
    eng = _engine(game.json_str, 0); eng.set_track_rules_fired(True)
    total = eng.get_rule_count()
    if total == 0:
        return None, 0
    B = model.init_belief(torch.from_numpy(_read_padded(eng, game.n_obj, perm, cm, hm, wm))[None].to(device)) * cellT
    fired = set()
    for _ in range(steps):
        if mode == "random" or rng.random() < eps:
            ai = rng.randrange(NA)
        else:
            igs = [model.information_gain(B, torch.tensor([k], device=device), n_ig, vmask) for k in range(NA)]
            ai = int(np.argmax(igs))
        eng.clear_rules_fired()
        eng.process_input(V.ACTION_TO_INPUT[V.ACTIONS[ai]])
        n = 0
        while eng.is_againing() and n < 50:
            eng.process_input(-1); n += 1
        fired |= set(eng.get_rules_fired())
        og = torch.from_numpy(_read_padded(eng, game.n_obj, perm, cm, hm, wm))[None].to(device)
        B = model.update_belief(B, og, torch.tensor([ai], device=device), cell_mask=cellT)
    return fired, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="nca_wm/active_learning/ckpts/multigame_bootstrap.pt")
    p.add_argument("--n-games", type=int, default=40)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location=device)
    cmax = ck["cfg"]["n_obj"]
    model = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device); model.load_state_dict(ck["model_state"]); model.eval()
    mg = MultiGameSet(cmax=cmax, split="train", seed=0)
    rng = random.Random(args.seed)
    games = rng.sample(mg.games, min(args.n_games, len(mg.games)))

    shared, ig_only, rnd_only, union, igc, rndc, jac = [], [], [], [], [], [], []
    for i, g in enumerate(games):
        fi, tot = fired_set(model, g, mg, device, args.steps, "ig", seed=i)
        fr, _ = fired_set(model, g, mg, device, args.steps, "random", seed=i)
        if fi is None or fr is None:
            continue
        u = fi | fr
        shared.append(len(fi & fr)); ig_only.append(len(fi - fr)); rnd_only.append(len(fr - fi))
        union.append(len(u)); igc.append(len(fi)); rndc.append(len(fr))
        jac.append(len(fi & fr) / max(len(u), 1))
    m = lambda x: float(np.mean(x))
    print(f"ckpt {args.ckpt} | n={len(igc)} games", flush=True)
    print(f"  fired count:  IG {m(igc):.2f}   random {m(rndc):.2f}   UNION {m(union):.2f}", flush=True)
    print(f"  composition:  shared {m(shared):.2f}   IG-only {m(ig_only):.2f}   random-only {m(rnd_only):.2f}", flush=True)
    print(f"  Jaccard(IG,random) {m(jac):.3f}   (1.0 = identical sets, 0 = disjoint)", flush=True)
    print(f"  -> union/IG ratio {m(union)/max(m(igc),1e-9):.3f}  (how much random ADDS beyond IG)", flush=True)


if __name__ == "__main__":
    main()
