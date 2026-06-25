"""Agent rule-coverage + IG-vs-random rule-set overlap on the attention belief.

The decisive test: now that the belief identifies dynamics in-context (per-step NLL
drops), does the greedy-IG agent fire MORE / DIFFERENT mechanics than random?

    .venv/bin/python -u -m nca_wm.active_learning.attn_coverage --ckpt <path>
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning.attn_belief_model import (AttnBeliefModel, AttnConfig,
                                                      masked_pool)
from nca_wm.active_learning.multigame_data import (MultiGameSet, _engine,
                                                   _masks, _perm, _read_padded)

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

    def read_spatial():
        o = torch.from_numpy(_read_padded(eng, game.n_obj, perm, cm, hm, wm))[None].to(device)
        sp = model.encode_frame(o)
        return sp, masked_pool(sp, cellT)

    sp, pooled = read_spatial()
    pooled_seq = [pooled]; prev_acts = [NA]            # NA = "no prev action"
    fired = set()
    for _ in range(steps):
        if mode == "random" or rng.random() < eps:
            ai = rng.randrange(NA)
        else:
            bel = model.belief_now(torch.stack(pooled_seq, 1),
                                   torch.tensor([prev_acts], device=device))
            igs = [model.information_gain(bel, sp, torch.tensor([k], device=device),
                                          cellT, vmask, n_ig) for k in range(NA)]
            ai = int(np.argmax(igs))
        eng.clear_rules_fired()
        eng.process_input(V.ACTION_TO_INPUT[V.ACTIONS[ai]])
        n = 0
        while eng.is_againing() and n < 50:
            eng.process_input(-1); n += 1
        fired |= set(eng.get_rules_fired())
        sp, pooled = read_spatial()
        pooled_seq.append(pooled); prev_acts.append(ai)
    return fired, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="nca_wm/active_learning/ckpts/attn_belief.pt")
    p.add_argument("--n-games", type=int, default=60)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--n-games-train", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location=device)
    model = AttnBeliefModel(AttnConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    mg = MultiGameSet(cmax=ck["cfg"]["n_obj"], split="train", seed=0)
    mg.games = mg.games[:args.n_games_train]
    rng = random.Random(args.seed)
    games = rng.sample(mg.games, min(args.n_games, len(mg.games)))

    cov_ig, cov_rnd, shared, ig_only, rnd_only, jac = [], [], [], [], [], []
    for i, g in enumerate(games):
        fi, tot = fired_set(model, g, mg, device, args.steps, "ig", seed=i)
        fr, _ = fired_set(model, g, mg, device, args.steps, "random", seed=i)
        if fi is None or fr is None:
            continue
        cov_ig.append(len(fi) / tot); cov_rnd.append(len(fr) / tot)
        u = fi | fr
        shared.append(len(fi & fr)); ig_only.append(len(fi - fr)); rnd_only.append(len(fr - fi))
        jac.append(len(fi & fr) / max(len(u), 1))
    m = lambda x: float(np.mean(x))
    print(f"ckpt {args.ckpt} | n={len(cov_ig)} games | {args.steps} steps", flush=True)
    print(f"  RULE COVERAGE:  IG-agent {m(cov_ig):.3f}  vs random {m(cov_rnd):.3f}  "
          f"(delta {m(cov_ig)-m(cov_rnd):+.3f})", flush=True)
    print(f"  composition:    shared {m(shared):.2f}  IG-only {m(ig_only):.2f}  "
          f"random-only {m(rnd_only):.2f}  Jaccard {m(jac):.3f}", flush=True)


if __name__ == "__main__":
    main()
