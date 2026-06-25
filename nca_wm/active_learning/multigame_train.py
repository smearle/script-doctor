"""Train the belief model on many human games; measure MECHANIC ACTIVATION.

Goal (per user): an agent that explores human games effectively, measured by
rule-firing coverage (distinct rules fired / total) via the cpp engine's
set_track_rules_fired plumbing. We train the belief world model in-distribution on
the gist subset, then compare the greedy-IG agent's rule coverage to random.

    .venv/bin/python -u -m nca_wm.active_learning.multigame_train --updates 8000
"""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import torch

from nca_wm.active_learning import multigame_data as MD
from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.multigame_data import (MultiGameSet, _engine,
                                                   _masks, _perm, _read_padded)
from nca_wm.active_learning.nca_belief_model import BeliefConfig, NCABeliefModel

NA = len(V.ACTIONS)


def _usage_reg(logpi):
    mean = logpi.exp().mean(0).clamp_min(1e-8)
    return math.log(logpi.shape[-1]) + (mean * mean.log()).sum()


def roll_loss(model, O, A, R, CM, CH, usage_w):
    vmask = CH[:, :, None, None] * CM[:, None, :, :]           # (N,C,H,W)
    B = model.init_belief(O[:, 0]) * CM[:, None]
    T = A.shape[1]
    loss = 0.0
    for t in range(T):
        a, onext, op = A[:, t], O[:, t + 1], R[:, t]
        l0, p0 = model.q0_logits(B, a)
        l1, p1 = model.q1_logits(B, a, onext)
        loss = loss + model.mixture_nll(l0, p0, onext, vmask).mean() \
                    + model.mixture_nll(l1, p1, op, vmask).mean() \
                    + usage_w * (_usage_reg(p0) + _usage_reg(p1))
        B = model.update_belief(B, onext, a, cell_mask=CM)
    return loss / T


@torch.no_grad()
def agent_coverage(model, game, mgset, device, steps=40, mode="ig", n_ig=4,
                   eps=0.15, seed=0):
    """Roll an agent in `game`; return (distinct rules fired)/total."""
    rng = random.Random(seed)
    cm, hm, wm = mgset.cmax, mgset.hmax, mgset.wmax
    perm = _perm(game.n_obj, cm, rng)
    cell, chan = _masks(game.n_obj, game.H, game.W, perm, cm, hm, wm)
    cellT = torch.from_numpy(cell)[None].to(device)
    vmask = torch.from_numpy(chan[:, None, None] * cell[None]).to(device)[None]
    eng = _engine(game.json_str, 0)
    eng.set_track_rules_fired(True)
    total = eng.get_rule_count()
    if total == 0:
        return None
    og = torch.from_numpy(_read_padded(eng, game.n_obj, perm, cm, hm, wm))[None].to(device)
    B = model.init_belief(og) * cellT
    fired = set()
    for _ in range(steps):
        if mode == "random" or rng.random() < eps:
            ai = rng.randrange(NA)
        else:
            igs = [model.information_gain(B, torch.tensor([k], device=device),
                                          n_samples=n_ig, vmask=vmask) for k in range(NA)]
            ai = int(np.argmax(igs))
        eng.clear_rules_fired()
        eng.process_input(V.ACTION_TO_INPUT[V.ACTIONS[ai]])
        n = 0
        while eng.is_againing() and n < 50:
            eng.process_input(-1); n += 1
        fired |= set(eng.get_rules_fired())
        og = torch.from_numpy(_read_padded(eng, game.n_obj, perm, cm, hm, wm))[None].to(device)
        B = model.update_belief(B, og, torch.tensor([ai], device=device), cell_mask=cellT)
    return len(fired) / total


@torch.no_grad()
def eval_coverage(model, mgset, device, n_games=40, steps=40, seed=1):
    rng = random.Random(seed)
    games = rng.sample(mgset.games, min(n_games, len(mgset.games)))
    ig, rnd = [], []
    for i, g in enumerate(games):
        ci = agent_coverage(model, g, mgset, device, steps, "ig", seed=i)
        cr = agent_coverage(model, g, mgset, device, steps, "random", seed=i)
        if ci is not None and cr is not None:
            ig.append(ci); rnd.append(cr)
    return float(np.mean(ig)), float(np.mean(rnd)), len(ig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--cmax", type=int, default=24)
    p.add_argument("--hmax", type=int, default=16)
    p.add_argument("--wmax", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--usage-w", type=float, default=0.02)
    p.add_argument("--K", type=int, default=24)
    p.add_argument("--eval-games", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)

    mg = MultiGameSet(cmax=args.cmax, hmax=args.hmax, wmax=args.wmax, split="train", seed=args.seed)
    print(f"multigame train set: {len(mg)} games", flush=True)
    cfg = BeliefConfig(n_obj=args.cmax, n_act=NA, K=args.K)
    model = NCABeliefModel(cfg).to(device)
    print(f"belief model params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    rng = random.Random(args.seed)
    t0 = time.time(); model.train()
    for step in range(args.updates):
        O, A, R, CM, CH = MD.batch(mg, args.batch_size, args.n_steps, rng)
        O, A, R, CM, CH = [x.to(device) for x in (O, A, R, CM, CH)]
        for g in opt.param_groups:
            g["lr"] = args.lr * (0.5 * (1 + math.cos(math.pi * min(step / args.updates, 1))))
        loss = roll_loss(model, O, A, R, CM, CH, args.usage_w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  upd/s {(step+1)/max(time.time()-t0,1e-9):.1f}", flush=True)
        if step > 0 and step % args.eval_every == 0:
            model.eval()
            ig, rnd, n = eval_coverage(model, mg, device, args.eval_games, args.K)
            print(f"  [rule coverage @ {step}] IG-agent {ig:.3f} vs random {rnd:.3f}  (n={n} games)", flush=True)
            model.train()

    model.eval()
    ig, rnd, n = eval_coverage(model, mg, device, args.eval_games, args.K)
    print(f"\nFINAL rule coverage: IG-agent {ig:.3f} vs random {rnd:.3f}  (n={n} games)", flush=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__},
               "nca_wm/active_learning/ckpts/multigame_belief.pt")
    print("saved ckpts/multigame_belief.pt", flush=True)


if __name__ == "__main__":
    main()
