"""On-policy bootstrap: train the WM on the IG agent's OWN trajectories.

The note's loop: pi_k (IG planner over the current WM) collects haoo' data D_k ->
train WM p_k on D_k -> repeat. Fixes (a) coverage (the agent reaches mechanics
random misses) and (b) the train/test distribution shift (belief now sees the
informative histories the agent actually produces, not random walks).

A replay buffer is warmed with random trajectories, then periodically refreshed
with on-policy (greedy-IG, epsilon-mixed) trajectories from the current model.

    .venv/bin/python -u -m nca_wm.active_learning.multigame_bootstrap --updates 6000
"""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import torch

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.multigame_data import (MultiGameSet, _engine,
                                                   _masks, _perm, _read_padded)
from nca_wm.active_learning.multigame_train import eval_coverage, roll_loss
from nca_wm.active_learning.nca_belief_model import BeliefConfig, NCABeliefModel

NA = len(V.ACTIONS)


@torch.no_grad()
def collect_trajectory(model, game, mg, device, n_steps, mode="ig", n_ig=2,
                       eps=0.2, seed=0):
    """One haoo' trajectory (greedy-IG or random actions) + per-step resample."""
    rng = random.Random(seed)
    cm, hm, wm = mg.cmax, mg.hmax, mg.wmax
    perm = _perm(game.n_obj, cm, rng)
    cell, chan = _masks(game.n_obj, game.H, game.W, perm, cm, hm, wm)
    cellT = torch.from_numpy(cell)[None].to(device)
    vmask = torch.from_numpy(chan[:, None, None] * cell[None]).to(device)[None]
    eng = _engine(game.json_str, 0)

    def rd():
        return _read_padded(eng, game.n_obj, perm, cm, hm, wm)

    B = model.init_belief(torch.from_numpy(rd())[None].to(device)) * cellT
    grids, acts, resamps = [rd()], [], []
    for _ in range(n_steps):
        if mode == "random" or rng.random() < eps:
            ai = rng.randrange(NA)
        else:
            igs = [model.information_gain(B, torch.tensor([k], device=device),
                                          n_samples=n_ig, vmask=vmask) for k in range(NA)]
            ai = int(np.argmax(igs))
        a = V.ACTIONS[ai]
        s1, s2 = str(rng.getrandbits(40)), str(rng.getrandbits(40))
        bak = eng.backup_level()
        W.step_engine(eng, a, seed=s1); o1 = rd()
        eng.restore_level(bak); W.step_engine(eng, a, seed=s2); o2 = rd()
        eng.restore_level(bak); W.step_engine(eng, a, seed=s1)
        grids.append(o1); acts.append(ai); resamps.append(o2)
        B = model.update_belief(B, torch.from_numpy(o1)[None].to(device),
                                torch.tensor([ai], device=device), cell_mask=cellT)
    return (np.stack(grids), np.asarray(acts, np.int64), np.stack(resamps), cell, chan)


def _stack(trajs, idx, device):
    O = torch.from_numpy(np.stack([trajs[i][0] for i in idx])).to(device)
    A = torch.from_numpy(np.stack([trajs[i][1] for i in idx])).long().to(device)
    R = torch.from_numpy(np.stack([trajs[i][2] for i in idx])).to(device)
    CM = torch.from_numpy(np.stack([trajs[i][3] for i in idx])).to(device)
    CH = torch.from_numpy(np.stack([trajs[i][4] for i in idx])).to(device)
    return O, A, R, CM, CH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--cmax", type=int, default=32)
    p.add_argument("--hmax", type=int, default=16)
    p.add_argument("--wmax", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--usage-w", type=float, default=0.02)
    p.add_argument("--K", type=int, default=24)
    p.add_argument("--buffer", type=int, default=1200)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--collect-every", type=int, default=250)
    p.add_argument("--collect-n", type=int, default=64)
    p.add_argument("--onpolicy-eps", type=float, default=0.25)
    p.add_argument("--eval-games", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)

    mg = MultiGameSet(cmax=args.cmax, hmax=args.hmax, wmax=args.wmax, split="train", seed=args.seed)
    print(f"multigame train set: {len(mg)} games", flush=True)
    cfg = BeliefConfig(n_obj=args.cmax, n_act=NA, K=args.K)
    model = NCABeliefModel(cfg).to(device)
    print(f"belief model params {sum(q.numel() for q in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(args.seed)

    # warmup buffer with random trajectories
    buf = []
    for i in range(args.warmup):
        g = rng.choice(mg.games)
        buf.append(collect_trajectory(model, g, mg, device, args.n_steps, "random", seed=i))
    print(f"warmup buffer: {len(buf)} random trajectories", flush=True)

    t0 = time.time(); model.train(); cstep = 10**9
    for step in range(args.updates):
        if step > 0 and step % args.collect_every == 0:
            model.eval()
            for j in range(args.collect_n):
                g = rng.choice(mg.games)
                buf.append(collect_trajectory(model, g, mg, device, args.n_steps,
                                              "ig", eps=args.onpolicy_eps, seed=cstep + j))
            cstep += args.collect_n
            buf = buf[-args.buffer:]
            model.train()
            print(f"  [collect @ {step}] +{args.collect_n} on-policy | buffer {len(buf)}", flush=True)

        idx = [rng.randrange(len(buf)) for _ in range(args.batch_size)]
        O, A, R, CM, CH = _stack(buf, idx, device)
        for gp in opt.param_groups:
            gp["lr"] = args.lr * (0.5 * (1 + math.cos(math.pi * min(step / args.updates, 1))))
        loss = roll_loss(model, O, A, R, CM, CH, args.usage_w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  upd/s {(step+1)/max(time.time()-t0,1e-9):.1f}", flush=True)
        if step > 0 and step % args.eval_every == 0:
            model.eval()
            ig, rnd, n = eval_coverage(model, mg, device, args.eval_games, 24)
            print(f"  [rule coverage @ {step}] IG-agent {ig:.3f} vs random {rnd:.3f}  (n={n})", flush=True)
            model.train()

    model.eval()
    ig, rnd, n = eval_coverage(model, mg, device, args.eval_games, 24)
    print(f"\nFINAL rule coverage: IG-agent {ig:.3f} vs random {rnd:.3f}  (n={n})", flush=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__},
               "nca_wm/active_learning/ckpts/multigame_bootstrap.pt")
    print("saved ckpts/multigame_bootstrap.pt", flush=True)


if __name__ == "__main__":
    main()
