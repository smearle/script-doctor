"""Train the attention-belief WM on a mid-size game set; test in-context inference.

Hypothesis test: with attention over history, per-step q0 NLL should DECREASE over
a rollout (the belief identifies dynamics) — unlike the conv-NCA belief (flat/rising).

    .venv/bin/python -u -m nca_wm.active_learning.attn_belief_train --updates 4000 --n-games 256
"""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import torch

from nca_wm.active_learning import multigame_data as MD
from nca_wm.active_learning.multigame_data import MultiGameSet
from nca_wm.active_learning.attn_belief_model import AttnBeliefModel, AttnConfig


@torch.no_grad()
def per_step_nll(model, mgset, device, n=160, n_steps=8, seed=7):
    rng = random.Random(seed)
    O, A, R, CM, CH = MD.batch(mgset, n, n_steps, rng)
    O, A, R, CM, CH = [x.to(device) for x in (O, A, R, CM, CH)]
    q0, _ = model.forward_traj(O, A, R, CM, CH)
    return q0.mean(0).tolist()                          # per-step curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--n-games", type=int, default=256)
    p.add_argument("--cmax", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)

    mg = MultiGameSet(cmax=args.cmax, split="train", seed=args.seed)
    mg.games = mg.games[:args.n_games]
    held = MultiGameSet(cmax=args.cmax, split="holdout", seed=args.seed)
    print(f"train {len(mg)} games | held-out {len(held)}", flush=True)
    cfg = AttnConfig(n_obj=args.cmax)
    model = AttnBeliefModel(cfg).to(device)
    print(f"attn-belief params {sum(q.numel() for q in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    rng = random.Random(args.seed); t0 = time.time(); model.train()
    for step in range(args.updates):
        O, A, R, CM, CH = MD.batch(mg, args.batch_size, args.n_steps, rng)
        O, A, R, CM, CH = [x.to(device) for x in (O, A, R, CM, CH)]
        for g in opt.param_groups:
            g["lr"] = args.lr * (0.5 * (1 + math.cos(math.pi * min(step / args.updates, 1))))
        q0, q1 = model.forward_traj(O, A, R, CM, CH)
        loss = q0.mean() + q1.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 200 == 0:
            print(f"step {step:5d}  loss {loss.item():.3f}  upd/s {(step+1)/max(time.time()-t0,1e-9):.1f}", flush=True)
        if step > 0 and step % args.eval_every == 0:
            model.eval()
            tr = per_step_nll(model, mg, device); ho = per_step_nll(model, held, device)
            d_tr = tr[0] - tr[-1]; d_ho = ho[0] - ho[-1]
            print(f"  [per-step q0 NLL @ {step}]  TRAIN {' '.join(f'{x:.1f}' for x in tr)}  (drop {d_tr:+.1f})", flush=True)
            print(f"                              HELD  {' '.join(f'{x:.1f}' for x in ho)}  (drop {d_ho:+.1f})", flush=True)
            model.train()

    model.eval()
    tr = per_step_nll(model, mg, device); ho = per_step_nll(model, held, device)
    print(f"\nFINAL per-step q0 NLL:", flush=True)
    print(f"  TRAIN {' '.join(f'{x:.1f}' for x in tr)}  (drop {tr[0]-tr[-1]:+.1f})", flush=True)
    print(f"  HELD  {' '.join(f'{x:.1f}' for x in ho)}  (drop {ho[0]-ho[-1]:+.1f})", flush=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__},
               "nca_wm/active_learning/ckpts/attn_belief.pt")
    print("saved ckpts/attn_belief.pt", flush=True)


if __name__ == "__main__":
    main()
