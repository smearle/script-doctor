"""Held-out WM predictive validation: next-frame NLL on fresh rollouts.

Leak-proof split = held-out GAMES (the model never saw them at all, so no
transition-generator overlap is possible). We report per-frame q0 mixture-NLL
(masked to real cells x channels) on fresh random rollouts for both the TRAIN
games (in-distribution) and the HELD-OUT games (generalization). The gap measures
whether the WM has learned transferable dynamics vs memorized training games.

    .venv/bin/python -u -m nca_wm.active_learning.eval_wm_nll --ckpt <path>
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from nca_wm.active_learning import multigame_data as MD
from nca_wm.active_learning.multigame_data import MultiGameSet
from nca_wm.active_learning.nca_belief_model import BeliefConfig, NCABeliefModel


@torch.no_grad()
def split_nll(model, mgset, device, n=128, n_steps=8, seed=7):
    """Mean per-frame q0 NLL over fresh rollouts; also the per-step curve."""
    rng = random.Random(seed)
    O, A, R, CM, CH = MD.batch(mgset, n, n_steps, rng)
    O, A, CM, CH = O.to(device), A.to(device), CM.to(device), CH.to(device)
    vmask = CH[:, :, None, None] * CM[:, None, :, :]
    B = model.init_belief(O[:, 0]) * CM[:, None]
    per_step = []
    for t in range(n_steps):
        a, onext = A[:, t], O[:, t + 1]
        l0, p0 = model.q0_logits(B, a)
        per_step.append(model.mixture_nll(l0, p0, onext, vmask).mean().item())
        B = model.update_belief(B, onext, a, cell_mask=CM)
    return float(np.mean(per_step)), per_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="nca_wm/active_learning/ckpts/multigame_bootstrap.pt")
    p.add_argument("--cmax", type=int, default=32)
    p.add_argument("--hmax", type=int, default=16)
    p.add_argument("--wmax", type=int, default=20)
    p.add_argument("--n", type=int, default=160)
    p.add_argument("--holdout", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location=device)
    cmax = ck["cfg"]["n_obj"]
    model = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()

    train = MultiGameSet(cmax=cmax, hmax=args.hmax, wmax=args.wmax, split="train",
                         holdout=args.holdout, seed=args.seed)
    held = MultiGameSet(cmax=cmax, hmax=args.hmax, wmax=args.wmax, split="holdout",
                        holdout=args.holdout, seed=args.seed)
    print(f"ckpt {args.ckpt} | cmax {cmax} | train {len(train)} held-out {len(held)}", flush=True)

    tr, tr_curve = split_nll(model, train, device, args.n)
    ho, ho_curve = split_nll(model, held, device, args.n)
    print(f"per-frame q0 NLL (masked):  TRAIN {tr:.3f}   HELD-OUT {ho:.3f}   gap {ho-tr:+.3f}", flush=True)
    print(f"  train per-step:    " + " ".join(f"{x:.2f}" for x in tr_curve), flush=True)
    print(f"  held-out per-step: " + " ".join(f"{x:.2f}" for x in ho_curve), flush=True)


if __name__ == "__main__":
    main()
