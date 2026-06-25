"""Expectimax planner over the NCA-belief model (IG as per-step reward).

Mirrors planner.py but operates on belief states + the belief model's coherent
frame sampling, so it is tractable (conv passes, no per-cell autoregression).
Used as the learned active-collection policy in step 4.
"""
from __future__ import annotations

import torch

from nca_wm.active_learning import vocab as V


@torch.no_grad()
def plan_action(model, B, device, depth=2, n_chance=2, n_ig=6,
                actions=None, discount=1.0):
    """Return (best_action_str, per-action value) from belief state B (1,d_b,H,W)."""
    acts = actions if actions is not None else list(range(len(V.ACTIONS)))
    values = {}

    def expectimax(B, d):
        best = -float("inf")
        for ai in acts:
            a = torch.tensor([ai], device=device)
            ig = model.information_gain(B, a, n_samples=n_ig)
            future = 0.0
            if d > 1:
                logits0, logpi0 = model.q0_logits(B, a)
                probs0 = logpi0.exp()[0]
                tot = 0.0
                for _ in range(n_chance):
                    k = torch.multinomial(probs0, 1).item()
                    o = torch.bernoulli(torch.sigmoid(logits0[:, k]))
                    Bn = model.update_belief(B, o, a)
                    tot += expectimax(Bn, d - 1)
                future = discount * tot / n_chance
            val = ig + future
            if d == depth:
                values[ai] = val
            best = max(best, val)
        return best

    expectimax(B, depth)
    best = max(values, key=values.get)
    return V.ACTIONS[best], values
