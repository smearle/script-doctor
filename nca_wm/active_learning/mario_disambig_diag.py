"""Diagnose the disambiguating jump directly: what does each belief model predict
for the Step cell it rises into, BEFORE the jump, in each world?

The two worlds are observationally identical until a jump-into-Step-from-below, so
at the fresh under-platform state the model CANNOT know whether the Step will
break. A calibrated belief should predict P(step present | fresh, UP) ~ the train
prior (near 0.5 if both worlds equally seen); a rare-event-collapsed model
predicts ~1.0 (step stays) and is confidently wrong on breakable. We also report
the IG contribution localized to the break cell vs the whole-grid IG, to separate
"no uncertainty" from "uncertainty diluted across the grid".

    .venv/bin/python -u -m nca_wm.active_learning.mario_disambig_diag
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.mario_explore import break_cols, navigate_to_break, UP
from nca_wm.active_learning.mario_belief_compare import TfCtx, NcaCtx, H, W
from nca_wm.active_learning.attn_belief_model import AttnBeliefModel, AttnConfig
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig


def _break_cell(grid, pb, sb, fb):
    """The first Step cell directly above the player (the one a jump rises into)."""
    pa = np.argwhere((grid >> pb) & 1)
    if not len(pa):
        return None
    pr, pc = int(pa[0][0]), int(pa[0][1])
    for dr in range(1, 5):
        rr = pr - dr
        if rr < 0:
            break
        cell = int(grid[rr, pc])
        if cell & ((1 << sb) | (1 << fb)):
            return (rr, pc) if (cell >> sb) & 1 else None
    return None


@torch.no_grad()
def _q0_marginal(model, ctx, device):
    """Per-cell P(object present) under q0 for action UP at the current state."""
    a = torch.tensor([UP], device=device)
    if isinstance(model, NCABeliefModel):
        l0, lp0 = model.q0_logits(ctx.B, a)                       # (1,K,C,H,W),(1,K)
    else:
        bel = ctx._belief()
        cb = bel + model.emb_a(a)
        l0 = model._dec_all_k(model.dec0, model.emb_z0, ctx.sp, cb)
        lp0 = F.log_softmax(model.prior0(cb), -1)
    w = lp0.exp()[0]                                              # (K,)
    prob = (w[:, None, None, None] * torch.sigmoid(l0[0])).sum(0)  # (C,H,W) mixture marginal
    return prob.cpu().numpy()


@torch.no_grad()
def diag(model, make_ctx, label, games, device, n_samples, seed):
    print(f"\n=== {label} ===")
    for game in games:
        ctx = make_ctx(game, random.Random(seed)); ctx.ns = n_samples
        ok = navigate_to_break(ctx)
        if not ok:
            print(f"  {game.gist}: could not navigate under a Step"); continue
        sb = MB._bit(ctx.eng, "Step")
        cell = _break_cell(ctx.grid(), ctx.pb, sb, ctx.fb)
        # model's PRE-jump prediction at the break cell
        prob = _q0_marginal(model, ctx, device)
        p_step = float(prob[sb, cell[0], cell[1]]) if cell else float("nan")
        ig_full = ctx.ig(UP)
        # ground truth: jump, then is the Step still present at that cell?
        before = int((ctx.grid()[cell] >> sb) & 1) if cell else -1
        ctx.step(UP)
        after = int((ctx.grid()[cell] >> sb) & 1) if cell else -1
        gt = "STAYS" if after == before else "BREAKS"
        # is the probe state OOD? compare the model's whole next-frame prediction
        # to the engine ground truth over the real grid (cell-accuracy).
        gt_next = ctx._obs()[0].cpu().numpy()                     # (C,H,W) post-UP truth
        nC, nH, nW = game.n_obj, game.H, game.W
        pred = (prob[:nC, :nH, :nW] > 0.5)
        truth = (gt_next[:nC, :nH, :nW] > 0.5)
        cell_acc = float((pred == truth).mean())
        print(f"  {game.gist:16s} break-cell={cell}  "
              f"P_model(step present|fresh,UP)={p_step:.3f}  ground-truth={gt}  "
              f"whole-grid IG(UP)={ig_full:+.4f}  pred cell-acc={cell_acc:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="nca_wm/active_learning/ckpts/mario2_transformer/params_best.pkl")
    p.add_argument("--nca", default="nca_wm/active_learning/ckpts/mario2_nca_belief/params_best.pkl")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    games = MB.build_worlds()
    ck = torch.load(args.tf, map_location=device)
    tf = AttnBeliefModel(AttnConfig(**ck["cfg"])).to(device); tf.load_state_dict(ck["model_state"]); tf.eval()
    ckn = torch.load(args.nca, map_location=device)
    nca = NCABeliefModel(BeliefConfig(**ckn["cfg"])).to(device); nca.load_state_dict(ckn["model_state"]); nca.eval()
    print(f"[TF] step={ck.get('step')}  [NCA] step={ckn.get('step')}")
    diag(tf, lambda g, r: TfCtx(tf, g, device, r), "Belief Transformer", games, device, args.n_samples, args.seed)
    diag(nca, lambda g, r: NcaCtx(nca, g, device, r), "Belief Recurrent-NCA", games, device, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
