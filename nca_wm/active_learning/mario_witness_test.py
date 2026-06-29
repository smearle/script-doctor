"""Does the belief update after WITNESSING one non-breaking jump?

In the BASE world (Step never breaks), measure the model's P(step present | UP) at
the disambiguating cell BEFORE vs AFTER it observes one real jump-into-Step whose
outcome is "stays". The observation is teacher-forced (NcaCtx ingests the TRUE
engine next-frame into the belief B), so the hidden state evolves under the real
"stays" sequence even though the WM's own q0 predicts "break". If the belief works,
P should rise toward "stays" (present) after witnessing; if collapsed, it stays ~0.

    .venv/bin/python -u -m nca_wm.active_learning.mario_witness_test \
        --ckpts <ckpt1> <ckpt2> ...
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.mario_explore import break_cols, navigate_to_break, UP, TICK
from nca_wm.active_learning.mario_belief_compare import NcaCtx
from nca_wm.active_learning.mario_disambig_diag import _break_cell, _q0_marginal
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig


@torch.no_grad()
def _p_step(model, ctx, device):
    sb = MB._bit(ctx.eng, "Step")
    cell = _break_cell(ctx.grid(), ctx.pb, sb, ctx.fb)
    if cell is None:
        return None, None
    prob = _q0_marginal(model, ctx, device)
    return float(prob[sb, cell[0], cell[1]]), cell


@torch.no_grad()
def witness_test(model, device, seed=0, n_witness=1):
    game = [g for g in MB.build_worlds() if g.gist == "mario"][0]   # base: stays is truth
    ctx = NcaCtx(model, game, device, random.Random(seed)); ctx.ns = 1
    if not navigate_to_break(ctx):
        return None
    p_before, cell_b = _p_step(model, ctx, device)
    # WITNESS n_witness real jump-into-Step events (each leaves the step intact in base)
    for _ in range(n_witness):
        ctx.step(UP)                       # belief ingests the TRUE next frame (stays)
        for _ in range(4):
            ctx.step(TICK)                 # fall back to the ground, still witnessing
        navigate_to_break(ctx)
    p_after, cell_a = _p_step(model, ctx, device)
    return dict(p_before=p_before, cell_before=cell_b, p_after=p_after,
                cell_after=cell_a, breaks=ctx.n_break)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[
        "nca_wm/active_learning/ckpts/mario2_nca_belief_600k/params.pkl",
        "nca_wm/active_learning/ckpts/mario2_nca_belief_600k/params_step2500.pkl"])
    p.add_argument("--n-witness", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    print("BASE world (truth = STAYS). P(step present|UP): higher = predicts STAYS "
          f"(calibrated); ~0 = predicts BREAK (wrong/collapsed). Witness={args.n_witness}\n")
    for c in args.ckpts:
        ck = torch.load(c, map_location=device)
        m = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device)
        m.load_state_dict(ck["model_state"]); m.eval()
        r = witness_test(m, device, n_witness=args.n_witness)
        tag = c.split("/")[-1] + f" (step {ck.get('step')})"
        if r is None:
            print(f"  {tag}: could not navigate"); continue
        print(f"  {tag}")
        print(f"     P(step|UP) BEFORE witnessing = {r['p_before']:.3f}  (cell {r['cell_before']})")
        print(f"     P(step|UP) AFTER  witnessing = {r['p_after']:.3f}  (cell {r['cell_after']})  "
              f"-> {'UPDATED toward STAYS' if (r['p_after']-r['p_before'])>0.1 else 'no meaningful update'}")


if __name__ == "__main__":
    main()
