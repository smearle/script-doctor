"""Is the disambiguating-jump collapse HISTORY-driven?

For each trained model + world: navigate under a Step, then read q0's
P(step present | UP) at the break cell from two beliefs:
  P_full  -- the FULL-history belief accumulated while navigating.
  P_fresh -- a FRESH single-frame belief init_belief(current_obs), i.e. the
             no-history / k=0-equivalent first-tick prediction. This is
             in-distribution: q0 at t=0 of every training trajectory is computed
             exactly this way (from init_belief(O[0]), no update_belief yet).

The two worlds are observationally identical up to this state, so P_fresh MUST be
identical across worlds (same input). If P_fresh ~ 0.5 (uncertain) while
P_full ~ 0 (collapsed), the model is resolving the ambiguity from HISTORY --
leaking world-identity through the recurrent belief. If P_fresh is also ~0, the
collapse is already baked into the single-frame prediction.

    .venv/bin/python -u -m nca_wm.active_learning.mario_history_probe --ckpts ...
"""
from __future__ import annotations

import argparse
import random

import torch

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.mario_explore import navigate_to_break, UP
from nca_wm.active_learning.mario_belief_compare import NcaCtx
from nca_wm.active_learning.mario_disambig_diag import _break_cell
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig


@torch.no_grad()
def _p_step_at_break(model, B, ctx):
    a = torch.tensor([UP], device=B.device)
    l0, lp0 = model.q0_logits(B, a)
    w = lp0.exp()[0]
    prob = (w[:, None, None, None] * torch.sigmoid(l0[0])).sum(0).cpu().numpy()
    sb = MB._bit(ctx.eng, "Step")
    cell = _break_cell(ctx.grid(), ctx.pb, sb, ctx.fb)
    if cell is None:
        return float("nan"), None
    return float(prob[sb, cell[0], cell[1]]), cell


@torch.no_grad()
def probe(ckpt, device, seed):
    ck = torch.load(ckpt, map_location=device)
    model = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    tag = ckpt.split("/")[-2] if "/" in ckpt else ckpt
    print(f"\n=== {tag}  step={ck.get('step')}  K={ck['cfg'].get('K')} ===")
    for game in MB.build_worlds():
        ctx = NcaCtx(model, game, device, random.Random(seed))
        if not navigate_to_break(ctx):
            print(f"  {game.gist}: could not navigate under a Step"); continue
        p_full, cell = _p_step_at_break(model, ctx.B, ctx)
        B_fresh = model.init_belief(ctx._obs()) * ctx.cellT[:, None]
        p_fresh, _ = _p_step_at_break(model, B_fresh, ctx)
        print(f"  {game.gist:16s} break={cell}  "
              f"P_full(history)={p_full:.3f}   P_fresh(no-history)={p_fresh:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for c in args.ckpts:
        probe(c, device, args.seed)


if __name__ == "__main__":
    main()
