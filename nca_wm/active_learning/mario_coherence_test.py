"""Coherence test for the ambiguous under-the-platform jump.

The two Mario worlds are observationally identical until a jump-into-Step: then
the Step either STAYS (base) or BREAKS (breakable). Crucially the two outcomes
differ at SEVERAL correlated cells (the Step cell AND the player's resulting
position), so the next-observation distribution p(o|h,a) is jointly bimodal, not
a single independent Bernoulli.

This script asks, for a trained q0 head, whether it represents that bimodal
structure COHERENTLY:

  * marginal  P(step present | fresh, UP) at the break cell  -- the single-cell
    read; a factorized (K=1) head can match this (~0.5) and look "uncertain".
  * COHERENCE -- sample whole next-frames from q0 and check, over the cells where
    STAYS and BREAKS differ, whether each sample matches one FULL outcome (a
    coherent draw from a mode) or is an independent per-cell mixture (incoherent
    "half-broken" frames that never occur). This is what the K-mode mixture buys
    beyond global pooling / NCA repeats; a K=1 head cannot produce it.

    .venv/bin/python -u -m nca_wm.active_learning.mario_coherence_test \
        --ckpts <ckptA> <ckptB> ... --n-samples 400
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.mario_explore import navigate_to_break, UP
from nca_wm.active_learning.mario_belief_compare import NcaCtx
from nca_wm.active_learning.mario_disambig_diag import _break_cell
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig


@torch.no_grad()
def _pre_jump(model, game, device, seed):
    """Navigate under a Step; return (belief B before the jump, pre-obs, post-UP
    truth, Step-bit, break-cell)."""
    ctx = NcaCtx(model, game, device, random.Random(seed))
    if not navigate_to_break(ctx):
        return None
    B = ctx.B.clone()
    pre = ctx._obs()[0].cpu().numpy()
    sb = MB._bit(ctx.eng, "Step")
    cell = _break_cell(ctx.grid(), ctx.pb, sb, ctx.fb)
    ctx.step(UP)
    post = ctx._obs()[0].cpu().numpy()
    return B, pre, post, sb, cell


@torch.no_grad()
def coherence(ckpt_path, device, n_samples=400, seed=0):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = BeliefConfig(**ck["cfg"])
    model = NCABeliefModel(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    worlds = {w.gist: w for w in MB.build_worlds()}
    s = _pre_jump(model, worlds["mario"], device, seed)
    b = _pre_jump(model, worlds["mario_breakable"], device, seed)
    if s is None or b is None:
        print(f"  [{ckpt_path}] could not navigate under a Step"); return None
    B, pre_s, o_stay, sb, cell = s
    _, pre_b, o_break, _, _ = b

    g = worlds["mario"]
    nC, nH, nW = g.n_obj, g.H, g.W
    sl = (slice(0, nC), slice(0, nH), slice(0, nW))
    occ_stay = (o_stay[sl] > 0.5)
    occ_break = (o_break[sl] > 0.5)
    D = occ_stay != occ_break                                   # disambiguating cells
    nD = int(D.sum())
    pre_identical = bool(np.array_equal(pre_s[sl] > 0.5, pre_b[sl] > 0.5))

    # q0 at the (shared) pre-jump belief
    a = torch.tensor([UP], device=device)
    l0, lp0 = model.q0_logits(B, a)                              # (1,K,C,H,W),(1,K)
    K = l0.shape[1]
    probs0 = lp0.exp()[0]                                        # (K,)
    # single-cell marginal at the break cell (the naive "is it uncertain?" read)
    w = probs0[:, None, None, None]
    marg = (w * torch.sigmoid(l0[0])).sum(0).cpu().numpy()       # (C,H,W)
    p_step = float(marg[sb, cell[0], cell[1]]) if cell else float("nan")

    stay_t = torch.tensor(occ_stay, device=device)
    break_t = torch.tensor(occ_break, device=device)
    Dt = torch.tensor(D, device=device)
    l0D = l0[0][:, :nC, :nH, :nW]                                # (K,nC,nH,nW)

    n_stay = n_break = n_coherent = 0
    frac_break = []                                             # per-sample frac of D matching BREAK
    used_modes = np.zeros(K, dtype=np.int64)
    for _ in range(n_samples):
        k = int(torch.multinomial(probs0, 1).item())
        used_modes[k] += 1
        o = (torch.bernoulli(torch.sigmoid(l0D[k])) > 0.5)      # (nC,nH,nW)
        onD = o[Dt]
        m_stay = bool((onD == stay_t[Dt]).all())
        m_break = bool((onD == break_t[Dt]).all())
        frac_break.append(float((onD == break_t[Dt]).float().mean().item()))
        n_stay += m_stay; n_break += m_break
        n_coherent += (m_stay or m_break)
    frac_break = np.array(frac_break)

    # If outcomes were independent per cell at p~0.5, P(a sample matches EITHER
    # full outcome) ~ 2 * 0.5^nD -- vanishes for nD>~3. Coherent mixtures keep it
    # near 1.0 and split frac_break into a 0/1 bimodal histogram.
    print(f"\n=== {ckpt_path}  (K={K}, step={ck.get('step')}) ===")
    print(f"  pre-jump states identical across worlds: {pre_identical}")
    print(f"  disambiguating cells |D| = {nD}  (Step cell + player-position cells)")
    print(f"  break cell {cell}: marginal P(step present|fresh,UP) = {p_step:.3f}")
    print(f"  COHERENT samples (match a full outcome): {n_coherent}/{n_samples} "
          f"= {n_coherent/n_samples:.3f}   [chance if independent ~ {2*0.5**nD:.1e}]")
    print(f"    matched STAYS: {n_stay/n_samples:.3f}   matched BREAKS: {n_break/n_samples:.3f}")
    bins = np.histogram(frac_break, bins=np.linspace(0, 1, 11))[0]
    print(f"    frac-of-D-matching-BREAK histogram (0..1): {bins.tolist()}")
    print(f"    -> bimodal (mass at 0 and 1) = coherent; peak at 0.5 = incoherent")
    print(f"  modes used (of K={K}): {(used_modes>0).sum()} distinct; counts={used_modes.tolist()}")
    return dict(ckpt=ckpt_path, K=K, step=ck.get("step"), nD=nD, p_step=p_step,
                coherent=n_coherent / n_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for c in args.ckpts:
        r = coherence(c, device, args.n_samples, args.seed)
        if r:
            rows.append(r)
    if rows:
        print("\n=== summary ===")
        print(f"{'ckpt':50s} {'K':>3} {'step':>7} {'|D|':>4} {'P(step)':>8} {'coherent':>9}")
        for r in rows:
            print(f"{r['ckpt'][-50:]:50s} {r['K']:>3} {str(r['step']):>7} "
                  f"{r['nD']:>4} {r['p_step']:>8.3f} {r['coherent']:>9.3f}")


if __name__ == "__main__":
    main()
