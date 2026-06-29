"""What does the CONVERGED belief model predict at the disambiguating jump,
in-distribution (proper k=8 windows from the data) — uncertain (~0.5) or committed?

For UP transitions where the player is grounded under a Step (the disambiguating
config), in BOTH worlds, build the same k=8 predecessor-chain window the model was
trained on, roll the belief, and read q0 P(step present) at that Step cell. Also
checks whether base/breakable windows are byte-identical (=> the model MUST be
uncertain) or distinguishable. This bypasses the long out-of-horizon navigation
the interactive probe used.

    .venv/bin/python -u -m nca_wm.active_learning.mario_indist_test --ckpt <ckpt>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from nca_wm.train_recurrent import GameData, build_trajectory_batch
from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.multigame_data import _engine
from nca_wm.active_learning.mario_transformer_baseline import _to_attn_batch
from nca_wm.active_learning.mario_nca_belief import load_dataset_for_algo, _vmask
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig
from nca_wm.state_ops import _unpack_states

UP = 0


def _disambig_rows(S, A, sb, pb, fb, H):
    """Rows where action==UP and player is grounded under a Step (connecting config)."""
    out = []
    for i in range(len(S)):
        if int(A[i]) != UP:
            continue
        pl = np.argwhere(S[i, pb] > 0)
        if not len(pl):
            continue
        pr, pc = int(pl[0][0]), int(pl[0][1])
        if not (pr + 1 < H and (S[i, sb, pr + 1, pc] or S[i, fb, pr + 1, pc])):
            continue                                  # not grounded
        for dr in range(1, 5):
            rr = pr - dr
            if rr < 0:
                break
            if S[i, sb, rr, pc] or S[i, fb, rr, pc]:
                if S[i, sb, rr, pc]:
                    out.append((i, rr, pc))           # disambiguating: Step above
                break
    return out


@torch.no_grad()
def _q0_P_at(model, O, A, CM, cells):
    """Belief roll over the window; q0 mixture-marginal P(step present) at `cells`."""
    cm = CM
    B = model.init_belief(O[:, 0]) * cm[:, None]
    T = A.shape[1]
    for t in range(T - 1):
        B = model.update_belief(B, O[:, t + 1], A[:, t], cell_mask=cm)
    l0, p0 = model.q0_logits(B, A[:, T - 1])
    w = F.softmax(p0, -1)
    prob = (w[..., None, None, None] * torch.sigmoid(l0)).sum(1)        # (B,C,H,W)
    return prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="nca_wm/active_learning/ckpts/mario2_nca_belief_600k/params.pkl")
    p.add_argument("--algo", default="bfs")
    p.add_argument("--cap", type=int, default=200000)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    model = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device); model.load_state_dict(ck["model_state"]); model.eval()
    e0 = _engine(MB.build_worlds()[0].json_str, 0)
    sb_bit, pb, fb = MB._bit(e0, "Step"), MB._bit(e0, "Player"), MB._bit(e0, "Floor")

    names = ["mario", "mario_breakable"]
    dataset, infos = load_dataset_for_algo(names, [args.algo], args.cap, 0.1, 30, 0)
    games = [GameData(g, dataset, info) for g, info in enumerate(infos)]
    maxC = max(g.n_objs for g in games); maxH = max(g.H for g in games); maxW = max(g.W for g in games)
    print(f"\nCONVERGED model {args.ckpt} (step {ck.get('step')}); algo={args.algo}; in-distribution k={args.k}\n"
          f"P(step present|UP) at the disambiguating cell — ~0.5=uncertain, ~1=STAYS, ~0=BREAK:\n")
    rng = np.random.default_rng(0)
    for gd, info in zip(games, infos):
        S = _unpack_states(gd.states_packed, gd.W).astype(np.int8)
        A = gd.actions
        dis = _disambig_rows(S, A, sb_bit, pb, fb, gd.H)
        if not dis:
            print(f"  {info['name']:16s}: no disambiguating rows found"); continue
        sel = [dis[i] for i in rng.choice(len(dis), min(args.n, len(dis)), replace=False)]
        rows = np.array([r[0] for r in sel])
        sb = build_trajectory_batch(gd, rows, args.k, rng, maxC, maxH, maxW)
        O, Aa, R, CM, CH, _v = [x.to(device) for x in _to_attn_batch(*sb, gd.n_objs)]
        prob = _q0_P_at(model, O, Aa, CM, sel)
        Ps = np.array([float(prob[j, sb_bit, sel[j][1], sel[j][2]]) for j in range(len(sel))])
        print(f"  {info['name']:16s}: n={len(sel)}  mean P={Ps.mean():.3f}  "
              f"median={np.median(Ps):.3f}  frac(P>0.5)={float((Ps>0.5).mean()):.3f}  "
              f"[min {Ps.min():.3f}, max {Ps.max():.3f}]")


if __name__ == "__main__":
    main()
