"""Run the trusted per-world disambiguation diagnostic (mario_disambig_diag.diag)
on one or more NCABeliefModel checkpoints, without needing a transformer ckpt.

    .venv/bin/python -u -m nca_wm.active_learning._diag_nca --ckpts <a.pkl> <b.pkl>
"""
import argparse

import torch

from nca_wm.active_learning import mario_disambig_diag as D, mario_belief as MB
from nca_wm.active_learning.mario_belief_compare import NcaCtx
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    games = MB.build_worlds()
    for p in args.ckpts:
        ck = torch.load(p, map_location=dev)
        m = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(dev)
        m.load_state_dict(ck["model_state"])
        m.eval()
        lbl = f"{(p.split('/')[-2] if '/' in p else p)} step={ck.get('step')} K={ck['cfg'].get('K')}"
        D.diag(m, lambda g, r, m=m: NcaCtx(m, g, dev, r), lbl, games, dev,
               args.n_samples, args.seed)


if __name__ == "__main__":
    main()
