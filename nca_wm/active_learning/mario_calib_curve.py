"""OBSERVE the belief-calibration-vs-training curve (NOT for selection).

Probes every saved `params_step{N}.pkl` checkpoint and reports, per world, the
model's P(step present | fresh under-platform, UP) at the disambiguating cell and
its q0 prior entropy over the K latent modes. Answers: does the belief un-collapse
(P -> the data marginal ~0.75 "stays") with more training, or stay collapsed
(P -> 0, confident "breaks")? Pure measurement — checkpoint selection must still
use a generic criterion, never this.

    .venv/bin/python -u -m nca_wm.active_learning.mario_calib_curve \
        --dir nca_wm/active_learning/ckpts/mario2_nca_belief_600k
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re

import numpy as np
import torch

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.mario_explore import break_cols, navigate_to_break, UP
from nca_wm.active_learning.mario_belief_compare import NcaCtx
from nca_wm.active_learning.mario_disambig_diag import _break_cell, _q0_marginal
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig


def _step_of(path):
    m = re.search(r"params_step(\d+)\.pkl", path)
    return int(m.group(1)) if m else -1


@torch.no_grad()
def probe(model, games, device, seed=0):
    out = {}
    for game in games:
        ctx = NcaCtx(model, game, device, random.Random(seed)); ctx.ns = 1
        if not navigate_to_break(ctx):
            out[game.gist] = None; continue
        sb = MB._bit(ctx.eng, "Step")
        cell = _break_cell(ctx.grid(), ctx.pb, sb, ctx.fb)
        prob = _q0_marginal(model, ctx, device)
        out[game.gist] = float(prob[sb, cell[0], cell[1]]) if cell else float("nan")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="nca_wm/active_learning/ckpts/mario2_nca_belief_600k")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    games = MB.build_worlds()
    ckpts = sorted(glob.glob(os.path.join(args.dir, "params_step*.pkl")), key=_step_of)
    if not ckpts:
        print(f"no params_step*.pkl in {args.dir}"); return
    print(f"{'step':>7}  {'P(step|fresh,UP) mario':>24}  {'mario_breakable':>16}")
    print("  (data marginal ~0.75 'stays' = calibrated/uncertain; ->0 = collapsed 'breaks')")
    for c in ckpts:
        ck = torch.load(c, map_location=device)
        m = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device)
        m.load_state_dict(ck["model_state"]); m.eval()
        r = probe(m, games, device)
        pm = r.get("mario"); pb = r.get("mario_breakable")
        print(f"{_step_of(c):>7}  {('%.3f'%pm) if pm is not None else 'n/a':>24}  "
              f"{('%.3f'%pb) if pb is not None else 'n/a':>16}", flush=True)


if __name__ == "__main__":
    main()
