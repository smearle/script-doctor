#!/usr/bin/env python3
"""Evaluate a recurrent run on a game's HELD-OUT human val episodes ({game}_human_val_seq.npz,
written by build_dataset). This is the non-leaked human-distribution promotion gate: teacher-
forced changed-cell accuracy on human users that were excluded from training."""
import argparse, numpy as np
from nca_wm.autumn import infer as I

def eval_run(run, val_npz, device="cuda"):
    model, cfg = I.load_run(f"nca_wm/autumn/runs/{run}", device=device)
    d = dict(np.load(val_npz, allow_pickle=True))
    S, A = d["states"], d["actions"]; lens = d.get("lengths")
    rec = cfg.get("recurrent")
    chg = ok = cell = cok = 0
    for e in range(len(S)):
        L = int(lens[e]) if lens is not None else A.shape[1]
        h = None; prev = S[e, 0]
        for t in range(L):
            a = [int(x) for x in A[e, t]]
            act = ("click", a[1], a[2]) if a[0] == 1 else (["noop","click","up","down","left","right"][a[0]],)
            if rec: pred, h = I.recurrent_step(model, cfg, S[e, t], act, h)
            else: pred = I.wm_step(model, cfg, S[e, t], act, prev_board=prev)
            tgt, cur = S[e, t+1], S[e, t]; ch = tgt != cur
            chg += int(ch.sum()); ok += int(((pred == tgt) & ch).sum())
            cell += tgt.size; cok += int((pred == tgt).sum()); prev = cur
    return 100*ok/max(chg,1), 100*cok/max(cell,1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of run names to compare")
    ap.add_argument("--val", required=True, help="path to {game}_human_val_seq.npz")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    for run in a.runs.split(","):
        try:
            chg, cell = eval_run(run.strip(), a.val, a.device)
            print(f"  {run.strip():28s} human-val changed-cell={chg:5.1f}%  cell={cell:6.2f}%")
        except Exception as e:
            print(f"  {run.strip():28s} ERROR {str(e)[:80]}")
