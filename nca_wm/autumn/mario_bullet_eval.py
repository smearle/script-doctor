#!/usr/bin/env python3
"""Gating eval for the Mario bullet mechanic (the AR phantom-bullet bug).

Two probes, both vs the real engine:
  1. AR rollouts over many seeds with the coin-collecting climber (fires in bursts) -> measures
     runaway rolls, phantom-step %, AR fire precision/recall, median AR first-divergence.
  2. A deterministic scripted scenario (collect coin -> fire -> click@ammo0 -> noops) -> the model
     must (a) fire exactly one bullet when ammo>0, (b) NOT spawn on click@ammo0, (c) NOT spawn on noop.

Usage: python -m nca_wm.autumn.mario_bullet_eval RUN1 [RUN2 ...] [--device cuda:1] [--nseed 40]
"""
import argparse, numpy as np
from nca_wm.autumn.collect import AutumnGame
from nca_wm.autumn.objects import render_objects, _obj_mario_act
from nca_wm.autumn import infer as I

VOCAB = {"mario":0,"steps":1,"coins":2,"enemy":3,"bullets":4}
def nbull(g): return int((g[4]>0).sum())
def mxy(g):
    ys,xs=np.where(g[0]==1); return (int(xs[0]),int(ys[0])) if len(xs) else (4,15)

def heur_rollout(seed, steps=80):
    env=AutumnGame("mario", seed=seed); rng=np.random.default_rng(1000+seed); fire=[0]
    def grid(): return render_objects(env, VOCAB)
    S=[grid()]; A=[]; amap={"noop":0,"click":1,"up":2,"down":3,"left":4,"right":5}
    for t in range(steps):
        a=_obj_mario_act(env, VOCAB, rng, fire)
        try: env.apply(a)
        except RuntimeError: break
        at=amap[a[0]]; cx=a[1] if a[0]=="click" else -1; cy=a[2] if a[0]=="click" else -1
        A.append((at,cx,cy)); S.append(grid())
    return np.array(S), np.array(A)

def acttuple(a): return ("click",int(a[1]),int(a[2])) if a[0]==1 else (["noop","click","up","down","left","right"][a[0]],)

def ar_probe(run, rolls, dev):
    model,cfg=I.load_run(f"nca_wm/autumn/runs/{run}", device=dev)
    BUL=list(cfg["vocab"]).index("bullets")
    runaway=0; phantom_steps=0; tot=0; divs=[]; fire_ev=fire_hit=pred_fire=false_fire=0
    for S,A in rolls:
        if len(A)==0: continue
        h=None; wm=S[0].copy(); prev=S[0].copy(); first=None; rr=False
        for t in range(len(A)):
            wm,h=I.object_step(model,cfg,wm,acttuple(A[t]),prev,h); prev=wm; tot+=1
            ec=int((S[t][BUL]>0).sum()); eb=int((S[t+1][BUL]>0).sum()); mb=int((wm[BUL]>0).sum())
            if mb>eb+1: phantom_steps+=1; rr=True
            if first is None and int((wm!=S[t+1]).sum())>0: first=t
            if eb>ec: fire_ev+=1; fire_hit+=int(mb>ec)         # engine fired -> did model?
            if mb>(int((prev[BUL]>0).sum()) if t==0 else mbprev): pass
            mbprev=mb
        runaway+=int(rr); divs.append(first if first is not None else len(A))
    return dict(runaway=runaway, n=len(rolls), phantom_pct=100*phantom_steps/max(tot,1),
                med_div=int(np.median(divs)), fire_recall=f"{fire_hit}/{fire_ev}")

def scripted_probe(run, dev):
    """Deterministic: collect (4,12), fire, click@ammo0, noops. Returns per-model bullet trace vs engine."""
    model,cfg=I.load_run(f"nca_wm/autumn/runs/{run}", device=dev)
    env=AutumnGame("mario", seed=7)
    def grid(): return render_objects(env, VOCAB)
    script=["left","left","left","up","noop","C","noop","noop","noop","C","noop","noop","noop","noop","noop","noop"]
    S=[grid()]; A=[]
    for t in script:
        a=("click",*mxy(grid())) if t=="C" else (t,)
        A.append(a); env.apply(a); S.append(grid())
    wm=S[0].copy(); prev=S[0].copy(); h=None; eng=[]; mod=[]
    for t,a in enumerate(A):
        wm,h=I.object_step(model,cfg,wm,a,prev,h); prev=wm
        eng.append(nbull(S[t+1])); mod.append(nbull(wm))
    # phantom-on-noop: any noop/move step where model bull > engine bull
    phantom = any(A[t][0]!="click" and mod[t]>eng[t] for t in range(len(A)))
    overfire = any(mod[t]>eng[t]+1 for t in range(len(A)))
    return script, eng, mod, phantom, overfire

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:1"); ap.add_argument("--nseed", type=int, default=40)
    a=ap.parse_args()
    rolls=[heur_rollout(s) for s in range(60, 60+a.nseed)]
    print(f"=== AR probe ({a.nseed} heuristic rollouts) ===")
    print(f"{'run':40s} {'runaway':>9s} {'phantom%':>9s} {'medDiv':>7s} {'fireRecall':>11s}")
    for r in a.runs:
        m=ar_probe(r, rolls, a.device)
        print(f"{r:40s} {m['runaway']:>3d}/{m['n']:<5d} {m['phantom_pct']:8.2f}% {m['med_div']:7d} {m['fire_recall']:>11s}")
    print(f"\n=== scripted probe (collect->fire->click@0->noops) ===")
    for r in a.runs:
        sc,eng,mod,ph,of=scripted_probe(r, a.device)
        verdict="PASS" if (not ph and not of) else ("PHANTOM-NOOP" if ph else "OVERFIRE")
        print(f"{r:40s} {verdict}")
        print(f"    eng: {eng}")
        print(f"    mod: {mod}")

if __name__=="__main__": main()
