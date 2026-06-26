#!/usr/bin/env python3
"""Targeted 'ammo-discipline' curriculum for the Mario bullet mechanic.

Generates clean, long, decorrelated trajectories that isolate the hidden mario.bullets counter:
  collect (4,12) coin (ammo->1) -> LONG no-click run (ammo persists, NO bullet) -> click (fire 1,
  ammo->0) -> noops (bullet travels up / despawns) -> click@ammo0 (NO bullet) -> noops.
Interleaves random arrow moves for diversity. Guards the enemy-death interpreter crash by truncating.
Saves (E,L+1,C,H,W) object sequences matching run_recurrent's loader.
"""
import argparse, numpy as np
from nca_wm.autumn.collect import AutumnGame, _suppress
from nca_wm.autumn.objects import render_objects

VOCAB={"mario":0,"steps":1,"coins":2,"enemy":3,"bullets":4}
AMAP={"noop":0,"click":1,"up":2,"down":3,"left":4,"right":5}

def mxy(g):
    ys,xs=np.where(g[0]==1); return (int(xs[0]),int(ys[0])) if len(xs) else (4,15)

def gen_episode(seed, rng, maxL=70):
    env=AutumnGame("mario", seed=seed)
    def grid(): return render_objects(env, VOCAB)
    S=[grid()]; A=[]
    def step(a):
        # a is ("noop",)/("up",)/... or ("click",x,y)
        try:
            env.apply(a)
        except RuntimeError:
            return False
        at=AMAP[a[0]]; cx=a[1] if a[0]=="click" else -1; cy=a[2] if a[0]=="click" else -1
        A.append((at,cx,cy)); S.append(grid()); return True

    # 1) go to x=4 and jump to collect (4,12); sprinkle a stray noop for phase diversity
    plan = ["left","left","left"] + (["noop"] if rng.random()<0.5 else []) + ["up","noop","noop"]
    for t in plan:
        if not step((t,)): return None
    # 2) LONG no-click run (ammo held silently): mix noop + arrows, NO clicks
    nhold = int(rng.integers(8, 22))
    for _ in range(nhold):
        r=rng.random()
        a = ("noop",) if r<0.55 else ((["left","right","up","down"][rng.integers(4)],))
        if not step(a): break
    # 3) fire once (ammo>0 -> bullet). click at mario's cell.
    g=S[-1]; mx,my=mxy(g)
    if not step(("click",mx,my)): 
        pass
    # 4) noops while bullet travels (guard crash on enemy hit -> truncates)
    for _ in range(int(rng.integers(3, 7))):
        if not step(("noop",)): break
    # 5) click again @ ammo0 -> NO bullet (the key negative), then noops
    g=S[-1]; mx,my=mxy(g)
    step(("click",mx,my))
    for _ in range(int(rng.integers(4, 10))):
        r=rng.random()
        a=("noop",) if r<0.7 else ((["left","right"][rng.integers(2)],))
        if not step(a): break
    # 6) maybe a third click@0 for good measure
    if rng.random()<0.5:
        g=S[-1]; mx,my=mxy(g); step(("click",mx,my))
        for _ in range(int(rng.integers(2,6))):
            if not step(("noop",)): break
    if len(A) < 6: return None
    return np.array(S[:maxL+1]), np.array(A[:maxL])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="nca_wm/autumn/data/mario_obj_ammo.npz")
    a=ap.parse_args()
    rng=np.random.default_rng(a.seed)
    eps=[]
    with _suppress():
        for i in range(a.n):
            r=gen_episode(int(rng.integers(1<<30)), rng)
            if r is not None: eps.append(r)
    L=max(len(A) for _,A in eps); C=5; gs=16
    E=len(eps)
    S=np.zeros((E,L+1,C,gs,gs),np.uint8); Aout=np.zeros((E,L,3),np.int64)
    for i,(s,ac) in enumerate(eps):
        l=len(ac)
        S[i,:l+1]=s
        for t in range(l+1,L+1): S[i,t]=s[-1]   # pad with last state
        Aout[i,:l]=ac
    # quick coverage stats
    BUL=4; clk=int((Aout[:,:,0]==1).sum())
    fire=nofire=0
    for e in range(E):
        for t in range(L):
            if int(Aout[e,t,0])==1:
                c=int((S[e,t,BUL]>0).sum()); n=int((S[e,t+1,BUL]>0).sum())
                if n>c: fire+=1
                else: nofire+=1
    print(f"[mario/ammo] {E} eps, L={L}, clicks={clk} fire(ammo>0)={fire} no-fire(ammo0)={nofire}")
    np.savez_compressed(a.out, states=S, actions=Aout, vocab=np.array(list(VOCAB.keys())),
                        grid_size=np.int32(gs), game=np.str_("mario"))
    print(f"saved {a.out}")

if __name__=="__main__": main()
