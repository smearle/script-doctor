"""Belief-NCA on the two Mario worlds — the NCA counterpart of mario_belief.py.

Same active-learning experiment (world disambiguation by information gain), same
data pipeline / policies / BFS start-states / IG probe as `mario_belief.py` +
`mario_explore.py`, but the belief backbone is the recurrent **spatial** NCA
(`NCABeliefModel`) instead of the causal Transformer (`AttnBeliefModel`). This is
the apples-to-apples AL comparison: identical mixture/q0/q1/IG heads, only the
belief recurrence differs. Param-matched to the AL transformer (~4.7M).

The belief is a hidden GRID carried across ticks via init_belief / update_belief
(global+axis pooling lets a single broken-Step cell propagate so the belief
sharpens — the fix for the prior NCA-belief that "did not sharpen").

    .venv/bin/python -u -m nca_wm.active_learning.mario_nca_belief --train --search --updates 12000
    .venv/bin/python -u -m nca_wm.active_learning.mario_nca_belief --probe
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.multigame_data import _engine, _masks, _perm, _read_padded
from nca_wm.active_learning.mario_explore import break_cols, navigate_to_break, UP, TICK
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig

CKPT = Path("nca_wm/active_learning/ckpts/mario_nca_belief.pt")


# ----------------------------- training -----------------------------
def _vmask(CM, CH):
    """(N,H,W) cell mask + (N,C) channel mask -> (N,C,H,W) validity mask."""
    return CH[:, :, None, None] * CM[:, None, :, :]


def roll_loss(model, O, A, R, CM, CH, usage_w=0.0):
    """Recurrent belief roll: per-tick q0/q1 mixture NLL, masked to the real grid.

    B starts from the first frame, the head at tick t predicts O[t+1] from the
    belief BEFORE ingesting it (so the belief must carry hidden dynamics), then
    the belief ingests (O[t+1], A[t]). Mirrors mario_belief.forward_traj but with
    the spatial-NCA recurrence."""
    cm = CM
    vmask = _vmask(CM, CH)
    B = model.init_belief(O[:, 0]) * cm[:, None]
    T = A.shape[1]
    q0_sum = q1_sum = 0.0
    for t in range(T):
        a, onext, oresamp = A[:, t], O[:, t + 1], R[:, t]
        l0, p0 = model.q0_logits(B, a)
        l1, p1 = model.q1_logits(B, a, onext)
        q0 = model.mixture_nll(l0, p0, onext, vmask).mean()
        q1 = model.mixture_nll(l1, p1, oresamp, vmask).mean()
        q0_sum = q0_sum + q0
        q1_sum = q1_sum + q1
        B = model.update_belief(B, onext, a, cell_mask=cm)
    return (q0_sum + q1_sum) / T, q0_sum / T


@torch.no_grad()
def per_world_nll(model, games, device, n=120, n_steps=10, seed=7, start_pools=None):
    rng = random.Random(seed)
    policies = MB.make_policies(games)
    out = {}
    for g in games:
        sp = {g.gist: start_pools[g.gist]} if start_pools else None
        O, A, R, CM, CH = MB.mario_batch([g], n, n_steps, rng, policies, sp)
        O, A, R, CM, CH = [x.to(device) for x in (O, A, R, CM, CH)]
        cm, vmask = CM, _vmask(CM, CH)
        B = model.init_belief(O[:, 0]) * cm[:, None]
        curve = []
        for t in range(n_steps):
            l0, p0 = model.q0_logits(B, A[:, t])
            curve.append(model.mixture_nll(l0, p0, O[:, t + 1], vmask).mean().item())
            B = model.update_belief(B, O[:, t + 1], A[:, t], cell_mask=cm)
        out[g.gist] = (float(np.mean(curve)), curve)
    return out


# ----------------------------- IG probe -----------------------------
class NcaMarioCtx:
    """Engine + carried spatial belief B (init_belief/update_belief). Exposes the
    attrs mario_explore.navigate_to_break expects (grid/pb/sb/fb/step)."""
    def __init__(self, model, game, device, rng, n_samples=12):
        self.m, self.dev, self.game, self.ns = model, device, game, n_samples
        self.perm = _perm(game.n_obj, MB.CMAX, rng)
        cell, chan = _masks(game.n_obj, game.H, game.W, self.perm, MB.CMAX, MB.HMAX, MB.WMAX)
        self.cellT = torch.from_numpy(cell)[None].to(device)
        self.vmask = torch.from_numpy(chan[:, None, None] * cell[None]).to(device)[None]
        self.eng = _engine(game.json_str, 0)
        self.pb, self.sb, self.fb = (MB._bit(self.eng, n) for n in ("Player", "Step", "Floor"))
        self.B = self.m.init_belief(self._read()) * self.cellT[:, None]
        self.last_steps = self._stepcount(); self.n_break = 0; self.n_disambig_jump = 0

    def _read(self):
        return torch.from_numpy(_read_padded(self.eng, self.game.n_obj, self.perm,
                                             MB.CMAX, MB.HMAX, MB.WMAX))[None].to(self.dev)

    def grid(self):
        return MB._grid(self.eng)

    def _stepcount(self):
        return int(((self.grid() >> self.sb) & 1).sum())

    @torch.no_grad()
    def ig(self, ai):
        return self.m.information_gain(self.B, torch.tensor([ai], device=self.dev),
                                       n_samples=self.ns, vmask=self.vmask)

    def is_disambig_jump(self, ai):
        cols, info = break_cols(self.grid(), self.pb, self.sb, self.fb)
        return ai == UP and info is not None and info[2] and info[1] in cols

    @torch.no_grad()
    def step(self, ai):
        dis = self.is_disambig_jump(ai)
        self.eng.process_input(ai)
        k = 0
        while self.eng.is_againing() and k < 50:
            self.eng.process_input(-1); k += 1
        sc = self._stepcount()
        if sc < self.last_steps:
            self.n_break += 1
        self.last_steps = sc
        if dis:
            self.n_disambig_jump += 1
        self.B = self.m.update_belief(self.B, self._read(),
                                      torch.tensor([ai], device=self.dev), cell_mask=self.cellT)


@torch.no_grad()
def ig_probe(model, games, device, n_samples=12, seed=0):
    """Disambiguating-jump IG: fresh (world unresolved) vs known (after 1 jump).
    Expect fresh >> known, UP >> other actions, open-air ~0 — same as mario_explore."""
    from nca_wm.active_learning.mario_explore import ACTIONS, NA
    print(f"NCA-belief IG probe (n_samples={n_samples})  [UP=jump]")
    for game in games:
        rng = random.Random(seed)
        ctx = NcaMarioCtx(model, game, device, rng, n_samples)
        cols0, info0 = break_cols(ctx.grid(), ctx.pb, ctx.sb, ctx.fb)
        ig_air = ctx.ig(UP)
        ok = navigate_to_break(ctx)
        per_act = {ACTIONS[a]: ctx.ig(a) for a in range(NA)} if ok else {}
        ig_fresh = per_act.get("UP")
        if ok:
            ctx.step(UP)
        for _ in range(3):
            ctx.step(TICK)
        ok2 = navigate_to_break(ctx)
        ig_known = ctx.ig(UP) if ok2 else None
        print(f"\n  world {game.gist}")
        print(f"    open-air UP IG (control)      : {ig_air:+.3f}  [break-cols elsewhere={cols0}]")
        if ok:
            print(f"    fresh under-platform, per-act : " +
                  "  ".join(f"{k} {v:+.3f}" for k, v in per_act.items()))
            print(f"    -> fresh UP IG               : {ig_fresh:+.3f}")
        print(f"    known UP IG (after 1 jump)    : "
              f"{ig_known:+.3f}" if ig_known is not None else "    known UP IG: (no 2nd break col)")
        print(f"    breaks={ctx.n_break} disambig-jumps={ctx.n_disambig_jump}")


# ----------------------------- main -----------------------------
def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)
    games = MB.build_worlds()
    pol = MB.make_policies(games, eps=args.eps, use_explorer=not args.search)
    start_pools = None
    if args.search:
        start_pools = {}
        for g in games:
            t = time.time()
            start_pools[g.gist] = MB.collect_start_states(g, args.pool, args.bfs_cap, args.seed)
            print(f"BFS start-states {g.gist}: pool {len(start_pools[g.gist])} "
                  f"({time.time()-t:.0f}s)", flush=True)
    cfg = BeliefConfig(n_obj=MB.CMAX, n_act=MB.N_ACT, d=args.d, d_b=args.d_b,
                       d_cond=args.d_cond, K=args.K, nca_steps=args.nca_steps)
    model = NCABeliefModel(cfg).to(device)
    print(f"NCA-belief params {sum(p.numel() for p in model.parameters()):,} | "
          f"d={cfg.d} d_b={cfg.d_b} nca_steps={cfg.nca_steps} K={cfg.K}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(args.seed); t0 = time.time(); model.train()
    for step in range(args.updates):
        O, A, R, CM, CH = MB.mario_batch(games, args.batch_size, args.n_steps, rng, pol, start_pools)
        O, A, R, CM, CH = [x.to(device) for x in (O, A, R, CM, CH)]
        for g in opt.param_groups:
            g["lr"] = args.lr * (0.5 * (1 + math.cos(math.pi * min(step / args.updates, 1))))
        loss, _q0 = roll_loss(model, O, A, R, CM, CH, args.usage_w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 200 == 0:
            print(f"step {step:5d}  loss {loss.item():.3f}  upd/s "
                  f"{(step+1)/max(time.time()-t0,1e-9):.1f}", flush=True)
        if step > 0 and step % args.eval_every == 0:
            model.eval()
            for k, (m, curve) in per_world_nll(model, games, device, start_pools=start_pools).items():
                print(f"  [{step}] {k:16s} q0NLL mean {m:.3f}  per-step "
                      f"{' '.join(f'{x:.2f}' for x in curve)}", flush=True)
            model.train()
            CKPT.parent.mkdir(exist_ok=True)
            torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                        "worlds": [w[0] for w in MB.WORLDS]}, CKPT)
    model.eval()
    print("\nFINAL per-world q0 NLL:")
    for k, (m, curve) in per_world_nll(model, games, device, start_pools=start_pools).items():
        print(f"  {k:16s} mean {m:.4f}  per-step {' '.join(f'{x:.2f}' for x in curve)}")
    CKPT.parent.mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                "worlds": [w[0] for w in MB.WORLDS]}, CKPT)
    print(f"saved {CKPT}", flush=True)
    print("\n=== IG probe (frozen, just-trained model) ===")
    ig_probe(model, games, device, args.n_samples, args.seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--updates", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=16)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--d-b", type=int, default=256)
    p.add_argument("--d-cond", type=int, default=160)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--nca-steps", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--usage-w", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--search", action="store_true")
    p.add_argument("--pool", type=int, default=30000)
    p.add_argument("--bfs-cap", type=int, default=90000)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    if args.train:
        train(args)
    elif args.probe:
        device = torch.device(args.device)
        ck = torch.load(CKPT, map_location=device)
        model = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model_state"]); model.eval()
        for q in model.parameters():
            q.requires_grad_(False)
        ig_probe(model, MB.build_worlds(), device, args.n_samples, args.seed)
    else:
        print("pass --train or --probe")


if __name__ == "__main__":
    main()
