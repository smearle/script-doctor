"""IG probe + A2C explorer for the two-world Mario belief WM.

Loads ckpts/mario_belief.pt (trained by mario_belief.py) and:
  - ig_probe: jump-into-platform IG when the world is UNRESOLVED (fresh) vs AFTER
    one disambiguating jump (known), on BOTH worlds, plus an open-air control and
    a per-action breakdown. Expect: fresh >> known; UP >> other actions; open-air ~0.
  - explore: train a small A2C policy to maximize cumulative IG (intrinsic reward
    from the frozen belief WM); eval = disambiguating-jump frequency (and breaks in
    the variant) vs a random baseline.

    .venv/bin/python -u -m nca_wm.active_learning.mario_explore --probe
    .venv/bin/python -u -m nca_wm.active_learning.mario_explore --explore --updates 600
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca_wm.active_learning.attn_belief_model import AttnBeliefModel, AttnConfig, masked_pool
from nca_wm.active_learning.multigame_data import _engine, _masks, _perm, _read_padded
from nca_wm.active_learning import mario_belief as MB

ACTIONS = MB.ACTIONS
NA = MB.N_ACT                  # also the "no prev action" token id
UP, LEFT, DOWN, RIGHT, ACTION, TICK = 0, 1, 2, 3, 4, 5


def break_cols(g, pb, sb, fb):
    """Columns where a jump from the player's row breaks a Step (first solid <=4
    rows above is a Step). Returns (cols, (pr,pc,grounded)) or ([], None)."""
    H, Wd = g.shape
    pa = np.argwhere((g >> pb) & 1)
    if not len(pa):
        return [], None
    pr, pc = int(pa[0][0]), int(pa[0][1])

    def solid(c):
        return bool(c & ((1 << sb) | (1 << fb)))

    cols = []
    for c in range(Wd):
        for dr in range(1, 5):
            rr = pr - dr
            if rr < 0:
                break
            cell = int(g[rr, c])
            if solid(cell):
                if (cell >> sb) & 1:
                    cols.append(c)
                break
    grounded = pr + 1 < H and solid(int(g[pr + 1, pc]))
    return cols, (pr, pc, grounded)


class MarioCtx:
    """Per-episode engine + belief bookkeeping (mirrors explore_policy._GameCtx)."""
    def __init__(self, wm, game, device, rng):
        self.wm, self.dev, self.game = wm, device, game
        self.perm = _perm(game.n_obj, MB.CMAX, rng)
        cell, chan = _masks(game.n_obj, game.H, game.W, self.perm, MB.CMAX, MB.HMAX, MB.WMAX)
        self.cellT = torch.from_numpy(cell)[None].to(device)
        self.vmask = torch.from_numpy(chan[:, None, None] * cell[None]).to(device)[None]
        self.eng = _engine(game.json_str, 0); self.eng.set_track_rules_fired(True)
        self.pb, self.sb, self.fb = (MB._bit(self.eng, n) for n in ("Player", "Step", "Floor"))
        self.sp, self.pooled = self._read()
        self.ps, self.pa = [self.pooled], [NA]
        self.last_steps = self._stepcount(); self.n_break = 0; self.n_disambig_jump = 0

    def _read(self):
        o = torch.from_numpy(_read_padded(self.eng, self.game.n_obj, self.perm,
                                          MB.CMAX, MB.HMAX, MB.WMAX))[None].to(self.dev)
        sp = self.wm.encode_frame(o)
        return sp, masked_pool(sp, self.cellT)

    def grid(self):
        return MB._grid(self.eng)

    def _stepcount(self):
        return int(((self.grid() >> self.sb) & 1).sum())

    @torch.no_grad()
    def belief(self):
        return self.wm.belief_now(torch.stack(self.ps, 1),
                                  torch.tensor([self.pa], device=self.dev))

    @torch.no_grad()
    def ig(self, bel, ai, n):
        return self.wm.information_gain(bel, self.sp, torch.tensor([ai], device=self.dev),
                                        self.cellT, self.vmask, n)

    def is_disambig_jump(self, ai):
        cols, info = break_cols(self.grid(), self.pb, self.sb, self.fb)
        return ai == UP and info is not None and info[2] and info[1] in cols

    def step(self, ai):
        dis = self.is_disambig_jump(ai)
        self.eng.clear_rules_fired()
        self.eng.process_input(ai)            # action index == engine input id (5 = realtime tick)
        k = 0
        while self.eng.is_againing() and k < 50:
            self.eng.process_input(-1); k += 1
        sc = self._stepcount()
        if sc < self.last_steps:
            self.n_break += 1
        self.last_steps = sc
        if dis:
            self.n_disambig_jump += 1
        self.sp, self.pooled = self._read()
        self.ps.append(self.pooled); self.pa.append(ai)


def navigate_to_break(ctx, max_steps=24):
    """Walk (LEFT/RIGHT) toward the nearest break column, settling with TICK (gravity),
    WITHOUT jumping. Returns True when grounded on a break column."""
    for _ in range(max_steps):
        cols, info = break_cols(ctx.grid(), ctx.pb, ctx.sb, ctx.fb)
        if info is None or not cols:
            return False
        pr, pc, grounded = info
        if not grounded:
            ctx.step(TICK); continue
        if pc in cols:
            return True
        tgt = min(cols, key=lambda c: abs(c - pc))
        ctx.step(RIGHT if tgt > pc else LEFT)
    return False


@torch.no_grad()
def ig_probe(wm, games, device, n_samples=12, seed=0):
    print(f"IG probe (n_samples={n_samples})  [UP=jump]")
    for game in games:
        rng = random.Random(seed)
        ctx = MarioCtx(wm, game, device, rng)
        # control: open-air jump (player starts where no platform is within reach)
        cols0, info0 = break_cols(ctx.grid(), ctx.pb, ctx.sb, ctx.fb)
        ig_air = ctx.ig(ctx.belief(), UP, n_samples)
        # navigate under a platform (belief still fresh = world unresolved)
        ok = navigate_to_break(ctx)
        bel = ctx.belief()
        per_act = {ACTIONS[a]: ctx.ig(bel, a, n_samples) for a in range(NA)} if ok else {}
        ig_fresh = per_act.get("UP")
        # execute the disambiguating jump -> belief resolves which world
        if ok:
            ctx.step(UP)
        # settle (gravity ticks) + navigate to another break column, measure IG again (known)
        for _ in range(3):
            ctx.step(TICK)
        ok2 = navigate_to_break(ctx)
        ig_known = ctx.ig(ctx.belief(), UP, n_samples) if ok2 else None
        print(f"\n  world {game.gist}")
        pc0 = info0[1] if info0 else None
        print(f"    open-air UP IG (control)      : {ig_air:+.3f}  "
              f"[player col {pc0}, not under a platform; break-cols elsewhere={cols0}]")
        if ok:
            print(f"    fresh under-platform, per-act : " +
                  "  ".join(f"{k} {v:+.3f}" for k, v in per_act.items()))
            print(f"    -> fresh UP IG               : {ig_fresh:+.3f}")
        print(f"    known UP IG (after 1 jump)    : "
              f"{ig_known:+.3f}" if ig_known is not None else "    known UP IG: (no 2nd break col)")
        print(f"    breaks this episode={ctx.n_break} disambig-jumps={ctx.n_disambig_jump}")


# ----------------------------- A2C explorer -----------------------------
class Policy(nn.Module):
    def __init__(self, d_in, hidden=192):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, NA); self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


def rollout(wm, policy, game, device, T, rsamp, rng, mode="train"):
    ctx = MarioCtx(wm, game, device, rng)
    logps, vals, rews, ents = [], [], [], []
    for _ in range(T):
        bel = ctx.belief()
        feat = torch.cat([bel, ctx.pooled], -1).detach()
        logits, value = policy(feat)
        dist = torch.distributions.Categorical(logits=logits)
        ai = logits.argmax(-1) if mode == "greedy" else dist.sample()
        if mode == "train":
            r = ctx.ig(bel, int(ai), rsamp)
            logps.append(dist.log_prob(ai)); vals.append(value)
            rews.append(r); ents.append(dist.entropy())
        ctx.step(int(ai))
    return logps, vals, rews, ents, ctx


def train_explorer(wm, policy, games, device, args):
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng = random.Random(0)
    for upd in range(args.updates):
        L, Vv, Rr, Ee = [], [], [], []
        rsum = 0.0; n_ep = 0
        for _ in range(args.episodes):
            g = rng.choice(games)
            logps, vals, rews, ents, _ = rollout(wm, policy, g, device, args.T, args.rsamp, rng)
            if not rews:
                continue
            R = 0.0; returns = []
            for r in reversed(rews):
                R = r + args.gamma * R; returns.insert(0, R)
            Rr += returns; L += logps; Vv += vals; Ee += ents
            rsum += float(np.mean(rews)); n_ep += 1
        if not Rr:
            continue
        returns = torch.tensor(Rr, device=device)
        vals = torch.stack(Vv); logps = torch.stack(L); ents = torch.stack(Ee)
        adv = (returns - vals).detach(); adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = -(logps * adv).mean() + 0.5 * F.mse_loss(vals, returns) - args.ent_w * ents.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        if upd % 50 == 0:
            print(f"upd {upd:4d}  mean-IG-reward {rsum/max(n_ep,1):+.3f}  loss {loss.item():.2f}", flush=True)


@torch.no_grad()
def eval_explorer(wm, policy, games, device, T, n_ep=40, seed=1):
    for game in games:
        pj = pb_ = rj = rb_ = 0.0
        for i in range(n_ep):
            _, _, _, _, ctx = rollout(wm, policy, game, device, T, 0, random.Random(seed + i), "sample")
            pj += ctx.n_disambig_jump; pb_ += ctx.n_break
            r2 = random.Random(1000 + seed + i)
            rc = MarioCtx(wm, game, device, random.Random(seed + i))
            for _ in range(T):
                rc.step(r2.randrange(NA))
            rj += rc.n_disambig_jump; rb_ += rc.n_break
        n = n_ep
        print(f"  {game.gist:16s} policy disambig-jumps/ep {pj/n:.2f} breaks/ep {pb_/n:.2f} | "
              f"random disambig-jumps/ep {rj/n:.2f} breaks/ep {rb_/n:.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="nca_wm/active_learning/ckpts/mario_belief.pt")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--explore", action="store_true")
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--rsamp", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ent-w", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device)
    wm = AttnBeliefModel(AttnConfig(**ck["cfg"])).to(device)
    wm.load_state_dict(ck["model_state"]); wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    games = MB.build_worlds()
    print(f"WM {args.ckpt} frozen | worlds {[g.gist for g in games]}", flush=True)

    if args.probe or not args.explore:
        ig_probe(wm, games, device, args.n_samples, args.seed)
    if args.explore:
        policy = Policy(ck["cfg"]["d_cond"] + ck["cfg"]["d"]).to(device)
        print("[before]"); eval_explorer(wm, policy, games, device, args.T)
        train_explorer(wm, policy, games, device, args)
        print("[after]"); eval_explorer(wm, policy, games, device, args.T)
        torch.save(policy.state_dict(), "nca_wm/active_learning/ckpts/mario_explore.pt")
        print("saved ckpts/mario_explore.pt", flush=True)


if __name__ == "__main__":
    main()
