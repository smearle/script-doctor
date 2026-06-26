"""Learned exploration policy (A2C) maximizing CUMULATIVE information gain.

Greedy depth-1 IG is myopic (can't navigate to set up multi-step mechanics). Here a
small policy/value net is trained by advantage actor-critic to maximize the
discounted sum of per-step IG (the intrinsic reward from the frozen attention WM).
Temporal credit assignment lets it head toward high-IG regions even when the
immediate IG is 0. Eval = rule-firing coverage vs random.

    .venv/bin/python -u -m nca_wm.active_learning.explore_policy --wm <ckpt> --updates 800
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning.attn_belief_model import (AttnBeliefModel, AttnConfig,
                                                      masked_pool)
from nca_wm.active_learning.multigame_data import (MultiGameSet, _engine,
                                                   _masks, _perm, _read_padded)

NA = len(V.ACTIONS)


class Policy(nn.Module):
    def __init__(self, d_in, hidden=192):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, NA)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


class _GameCtx:
    """Per-episode engine + WM-belief bookkeeping."""
    def __init__(self, wm, game, mg, device, rng):
        self.wm, self.dev = wm, device
        self.perm = _perm(game.n_obj, mg.cmax, rng)
        cell, chan = _masks(game.n_obj, game.H, game.W, self.perm, mg.cmax, mg.hmax, mg.wmax)
        self.cellT = torch.from_numpy(cell)[None].to(device)
        self.vmask = torch.from_numpy(chan[:, None, None] * cell[None]).to(device)[None]
        self.eng = _engine(game.json_str, 0); self.eng.set_track_rules_fired(True)
        self.total = self.eng.get_rule_count()
        self.g, self.mg = game, mg
        self.sp, self.pooled = self._read()
        self.ps, self.pa = [self.pooled], [NA]
        self.fired = set()

    def _read(self):
        o = torch.from_numpy(_read_padded(self.eng, self.g.n_obj, self.perm,
                                          self.mg.cmax, self.mg.hmax, self.mg.wmax))[None].to(self.dev)
        sp = self.wm.encode_frame(o)
        return sp, masked_pool(sp, self.cellT)

    @torch.no_grad()
    def belief(self):
        return self.wm.belief_now(torch.stack(self.ps, 1), torch.tensor([self.pa], device=self.dev))

    @torch.no_grad()
    def ig(self, bel, ai, n):
        return self.wm.information_gain(bel, self.sp, torch.tensor([ai], device=self.dev),
                                        self.cellT, self.vmask, n)

    def step(self, ai):
        self.eng.clear_rules_fired()
        self.eng.process_input(V.ACTION_TO_INPUT[V.ACTIONS[ai]])
        n = 0
        while self.eng.is_againing() and n < 50:
            self.eng.process_input(-1); n += 1
        self.fired |= set(self.eng.get_rules_fired())
        self.sp, self.pooled = self._read()
        self.ps.append(self.pooled); self.pa.append(ai)


def rollout(wm, policy, game, mg, device, T, rsamp, rng, mode="train"):
    ctx = _GameCtx(wm, game, mg, device, rng)
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
    return logps, vals, rews, ents, ctx.fired, ctx.total


def train(wm, policy, mg, device, args):
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng = random.Random(0)
    for upd in range(args.updates):
        L, Vv, Rr, Ee = [], [], [], []          # batch-pooled logp / value / return / entropy
        rsum = 0.0; n_ep = 0
        for _ in range(args.episodes):
            g = rng.choice(mg.games)
            logps, vals, rews, ents, _, _ = rollout(wm, policy, g, mg, device, args.T, args.rsamp, rng)
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
        adv = (returns - vals).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)        # normalize ADVANTAGE across batch
        ploss = -(logps * adv).mean()
        vloss = F.mse_loss(vals, returns)                    # value predicts un-normalized returns
        loss = ploss + 0.5 * vloss - args.ent_w * ents.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        if upd % 50 == 0:
            print(f"upd {upd:4d}  mean-IG-reward {rsum/max(n_ep,1):.2f}  Vpred {vals.mean().item():.1f}  "
                  f"loss {loss.item():.2f}", flush=True)


@torch.no_grad()
def eval_coverage(wm, policy, mg, device, n_games, T, seed=1):
    rng = random.Random(seed)
    games = rng.sample(mg.games, min(n_games, len(mg.games)))
    cp, cr, ig_only, rnd_only = [], [], [], []
    for i, g in enumerate(games):
        r2 = random.Random(i)
        _, _, _, _, fp, tot = rollout(wm, policy, g, mg, device, T, 0, random.Random(i), "sample")
        if tot == 0:
            continue
        # random baseline (paired)
        ctx = _GameCtx(wm, g, mg, device, random.Random(i))
        for _ in range(T):
            ctx.step(random.Random(1000 + i).randrange(NA) if False else r2.randrange(NA))
        cp.append(len(fp) / tot); cr.append(len(ctx.fired) / tot)
        ig_only.append(len(fp - ctx.fired)); rnd_only.append(len(ctx.fired - fp))
    m = lambda x: float(np.mean(x))
    print(f"  POLICY coverage {m(cp):.3f} vs random {m(cr):.3f}  (delta {m(cp)-m(cr):+.3f}) | "
          f"policy-only {m(ig_only):.2f} random-only {m(rnd_only):.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", default="nca_wm/active_learning/ckpts/attn_belief.pt")
    p.add_argument("--updates", type=int, default=800)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--rsamp", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ent-w", type=float, default=0.01)
    p.add_argument("--n-games-train", type=int, default=256)
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)

    ck = torch.load(args.wm, map_location=device)
    wm = AttnBeliefModel(AttnConfig(**ck["cfg"])).to(device)
    wm.load_state_dict(ck["model_state"]); wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    mg = MultiGameSet(cmax=ck["cfg"]["n_obj"], split="train", seed=0)
    mg.games = mg.games[:args.n_games_train]
    policy = Policy(ck["cfg"]["d_cond"] + ck["cfg"]["d"]).to(device)
    print(f"WM {args.wm} frozen | policy params {sum(q.numel() for q in policy.parameters()):,} | "
          f"{len(mg)} games", flush=True)

    print("[before training]"); eval_coverage(wm, policy, mg, device, args.eval_games, args.T)
    train(wm, policy, mg, device, args)
    print("[after training]"); eval_coverage(wm, policy, mg, device, args.eval_games, args.T)
    torch.save(policy.state_dict(), "nca_wm/active_learning/ckpts/explore_policy.pt")
    print("saved ckpts/explore_policy.pt", flush=True)


if __name__ == "__main__":
    main()
