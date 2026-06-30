"""Train an RL player whose ONLY reward is information gain from the FROZEN jax
NCA world model + adapter (q0/q1), and test whether it learns to seek + break
the Step -- the jax analogue of mario_nca_explore.py.

The world model (NCAWorldModel) and the IG adapter (AdapterHead) are frozen; an
A2C policy is rewarded by the per-step IG (the same q0/q1 estimate the viewer
shows). If IG is calibrated (high on the disambiguating jump-into-Step, ~0
elsewhere), a reward-maximizing policy should navigate under a Step and jump far
more than a random policy. The policy body is a small NCA-like conv net (mirrors
the WM's spatial structure) with a global-pooled readout to a discrete action +
value head. Feedforward for now (no recurrence).

Reward, env, IG, and sprite rendering all reuse nca_wm.active_learning.mario_nca_serve.

  python -m nca_wm.active_learning.ig_rl --train --eval --gif
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca_wm.active_learning.mario_nca_serve import (
    load_base, load_adapter, build_render_backends, NCACtx, N_ACTIONS, CMAX,
)
from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.mario_explore import break_cols, UP
from nca_wm.state_ops import _multihot_to_objects

NA = N_ACTIONS


# ----------------------------- NCA-like policy -----------------------------
class NCAPolicy(nn.Module):
    """Small NCA-ish conv encoder over the (C,H,W) board -> global mean+max pool
    -> discrete action logits + scalar value. Feedforward."""
    def __init__(self, c_in=CMAX, hid=96, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(c_in, hid, 3, padding=1), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(hid, hid, 3, padding=1), nn.GELU()]
        self.enc = nn.Sequential(*layers)
        self.pi = nn.Linear(2 * hid, NA)
        self.v = nn.Linear(2 * hid, 1)

    def forward(self, obs):                       # obs (B,C,H,W)
        h = self.enc(obs)
        feat = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=-1)
        return self.pi(feat), self.v(feat).squeeze(-1)


def _obs_t(ctx, device):
    return torch.from_numpy(ctx._engine_obs()[None]).float().to(device)


def _under_platform(ctx):
    cols, info = break_cols(ctx._grid(), ctx.pb, ctx.sb, ctx.fb)
    return info is not None and info[2] and info[1] in cols


# ----------------------------- rollout + A2C -----------------------------
def rollout(q0, q1, policy, game, backend, device, T, rsamp, seed, mode="train"):
    ctx = NCACtx(q0, q1, game, backend, random.Random(seed))
    logps, vals, rews, ents = [], [], [], []
    n_under_up = 0
    for _ in range(T):
        obs = _obs_t(ctx, device)
        logits, value = policy(obs)
        dist = torch.distributions.Categorical(logits=logits[0])
        ai_t = logits[0].argmax(-1) if mode == "greedy" else dist.sample()
        ai = int(ai_t)
        if ai == UP and _under_platform(ctx):
            n_under_up += 1
        if mode == "train":
            r = max(0.0, ctx._ig(ai, rsamp))      # intrinsic reward = WM info-gain
            logps.append(dist.log_prob(ai_t)); vals.append(value[0])
            rews.append(r); ents.append(dist.entropy())
        ctx.step(ai)
    ctx.n_under_up = n_under_up
    return logps, vals, rews, ents, ctx


def train_explorer(q0, q1, policy, games, backends, device, args, best_path=None):
    import copy
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng = random.Random(0)
    best_r, best_state, recent = -1.0, None, []
    for upd in range(args.updates):
        L, Vv, Rr, Ee = [], [], [], []
        rsum = 0.0; n_ep = 0
        for _ in range(args.episodes):
            gi = rng.randrange(len(games)); g = games[gi]
            logps, vals, rews, ents, _ = rollout(
                q0, q1, policy, g, backends[g.gist], device, args.T, args.rsamp,
                rng.randrange(1 << 30))
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
        mr = rsum / max(n_ep, 1)
        recent.append(mr); recent = recent[-20:]
        sm = float(np.mean(recent))
        if sm > best_r:
            best_r = sm; best_state = copy.deepcopy(policy.state_dict())
        if upd % 10 == 0:
            print(f"upd {upd:4d}  mean-IG-reward {mr:+.3f}  smoothed {sm:+.3f}  "
                  f"best {best_r:+.3f}  loss {loss.item():.3f}", flush=True)
    if best_state is not None:
        policy.load_state_dict(best_state)
        if best_path:
            Path(best_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, best_path)
        print(f"[train] restored best policy (smoothed reward {best_r:+.3f})", flush=True)


@torch.no_grad()
def eval_explorer(q0, q1, policy, games, backends, device, T, n_ep=30, seed=1):
    print("\n=== eval: IG-policy vs random (under-platform jumps & breaks per episode) ===")
    for game in games:
        pj = pb = rj = rb = 0.0
        for i in range(n_ep):
            _, _, _, _, ctx = rollout(q0, q1, policy, game, backends[game.gist],
                                      device, T, 0, seed + i, "greedy")
            pj += ctx.n_under_up; pb += ctx.n_break
            rc = NCACtx(q0, q1, game, backends[game.gist], random.Random(1000 + seed + i))
            for _ in range(T):
                a = random.Random(2000 + seed + i + _).randrange(NA)
                if a == UP and _under_platform(rc):
                    rj += 1
                rc.step(a)
            rb += rc.n_break
        n = n_ep
        print(f"  {game.gist:16s} POLICY under-platform-UP/ep {pj/n:.2f} breaks/ep {pb/n:.2f} | "
              f"RANDOM under-platform-UP/ep {rj/n:.2f} breaks/ep {rb/n:.2f}", flush=True)


@torch.no_grad()
def make_gif(q0, q1, policy, game, backend, device, T, path, seed=7):
    import imageio
    ctx = NCACtx(q0, q1, game, backend, random.Random(seed))
    frames = []

    def render():
        o = ctx._engine_obs()
        crop = (o[:ctx.n_obj, :ctx.H, :ctx.W] > 0.5).astype(np.uint8)
        fr = backend.render_frame_from_objects(_multihot_to_objects(crop), ctx.W, ctx.H)
        import PIL.Image
        frames.append(np.array(PIL.Image.fromarray(fr).resize(
            (ctx.W * 16, ctx.H * 16), PIL.Image.NEAREST)))

    render()
    for _ in range(T):
        obs = _obs_t(ctx, device)
        logits, _ = policy(obs)
        ai = int(logits[0].argmax(-1))
        ctx.step(ai)
        render()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=0.25)
    print(f"  wrote {path}  ({len(frames)} frames, {ctx.n_break} breaks)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--rsamp", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent_w", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")     # policy on CPU; jax IG on GPU
    ap.add_argument("--ckpt", default="nca_wm/active_learning/ckpts/ig_rl_policy.pt")
    ap.add_argument("--gif_dir", default="/tmp/ig_rl_gifs")
    args = ap.parse_args()

    device = torch.device(args.device)
    q0 = load_base()
    q1 = load_adapter()
    _games_by_gist, backends = build_render_backends()   # backends keyed by gist(==name)
    games = MB.build_worlds()
    policy = NCAPolicy().to(device)
    nparam = sum(p.numel() for p in policy.parameters())
    print(f"[ig_rl] policy params: {nparam:,}  games={[g.gist for g in games]}", flush=True)

    if args.train:
        train_explorer(q0, q1, policy, games, backends, device, args, best_path=args.ckpt)
    elif Path(args.ckpt).is_file():
        policy.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"[ig_rl] loaded policy {args.ckpt}", flush=True)
    if args.eval:
        eval_explorer(q0, q1, policy, games, backends, device, args.T)
    if args.gif:
        for g in games:
            make_gif(q0, q1, policy, g, backends[g.gist], device, args.T,
                     f"{args.gif_dir}/{g.gist}_ig_policy.gif")


if __name__ == "__main__":
    main()
