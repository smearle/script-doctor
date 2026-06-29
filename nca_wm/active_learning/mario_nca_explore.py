"""Train an RL agent whose ONLY reward is information gain from the frozen
belief Recurrent-NCA, and test whether it learns to seek + break the Step.

Intrinsic-reward RL: the belief WM (NCABeliefModel) is frozen; an A2C policy is
rewarded by the WM's per-step information_gain. If the belief is IG-calibrated
(high IG on the disambiguating jump-into-Step, ~0 elsewhere), a reward-maximizing
policy should learn to navigate under a Step and jump (which in mario_breakable
breaks it) far more than a random policy. Mirrors mario_explore.py's A2C but uses
the NCA belief (spatial hidden grid carried via update_belief) instead of the
Transformer belief.

  --train : train the policy (saves ckpts/mario_nca_explore.pt)
  --eval  : disambig-jump / break frequency, policy vs random
  --gif   : render policy vs random rollouts to figures/

    .venv/bin/python -u -m nca_wm.active_learning.mario_nca_explore \
        --wm nca_wm/active_learning/ckpts/mario2_nca_belief_600k/params_best.pkl \
        --train --eval --gif
"""
from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.attn_belief_model import masked_pool
from nca_wm.active_learning.mario_explore import NA, TICK
from nca_wm.active_learning.mario_belief_compare import NcaCtx
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig

FIGDIR = Path("nca_wm/active_learning/figures")


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


def _feat(ctx):
    """Policy input = masked-pool of the carried belief grid B (1, d_b)."""
    return masked_pool(ctx.B, ctx.cellT).detach()


def rollout(wm, policy, game, device, T, rsamp, rng, mode="train"):
    ctx = NcaCtx(wm, game, device, rng); ctx.ns = rsamp
    logps, vals, rews, ents = [], [], [], []
    for _ in range(T):
        feat = _feat(ctx)
        logits, value = policy(feat)                  # (1,NA), (1,)
        dist = torch.distributions.Categorical(logits=logits[0])   # batch-free
        ai_t = logits[0].argmax(-1) if mode == "greedy" else dist.sample()  # scalar
        ai = int(ai_t)
        if mode == "train":
            r = ctx.ig(ai)                            # intrinsic reward = WM info-gain
            logps.append(dist.log_prob(ai_t)); vals.append(value[0])   # scalars
            rews.append(r); ents.append(dist.entropy())
        ctx.step(ai)
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
        if upd % 25 == 0:
            print(f"upd {upd:4d}  mean-IG-reward {rsum/max(n_ep,1):+.3f}  loss {loss.item():.3f}", flush=True)


@torch.no_grad()
def eval_explorer(wm, policy, games, device, T, n_ep=40, seed=1):
    print("\n=== eval: IG-policy vs random (disambig-jumps & breaks per episode) ===")
    for game in games:
        pj = pb = rj = rb = 0.0
        for i in range(n_ep):
            _, _, _, _, ctx = rollout(wm, policy, game, device, T, 0, random.Random(seed + i), "greedy")
            pj += ctx.n_disambig_jump; pb += ctx.n_break
            rc = NcaCtx(wm, game, device, random.Random(seed + i)); rc.ns = 0
            rr = random.Random(1000 + seed + i)
            for _ in range(T):
                rc.step(rr.randrange(NA))
            rj += rc.n_disambig_jump; rb += rc.n_break
        n = n_ep
        print(f"  {game.gist:16s} POLICY disambig-jumps/ep {pj/n:.2f} breaks/ep {pb/n:.2f} | "
              f"RANDOM disambig-jumps/ep {rj/n:.2f} breaks/ep {rb/n:.2f}", flush=True)


# ----------------------------- GIF rendering -----------------------------
def _backends():
    from nca_wm import game_curriculum as gc
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser
    MB._CACHE.mkdir(exist_ok=True); (MB._CACHE / "_scratch").mkdir(exist_ok=True)
    gc._set_materialize_dir(MB._CACHE / "_scratch")
    parser = init_ps_lark_parser(); MB.build_worlds()
    bk = {}
    for name, rel in MB.WORLDS:
        gc._materialize_game(name, (MB.ROOT / rel).read_text())
        b = CppPuzzleScriptBackend(); b.compile_game(parser, name); b.cpp_engine.load_level(0)
        bk[name] = b
    return bk


@torch.no_grad()
def make_gif(wm, policy, game, backend, device, T, path, mode="greedy", seed=7):
    """Roll a policy on `game`, render each engine frame, save an animated GIF."""
    import PIL.Image
    from nca_wm.state_ops import _multihot_to_objects
    ctx = NcaCtx(wm, game, device, random.Random(seed)); ctx.ns = 0
    rr = random.Random(seed)
    frames = []

    def render():
        o = ctx._obs()[0].cpu().numpy()
        crop = (o[:game.n_obj, :game.H, :game.W] > 0.5).astype(np.uint8)
        fr = backend.render_frame_from_objects(_multihot_to_objects(crop), game.W, game.H)
        frames.append(PIL.Image.fromarray(fr).resize((game.W * 16, game.H * 16), PIL.Image.NEAREST))

    render()
    for _ in range(T):
        if mode == "random":
            ai = rr.randrange(NA)
        else:
            logits, _ = policy(_feat(ctx)); ai = int(logits.argmax(-1))
        ctx.step(ai); render()
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"  wrote {path}  (breaks={ctx.n_break} disambig-jumps={ctx.n_disambig_jump})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", default="nca_wm/active_learning/ckpts/mario2_nca_belief_600k/params_best.pkl")
    p.add_argument("--policy", default="nca_wm/active_learning/ckpts/mario_nca_explore.pt")
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--gif", action="store_true")
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

    ck = torch.load(args.wm, map_location=device)
    wm = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device)
    wm.load_state_dict(ck["model_state"]); wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    games = MB.build_worlds()
    print(f"WM {args.wm} (step={ck.get('step')}) frozen | worlds {[g.gist for g in games]}", flush=True)

    policy = Policy(ck["cfg"]["d_b"]).to(device)
    pol_path = Path(args.policy)
    if args.train:
        print("[before]"); eval_explorer(wm, policy, games, device, args.T)
        train_explorer(wm, policy, games, device, args)
        pol_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), pol_path); print(f"saved {pol_path}", flush=True)
    elif pol_path.exists():
        policy.load_state_dict(torch.load(pol_path, map_location=device))
    policy.eval()

    if args.eval:
        print("[after]" if args.train else "[loaded policy]")
        eval_explorer(wm, policy, games, device, args.T)
    if args.gif:
        bk = _backends()
        for g in games:
            make_gif(wm, policy, g, bk[g.gist], device, args.T,
                     FIGDIR / f"nca_explore_{g.gist}_policy.gif", mode="greedy")
            make_gif(wm, policy, g, bk[g.gist], device, args.T,
                     FIGDIR / f"nca_explore_{g.gist}_random.gif", mode="random")


if __name__ == "__main__":
    main()
