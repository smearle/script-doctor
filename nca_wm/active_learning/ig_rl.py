"""RL player trained ONLY on information gain from the FROZEN jax world model
(NCAWorldModel q0) + adapter (q1). The reward at each step is the per-step IG of
the action taken; the player should learn to seek the genuinely informative
transitions (here: the jump into a platform from below, the sole world-
disambiguating action) with no environment-specific shaping.

Design choices (kept general, no Mario-specific tricks):
  * Policy body = a small conv net over the (C,H,W) board (NCA-like local
    processing). The spatial feature map is FLATTENED into the readout (not
    collapsed by global pooling), so the action can depend on *where* structure
    is -- without assuming a unique "player" cell.
  * Vectorized: B parallel engines stepped in lockstep; the policy forward and
    the IG estimate are BATCHED over all B envs (one q0 + rsamp q1 jax calls per
    timestep, instead of B*T*rsamp individual calls). Engine stepping is ~free.
  * Every episode starts from level 0; stochastic action sampling is the only
    source of trajectory diversity (no exploring starts / no planted positions).

  python -m nca_wm.active_learning.ig_rl --train --eval --gif
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from nca_wm.active_learning.mario_nca_serve import (
    load_base, load_adapter, build_render_backends, N_ACTIONS, CMAX,
)
from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.multigame_data import _engine, _read_padded, _perm
from nca_wm.active_learning.mario_explore import break_cols, UP
from nca_wm.state_ops import _multihot_to_objects

NA = N_ACTIONS
EPS = 1e-6


# ----------------------------- policy -----------------------------
class ConvPolicy(nn.Module):
    """Conv trunk -> 1x1 channel neck -> FLATTEN (spatial preserved) -> MLP ->
    action logits + value. No global-pool-only collapse, no player gather."""
    def __init__(self, c_in, H, W, hid=64, n_conv=4, neck=24, mlp=256):
        super().__init__()
        convs = [nn.Conv2d(c_in, hid, 3, padding=1), nn.GELU()]
        for _ in range(n_conv - 1):
            convs += [nn.Conv2d(hid, hid, 3, padding=1), nn.GELU()]
        convs += [nn.Conv2d(hid, neck, 1), nn.GELU()]
        self.enc = nn.Sequential(*convs)
        self.head = nn.Sequential(nn.Linear(neck * H * W, mlp), nn.GELU())
        self.pi = nn.Linear(mlp, NA)
        self.v = nn.Linear(mlp, 1)

    def forward(self, obs):                  # obs (B,C,H,W)
        z = self.head(self.enc(obs).flatten(1))
        return self.pi(z), self.v(z).squeeze(-1)


# ----------------------------- batched env -----------------------------
class BatchEnv:
    """B independent C++ engines for one world, stepped in lockstep."""
    def __init__(self, game, B):
        self.game, self.B = game, B
        self.engs = [_engine(game.json_str, 0) for _ in range(B)]
        self.perm = _perm(game.n_obj, CMAX, None)
        self.pb = MB._bit(self.engs[0], "Player")
        self.sb = MB._bit(self.engs[0], "Step")
        self.fb = MB._bit(self.engs[0], "Floor")
        self.reset()

    def _sc(self, e):
        return int(((MB._grid(e) >> self.sb) & 1).sum())

    def reset(self):
        for e in self.engs:
            e.load_level(0)
        self.last = [self._sc(e) for e in self.engs]
        self.n_break = [0] * self.B

    def obs(self):
        return np.stack([_read_padded(e, self.game.n_obj, self.perm, CMAX,
                                      self.game.H, self.game.W) for e in self.engs])

    def step(self, actions):
        for i, e in enumerate(self.engs):
            e.process_input(int(actions[i]))
            k = 0
            while e.is_againing() and k < 50:
                e.process_input(-1); k += 1
            sc = self._sc(e)
            if sc < self.last[i]:
                self.n_break[i] += 1
            self.last[i] = sc

    def under_platform(self):                # diagnostic only (not used in reward)
        out = np.zeros(self.B, bool)
        for i, e in enumerate(self.engs):
            cols, info = break_cols(MB._grid(e), self.pb, self.sb, self.fb)
            out[i] = info is not None and info[2] and info[1] in cols
        return out


class MultiBatchEnv:
    """Several BatchEnvs (one per world) presented as a single batch. The frozen
    WM/adapter are world-agnostic, so the policy forward and the IG reward batch
    over all worlds at once; only the engines (breakable vs not) differ."""
    def __init__(self, games, B_each):
        self.envs = [BatchEnv(g, B_each) for g in games]
        self.B = sum(e.B for e in self.envs)
        self._splits = np.cumsum([e.B for e in self.envs])[:-1]

    def reset(self):
        for e in self.envs:
            e.reset()

    def obs(self):
        return np.concatenate([e.obs() for e in self.envs], axis=0)

    def step(self, actions):
        for e, part in zip(self.envs, np.split(np.asarray(actions), self._splits)):
            e.step(part)

    def under_platform(self):
        return np.concatenate([e.under_platform() for e in self.envs])


# ----------------------------- batched IG reward -----------------------------
def batched_ig(q0, q1, states, actions, rng, n_samples):
    """IG(s,a) = E_{o~q0}[ sum_realcells log q1(o|s,a,o) - log q0(o|s,a) ], per env."""
    Sj, Aj = jnp.asarray(states), jnp.asarray(actions)
    P0 = np.asarray(q0(Sj, Aj)).clip(EPS, 1 - EPS)            # (B,C,H,W)
    m = (states.sum(1, keepdims=True) > 0)                    # real (non-pad) cells
    ig = np.zeros(len(states))
    for _ in range(n_samples):
        o = (rng.random(P0.shape) < P0).astype(np.float32)
        P1 = np.asarray(q1(Sj, Aj, jnp.asarray(o))).clip(EPS, 1 - EPS)
        lq0 = o * np.log(P0) + (1 - o) * np.log(1 - P0)
        lq1 = o * np.log(P1) + (1 - o) * np.log(1 - P1)
        ig += ((lq1 - lq0) * m).sum(axis=(1, 2, 3))
    return ig / n_samples


# ----------------------------- rollout + A2C -----------------------------
def rollout(env, policy, q0, q1, device, T, rsamp, rng_np):
    env.reset()
    logps, vals, rews, ents = [], [], [], []
    for _ in range(T):
        obs = env.obs()
        logits, value = policy(torch.from_numpy(obs).float().to(device))
        dist = Categorical(logits=logits)
        a = dist.sample()
        a_np = a.detach().cpu().numpy()
        ig = np.maximum(0.0, batched_ig(q0, q1, obs, a_np, rng_np, rsamp))
        logps.append(dist.log_prob(a)); vals.append(value)
        rews.append(torch.from_numpy(ig).float().to(device)); ents.append(dist.entropy())
        env.step(a_np)
    return torch.stack(logps), torch.stack(vals), torch.stack(rews), torch.stack(ents)


def train(env, policy, q0, q1, device, args, best_path):
    import copy
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng_np = np.random.default_rng(0)
    best, best_state, recent = -1.0, None, []
    for upd in range(args.updates):
        logps, vals, rews, ents = rollout(env, policy, q0, q1, device, args.T, args.rsamp, rng_np)
        # discounted returns over time (T,B)
        returns = torch.zeros_like(rews)
        R = torch.zeros(env.B, device=device)
        for t in range(args.T - 1, -1, -1):
            R = rews[t] + args.gamma * R
            returns[t] = R
        adv = (returns - vals).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = -(logps * adv).mean() + 0.5 * F.mse_loss(vals, returns) - args.ent_w * ents.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        mr = float(rews.mean())
        recent.append(mr); recent = recent[-25:]; sm = float(np.mean(recent))
        if sm > best:
            best = sm; best_state = copy.deepcopy(policy.state_dict())
        if upd % 10 == 0:
            print(f"upd {upd:4d}  mean-IG {mr:+.3f}  smoothed {sm:+.3f}  best {best:+.3f}  "
                  f"max-step-IG {float(rews.max()):.3f}  loss {loss.item():.3f}", flush=True)
    if best_state is not None:
        policy.load_state_dict(best_state)
        Path(best_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, best_path)
        print(f"[train] restored + saved best policy (smoothed IG {best:+.3f}) -> {best_path}", flush=True)


@torch.no_grad()
def evaluate(env, policy, q0, q1, device, T, rsamp=32, seed=123):
    print("\n=== eval: greedy IG-policy vs random ===")

    def run(greedy):
        rng = np.random.default_rng(seed)
        env.reset(); ig_steps = []; under_up = 0
        for _ in range(T):
            obs = env.obs()
            if greedy:
                logits, _ = policy(torch.from_numpy(obs).float().to(device))
                a = logits.argmax(-1).cpu().numpy()
            else:
                a = rng.integers(NA, size=env.B)
            ig_steps.append(batched_ig(q0, q1, obs, a, rng, rsamp))
            under_up += int(((a == UP) & env.under_platform()).sum())
            env.step(a)
        return np.mean(ig_steps), under_up / env.B, sum(env.n_break) / env.B

    pol = run(True); rnd = run(False)
    print(f"  {env.game.gist:16s}  mean IG/step   POLICY {pol[0]:.3f}  RANDOM {rnd[0]:.3f}")
    print(f"  {'':16s}  under-plat-UP/ep POLICY {pol[1]:.2f}  RANDOM {rnd[1]:.2f}")
    print(f"  {'':16s}  breaks/ep        POLICY {pol[2]:.2f}  RANDOM {rnd[2]:.2f}", flush=True)
    return pol, rnd


@torch.no_grad()
def make_gif(game, backend, policy, q0, q1, device, T, path):
    import imageio, PIL.Image
    env = BatchEnv(game, 1)
    frames = []

    def render():
        o = env.obs()[0]
        crop = (o[:game.n_obj, :game.H, :game.W] > 0.5).astype(np.uint8)
        fr = backend.render_frame_from_objects(_multihot_to_objects(crop), game.W, game.H)
        frames.append(np.array(PIL.Image.fromarray(fr).resize(
            (game.W * 16, game.H * 16), PIL.Image.NEAREST)))

    render()
    for _ in range(T):
        logits, _ = policy(torch.from_numpy(env.obs()).float().to(device))
        env.step(logits.argmax(-1).cpu().numpy())
        render()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=0.25)
    print(f"  wrote {path}  ({len(frames)} frames, {env.n_break[0]} breaks)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--world", default="mario")          # which world to train on
    ap.add_argument("--n_envs", type=int, default=48)
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--rsamp", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent_w", type=float, default=0.02)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--n_conv", type=int, default=4)
    ap.add_argument("--neck", type=int, default=24)
    ap.add_argument("--mlp", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="nca_wm/active_learning/ckpts/ig_rl_policy.pt")
    ap.add_argument("--gif_dir", default="/tmp/ig_rl_gifs")
    args = ap.parse_args()

    device = torch.device(args.device)
    q0, q1 = load_base(), load_adapter()
    games_by_gist, backends = build_render_backends()

    worlds = (["mario", "mario_breakable"] if args.world in ("both", "all")
              else args.world.split(","))
    train_games = [games_by_gist[w] for w in worlds]
    g0 = train_games[0]
    if len(train_games) == 1:
        env = BatchEnv(g0, args.n_envs)
    else:
        env = MultiBatchEnv(train_games, max(1, args.n_envs // len(train_games)))
    policy = ConvPolicy(CMAX, g0.H, g0.W, args.hid, args.n_conv, args.neck, args.mlp).to(device)
    nparam = sum(p.numel() for p in policy.parameters())
    print(f"[ig_rl] train_worlds={worlds} total_B={env.B} policy params={nparam:,}", flush=True)

    if args.train:
        train(env, policy, q0, q1, device, args, args.ckpt)
    elif Path(args.ckpt).is_file():
        policy.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"[ig_rl] loaded {args.ckpt}", flush=True)
    if args.eval:
        for gist, g in games_by_gist.items():
            evaluate(BatchEnv(g, args.n_envs), policy, q0, q1, device, args.T)
    if args.gif:
        for gist, g in games_by_gist.items():
            make_gif(g, backends[gist], policy, q0, q1, device, args.T,
                     f"{args.gif_dir}/{gist}_ig_policy.gif")


if __name__ == "__main__":
    main()
