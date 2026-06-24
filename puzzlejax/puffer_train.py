"""cleanrl-style recurrent (GRU) PPO on the C++ PuzzleScript vec env.

No JAX, no XLA compile: envs step in C++ (CPU, OpenMP), policy on GPU (torch).
This is path (A) of PUFFERLIB_PORT_PLAN.md — the C++ env already provides the
vectorization PufferLib would, so we drive it directly.

Usage:
    python -m puzzlejax.puffer_train --games Slidings --num_envs 512 \
        --num_steps 64 --total_timesteps 3000000 --device cuda
    python -m puzzlejax.puffer_train --games kettle,Slidings,sokodig,Travelling_salesman \
        --num_envs 1024 --total_timesteps 8000000
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from puzzlejax.puffer_ps_env import PuzzleScriptVecEnv
from puzzlejax.puffer_models import RecurrentPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=str, default="Slidings")
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=64)
    p.add_argument("--total_timesteps", type=int, default=3_000_000)
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--game_embed", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.02)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--num_threads", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    games = args.games.split(",")
    dev = args.device

    env = PuzzleScriptVecEnv(games, num_envs=args.num_envs,
                             max_episode_steps=args.max_episode_steps,
                             num_threads=args.num_threads)
    N = env.num_envs
    C = env.single_observation_space.shape[0]
    nA = env.single_action_space.n
    gids = torch.as_tensor(env.game_ids, device=dev)
    game_ids_np = env.game_ids

    policy = RecurrentPolicy(n_objs=C, n_actions=nA, hidden=args.hidden,
                             n_games=len(games), game_embed=args.game_embed).to(dev)
    # Force lazy proj init so the optimizer sees all params.
    _o, _ = env.reset()
    policy(torch.as_tensor(_o, device=dev), policy.initial_state(N, dev),
           torch.zeros(N, dtype=torch.bool, device=dev), gids)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    T = args.num_steps
    obs_buf = torch.zeros((T, N, C, *env.single_observation_space.shape[1:]),
                          dtype=torch.uint8, device=dev)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=dev)
    logp_buf = torch.zeros((T, N), device=dev)
    val_buf = torch.zeros((T, N), device=dev)
    rew_buf = torch.zeros((T, N), device=dev)
    done_buf = torch.zeros((T, N), device=dev)

    obs, _ = env.reset()
    obs = torch.as_tensor(obs, device=dev)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    hx = policy.initial_state(N, dev)

    num_updates = args.total_timesteps // (T * N)
    mb_envs = N // args.num_minibatches
    global_step = 0
    t_start = time.time()

    for update in range(1, num_updates + 1):
        init_hx = hx.detach().clone()
        # done = win (C++); truncated = timeout. A completed episode is either.
        win_pg = np.zeros(len(games)); ep_pg = np.zeros(len(games))
        for t in range(T):
            obs_buf[t] = obs
            done_buf[t] = done.float()
            with torch.no_grad():
                logits, value, hx = policy(obs.float(), hx, done, gids)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            act_buf[t] = action
            logp_buf[t] = dist.log_prob(action)
            val_buf[t] = value
            obs_np, rew, d, tr, info = env.step(action.cpu().numpy())
            obs = torch.as_tensor(obs_np, device=dev)
            done = torch.as_tensor(d, dtype=torch.bool, device=dev)
            rew_buf[t] = torch.as_tensor(rew, device=dev)
            ended = np.asarray(d) | np.asarray(tr)
            won = np.asarray(d)
            for gi in range(len(games)):
                m = game_ids_np == gi
                ep_pg[gi] += (ended & m).sum(); win_pg[gi] += (won & m).sum()
            global_step += N

        with torch.no_grad():
            _, next_value, _ = policy(obs.float(), hx, done, gids)
            adv = torch.zeros_like(rew_buf); lastgae = 0.0
            for t in reversed(range(T)):
                nextnonterm = 1.0 - (done.float() if t == T - 1 else done_buf[t + 1])
                nextval = next_value if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nextval * nextnonterm - val_buf[t]
                lastgae = delta + args.gamma * args.gae_lambda * nextnonterm * lastgae
                adv[t] = lastgae
            returns = adv + val_buf

        # PPO update — minibatch over env axis, replay GRU over time (BPTT).
        env_idx = np.arange(N)
        for _ in range(args.update_epochs):
            np.random.shuffle(env_idx)
            for mb in range(args.num_minibatches):
                idx = env_idx[mb * mb_envs:(mb + 1) * mb_envs]
                idx_t = torch.as_tensor(idx, device=dev)
                hx_mb = init_hx[idx_t]
                new_logits, new_vals = [], []
                for t in range(T):
                    lg, vl, hx_mb = policy(obs_buf[t, idx_t].float(), hx_mb,
                                           done_buf[t, idx_t].bool(), gids[idx_t])
                    new_logits.append(lg); new_vals.append(vl)
                new_logits = torch.stack(new_logits); new_vals = torch.stack(new_vals)
                dist = torch.distributions.Categorical(logits=new_logits)
                a_mb = act_buf[:, idx_t]
                new_logp = dist.log_prob(a_mb)
                entropy = dist.entropy().mean()
                logratio = new_logp - logp_buf[:, idx_t]
                ratio = logratio.exp()
                mb_adv = adv[:, idx_t]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg = torch.maximum(-mb_adv * ratio,
                                   -mb_adv * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
                v_loss = 0.5 * ((new_vals - returns[:, idx_t]) ** 2).mean()
                loss = pg + args.vf_coef * v_loss - args.ent_coef * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()

        if update % 5 == 0 or update == 1:
            sps = global_step / (time.time() - t_start)
            wr = win_pg.sum() / max(ep_pg.sum(), 1)
            pg = " ".join(f"{g[:6]}:{(win_pg[i]/max(ep_pg[i],1)):.2f}" for i, g in enumerate(games))
            print(f"upd {update}/{num_updates} step {global_step:,} "
                  f"win_rate {wr:.2f} | {pg} | SPS {sps:,.0f}", flush=True)

    print("done")


if __name__ == "__main__":
    main()
