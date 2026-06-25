"""PyTorch recurrent actor-critic for the PufferLib PuzzleScript port — the
torch counterpart of the JAX ActorCriticRNN. Conv encoder over a multihot
(C,H,W) board -> GRU -> actor (Discrete) + critic. Optional game-id embedding.

Stateful API for rollout: forward(obs, hxs, dones) resets the GRU carry where
done, mirroring ScannedRNN's done-masked reset.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ortho(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


class RecurrentPolicy(nn.Module):
    def __init__(self, n_objs: int, n_actions: int = 5, hidden: int = 128,
                 n_games: int = 1, game_embed: int = 0, pool: int = 12):
        super().__init__()
        self.hidden = hidden
        self.game_embed = game_embed
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(n_objs, 64, 3, padding=1), 2 ** 0.5), nn.ReLU(),
            _ortho(nn.Conv2d(64, 64, 3, padding=1), 2 ** 0.5), nn.ReLU(),
        )
        # Adaptive pool caps the flattened dim regardless of (padded) board size,
        # so the net scales to many varied-size games. For boards smaller than
        # `pool` this is a no-op (pool>=size keeps full resolution).
        self.pool = nn.AdaptiveMaxPool2d(pool)
        self._enc_dim = None  # lazily sized dense after flatten
        self.proj = None
        if game_embed > 0:
            self.embed = nn.Embedding(n_games, game_embed)
        self.gru = nn.GRUCell(hidden + game_embed, hidden)
        self.actor = nn.Sequential(_ortho(nn.Linear(hidden, hidden), 2 ** 0.5), nn.ReLU(),
                                   _ortho(nn.Linear(hidden, n_actions), 0.01))
        self.critic = nn.Sequential(_ortho(nn.Linear(hidden, hidden), 2 ** 0.5), nn.ReLU(),
                                     _ortho(nn.Linear(hidden, 1), 1.0))

    def encode(self, obs):
        # obs: (B, C, H, W) float
        x = self.conv(obs)
        H, W = x.shape[2], x.shape[3]
        # Cap spatial size only when the board is larger than the pool target.
        if H > self.pool.output_size or W > self.pool.output_size:
            x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        if self.proj is None:
            self.proj = _ortho(nn.Linear(x.shape[1], self.hidden), 2 ** 0.5).to(x.device)
        return F.relu(self.proj(x))

    def initial_state(self, batch, device):
        return torch.zeros(batch, self.hidden, device=device)

    def forward(self, obs, hx, done, game_ids=None):
        """One step. obs (B,C,H,W) uint8/float, hx (B,hidden), done (B,) bool
        (whether the PREVIOUS step ended -> reset carry). Returns logits, value, hx."""
        obs = obs.float()
        feat = self.encode(obs)
        if self.game_embed > 0 and game_ids is not None:
            feat = torch.cat([feat, self.embed(game_ids)], dim=-1)
        hx = hx * (~done).float().unsqueeze(-1)   # reset carry where done
        hx = self.gru(feat, hx)
        logits = self.actor(hx)
        value = self.critic(hx).squeeze(-1)
        return logits, value, hx


if __name__ == "__main__":
    import numpy as np
    from puzzlejax.puffer_ps_env import PuzzleScriptVecEnv

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    games = ["sokoban_basic", "kettle", "Slidings", "Travelling_salesman"]
    env = PuzzleScriptVecEnv(games, num_envs=512, max_episode_steps=200)
    obs, _ = env.reset()
    C = obs.shape[1]
    policy = RecurrentPolicy(n_objs=C, n_actions=5, hidden=128,
                             n_games=len(games), game_embed=16).to(dev)
    hx = policy.initial_state(env.num_envs, dev)
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    gids = torch.as_tensor(env.game_ids, device=dev)
    wins = 0
    for t in range(50):
        obs_t = torch.as_tensor(obs, device=dev)
        with torch.no_grad():
            logits, value, hx = policy(obs_t, hx, done, gids)
            actions = torch.distributions.Categorical(logits=logits).sample()
        obs, rew, d, tr, info = env.step(actions.cpu().numpy())
        done = torch.as_tensor(d, dtype=torch.bool, device=dev)
        wins += int(info["won"].sum())
    nparams = sum(p.numel() for p in policy.parameters())
    print(f"device={dev} obs={obs.shape} hx={tuple(hx.shape)} value={tuple(value.shape)} "
          f"params={nparams:,} | env<->torch pipeline OK; wins(random)={wins}")
