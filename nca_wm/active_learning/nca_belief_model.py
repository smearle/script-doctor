"""2-D NCA-belief world model for information-gain active learning.

Predicts a whole next frame in one conv pass (no per-cell autoregression) and
carries a recurrent spatial belief state across frames (the in-context posterior
over the hidden dynamics theta). A small discrete latent z (K modes) gives joint
coherence with an EXACT mixture likelihood (no Gumbel). Two decoder heads:
  q0(o | B, a)        -- next-frame distribution
  q1(o'| B, a, e(o))  -- re-roll distribution, conditioned on the first obs o,
                         trained on haoo' pairs so it learns o' ⊥ o for noise
                         (the chaos / not-mesmerized correctness requirement).

See NCA_BELIEF_MODEL_SPEC.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BeliefConfig:
    n_obj: int = 5          # canonical object channels (C)
    n_act: int = 5          # actions
    d: int = 96             # encoder feature channels
    d_b: int = 128          # belief channels
    d_cond: int = 64        # action+latent conditioning width
    K: int = 16             # discrete latent modes
    nca_steps: int = 3


def _logbern(logits, x):
    """Sum log Bernoulli(x | sigmoid(logits)) over channel+space. logits,x: (...,C,H,W)."""
    # log p = x*logsigmoid(l) + (1-x)*logsigmoid(-l)
    lp = x * F.logsigmoid(logits) + (1.0 - x) * F.logsigmoid(-logits)
    return lp.flatten(-3).sum(-1)  # sum over C,H,W -> (...)


class _FiLMDecoder(nn.Module):
    """context (B,d_b,H,W) + cond vector -> per-cell object logits (B,C,H,W)."""
    def __init__(self, cfg: BeliefConfig):
        super().__init__()
        self.film = nn.Linear(cfg.d_cond, 2 * cfg.d_b)
        self.c1 = nn.Conv2d(cfg.d_b, cfg.d_b, 3, padding=1)
        self.c2 = nn.Conv2d(cfg.d_b, cfg.n_obj, 1)

    def forward(self, ctx, cond):
        # ctx: (N,d_b,H,W)  cond: (N,d_cond) -> logits (N,C,H,W)
        g, b = self.film(cond).chunk(2, dim=-1)
        h = ctx * (1 + g[..., None, None]) + b[..., None, None]
        return self.c2(F.relu(self.c1(F.relu(h))))


class NCABeliefModel(nn.Module):
    def __init__(self, cfg: BeliefConfig):
        super().__init__()
        self.cfg = cfg
        C, d, d_b = cfg.n_obj, cfg.d, cfg.d_b
        # encoder (shared)
        self.enc = nn.Sequential(nn.Conv2d(C, d, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(d, d, 3, padding=1), nn.ReLU())
        self.b_init = nn.Conv2d(d, d_b, 3, padding=1)
        # belief update: prev belief + encoded obs + action planes -> belief; then NCA
        self.b_in = nn.Conv2d(d_b + d + cfg.n_act, d_b, 3, padding=1)
        self.nca_p = nn.Conv2d(d_b, d_b, 3, padding=1)
        self.nca_u = nn.Conv2d(d_b, d_b, 1)
        # conditioning embeddings
        self.emb_a = nn.Embedding(cfg.n_act, cfg.d_cond)
        self.emb_z0 = nn.Embedding(cfg.K, cfg.d_cond)
        self.emb_z1 = nn.Embedding(cfg.K, cfg.d_cond)
        # priors (MLP on pooled context)
        self.prior0 = nn.Sequential(nn.Linear(d_b + cfg.d_cond, d_b), nn.ReLU(),
                                    nn.Linear(d_b, cfg.K))
        self.prior1 = nn.Sequential(nn.Linear(d_b + d + cfg.d_cond, d_b), nn.ReLU(),
                                    nn.Linear(d_b, cfg.K))
        # q1 context fuses belief with encoded first-obs
        self.q1_ctx = nn.Conv2d(d_b + d, d_b, 3, padding=1)
        self.dec0 = _FiLMDecoder(cfg)
        self.dec1 = _FiLMDecoder(cfg)

    # --- belief recurrence ---
    def encode(self, o):                       # (N,C,H,W)->(N,d,H,W)
        return self.enc(o)

    def init_belief(self, o0):
        return self.b_init(self.encode(o0))

    def _act_planes(self, a, H, W):
        # a: (N,) long -> (N, n_act, H, W) one-hot planes
        oh = F.one_hot(a, self.cfg.n_act).float()
        return oh[..., None, None].expand(-1, -1, H, W)

    def update_belief(self, B, o, a, cell_mask=None):
        e = self.encode(o)
        H, W = B.shape[-2:]
        x = torch.cat([B, e, self._act_planes(a, H, W)], dim=1)
        B = self.b_in(x)
        for _ in range(self.cfg.nca_steps):
            B = B + self.nca_u(F.relu(self.nca_p(B)))
        if cell_mask is not None:                  # zero belief outside the real grid
            B = B * cell_mask[:, None]
        return B

    # --- heads (all-K, exact mixture) ---
    def _dec_all_k(self, dec, emb_z, ctx, a):
        """Return per-mode logits (N,K,C,H,W) and prior-cond (N,d_cond) action part."""
        N = ctx.shape[0]
        K = self.cfg.K
        a_emb = self.emb_a(a)                                   # (N,d_cond)
        z = emb_z.weight                                        # (K,d_cond)
        cond = a_emb[:, None, :] + z[None, :, :]                # (N,K,d_cond)
        ctxK = ctx[:, None].expand(-1, K, -1, -1, -1).reshape(N * K, *ctx.shape[1:])
        logits = dec(ctxK, cond.reshape(N * K, -1))
        return logits.view(N, K, *logits.shape[1:]), a_emb

    def q0_logits(self, B, a):
        logits, a_emb = self._dec_all_k(self.dec0, self.emb_z0, B, a)
        pool = B.mean(dim=(-2, -1))                             # (N,d_b)
        logpi = F.log_softmax(self.prior0(torch.cat([pool, a_emb], -1)), -1)
        return logits, logpi                                   # (N,K,C,H,W),(N,K)

    def q1_logits(self, B, a, o):
        e = self.encode(o)
        ctx = self.q1_ctx(torch.cat([B, e], dim=1))
        logits, a_emb = self._dec_all_k(self.dec1, self.emb_z1, ctx, a)
        pool = torch.cat([B.mean(dim=(-2, -1)), e.mean(dim=(-2, -1)), a_emb], -1)
        logpi = F.log_softmax(self.prior1(pool), -1)
        return logits, logpi

    @staticmethod
    def mixture_nll(logits, logpi, target, vmask=None):
        # logits:(N,K,C,H,W) logpi:(N,K) target:(N,C,H,W) vmask:(N,C,H,W)|None -> (N,)
        t = target[:, None]
        lp = t * F.logsigmoid(logits) + (1.0 - t) * F.logsigmoid(-logits)  # (N,K,C,H,W)
        if vmask is not None:
            lp = lp * vmask[:, None]
        lp = lp.flatten(-3).sum(-1)                            # (N,K)
        return -torch.logsumexp(logpi + lp, dim=-1)

    # --- information gain ---
    @torch.no_grad()
    def information_gain(self, B, a, n_samples=8, vmask=None):
        """IG(B,a) = E_{o~q0}[ log q1(o|B,a,o) - log q0(o|B,a) ], one belief state."""
        logits0, logpi0 = self.q0_logits(B, a)                 # (1,K,C,H,W),(1,K)
        probs0 = logpi0.exp()
        total = 0.0
        for _ in range(n_samples):
            k = torch.multinomial(probs0[0], 1).item()
            o = torch.bernoulli(torch.sigmoid(logits0[:, k]))  # (1,C,H,W)
            if vmask is not None:
                o = o * vmask
            logq0 = -self.mixture_nll(logits0, logpi0, o, vmask)
            logits1, logpi1 = self.q1_logits(B, a, o)
            logq1 = -self.mixture_nll(logits1, logpi1, o, vmask)
            total += (logq1 - logq0).item()
        return total / n_samples
