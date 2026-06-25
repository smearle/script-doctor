"""Attention-over-history belief world model (multi-game ICL).

Replaces the conv-NCA belief recurrence (which did not sharpen over a rollout)
with a CAUSAL TRANSFORMER over per-step (frame-summary, action) tokens — the
attention is the in-context-inference mechanism. Spatial detail comes from a conv
encoder of the current frame; the transformer belief injects "what the dynamics
are" via FiLM into the joint (one-pass) frame decoder. Discrete-latent mixture +
two haoo' heads (q0, q1) as before.

North-star test: per-step q0 NLL should DECREASE over a rollout.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttnConfig:
    n_obj: int = 32
    n_act: int = 5
    d: int = 96            # conv feature width
    d_model: int = 160     # transformer width
    n_layer: int = 4
    n_head: int = 4
    d_cond: int = 96       # FiLM conditioning width
    K: int = 16
    max_T: int = 16


def masked_pool(x, cell_mask):
    # x:(N,d,H,W) cell_mask:(N,H,W) -> (N,d)
    m = cell_mask[:, None]
    return (x * m).flatten(2).sum(-1) / m.flatten(2).sum(-1).clamp_min(1.0)


class _FiLMDec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.film = nn.Linear(cfg.d_cond, 2 * cfg.d)
        self.c1 = nn.Conv2d(cfg.d, cfg.d, 3, padding=1)
        self.c2 = nn.Conv2d(cfg.d, cfg.n_obj, 1)

    def forward(self, ctx, cond):
        g, b = self.film(cond).chunk(2, -1)
        h = ctx * (1 + g[..., None, None]) + b[..., None, None]
        return self.c2(F.relu(self.c1(F.relu(h))))


class AttnBeliefModel(nn.Module):
    def __init__(self, cfg: AttnConfig):
        super().__init__()
        self.cfg = cfg
        C, d = cfg.n_obj, cfg.d
        self.enc = nn.Sequential(nn.Conv2d(C, d, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(d, d, 3, padding=1), nn.ReLU())
        self.emb_a = nn.Embedding(cfg.n_act + 1, cfg.d_cond)  # +1 = "no prev action"
        self.tok_in = nn.Linear(d + cfg.d_cond, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(cfg.max_T, cfg.d_model))
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_head, cfg.d_model * 2,
                                           batch_first=True, dropout=0.0, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, cfg.n_layer)
        self.belief_proj = nn.Linear(cfg.d_model, cfg.d_cond)
        self.emb_z0 = nn.Embedding(cfg.K, cfg.d_cond)
        self.emb_z1 = nn.Embedding(cfg.K, cfg.d_cond)
        self.prior0 = nn.Sequential(nn.Linear(cfg.d_cond, cfg.d_cond), nn.ReLU(), nn.Linear(cfg.d_cond, cfg.K))
        self.prior1 = nn.Sequential(nn.Linear(cfg.d_cond + d, cfg.d_cond), nn.ReLU(), nn.Linear(cfg.d_cond, cfg.K))
        self.dec0 = _FiLMDec(cfg)
        self.dec1 = _FiLMDec(cfg)

    def _beliefs(self, pooled, acts):
        """pooled:(N,T+1,d) acts:(N,T) -> belief per step (N,T,d_cond).

        Token t = [pooled[t], act_emb(action INTO frame t)]; causal TF; belief_t
        summarizes frames 0..t and is used (with the next action) to predict t+1.
        """
        N, Tp1, d = pooled.shape
        T = Tp1 - 1
        prev_a = torch.full((N, T), self.cfg.n_act, device=acts.device, dtype=torch.long)
        prev_a[:, 1:] = acts[:, :-1]                      # action into frame t (frame 0 -> "none")
        tok = self.tok_in(torch.cat([pooled[:, :T], self.emb_a(prev_a)], -1))  # (N,T,d_model)
        tok = tok + self.pos[:T][None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=tok.device), 1)
        h = self.tf(tok, mask=mask, is_causal=True)
        return self.belief_proj(h)                        # (N,T,d_cond)

    def _dec_all_k(self, dec, emb_z, ctx, cond_base):
        # ctx:(M,d,H,W) cond_base:(M,d_cond) -> logits (M,K,C,H,W)
        M, K = ctx.shape[0], self.cfg.K
        cond = cond_base[:, None, :] + emb_z.weight[None]      # (M,K,d_cond)
        ctxK = ctx[:, None].expand(-1, K, -1, -1, -1).reshape(M * K, *ctx.shape[1:])
        logits = dec(ctxK, cond.reshape(M * K, -1))
        return logits.view(M, K, *logits.shape[1:])

    @staticmethod
    def mixture_nll(logits, logpi, target, vmask=None):
        t = target[:, None]
        lp = t * F.logsigmoid(logits) + (1 - t) * F.logsigmoid(-logits)
        if vmask is not None:
            lp = lp * vmask[:, None]
        return -torch.logsumexp(logpi + lp.flatten(-3).sum(-1), -1)

    # --- incremental inference for the agent ---
    def encode_frame(self, o):
        return self.enc(o)                                 # (1,d,H,W)

    def belief_now(self, pooled_seq, prev_acts):
        """pooled_seq:(1,L,d) prev_acts:(1,L) [action into each frame] -> belief (1,d_cond)."""
        L = pooled_seq.shape[1]
        tok = self.tok_in(torch.cat([pooled_seq, self.emb_a(prev_acts)], -1)) + self.pos[:L][None]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=tok.device), 1)
        h = self.tf(tok, mask=mask, is_causal=True)
        return self.belief_proj(h[:, -1])                  # (1,d_cond)

    @torch.no_grad()
    def information_gain(self, belief, spatial_cur, action, cell_mask, vmask, n_samples=4):
        """IG of taking `action` (idx tensor (1,)) from the current state+belief."""
        cb = belief + self.emb_a(action)                   # (1,d_cond)
        l0 = self._dec_all_k(self.dec0, self.emb_z0, spatial_cur, cb)   # (1,K,C,H,W)
        p0 = F.log_softmax(self.prior0(cb), -1)
        probs0 = p0.exp()[0]
        total = 0.0
        for _ in range(n_samples):
            k = torch.multinomial(probs0, 1).item()
            o = torch.bernoulli(torch.sigmoid(l0[:, k])) * vmask
            logq0 = -self.mixture_nll(l0, p0, o, vmask)
            eo = masked_pool(self.enc(o), cell_mask)
            l1 = self._dec_all_k(self.dec1, self.emb_z1, spatial_cur, cb)
            p1 = F.log_softmax(self.prior1(torch.cat([cb, eo], -1)), -1)
            logq1 = -self.mixture_nll(l1, p1, o, vmask)
            total += (logq1 - logq0).item()
        return total / n_samples

    def forward_traj(self, O, A, R, CM, CH):
        """Return per-step q0 and q1 NLL (N,T) given a batch of trajectories."""
        N, Tp1, C, H, Wd = O.shape
        T = Tp1 - 1
        flat = self.enc(O.reshape(N * Tp1, C, H, Wd))               # (N*Tp1,d,H,W)
        spatial = flat.view(N, Tp1, -1, H, Wd)
        cmf = CM[:, None].expand(-1, Tp1, -1, -1).reshape(N * Tp1, H, Wd)
        pooled = masked_pool(flat, cmf).view(N, Tp1, -1)            # (N,Tp1,d)
        beliefs = self._beliefs(pooled, A)                         # (N,T,d_cond)

        vmask = (CH[:, :, None, None] * CM[:, None, :, :])         # (N,C,H,W)
        a_emb = self.emb_a(A)                                      # (N,T,d_cond)  (next action)
        q0nll = torch.zeros(N, T, device=O.device)
        q1nll = torch.zeros(N, T, device=O.device)
        for t in range(T):
            ctx = spatial[:, t]                                    # enc(obs[t])  (N,d,H,W)
            cb0 = beliefs[:, t] + a_emb[:, t]                      # (N,d_cond)
            l0 = self._dec_all_k(self.dec0, self.emb_z0, ctx, cb0)
            p0 = F.log_softmax(self.prior0(cb0), -1)
            q0nll[:, t] = self.mixture_nll(l0, p0, O[:, t + 1], vmask)
            # q1: condition on first obs o = O[t+1] (its pooled enc); target resamp R[t]
            eo = masked_pool(spatial[:, t + 1], CM)                # (N,d)
            cb1 = beliefs[:, t] + a_emb[:, t]
            l1 = self._dec_all_k(self.dec1, self.emb_z1, ctx, cb1)
            p1 = F.log_softmax(self.prior1(torch.cat([cb1, eo], -1)), -1)
            q1nll[:, t] = self.mixture_nll(l1, p1, R[:, t], vmask)
        return q0nll, q1nll
