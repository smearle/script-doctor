"""Conditional Autumn NCA world model (PyTorch).

One world model that serves many Autumn games, conditioned on the program via a
Perceiver-style rule-slot encoder — the PyTorch analog of the JAX
`rule_attn_model.RuleSlotEncoder` / `RuleAttnNCAWorldModel` used on the
PuzzleScript side. Replaces the per-game `AutumnNCA` for multi-game / held-out
generalization experiments.

Pipeline:
  RuleSlotEncoder: program tokens (B,S) -> K rule slots (B,K,d_slot).
  ConditionalAutumnNCA: shared-weight conv NCA whose every step does per-cell
  cross-attention into the K slots (so conditioning stays spatial + expressive
  throughout the rollout), plus the global-pool summary and copy-skip from the
  base AutumnNCA.

Conventions match `model.py` so encode/eval code is reusable:
  state_onehot (B,C,H,W) one-hot over the game's color palette, padded to a
  common C across games; atype_onehot (B,N_ATYPES); click_map (B,1,H,W).
Token `COLORi` is aligned (by the tokenizer's `color_order`) to channel `i`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nca_wm.autumn.model import N_ATYPES
from nca_wm.autumn.tokenize_program import VOCAB_SIZE, PAD


class RuleSlotEncoder(nn.Module):
    """Tokens -> K x d_slot rule slots (PyTorch port of the JAX encoder).

    K learned query vectors cross-attend to the self-attended token sequence,
    giving a fixed-size latent that scales sub-linearly with program length —
    important here because linearized Autumn programs run to thousands of tokens.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, max_seq_len=2048, d_model=128,
                 n_self_layers=2, n_slots=16, d_slot=128, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.self_attn = nn.ModuleList()
        self.self_ln1 = nn.ModuleList()
        self.self_ln2 = nn.ModuleList()
        self.self_ff = nn.ModuleList()
        for _ in range(n_self_layers):
            self.self_attn.append(nn.MultiheadAttention(
                d_model, n_heads, batch_first=True))
            self.self_ln1.append(nn.LayerNorm(d_model))
            self.self_ln2.append(nn.LayerNorm(d_model))
            self.self_ff.append(nn.Sequential(
                nn.Linear(d_model, d_model * 4), nn.GELU(),
                nn.Linear(d_model * 4, d_model)))
        self.ln_final = nn.LayerNorm(d_model)
        self.slot_queries = nn.Parameter(torch.randn(n_slots, d_slot) * 0.02)
        self.slot_xattn = nn.MultiheadAttention(
            d_slot, n_heads, kdim=d_model, vdim=d_model, batch_first=True)
        self.slot_ln = nn.LayerNorm(d_slot)

    def forward(self, token_ids, key_padding_mask):
        # token_ids (B,S) long; key_padding_mask (B,S) bool, True = PAD (ignore).
        B, S = token_ids.shape
        pos = torch.arange(S, device=token_ids.device)[None, :]
        x = self.tok_embed(token_ids) + self.pos_embed(pos)
        for attn, ln1, ln2, ff in zip(
                self.self_attn, self.self_ln1, self.self_ln2, self.self_ff):
            y = ln1(x)
            y, _ = attn(y, y, y, key_padding_mask=key_padding_mask,
                        need_weights=False)
            x = x + y
            x = x + ff(ln2(x))
        x = self.ln_final(x)
        q = self.slot_queries[None].expand(B, -1, -1)
        slots, _ = self.slot_xattn(q, x, x, key_padding_mask=key_padding_mask,
                                   need_weights=False)
        return self.slot_ln(slots)  # (B, K, d_slot)


class ConditionalAutumnNCA(nn.Module):
    """Shared-weight conv NCA with per-step cross-attention to program slots."""

    def __init__(self, n_colors, n_hid=128, n_steps=10, n_slots=16, d_slot=128,
                 n_attn_heads=4, global_pool=True, copy_skip=5.0, history=0,
                 enc_d_model=128, enc_n_self_layers=2, max_seq_len=2048):
        super().__init__()
        self.n_colors = n_colors
        self.n_steps = n_steps
        self.global_pool = global_pool
        self.history = history
        self.encoder = RuleSlotEncoder(
            max_seq_len=max_seq_len, d_model=enc_d_model,
            n_self_layers=enc_n_self_layers, n_slots=n_slots, d_slot=d_slot,
            n_heads=n_attn_heads)

        # Input-level conditioning: a (normal-init) projection of the pooled
        # slots, broadcast spatially and concatenated to the embed input, so
        # the hidden state is program-specific from step 0. Without this the
        # model is program-blind at init (FiLM starts at identity) and, across
        # many games, the shared conv's per-game gradients cancel to a copy and
        # never escape — observed as a hard stall at 8+ games. Output is still
        # a copy at init (readout zero-init + copy_skip), so stability holds.
        self.cond_dim = 32
        self.cond_in = nn.Linear(d_slot, self.cond_dim)
        in_ch = n_colors * (1 + history) + N_ATYPES + 1 + self.cond_dim
        self.embed = nn.Conv2d(in_ch, n_hid, 1)
        self.perceive = nn.Conv2d(n_hid, n_hid, 3, padding=1)
        upd_in = n_hid * (3 if global_pool else 2)
        self.upd1 = nn.Conv2d(upd_in, n_hid, 1)
        self.upd2 = nn.Conv2d(n_hid, n_hid, 1)
        # FiLM conditioning: pooled slots -> per-channel (scale, shift) applied
        # to the hidden state every step. This is the workhorse: it puts the
        # game identity *inside* the shared conv computation, so the conv path
        # can compute different dynamics per game instead of averaging 8 games'
        # conflicting gradients into a copy (the failure seen with attention as
        # a mere additive side-channel). Zero-init -> gamma=1, beta=0 at start
        # (identity), so combined with zero-init upd2 the model begins as a
        # clean copy NCA and grows into using conditioning.
        self.film = nn.Linear(d_slot, 2 * n_hid)
        # Per-cell cross-attention into the rule slots (shared across steps),
        # routed through the same zero-init upd2 output so spatial conditioning
        # is integral to the update rather than an independent branch.
        self.attn_ln = nn.LayerNorm(n_hid)
        self.slot_proj_ln = nn.LayerNorm(d_slot)
        self.xattn = nn.MultiheadAttention(
            n_hid, n_attn_heads, kdim=d_slot, vdim=d_slot, batch_first=True)
        self.norm = nn.GroupNorm(1, n_hid)
        self.readout = nn.Conv2d(n_hid, n_colors, 1)
        self.copy_skip = nn.Parameter(torch.tensor(float(copy_skip)))
        nn.init.zeros_(self.upd2.weight); nn.init.zeros_(self.upd2.bias)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def encode(self, tokens, tok_mask):
        """tokens (B,S) long, tok_mask (B,S) bool True=real -> slots (B,K,d_slot)."""
        return self.encoder(tokens, key_padding_mask=~tok_mask)

    def forward(self, state_onehot, atype_onehot, click_map, tokens, tok_mask,
                hist_onehot=None, slots=None):
        B, _, H, W = state_onehot.shape
        if slots is None:
            # Batches are single-game (shared tokens), so encode the program
            # ONCE (B'=1 or however many distinct rows are passed) and broadcast
            # the slots to the batch — the encoder's O(S^2) token self-attention
            # is the memory bottleneck, so this is a large saving vs encoding
            # one identical copy per batch element.
            slots = self.encode(tokens, tok_mask)
        if slots.shape[0] == 1 and B > 1:
            slots = slots.expand(B, -1, -1)
        slots_k = self.slot_proj_ln(slots)

        # FiLM scale/shift from pooled slots (gamma=1, beta=0 at init).
        gamma, beta = self.film(slots.mean(dim=1)).chunk(2, dim=-1)  # (B, n_hid)
        gamma = (1.0 + gamma)[:, :, None, None]
        beta = beta[:, :, None, None]

        at = atype_onehot[:, :, None, None].expand(B, N_ATYPES, H, W)
        cvec = self.cond_in(slots.mean(dim=1))[:, :, None, None]  # (B,cond_dim,1,1)
        cvec = cvec.expand(B, self.cond_dim, H, W)
        parts = [state_onehot]
        if self.history:
            parts.append(hist_onehot)
        parts += [at, click_map, cvec]
        h = self.embed(torch.cat(parts, dim=1))

        for _ in range(self.n_steps):
            hf = gamma * h + beta                             # FiLM-modulate
            perc = self.perceive(hf)
            feats = [hf, perc]
            if self.global_pool:
                feats.append(hf.mean(dim=(2, 3), keepdim=True).expand_as(hf))
            u = F.gelu(self.upd1(torch.cat(feats, dim=1)))    # conv update feats
            # per-cell cross-attention into rule slots (spatial conditioning)
            hq = self.attn_ln(hf.flatten(2).transpose(1, 2))  # (B, HW, n_hid)
            attn_out, _ = self.xattn(hq, slots_k, slots_k, need_weights=False)
            attn_grid = attn_out.transpose(1, 2).reshape(B, -1, H, W)
            delta = self.upd2(u + attn_grid)                  # single zero-init out
            h = self.norm(h + delta)

        return self.readout(h) + self.copy_skip * state_onehot
