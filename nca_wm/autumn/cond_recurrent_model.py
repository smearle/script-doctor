"""Conditional RECURRENT Autumn NCA world model (PyTorch).

One recurrent world model that serves many Autumn games, conditioned on each
game's program. Combines two ideas already validated separately in this module:
  * `RecurrentAutumnNCA` — a hidden grid `h` carried ACROSS env steps, so the
    model can integrate unobservable episode-long state (Mario's bullet counter,
    paint's currColor, sand's clickType mode) that no single frame reveals.
  * `ConditionalAutumnNCA` — program-slot conditioning (rule-slot encoder +
    FiLM + input conditioning + per-cell cross-attention) so one set of weights
    fits many games.

This is the path to *perfect* in-distribution modelling: recurrence resolves the
hidden state that caps single-frame accuracy, and conditioning lets a single
model do it for every game. Trained with BPTT over episode sequences
(`*_seq.npz`); `h` is reset to zeros at each episode start.

Reuses the copy-basin lessons from `cond_model`:
  * zero-init update output + `copy_skip` so the model starts as a clean copy;
  * non-zero input-level conditioning so `h` is program-specific from step 0
    (else the shared weights average conflicting per-game gradients into a copy);
  * moderate `copy_skip` so a wider model can still leave the copy basin.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nca_wm.autumn.model import N_ATYPES
from nca_wm.autumn.cond_model import RuleSlotEncoder
from nca_wm.autumn.tokenize_program import VOCAB_SIZE


class ConditionalRecurrentAutumnNCA(nn.Module):
    def __init__(self, n_colors, n_hid=128, n_micro=6, n_slots=24, d_slot=160,
                 n_attn_heads=4, pool="meanmax", copy_skip=2.5,
                 enc_d_model=128, enc_n_self_layers=2, max_seq_len=1024):
        super().__init__()
        self.n_colors = n_colors
        self.n_hid = n_hid
        self.n_micro = n_micro
        self.pool = pool
        n_pool = {"none": 0, "mean": 1, "max": 1, "meanmax": 2}[pool]

        self.encoder = RuleSlotEncoder(
            vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len, d_model=enc_d_model,
            n_self_layers=enc_n_self_layers, n_slots=n_slots, d_slot=d_slot,
            n_heads=n_attn_heads)

        # Input-level conditioning (program -> hidden, from step 0).
        self.cond_dim = 32
        self.cond_in = nn.Linear(d_slot, self.cond_dim)
        self.obs_embed = nn.Conv2d(
            n_colors + N_ATYPES + 1 + self.cond_dim, n_hid, 1)

        self.perceive = nn.Conv2d(n_hid, n_hid, 3, padding=1)
        self.upd1 = nn.Conv2d(n_hid * (2 + n_pool), n_hid, 1)
        self.upd2 = nn.Conv2d(n_hid, n_hid, 1)
        # FiLM (per-channel scale/shift from pooled slots), zero-init -> identity.
        self.film = nn.Linear(d_slot, 2 * n_hid)
        # Per-cell cross-attention into the rule slots (shared across micro-steps).
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

    def init_hidden(self, B, H, W, device):
        return torch.zeros(B, self.n_hid, H, W, device=device)

    def step(self, state_onehot, atype_onehot, click_map, h, slots):
        """One env step under program conditioning. Returns (logits, new_h)."""
        B, _, H, W = state_onehot.shape
        slots_k = self.slot_proj_ln(slots)
        pooled = slots.mean(dim=1)                                # (B, d_slot)
        gamma, beta = self.film(pooled).chunk(2, dim=-1)
        gamma = (1.0 + gamma)[:, :, None, None]
        beta = beta[:, :, None, None]
        cvec = self.cond_in(pooled)[:, :, None, None].expand(B, self.cond_dim, H, W)

        at = atype_onehot[:, :, None, None].expand(B, N_ATYPES, H, W)
        h = h + self.obs_embed(torch.cat([state_onehot, at, click_map, cvec], dim=1))
        for _ in range(self.n_micro):
            hf = gamma * h + beta
            perc = self.perceive(hf)
            feats = [hf, perc]
            if self.pool in ("mean", "meanmax"):
                feats.append(hf.mean(dim=(2, 3), keepdim=True).expand_as(hf))
            if self.pool in ("max", "meanmax"):
                feats.append(hf.amax(dim=(2, 3), keepdim=True).expand_as(hf))
            u = F.gelu(self.upd1(torch.cat(feats, dim=1)))
            hq = self.attn_ln(hf.flatten(2).transpose(1, 2))       # (B, HW, n_hid)
            attn_out, _ = self.xattn(hq, slots_k, slots_k, need_weights=False)
            attn_grid = attn_out.transpose(1, 2).reshape(B, -1, H, W)
            dh = self.upd2(u + attn_grid)                          # zero-init out
            h = self.norm(h + dh)
        logits = self.readout(h) + self.copy_skip * state_onehot
        return logits, h
