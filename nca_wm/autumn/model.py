"""PyTorch NCA world model for a single Autumn game (discrete color grid).

Input  : per-cell color one-hot (C channels)
       + action-type one-hot broadcast spatially ({noop,click,up,down,left,right})
       + a click-location channel (1.0 at the clicked cell, else 0).
Body   : shared-weight cellular update applied n_steps times; each step mixes a
         3x3 perception of the hidden state with a global-pooled summary, so a
         single click anywhere (e.g. a global button trigger) can influence
         every cell.
Readout: per-cell logits over the C colors, with a learnable skip from the input
         one-hot so "copy the current color" is the easy default.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_TYPES = ["noop", "click", "up", "down", "left", "right"]
ATYPE_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
N_ATYPES = len(ACTION_TYPES)


def action_to_fields(action):
    """('noop',)/('click',x,y)/('up',)... -> (atype_idx, x, y)."""
    t = action[0]
    if t == "click":
        return ATYPE_IDX["click"], int(action[1]), int(action[2])
    return ATYPE_IDX[t], -1, -1


class AutumnNCA(nn.Module):
    def __init__(self, n_colors, n_hid=96, n_steps=10, global_pool=True,
                 copy_skip=5.0, history=0):
        super().__init__()
        self.n_colors = n_colors
        self.n_steps = n_steps
        self.global_pool = global_pool
        self.history = history  # number of preceding frames fed as extra channels
        in_ch = n_colors * (1 + history) + N_ATYPES + 1
        self.embed = nn.Conv2d(in_ch, n_hid, 1)
        self.perceive = nn.Conv2d(n_hid, n_hid, 3, padding=1)
        upd_in = n_hid * (3 if global_pool else 2)
        self.upd1 = nn.Conv2d(upd_in, n_hid, 1)
        self.upd2 = nn.Conv2d(n_hid, n_hid, 1)
        self.norm = nn.GroupNorm(1, n_hid)
        self.readout = nn.Conv2d(n_hid, n_colors, 1)
        self.copy_skip = nn.Parameter(torch.tensor(float(copy_skip)))
        nn.init.zeros_(self.upd2.weight)
        nn.init.zeros_(self.upd2.bias)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, state_onehot, atype_onehot, click_map, hist_onehot=None):
        # state_onehot (B,C,H,W) one-hot colors OR multi-hot object channels;
        # atype_onehot (B,N_ATYPES); click_map (B,1,H,W);
        # hist_onehot (B, history*C, H, W) oldest->newest.
        B, _, H, W = state_onehot.shape
        at = atype_onehot[:, :, None, None].expand(B, N_ATYPES, H, W)
        parts = [state_onehot]
        if self.history:
            parts.append(hist_onehot)
        parts += [at, click_map]
        x = torch.cat(parts, dim=1)
        h = self.embed(x)
        for _ in range(self.n_steps):
            perc = self.perceive(h)
            feats = [h, perc]
            if self.global_pool:
                g = h.mean(dim=(2, 3), keepdim=True).expand_as(h)
                feats.append(g)
            dh = self.upd2(F.gelu(self.upd1(torch.cat(feats, dim=1))))
            h = self.norm(h + dh)
        logits = self.readout(h) + self.copy_skip * state_onehot
        return logits


class RecurrentAutumnNCA(nn.Module):
    """NCA world model with hidden channels carried ACROSS env steps.

    Unlike AutumnNCA (stateless f(state,action)->next), this maintains a hidden
    grid h that persists between env steps, so it can integrate unobservable
    episode-long state (e.g. Mario's bullet counter: ++ on coin pickup, -- on
    fire) that no single frame reveals. Each env step injects the observed frame
    + action into h, runs n_micro cellular updates, and reads out the next frame.
    Trained over sequences with BPTT; h reset to zeros at episode start.
    """

    def __init__(self, n_colors, n_hid=96, n_micro=6, global_pool=True, copy_skip=5.0, pool=None):
        super().__init__()
        self.n_colors = n_colors
        self.n_hid = n_hid
        self.n_micro = n_micro
        self.global_pool = global_pool
        # pool: how the grid-wide summary is reduced. "mean" dilutes sparse signals
        # (a 1-cell button press -> 1/HW of the mean); "max" preserves them (a single
        # active cell survives), so a corner button press reaches every cell in one
        # micro-step. "meanmax" gives both. Defaults derive from global_pool so old
        # checkpoints load unchanged.
        self.pool = pool if pool is not None else ("mean" if global_pool else "none")
        n_pool = {"none": 0, "mean": 1, "max": 1, "meanmax": 2}[self.pool]
        self.obs_embed = nn.Conv2d(n_colors + N_ATYPES + 1, n_hid, 1)
        self.perceive = nn.Conv2d(n_hid, n_hid, 3, padding=1)
        upd_in = n_hid * (2 + n_pool)
        self.upd1 = nn.Conv2d(upd_in, n_hid, 1)
        self.upd2 = nn.Conv2d(n_hid, n_hid, 1)
        self.norm = nn.GroupNorm(1, n_hid)
        self.readout = nn.Conv2d(n_hid, n_colors, 1)
        self.copy_skip = nn.Parameter(torch.tensor(float(copy_skip)))
        nn.init.zeros_(self.upd2.weight); nn.init.zeros_(self.upd2.bias)
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def init_hidden(self, B, H, W, device):
        return torch.zeros(B, self.n_hid, H, W, device=device)

    def step(self, state_onehot, atype_onehot, click_map, h):
        """One env step: returns (next_logits, new_hidden)."""
        B, _, H, W = state_onehot.shape
        at = atype_onehot[:, :, None, None].expand(B, N_ATYPES, H, W)
        h = h + self.obs_embed(torch.cat([state_onehot, at, click_map], dim=1))
        for _ in range(self.n_micro):
            perc = self.perceive(h)
            feats = [h, perc]
            if self.pool in ("mean", "meanmax"):
                feats.append(h.mean(dim=(2, 3), keepdim=True).expand_as(h))
            if self.pool in ("max", "meanmax"):
                feats.append(h.amax(dim=(2, 3), keepdim=True).expand_as(h))
            dh = self.upd2(F.gelu(self.upd1(torch.cat(feats, dim=1))))
            h = self.norm(h + dh)
        logits = self.readout(h) + self.copy_skip * state_onehot
        return logits, h


def _onehot(states, n_colors, device):
    s = torch.as_tensor(np.asarray(states), dtype=torch.long, device=device)
    return F.one_hot(s, n_colors).permute(0, 3, 1, 2).float()


def encode_batch(states, atype, click_x, click_y, n_colors, device, hist_states=None):
    """states (B,H,W) uint8 indices; atype (B,) int; click_x/y (B,).
    hist_states: optional list of (B,H,W) preceding frames (oldest->newest).
    Returns (onehot, atype_onehot, click_map, hist_onehot|None)."""
    onehot = _onehot(states, n_colors, device)
    B, _, H, W = onehot.shape
    at = torch.as_tensor(np.asarray(atype), dtype=torch.long, device=device)
    atype_onehot = F.one_hot(at, N_ATYPES).float()
    click_map = torch.zeros(B, 1, H, W, device=device)
    cx = torch.as_tensor(np.asarray(click_x), dtype=torch.long, device=device)
    cy = torch.as_tensor(np.asarray(click_y), dtype=torch.long, device=device)
    is_click = at == ATYPE_IDX["click"]
    idx = torch.nonzero(is_click, as_tuple=True)[0]
    if idx.numel():
        click_map[idx, 0, cy[idx], cx[idx]] = 1.0
    hist_onehot = None
    if hist_states:
        hist_onehot = torch.cat([_onehot(h, n_colors, device) for h in hist_states], dim=1)
    return onehot, atype_onehot, click_map, hist_onehot
