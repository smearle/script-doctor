"""PyTorch mirror of UnifiedNCAWorldModel for JAX-free serving.

Line-for-line port of ``nca_wm/unified_model.py``. To make weight conversion a
pure copy (no transpose bookkeeping), every parameter is stored in its *Flax
native shape* and all layout-sensitive math is done in ``forward`` with einsum:

    Dense  kernel (in, out)            x @ kernel + bias
    Conv   kernel (kH, kW, in, out)    permuted to (out,in,kH,kW) only at the
                                       F.conv2d call site
    MHA    q/k/v kernel (in, H, Dh)    einsum('bqi,ihd->bqhd', ...)
           out   kernel (H, Dh, out)   einsum('bqhd,hdo->bqo', ...)

So ``load_flax_params`` just copies each leaf into the identically-shaped torch
Parameter. Hidden state is kept NHWC throughout (as in Flax); only conv and
GroupNorm transiently permute to NCHW.

Conventions matched to Flax exactly: LayerNorm/GroupNorm eps = 1e-6 (NOT torch's
1e-5 default); biased variance; attention scale = 1/sqrt(head_dim); GELU exact.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 6  # 0-3 move, 4 action, 5 no-op real-time tick (realtime games only)


def _t(arr) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def gelu(x):
    # Flax/JAX nn.gelu defaults to the tanh approximation; torch defaults to
    # exact erf. Match Flax.
    return F.gelu(x, approximate="tanh")


# ---------------------------------------------------------------------------
# Flax-native primitive layers (params stored in Flax shape; pure-copy load).
# ---------------------------------------------------------------------------
class FDense(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.kernel = nn.Parameter(torch.zeros(in_f, out_f))
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        return x @ self.kernel + self.bias

    def load(self, sub):
        self.kernel.data = _t(sub["kernel"])
        self.bias.data = _t(sub["bias"])


class FConv3x3(nn.Module):
    """3x3 SAME conv. kernel (kH, kW, in, out); NHWC in/out."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.kernel = nn.Parameter(torch.zeros(3, 3, in_f, out_f))
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):  # x: (B, H, W, Cin)
        w = self.kernel.permute(3, 2, 0, 1).contiguous()  # (out,in,kH,kW)
        x = x.permute(0, 3, 1, 2)                          # (B,Cin,H,W)
        y = F.conv2d(x, w, self.bias, padding=1)
        return y.permute(0, 2, 3, 1)                       # (B,H,W,Cout)

    def load(self, sub):
        self.kernel.data = _t(sub["kernel"])
        self.bias.data = _t(sub["bias"])


class FLayerNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):  # normalize over last dim, biased var (matches Flax)
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.scale + self.bias

    def load(self, sub):
        self.scale.data = _t(sub["scale"])
        self.bias.data = _t(sub["bias"])


class FGroupNorm1(nn.Module):
    """Flax GroupNorm(num_groups=1) on NHWC: normalize over (H, W, C)."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):  # x: (B, H, W, C)
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.scale + self.bias

    def load(self, sub):
        self.scale.data = _t(sub["scale"])
        self.bias.data = _t(sub["bias"])


class FEmbed(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(vocab, dim))

    def forward(self, idx):
        return self.embedding[idx]

    def load(self, sub):
        self.embedding.data = _t(sub["embedding"])


class FMHA(nn.Module):
    """Flax MultiHeadDotProductAttention, exact replica.

    q/k/v kernel (in, heads, head_dim); out kernel (heads, head_dim, out).
    """
    def __init__(self, in_q, in_kv, heads, head_dim, out_f):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.q_k = nn.Parameter(torch.zeros(in_q, heads, head_dim))
        self.q_b = nn.Parameter(torch.zeros(heads, head_dim))
        self.k_k = nn.Parameter(torch.zeros(in_kv, heads, head_dim))
        self.k_b = nn.Parameter(torch.zeros(heads, head_dim))
        self.v_k = nn.Parameter(torch.zeros(in_kv, heads, head_dim))
        self.v_b = nn.Parameter(torch.zeros(heads, head_dim))
        self.o_k = nn.Parameter(torch.zeros(heads, head_dim, out_f))
        self.o_b = nn.Parameter(torch.zeros(out_f))

    def forward(self, x_q, x_kv, mask=None):  # x_q (B,Lq,Dq), x_kv (B,Lk,Dkv)
        q = torch.einsum("bqi,ihd->bqhd", x_q, self.q_k) + self.q_b
        k = torch.einsum("bki,ihd->bkhd", x_kv, self.k_k) + self.k_b
        v = torch.einsum("bki,ihd->bkhd", x_kv, self.v_k) + self.v_b
        q = q / math.sqrt(self.head_dim)
        logits = torch.einsum("bqhd,bkhd->bhqk", q, k)
        if mask is not None:  # mask (B,1,1,Lk) bool, True = keep
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        w = torch.softmax(logits, dim=-1)
        o = torch.einsum("bhqk,bkhd->bqhd", w, v)
        return torch.einsum("bqhd,hdo->bqo", o, self.o_k) + self.o_b

    def load(self, sub):
        self.q_k.data = _t(sub["query"]["kernel"]); self.q_b.data = _t(sub["query"]["bias"])
        self.k_k.data = _t(sub["key"]["kernel"]);   self.k_b.data = _t(sub["key"]["bias"])
        self.v_k.data = _t(sub["value"]["kernel"]); self.v_b.data = _t(sub["value"]["bias"])
        self.o_k.data = _t(sub["out"]["kernel"]);   self.o_b.data = _t(sub["out"]["bias"])


# ---------------------------------------------------------------------------
# Encoders (mirror nca_wm.rule_attn_model.RuleSlotEncoder / models.GameSpecEncoder).
# ---------------------------------------------------------------------------
class TorchRuleSlotEncoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_self_layers,
                 n_slots, d_slot, n_heads):
        super().__init__()
        self.d_model, self.n_slots, self.d_slot = d_model, n_slots, d_slot
        self.n_self_layers, self.n_heads = n_self_layers, n_heads
        hd = d_model // n_heads
        self.tok_embed = FEmbed(vocab_size, d_model)
        self.pos_embed = FEmbed(max_seq_len, d_model)
        self.tok_ln1 = nn.ModuleList([FLayerNorm(d_model) for _ in range(n_self_layers)])
        self.tok_attn = nn.ModuleList(
            [FMHA(d_model, d_model, n_heads, hd, d_model) for _ in range(n_self_layers)])
        self.tok_ln2 = nn.ModuleList([FLayerNorm(d_model) for _ in range(n_self_layers)])
        self.tok_ff1 = nn.ModuleList([FDense(d_model, d_model * 4) for _ in range(n_self_layers)])
        self.tok_ff2 = nn.ModuleList([FDense(d_model * 4, d_model) for _ in range(n_self_layers)])
        self.tok_ln_final = FLayerNorm(d_model)
        self.slot_queries = nn.Parameter(torch.zeros(n_slots, d_slot))
        sd = d_slot // n_heads
        self.slot_xattn = FMHA(d_slot, d_model, n_heads, sd, d_slot)
        self.slot_ln = FLayerNorm(d_slot)

    def forward(self, token_ids, mask):  # (B,S) long, (B,S) bool
        B, S = token_ids.shape
        pos = torch.arange(S, device=token_ids.device)
        x = self.tok_embed(token_ids) + self.pos_embed(pos)[None]
        tok_mask = mask[:, None, None, :]
        for i in range(self.n_self_layers):
            y = self.tok_attn[i](self.tok_ln1[i](x), self.tok_ln1[i](x), tok_mask)
            x = x + y
            y = self.tok_ln2[i](x)
            y = self.tok_ff2[i](gelu(self.tok_ff1[i](y)))
            x = x + y
        x = self.tok_ln_final(x)
        q = self.slot_queries[None].expand(B, -1, -1)
        slots = self.slot_xattn(q, x, tok_mask)
        return self.slot_ln(slots)

    def load(self, p):
        self.tok_embed.load(p["tok_embed"]); self.pos_embed.load(p["pos_embed"])
        for i in range(self.n_self_layers):
            self.tok_ln1[i].load(p[f"tok_ln1_{i}"]); self.tok_attn[i].load(p[f"tok_attn_{i}"])
            self.tok_ln2[i].load(p[f"tok_ln2_{i}"]); self.tok_ff1[i].load(p[f"tok_ff1_{i}"])
            self.tok_ff2[i].load(p[f"tok_ff2_{i}"])
        self.tok_ln_final.load(p["tok_ln_final"])
        self.slot_queries.data = _t(p["slot_queries"])
        self.slot_xattn.load(p["slot_xattn"]); self.slot_ln.load(p["slot_ln"])


class TorchGameSpecEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_z, max_seq_len):
        super().__init__()
        self.vocab_size, self.d_model, self.n_layers = vocab_size, d_model, n_layers
        hd = d_model // n_heads
        self.tok_embed = FEmbed(vocab_size, d_model)
        self.pos_embed = FEmbed(max_seq_len, d_model)
        self.ln1 = nn.ModuleList([FLayerNorm(d_model) for _ in range(n_layers)])
        self.attn = nn.ModuleList([FMHA(d_model, d_model, n_heads, hd, d_model) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([FLayerNorm(d_model) for _ in range(n_layers)])
        self.ff1 = nn.ModuleList([FDense(d_model, d_model * 4) for _ in range(n_layers)])
        self.ff2 = nn.ModuleList([FDense(d_model * 4, d_model) for _ in range(n_layers)])
        self.ln_final = FLayerNorm(d_model)
        self.z_proj = FDense(d_model, d_z)

    def forward(self, token_ids, mask):
        B, S = token_ids.shape
        cls = torch.full((B, 1), self.vocab_size - 1, dtype=torch.long, device=token_ids.device)
        all_tok = torch.cat([cls, token_ids], dim=1)
        cls_mask = torch.ones((B, 1), dtype=torch.bool, device=token_ids.device)
        all_mask = torch.cat([cls_mask, mask], dim=1)
        L = all_tok.shape[1]
        pos = torch.arange(L, device=token_ids.device)
        x = self.tok_embed(all_tok) + self.pos_embed(pos)[None]
        attn_mask = all_mask[:, None, None, :]
        for i in range(self.n_layers):
            y = self.attn[i](self.ln1[i](x), self.ln1[i](x), attn_mask)
            x = x + y
            y = self.ff2[i](gelu(self.ff1[i](self.ln2[i](x))))
            x = x + y
        x = self.ln_final(x)
        return self.z_proj(x[:, 0, :])

    def load(self, p):
        self.tok_embed.load(p["tok_embed"]); self.pos_embed.load(p["pos_embed"])
        for i in range(self.n_layers):
            self.ln1[i].load(p[f"ln1_{i}"]); self.attn[i].load(p[f"attn_{i}"])
            self.ln2[i].load(p[f"ln2_{i}"]); self.ff1[i].load(p[f"ff1_{i}"]); self.ff2[i].load(p[f"ff2_{i}"])
        self.ln_final.load(p["ln_final"]); self.z_proj.load(p["z_proj"])


# ---------------------------------------------------------------------------
# Unified world model (mirror of UnifiedNCAWorldModel).
# ---------------------------------------------------------------------------
class TorchUnifiedNCAWorldModel(nn.Module):
    def __init__(self, *, cond="none", recurrent=False, n_hid=128, n_steps=4,
                 n_out=1, n_repeats=1, use_layernorm=False, input_skip=True,
                 axis_pool=False, axis_cummax=False, global_pool=False, history=0,
                 bptt_window=0, vocab_size=142, max_seq_len=192, enc_d_model=64,
                 enc_n_self_layers=2, n_slots=16, n_app_slots=0, d_slot=64,
                 n_attn_heads=4, d_model=64, n_heads=4, n_enc_layers=2, d_z=64):
        super().__init__()
        assert cond in ("none", "film", "rule_attn")
        assert n_steps % n_repeats == 0
        self.cond, self.recurrent = cond, recurrent
        self.n_hid, self.n_steps, self.n_out, self.n_repeats = n_hid, n_steps, n_out, n_repeats
        self.use_layernorm, self.input_skip = use_layernorm, input_skip
        self.axis_pool, self.axis_cummax, self.global_pool = axis_pool, axis_cummax, global_pool
        self.history, self.bptt_window = history, bptt_window
        self.n_slots, self.n_app_slots = n_slots, n_app_slots
        n_layers = n_steps // n_repeats
        self.n_layers = n_layers

        in_dim = n_out + N_ACTIONS + history * (n_out + N_ACTIONS)
        self.embed = FDense(in_dim, n_hid)
        self.step_ln = FLayerNorm(n_hid) if use_layernorm else None
        conv_in = 2 * n_hid if input_skip else n_hid
        self.convs = nn.ModuleList([FConv3x3(conv_in, n_hid) for _ in range(n_layers)])
        has_pool = axis_pool or axis_cummax or global_pool
        n_pool = (2 if axis_pool else 0) + (2 if axis_cummax else 0) + (1 if global_pool else 0)
        self.pool_projs = (nn.ModuleList([FDense(n_hid * (1 + n_pool), n_hid) for _ in range(n_layers)])
                           if has_pool else None)
        self.outs = nn.ModuleList([FDense(n_hid, n_hid) for _ in range(n_layers)])

        if cond == "rule_attn":
            hd = n_hid // n_attn_heads
            self.attn_lns = nn.ModuleList([FLayerNorm(n_hid) for _ in range(n_layers)])
            self.slot_lns = nn.ModuleList([FLayerNorm(d_slot) for _ in range(n_layers)])
            self.xattns = nn.ModuleList(
                [FMHA(n_hid, d_slot, n_attn_heads, hd, n_hid) for _ in range(n_layers)])
            self.encoder = TorchRuleSlotEncoder(
                vocab_size, max_seq_len, enc_d_model, enc_n_self_layers,
                n_slots, d_slot, n_attn_heads)
        elif cond == "film":
            self.encoder = TorchGameSpecEncoder(vocab_size, d_model, n_heads, n_enc_layers, d_z, max_seq_len)
            self.film_gamma = FDense(d_z, n_hid)
            self.film_beta = FDense(d_z, n_hid)

        self.carry_gn = FGroupNorm1(n_hid) if recurrent else None
        self.readout = FDense(n_hid, n_out)
        self.win_ln = FLayerNorm(n_hid)
        self.win_out = FDense(n_hid, 1)

    # ---- pool features (NHWC), mirror _pool_feats_inline ----
    def _pool(self, h):
        feats = []
        if self.axis_pool:
            feats.append(h.amax(dim=2, keepdim=True).expand_as(h))
            feats.append(h.amax(dim=1, keepdim=True).expand_as(h))
        if self.axis_cummax:
            feats.append(torch.cummax(h, dim=2).values)
            feats.append(torch.cummax(h, dim=1).values)
        if self.global_pool:
            feats.append(h.amax(dim=(1, 2), keepdim=True).expand_as(h))
        return feats

    def _layer(self, h, h_inp, mask, i, slots_dyn, gamma, beta):
        B, H, W, _ = h.shape
        h_step = self.step_ln(h) if self.step_ln is not None else h
        conv_in = torch.cat([h_step, h_inp], dim=-1) if self.input_skip else h_step
        h_conv = self.convs[i](conv_in)
        feats = self._pool(h)
        if feats:
            h_conv = self.pool_projs[i](torch.cat([h_conv] + feats, dim=-1))
        core = h_conv
        if self.cond == "rule_attn":
            h_flat = h.reshape(B, H * W, self.n_hid)
            attn_out = self.xattns[i](self.attn_lns[i](h_flat), self.slot_lns[i](slots_dyn))
            core = h_conv + attn_out.reshape(B, H, W, self.n_hid)
        delta = self.outs[i](gelu(core))
        if self.cond == "film":
            delta = gamma * delta + beta
        return (h + delta) * mask

    def _body(self, h, h_inp, mask, slots_dyn, gamma, beta):
        for _ in range(self.n_repeats):
            for i in range(self.n_layers):
                h = self._layer(h, h_inp, mask, i, slots_dyn, gamma, beta)
        return h

    def _win(self, h, mask):
        pooled = (h * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)
        return self.win_out(self.win_ln(pooled)).squeeze(-1)

    @torch.no_grad()
    def forward(self, state, action_onehot, game_tokens=None, game_mask=None):
        # ---- conditioning (computed once) ----
        slots_dyn = gamma = beta = None
        if self.cond == "rule_attn":
            slots = self.encoder(game_tokens, game_mask)
            n_dyn = self.n_slots - self.n_app_slots
            slots_dyn = slots[:, :n_dyn, :]
        elif self.cond == "film":
            z = self.encoder(game_tokens, game_mask)
            gamma = (self.film_gamma(z) + 1.0)[:, None, None, :]
            beta = self.film_beta(z)[:, None, None, :]

        if self.recurrent:
            B, L, C, H, W = state.shape
            h = state.new_zeros(B, H, W, self.n_hid)
            logits_seq, win_seq = [], []
            for t in range(L):
                x = state[:, t].permute(0, 2, 3, 1)
                mask = (x.sum(dim=-1, keepdim=True) > 0).float()
                act = action_onehot[:, t][:, None, None, :].expand(B, H, W, N_ACTIONS)
                h_inp = self.embed(torch.cat([x, act], dim=-1))
                h = (h + h_inp) * mask
                h = self._body(h, h_inp, mask, slots_dyn, gamma, beta)
                h = self.carry_gn(h) * mask
                logits_seq.append(self.readout(h).permute(0, 3, 1, 2))
                win_seq.append(self._win(h, mask))
            logits = torch.stack(logits_seq, dim=1)
            win = torch.stack(win_seq, dim=1)
            sprite = state.new_zeros(B, L, self.n_out, 5, 5, 4)
            return logits, win, sprite

        B, C, H, W = state.shape
        x = state.permute(0, 2, 3, 1)
        act = action_onehot[:, None, None, :].expand(B, H, W, N_ACTIONS)
        inp = torch.cat([x, act], dim=-1)  # history channels unsupported here (history=0 case)
        mask = (x.sum(dim=-1, keepdim=True) > 0).float()
        h_inp = self.embed(inp)
        h = h_inp * mask
        h = self._body(h, h_inp, mask, slots_dyn, gamma, beta)
        logits = self.readout(h).permute(0, 3, 1, 2)
        win = self._win(h, mask)
        sprite = state.new_zeros(B, self.n_out, 5, 5, 4)
        return logits, win, sprite

    # ---- load a Flax param pytree (the contents under params['params']) ----
    def load_flax_params(self, params):
        p = params["params"] if "params" in params else params
        self.embed.load(p["embed"])
        if self.step_ln is not None:
            self.step_ln.load(p["step_ln"])
        for i in range(self.n_layers):
            self.convs[i].load(p[f"conv_{i}"])
            if self.pool_projs is not None:
                self.pool_projs[i].load(p[f"pool_proj_{i}"])
            self.outs[i].load(p[f"out_{i}"])
        if self.cond == "rule_attn":
            for i in range(self.n_layers):
                self.attn_lns[i].load(p[f"attn_ln_{i}"])
                self.slot_lns[i].load(p[f"slot_ln_{i}"])
                self.xattns[i].load(p[f"cell_slot_xattn_{i}"])
            self.encoder.load(p["game_encoder"])
        elif self.cond == "film":
            self.encoder.load(p["game_encoder"])
            self.film_gamma.load(p["film_gamma"]); self.film_beta.load(p["film_beta"])
        if self.carry_gn is not None:
            self.carry_gn.load(p["carry_gn"])
        self.readout.load(p["readout"]); self.win_ln.load(p["win_ln"]); self.win_out.load(p["win_out"])
        return self
