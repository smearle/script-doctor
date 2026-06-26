"""Code-conditioned neural game engine on GENERATED games.

Counterpart to engine_train.py (in-context NCA-belief). Here the world model is
handed the game's tokenized SOURCE CODE and predicts transitions
f(state, action, code) -> next_state. We always have the code for generated
games, so this is the natural strong engine; comparing its held-out NLL to the
in-context model's measures how much the code is worth / how inferable dynamics
are from experience. Both judged against the no-rule (default movement) baseline.

    .venv/bin/python -u -m game_synth.code_cond_train --updates 6000
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm.active_learning.nca_belief_model import _FiLMDecoder, BeliefConfig
from nca_wm.tokenize_game import VOCAB_SIZE_BASE, get_game_tree_from_js, tokenize_game
from game_synth.engine_train import (B2I, N_ACT, OBJ6, _engine, compile_games,
                                     norule_next, sample_traj)

MAX_TOK = 256


def mixture_nll(logits, logpi, target):
    t = target[:, None]
    lp = t * F.logsigmoid(logits) + (1.0 - t) * F.logsigmoid(-logits)
    lp = lp.flatten(-3).sum(-1)
    return -torch.logsumexp(logpi + lp, dim=-1)


class RuleSlotEncoder(nn.Module):
    """Perceiver-style: game tokens -> K rule slots (learned queries cross-attend)."""
    def __init__(self, vocab, d=96, n_slots=16, n_self=2, heads=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, MAX_TOK, d))
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True), n_self)
        self.slot_q = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, tok, mask):                          # tok,mask (N,L)
        x = self.tok_emb(tok) + self.pos[:, :tok.shape[1]]
        h = self.self_attn(x, src_key_padding_mask=~mask)  # (N,L,d)
        q = self.slot_q[None].expand(tok.shape[0], -1, -1)  # (N,K,d)
        slots, _ = self.cross(q, h, h, key_padding_mask=~mask)
        return slots                                       # (N,K,d)


class RuleAttnCodeWorldModel(nn.Module):
    """Code-conditioned via rule slots + per-cell cross-attention (PuzzleScript-style)."""
    def __init__(self, n_obj=6, n_act=5, d_b=128, d=96, d_cond=64, K=16, n_slots=16,
                 vocab=VOCAB_SIZE_BASE):
        super().__init__()
        self.K = K
        self.cfg = BeliefConfig(n_obj=n_obj, n_act=n_act, d_b=d_b, d_cond=d_cond, K=K)
        self.code_enc = RuleSlotEncoder(vocab, d, n_slots)
        self.state_enc = nn.Sequential(nn.Conv2d(n_obj, d_b, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(d_b, d_b, 3, padding=1), nn.ReLU())
        self.slot_proj = nn.Linear(d, d_b)
        self.cell_attn = nn.MultiheadAttention(d_b, 4, batch_first=True)
        self.emb_a = nn.Embedding(n_act, d_cond)
        self.emb_z = nn.Embedding(K, d_cond)
        self.prior = nn.Sequential(nn.Linear(d_b + d_cond, d_b), nn.ReLU(), nn.Linear(d_b, K))
        self.dec = _FiLMDecoder(self.cfg)

    def logits(self, state, action, tok, mask):
        slots = self.slot_proj(self.code_enc(tok, mask))   # (N,Ks,d_b)
        ctx = self.state_enc(state)                        # (N,d_b,H,W)
        N, d_b, H, W = ctx.shape
        cells = ctx.flatten(2).transpose(1, 2)             # (N,HW,d_b)
        att, _ = self.cell_attn(cells, slots, slots)       # per-cell cross-attn to rule slots
        ctx = (cells + att).transpose(1, 2).reshape(N, d_b, H, W)
        a_emb = self.emb_a(action)
        cond = a_emb[:, None] + self.emb_z.weight[None]     # (N,K,d_cond)
        K = self.K
        ctxK = ctx[:, None].expand(-1, K, -1, -1, -1).reshape(N * K, d_b, H, W)
        lo = self.dec(ctxK, cond.reshape(N * K, -1)).view(N, K, self.cfg.n_obj, H, W)
        logpi = F.log_softmax(self.prior(torch.cat([ctx.mean(dim=(-2, -1)), a_emb], -1)), -1)
        return lo, logpi


def _pool_features(h, axis_pool, axis_cummax, global_pool):
    """Global-context features from NCHW hidden state, broadcast back to (N,C,H,W).
    Each op preserves the channel dim (one summary value per channel), adds no
    parameters. These give the otherwise-local conv body the reach to represent
    long-range PuzzleScript rules:
      axis_pool   (2): row max (over W) / col max (over H) -> "X exists in my row/col"
      axis_cummax (4): directional prefix max L->R / R->L / T->B / B->T (ellipsis/dir)
      global_pool (1): grid max -> "X exists somewhere" (multi-bracket [X] [Y])."""
    feats = []
    if axis_pool:
        feats.append(h.amax(dim=3, keepdim=True).expand_as(h))   # max over W (per row)
        feats.append(h.amax(dim=2, keepdim=True).expand_as(h))   # max over H (per col)
    if axis_cummax:
        feats.append(h.cummax(dim=3).values)                     # L->R along W
        feats.append(h.flip(3).cummax(dim=3).values.flip(3))     # R->L
        feats.append(h.cummax(dim=2).values)                     # T->B along H
        feats.append(h.flip(2).cummax(dim=2).values.flip(2))     # B->T
    if global_pool:
        feats.append(h.amax(dim=(2, 3), keepdim=True).expand_as(h))
    return torch.cat(feats, dim=1) if feats else None


class CodeCondWorldModel(nn.Module):
    """f(state, action, code) -> distribution over next state (K-mode mixture).

    The spatial body is an iterated, weight-shared NCA with global/axis pooling
    (not the old single local-conv pass), so it CAN represent ellipsis,
    multi-bracket, and chain-reaction (startloop) dynamics that a fixed ~7x7
    receptive field cannot. The action enters the body (one-hot planes) so the
    propagation is direction-aware; code conditions the body via FiLM."""
    def __init__(self, n_obj=6, n_act=5, d_b=128, d_tok=96, d_cond=64, K=16,
                 vocab=VOCAB_SIZE_BASE, n_steps=6, axis_pool=True, axis_cummax=True,
                 global_pool=True, input_skip=True, use_layernorm=True):
        super().__init__()
        self.K = K
        self.n_act = n_act
        self.n_steps = n_steps
        self.axis_pool, self.axis_cummax, self.global_pool = axis_pool, axis_cummax, global_pool
        self.input_skip = input_skip
        self.cfg = BeliefConfig(n_obj=n_obj, n_act=n_act, d_b=d_b, d_cond=d_cond, K=K)
        # code encoder: token transformer -> pooled code vector
        self.tok_emb = nn.Embedding(vocab, d_tok)
        self.pos = nn.Parameter(torch.zeros(1, MAX_TOK, d_tok))
        layer = nn.TransformerEncoderLayer(d_tok, 4, d_tok * 2, batch_first=True)
        self.code_enc = nn.TransformerEncoder(layer, 2)
        # NCA body: embed (state + action planes) -> code-FiLM -> iterated update
        self.embed = nn.Conv2d(n_obj + n_act, d_b, 3, padding=1)
        self.code_film = nn.Linear(d_tok, 2 * d_b)
        n_pool = (2 if axis_pool else 0) + (4 if axis_cummax else 0) + (1 if global_pool else 0)
        self.step_conv = nn.Conv2d(d_b, d_b, 3, padding=1)      # weight-shared across steps
        upd_in = d_b * (1 + n_pool) + (d_b if input_skip else 0)
        self.step_update = nn.Conv2d(upd_in, d_b, 1)
        self.ln = nn.GroupNorm(1, d_b) if use_layernorm else None
        # K-mode decoder
        self.emb_a = nn.Embedding(n_act, d_cond)
        self.emb_z = nn.Embedding(K, d_cond)
        self.prior = nn.Sequential(nn.Linear(d_b + d_cond + d_tok, d_b), nn.ReLU(),
                                   nn.Linear(d_b, K))
        self.dec = _FiLMDecoder(self.cfg)

    def encode_code(self, tok, mask):                         # tok,mask (N,L)
        x = self.tok_emb(tok) + self.pos[:, :tok.shape[1]]
        h = self.code_enc(x, src_key_padding_mask=~mask)      # (N,L,d_tok)
        m = mask.float()[..., None]
        return (h * m).sum(1) / m.sum(1).clamp_min(1)         # (N,d_tok)

    def _body(self, state, action, c):
        """Iterated NCA body -> per-cell context (N,d_b,H,W)."""
        N, _, H, W = state.shape
        a_oh = F.one_hot(action, self.n_act).float()[..., None, None].expand(-1, -1, H, W)
        h0 = self.embed(torch.cat([state, a_oh], 1))          # (N,d_b,H,W)
        g, b = self.code_film(c).chunk(2, -1)
        h0 = h0 * (1 + g[..., None, None]) + b[..., None, None]
        msk = (state.sum(1, keepdim=True) > 0).float()        # real cells (zero out padding)
        h = h0 * msk
        for _ in range(self.n_steps):
            hn = self.ln(h) if self.ln is not None else h
            parts = [self.step_conv(hn)]
            pf = _pool_features(hn, self.axis_pool, self.axis_cummax, self.global_pool)
            if pf is not None:
                parts.append(pf)
            if self.input_skip:
                parts.append(h0)
            h = (h + self.step_update(torch.cat(parts, 1))) * msk
        return h

    def logits(self, state, action, tok, mask):
        c = self.encode_code(tok, mask)                       # (N,d_tok)
        ctx = self._body(state, action, c)                    # (N,d_b,H,W)
        N, K = ctx.shape[0], self.K
        a_emb = self.emb_a(action)
        cond = a_emb[:, None] + self.emb_z.weight[None]       # (N,K,d_cond)
        ctxK = ctx[:, None].expand(-1, K, -1, -1, -1).reshape(N * K, *ctx.shape[1:])
        lo = self.dec(ctxK, cond.reshape(N * K, -1))          # (N*K, n_obj, H, W)
        lo = lo.view(N, K, *lo.shape[1:])                     # (N, K, n_obj, H, W)
        logpi = F.log_softmax(self.prior(torch.cat([ctx.mean(dim=(-2, -1)), a_emb, c], -1)), -1)
        return lo, logpi


def tokens_for(jsons, parser):
    """Tokenize each compiled game's mechanics -> padded token tensors."""
    toks = {}
    for name in jsons:
        try:
            tree, ids = get_game_tree_from_js(parser, name)
            t = tokenize_game(tree, ids, encode_sprites=False, include_levels=False)[:MAX_TOK]
            if t:
                toks[name] = t
        except Exception:
            continue
    return toks


def pad_tokens(seqs, device):
    L = max(len(s) for s in seqs)
    tok = torch.zeros(len(seqs), L, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        tok[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = True
    return tok.to(device), mask.to(device)


def transitions_batch(jsons, toks, names, n, device, rng):
    """n (state, action, next_state, code) transitions sampled across games."""
    S, A, NX, codes = [], [], [], []
    for _ in range(n):
        name = rng.choice(names)
        o, a, _ = sample_traj(jsons, name, rng.randint(2, 6), rng)
        t = rng.randrange(len(a))
        S.append(o[t]); A.append(a[t]); NX.append(o[t + 1]); codes.append(toks[name])
    tok, mask = pad_tokens(codes, device)
    return (torch.from_numpy(np.stack(S)).to(device), torch.tensor(A, device=device),
            torch.from_numpy(np.stack(NX)).to(device), tok, mask)


@torch.no_grad()
def eval_codecond(model, jsons, toks, names, device, n_per=6, seed=7):
    rng = random.Random(seed)
    mnll, ident, norule = [], [], []
    z1 = torch.zeros(1, 1, device=device)
    for name in names:
        if name not in toks:
            continue
        tok, mask = pad_tokens([toks[name]], device)
        for _ in range(n_per):
            o, a, _ = sample_traj(jsons, name, 6, rng)
            for t in range(len(a)):
                s = torch.from_numpy(o[t][None]).to(device)
                nx = torch.from_numpy(o[t + 1][None]).to(device)
                act = torch.tensor([a[t]], device=device)
                lo, lp = model.logits(s, act, tok, mask)
                mnll.append(mixture_nll(lo, lp, nx).item())
                ident.append(mixture_nll((s * 12 - 6)[:, None], z1, nx).item())
                nr = norule_next(s, int(a[t]))
                norule.append(mixture_nll((nr * 12 - 6)[:, None], z1, nx).item())
    return float(np.mean(mnll)), float(np.mean(ident)), float(np.mean(norule))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-glob", default="game_synth/fixed_s*/games/*.txt")
    ap.add_argument("--max-games", type=int, default=250)
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="pool", choices=["pool", "rule_attn"],
                    help="pool=mean-pooled FiLM; rule_attn=K rule slots + per-cell cross-attn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)

    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()
    paths = sorted(glob.glob(str(_REPO / args.games_glob)))
    random.Random(args.seed).shuffle(paths)
    print(f"compiling up to {args.max_games} games...", flush=True)
    jsons = compile_games(paths, parser, limit=args.max_games)
    print("tokenizing game code...", flush=True)
    toks = tokens_for(jsons, parser)
    names = [n for n in jsons if n in toks]
    random.Random(args.seed).shuffle(names)
    holdout = names[:args.holdout]; train = names[args.holdout:]
    vocab = max(max(t) for t in toks.values()) + 1
    print(f"usable games: {len(names)} | train {len(train)} | holdout {len(holdout)} | "
          f"median code len {int(np.median([len(toks[n]) for n in names]))} | "
          f"token vocab {vocab}", flush=True)

    V = max(vocab, VOCAB_SIZE_BASE)
    model = (RuleAttnCodeWorldModel(vocab=V) if args.encoder == "rule_attn"
             else CodeCondWorldModel(vocab=V)).to(device)
    print(f"encoder: {args.encoder}", flush=True)
    print(f"code-conditioned engine params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(args.seed); model.train(); t0 = time.time()
    for step in range(args.updates):
        s, a, nx, tok, mask = transitions_batch(jsons, toks, train, args.batch_size, device, rng)
        lo, lp = model.logits(s, a, tok, mask)
        loss = mixture_nll(lo, lp, nx).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  upd/s {(step+1)/max(time.time()-t0,1e-9):.1f}",
                  flush=True)

    model.eval()
    for label, gset in [("TRAINING games", train[:args.holdout]), ("HELD-OUT games", holdout)]:
        m, ident, nr = eval_codecond(model, jsons, toks, gset, device)
        beat = "BEATS" if m < nr else "ABOVE"
        print(f"\n=== {label}: code-conditioned q0 NLL ===", flush=True)
        print(f"  model {m:.4f} | identity {ident:.4f} | no-rule {nr:.4f} -> {beat} no-rule", flush=True)
    torch.save({"model_state": model.state_dict()}, "game_synth/code_engine.pt")
    print("saved game_synth/code_engine.pt", flush=True)


if __name__ == "__main__":
    main()
