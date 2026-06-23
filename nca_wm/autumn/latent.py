#!/usr/bin/env python3
"""Latent-variable (CVAE) stochastic world model — captures the JOINT next-state
distribution so samples are COHERENT (e.g. "exactly the spawned food, location
chosen by z"), unlike the per-cell marginal (which, sampled independently, gives
a variable/incoherent food count).

Encoder q(z | state, action, next) -> N(mean, var); NCA decoder p(next | state,
action, z); trained with the ELBO (CE reconstruction + beta*KL). At sampling,
z ~ N(0,I) -> a coherent next-state sample.

Focused experiment (ants, random food spawns):
    python -m nca_wm.autumn.latent --game ants --updates 6000
compares the sampled food-COUNT distribution: latent-z vs independent-marginal vs engine.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca_wm.autumn.model import N_ATYPES, ATYPE_IDX, action_to_fields
from nca_wm.autumn.train import pick_device


def _enc_inputs(states, nexts, at, cx, cy, n_colors, device):
    s = F.one_hot(torch.as_tensor(states, dtype=torch.long, device=device), n_colors).permute(0, 3, 1, 2).float()
    n = F.one_hot(torch.as_tensor(nexts, dtype=torch.long, device=device), n_colors).permute(0, 3, 1, 2).float()
    B, _, H, W = s.shape
    a = torch.as_tensor(at, dtype=torch.long, device=device)
    at_oh = F.one_hot(a, N_ATYPES).float()[:, :, None, None].expand(B, N_ATYPES, H, W)
    click = torch.zeros(B, 1, H, W, device=device)
    cxx = torch.as_tensor(cx, dtype=torch.long, device=device); cyy = torch.as_tensor(cy, dtype=torch.long, device=device)
    idx = torch.nonzero(a == ATYPE_IDX["click"], as_tuple=True)[0]
    if idx.numel():
        click[idx, 0, cyy[idx], cxx[idx]] = 1.0
    return s, n, at_oh, click


class Encoder(nn.Module):
    def __init__(self, n_colors, zdim):
        super().__init__()
        in_ch = n_colors * 2 + N_ATYPES + 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.mean = nn.Linear(64, zdim); self.logvar = nn.Linear(64, zdim)

    def forward(self, s, n, at_oh, click):
        h = self.net(torch.cat([s, n, at_oh, click], 1)).mean(dim=(2, 3))
        return self.mean(h), self.logvar(h)


class Decoder(nn.Module):
    """NCA conditioned on z (z broadcast as extra input channels)."""
    def __init__(self, n_colors, zdim, n_hid=96, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        in_ch = n_colors + N_ATYPES + 1 + zdim
        self.embed = nn.Conv2d(in_ch, n_hid, 1)
        self.perceive = nn.Conv2d(n_hid, n_hid, 3, padding=1)
        self.upd1 = nn.Conv2d(n_hid * 3, n_hid, 1); self.upd2 = nn.Conv2d(n_hid, n_hid, 1)
        self.norm = nn.GroupNorm(1, n_hid)
        self.readout = nn.Conv2d(n_hid, n_colors, 1)
        self.copy = nn.Parameter(torch.tensor(3.0))
        nn.init.zeros_(self.upd2.weight); nn.init.zeros_(self.upd2.bias)
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def forward(self, s, at_oh, click, z):
        B, _, H, W = s.shape
        zb = z[:, :, None, None].expand(B, z.shape[1], H, W)
        h = self.embed(torch.cat([s, at_oh, click, zb], 1))
        for _ in range(self.n_steps):
            perc = self.perceive(h)
            g = h.mean(dim=(2, 3), keepdim=True).expand_as(h)
            h = self.norm(h + self.upd2(F.gelu(self.upd1(torch.cat([h, perc, g], 1)))))
        return self.readout(h) + self.copy * s


class CVAE(nn.Module):
    def __init__(self, n_colors, zdim=16, n_hid=96, n_steps=8):
        super().__init__()
        self.zdim = zdim
        self.enc = Encoder(n_colors, zdim)
        self.dec = Decoder(n_colors, zdim, n_hid, n_steps)

    def elbo(self, s, n, at_oh, click, tgt, beta, w_chg=40.0, free_bits=0.5):
        mean, logvar = self.enc(s, n, at_oh, click)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        logits = self.dec(s, at_oh, click, z)
        # weight CHANGED cells (food spawns, moves) so the rare stochastic signal
        # isn't negligible in the loss -> decoder is forced to use z (no collapse).
        rec_pc = F.cross_entropy(logits, tgt, reduction="none")
        changed = (tgt != s.argmax(1)).float()
        w = 1.0 + changed * w_chg
        rec = (rec_pc * w).sum() / w.sum()
        # per-dim KL with free bits so z must carry >= free_bits nats
        kl_dim = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl = torch.clamp(kl_dim, min=free_bits / kl_dim.shape[1]).sum(1).mean()
        return rec + beta * kl, rec, kl

    @torch.no_grad()
    def sample(self, s, at_oh, click):
        z = torch.randn(s.shape[0], self.zdim, device=s.device)
        return self.dec(s, at_oh, click, z).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ants")
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--zdim", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--w_chg", type=float, default=40.0)
    ap.add_argument("--free_bits", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)

    d = np.load(f"nca_wm/autumn/data/{args.game}.npz", allow_pickle=True)
    pal = list(d["palette"]); C = len(pal); FOOD = pal.index("red") if "red" in pal else -1
    s_, ns_, at_, cx_, cy_ = d["states"], d["next_states"], d["action_type"], d["click_x"], d["click_y"]
    N = len(s_); rng = np.random.default_rng(0); perm = rng.permutation(N)
    val, train = perm[:int(0.1 * N)], perm[int(0.1 * N):]
    model = CVAE(C, zdim=args.zdim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"CVAE {args.game}: C={C} zdim={args.zdim} N={N} params={sum(p.numel() for p in model.parameters()):,}")

    for step in range(1, args.updates + 1):
        b = rng.choice(train, size=args.batch_size)
        s, n, at_oh, click = _enc_inputs(s_[b], ns_[b], at_[b], cx_[b], cy_[b], C, device)
        tgt = torch.as_tensor(ns_[b], dtype=torch.long, device=device)
        beta = args.beta * min(1.0, step / 1000)   # KL anneal to avoid posterior collapse
        loss, rec, kl = model.elbo(s, n, at_oh, click, tgt, beta, w_chg=args.w_chg, free_bits=args.free_bits)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == 1:
            print(f"[{step}] loss={loss.item():.4f} rec={rec.item():.4f} kl={kl.item():.3f}")

    # ---- experiment: sampled food-COUNT distribution on click transitions ----
    from nca_wm.autumn.model import AutumnNCA, encode_batch
    base, bcfg = None, None
    try:
        base, bcfg = __import__("nca_wm.autumn.infer", fromlist=["load_run"]).load_run(
            f"nca_wm/autumn/runs/{args.game}_singleframe", device=str(device))
    except Exception as e:
        print("baseline load failed:", e)
    clk = np.where(at_[val] == ATYPE_IDX["click"])[0]
    vi = val[clk][:600]
    s, _, at_oh, click = _enc_inputs(s_[vi], ns_[vi], at_[vi], cx_[vi], cy_[vi], C, device)
    cur_food = (s_[vi] == FOOD).sum((1, 2))
    true_new = ((ns_[vi] == FOOD) & ~(s_[vi] == FOOD)).sum((1, 2))   # engine new-food count

    def newcount(pred):  # food cells in pred not already present
        return ((pred == FOOD) & ~(s_[vi] == FOOD)).sum((1, 2))

    cvae_counts = newcount(model.sample(s, at_oh, click).cpu().numpy())
    if base is not None:
        oh, ao, cm, _ = encode_batch(s_[vi], at_[vi], cx_[vi], cy_[vi], C, str(device))
        with torch.no_grad():
            p = torch.softmax(base(oh, ao, cm), 1).permute(0, 2, 3, 1).reshape(-1, C)
            samp = torch.multinomial(p, 1).reshape(len(vi), s_.shape[1], s_.shape[2]).cpu().numpy()
        base_counts = newcount(samp)
    print("\n=== sampled NEW-food count per click (coherence of the joint) ===")
    print(f"  engine (truth):     mean={true_new.mean():.2f} std={true_new.std():.2f}")
    print(f"  CVAE (z-sample):    mean={cvae_counts.mean():.2f} std={cvae_counts.std():.2f}")
    if base is not None:
        print(f"  marginal (indep.):  mean={base_counts.mean():.2f} std={base_counts.std():.2f}")
    # closeness of count distribution to engine (lower = better)
    def hist(x): h = np.bincount(np.clip(x, 0, 6), minlength=7)[:7]; return h / h.sum()
    he = hist(true_new); hc = hist(cvae_counts)
    print(f"  count-dist L1 to engine: CVAE={np.abs(he-hc).sum():.3f}"
          + (f"  marginal={np.abs(he-hist(base_counts)).sum():.3f}" if base is not None else ""))


if __name__ == "__main__":
    main()
