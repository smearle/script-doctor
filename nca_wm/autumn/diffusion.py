#!/usr/bin/env python3
"""NCA-native joint sampler (MaskGIT-style discrete diffusion).

Tests the claim: an NCA can model the spatial JOINT next-state distribution if it
SAMPLES intermediate states and propagates them, rather than emitting one
factorized per-cell output. The NCA body is unchanged; we (a) feed a partially-
revealed next-state as extra input, (b) train with random masking, (c) sample
iteratively — committing confident cells over T rounds so that committing one
food cell conditions/suppresses the rest -> coherent count.

Focused ants experiment vs the independent marginal:
    python -m nca_wm.autumn.diffusion --game ants --updates 8000
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca_wm.autumn.model import N_ATYPES, ATYPE_IDX
from nca_wm.autumn.train import pick_device
from nca_wm.autumn.latent import _enc_inputs   # state/next/action/click one-hots


class MaskNCA(nn.Module):
    """NCA decoder p(next | state, action, partial_next). partial_next uses an
    extra MASK token (index n_colors) for not-yet-revealed cells."""
    def __init__(self, n_colors, n_hid=96, n_steps=10):
        super().__init__()
        self.n_colors = n_colors
        self.n_steps = n_steps
        in_ch = n_colors + N_ATYPES + 1 + (n_colors + 1)  # state + action + click + partial(+mask)
        self.embed = nn.Conv2d(in_ch, n_hid, 1)
        self.perceive = nn.Conv2d(n_hid, n_hid, 3, padding=1)
        self.upd1 = nn.Conv2d(n_hid * 3, n_hid, 1); self.upd2 = nn.Conv2d(n_hid, n_hid, 1)
        self.norm = nn.GroupNorm(1, n_hid)
        self.readout = nn.Conv2d(n_hid, n_colors, 1)
        self.copy = nn.Parameter(torch.tensor(3.0))
        nn.init.zeros_(self.upd2.weight); nn.init.zeros_(self.upd2.bias)
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def forward(self, s, at_oh, click, partial_oh):
        h = self.embed(torch.cat([s, at_oh, click, partial_oh], 1))
        for _ in range(self.n_steps):
            perc = self.perceive(h)
            g = h.mean(dim=(2, 3), keepdim=True).expand_as(h)
            h = self.norm(h + self.upd2(F.gelu(self.upd1(torch.cat([h, perc, g], 1)))))
        return self.readout(h) + self.copy * s

    @torch.no_grad()
    def sample(self, s, at_oh, click, T=10):
        """Iterative confidence-based decoding -> (B,H,W) coherent sample."""
        B, _, H, W = s.shape; C = self.n_colors; MASK = C
        cur = torch.full((B, H, W), MASK, dtype=torch.long, device=s.device)
        n = H * W
        for t in range(1, T + 1):
            partial = F.one_hot(cur, C + 1).permute(0, 3, 1, 2).float()
            logits = self.forward(s, at_oh, click, partial)
            probs = torch.softmax(logits, 1)                       # (B,C,H,W)
            samp = torch.multinomial(probs.permute(0, 2, 3, 1).reshape(-1, C), 1).reshape(B, H, W)
            conf = probs.gather(1, samp[:, None]).squeeze(1)       # confidence of the sample
            masked = cur == MASK
            conf = torch.where(masked, conf, torch.full_like(conf, 2.0))  # keep committed
            # number to KEEP masked after this round (cosine schedule -> 0)
            keep = int(np.floor(n * np.cos(np.pi / 2 * t / T)))
            flat_conf = conf.reshape(B, n)
            thresh = flat_conf.kthvalue(max(keep, 1), dim=1, keepdim=True).values if keep > 0 else flat_conf.min(1, keepdim=True).values - 1
            commit = (flat_conf > thresh).reshape(B, H, W) & masked
            cur = torch.where(commit, samp, cur)
            if t == T:
                cur = torch.where(cur == MASK, samp, cur)
        return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ants")
    ap.add_argument("--updates", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_chg", type=float, default=20.0)
    ap.add_argument("--T", type=int, default=10)
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
    model = MaskNCA(C).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"MaskNCA {args.game}: C={C} N={N} params={sum(p.numel() for p in model.parameters()):,}")

    for step in range(1, args.updates + 1):
        b = rng.choice(train, size=args.batch_size)
        s, _, at_oh, click = _enc_inputs(s_[b], ns_[b], at_[b], cx_[b], cy_[b], C, device)
        nxt = torch.as_tensor(ns_[b], dtype=torch.long, device=device)
        r = torch.rand(len(b), 1, 1, device=device)
        m = (torch.rand_like(nxt.float()) < r)                      # cells to MASK
        partial = torch.where(m, torch.full_like(nxt, C), nxt)
        partial_oh = F.one_hot(partial, C + 1).permute(0, 3, 1, 2).float()
        logits = model(s, at_oh, click, partial_oh)
        rec_pc = F.cross_entropy(logits, nxt, reduction="none")
        chg = (nxt != torch.as_tensor(s_[b], dtype=torch.long, device=device)).float()
        w = (1.0 + chg * args.w_chg) * m.float()                    # supervise masked cells (weight changed)
        loss = (rec_pc * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == 1:
            print(f"[{step}] loss={loss.item():.4f}")

    # ---- experiment: sampled NEW-food count distribution ----
    from nca_wm.autumn.model import encode_batch
    from nca_wm.autumn.infer import load_run
    clk = np.where(at_[val] == ATYPE_IDX["click"])[0]; vi = val[clk][:600]
    s, _, at_oh, click = _enc_inputs(s_[vi], ns_[vi], at_[vi], cx_[vi], cy_[vi], C, device)
    cur0 = (s_[vi] == FOOD)
    truth = ((ns_[vi] == FOOD) & ~cur0).sum((1, 2))
    diff_samp = model.sample(s, at_oh, click, T=args.T).cpu().numpy()
    diff_cnt = ((diff_samp == FOOD) & ~cur0).sum((1, 2))
    base, _ = load_run(f"nca_wm/autumn/runs/{args.game}_singleframe", device=str(device))
    oh, ao, cm, _ = encode_batch(s_[vi], at_[vi], cx_[vi], cy_[vi], C, str(device))
    with torch.no_grad():
        p = torch.softmax(base(oh, ao, cm), 1).permute(0, 2, 3, 1).reshape(-1, C)
        ms = torch.multinomial(p, 1).reshape(len(vi), s_.shape[1], s_.shape[2]).cpu().numpy()
    base_cnt = ((ms == FOOD) & ~cur0).sum((1, 2))

    def hist(x): h = np.bincount(np.clip(x, 0, 6), minlength=7)[:7]; return h / h.sum()
    he = hist(truth)
    print("\n=== sampled NEW-food count per click ===")
    print(f"  engine (truth):      mean={truth.mean():.2f} std={truth.std():.2f}")
    print(f"  NCA-diffusion:       mean={diff_cnt.mean():.2f} std={diff_cnt.std():.2f}  L1={np.abs(he-hist(diff_cnt)).sum():.3f}")
    print(f"  marginal (indep.):   mean={base_cnt.mean():.2f} std={base_cnt.std():.2f}  L1={np.abs(he-hist(base_cnt)).sum():.3f}")
    # spatial correlation of the diffusion sample distribution to engine
    emp = ((ns_[vi] == FOOD) & ~cur0).mean(0)
    dmap = ((diff_samp == FOOD) & ~cur0).mean(0)
    print(f"  NCA-diffusion spatial corr to engine: {np.corrcoef(dmap.ravel(), emp.ravel())[0,1]:.3f}")


if __name__ == "__main__":
    main()
