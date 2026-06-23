#!/usr/bin/env python3
"""Controlled validation: does the NCA-native joint sampler beat the independent
marginal when the random spawn is spatially CORRELATED (a connected piece)?

This isolates the conceptual variable. A "click" on a blank grid spawns a random
TETROMINO (one of 7 connected 4-cell shapes) at a random position — the cells are
correlated (they form a connected shape), unlike ants' independent food points.

Hypothesis: the marginal (independent per-cell sampling) scatters cells into
fragments (many connected components), while the MaskGIT-NCA (samples + propagates
intermediate states) produces COHERENT connected pieces (1 component, 4 cells).

    python -m nca_wm.autumn.correlated_test
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

from nca_wm.autumn.model import AutumnNCA, encode_batch, ATYPE_IDX
from nca_wm.autumn.diffusion import MaskNCA
from nca_wm.autumn.latent import _enc_inputs
from nca_wm.autumn.train import pick_device

H = W = 16
SHAPES = [  # 7 tetrominoes (relative cells)
    [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (1, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(1, 0), (1, 1), (1, 2), (0, 2)], [(1, 0), (2, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1)]]


def gen(n, seed=0):
    rng = np.random.default_rng(seed)
    states = np.zeros((n, H, W), np.uint8)            # blank
    nexts = np.zeros((n, H, W), np.uint8)
    for i in range(n):
        sh = SHAPES[rng.integers(len(SHAPES))]
        mx = max(c[0] for c in sh); my = max(c[1] for c in sh)
        ox = rng.integers(0, W - mx); oy = rng.integers(0, H - my)
        for dx, dy in sh:
            nexts[i, oy + dy, ox + dx] = 1            # piece color = 1
    at = np.full(n, ATYPE_IDX["click"], np.uint8)
    cx = rng.integers(0, W, n).astype(np.int16); cy = rng.integers(0, H, n).astype(np.int16)
    return states, nexts, at, cx, cy


def coherence(samples, states):
    """mean #connected-components and #cells among newly-spawned (non-bg) cells."""
    ncomp, ncell = [], []
    for s, st in zip(samples, states):
        new = (s == 1) & (st == 0)
        ncell.append(int(new.sum()))
        if new.sum():
            _, k = label(new)
            ncomp.append(k)
        else:
            ncomp.append(0)
    return np.mean(ncomp), np.std(ncell), np.mean(ncell)


def main():
    import sys
    w_chg = float(sys.argv[sys.argv.index("--w_chg") + 1]) if "--w_chg" in sys.argv else 10.0
    device = pick_device("auto")
    s_, ns_, at_, cx_, cy_ = gen(8000, seed=0)
    vs, vn, va, vcx, vcy = gen(800, seed=99)
    C = 2

    # ---- marginal model (plain CE single-frame) ----
    marg = AutumnNCA(C, n_hid=96, n_steps=10).to(device)
    om = torch.optim.Adam(marg.parameters(), 1e-3)
    for step in range(1, 5001):
        b = np.random.randint(0, len(s_), 64)
        oh, ao, cm, _ = encode_batch(s_[b], at_[b], cx_[b], cy_[b], C, str(device))
        tgt = torch.as_tensor(ns_[b], dtype=torch.long, device=device)
        loss = F.cross_entropy(marg(oh, ao, cm), tgt)
        om.zero_grad(); loss.backward(); om.step()
    print(f"marginal trained (loss {loss.item():.4f})")

    # ---- MaskGIT-NCA joint sampler ----
    diff = MaskNCA(C, n_hid=96, n_steps=10).to(device)
    od = torch.optim.Adam(diff.parameters(), 1e-3)
    for step in range(1, 6001):
        b = np.random.randint(0, len(s_), 64)
        s, _, ao, cm = _enc_inputs(s_[b], ns_[b], at_[b], cx_[b], cy_[b], C, device)
        nxt = torch.as_tensor(ns_[b], dtype=torch.long, device=device)
        r = torch.rand(len(b), 1, 1, device=device); m = torch.rand_like(nxt.float()) < r
        partial = torch.where(m, torch.full_like(nxt, C), nxt)
        po = F.one_hot(partial, C + 1).permute(0, 3, 1, 2).float()
        rec = F.cross_entropy(diff(s, ao, cm, po), nxt, reduction="none")
        chg = (nxt != torch.as_tensor(s_[b], dtype=torch.long, device=device)).float()
        wgt = (1 + chg * 10) * m.float()
        loss = (rec * wgt).sum() / wgt.sum().clamp(min=1)
        od.zero_grad(); loss.backward(); od.step()
    print(f"MaskGIT-NCA trained (loss {loss.item():.4f})")

    # ---- sample + measure coherence on held-out ----
    s, _, ao, cm = _enc_inputs(vs, vn, va, vcx, vcy, C, device)
    with torch.no_grad():
        oh, ao2, cm2, _ = encode_batch(vs, va, vcx, vcy, C, str(device))
        p = torch.softmax(marg(oh, ao2, cm2), 1).permute(0, 2, 3, 1).reshape(-1, C)
        marg_samp = torch.multinomial(p, 1).reshape(len(vs), H, W).cpu().numpy()
    diff_samp = diff.sample(s, ao, cm, T=12).cpu().numpy()

    tc, tstd, tmean = coherence(vn, vs)             # truth: 1 comp, 4 cells
    mc, mstd, mmean = coherence(marg_samp, vs)
    dc, dstd, dmean = coherence(diff_samp, vs)
    print("\n=== spawned-piece coherence (truth = 1 connected component, 4 cells) ===")
    print(f"  truth:          components={tc:.2f}  cells mean={tmean:.2f} std={tstd:.2f}")
    print(f"  marginal:       components={mc:.2f}  cells mean={mmean:.2f} std={mstd:.2f}")
    print(f"  MaskGIT-NCA:    components={dc:.2f}  cells mean={dmean:.2f} std={dstd:.2f}")
    print(f"\n  => coherent if components≈1 & cells≈4. "
          f"NCA-joint {'WINS' if dc < mc - 0.3 else 'ties/loses'} on connectedness "
          f"({dc:.2f} vs marginal {mc:.2f} components)")


if __name__ == "__main__":
    main()
