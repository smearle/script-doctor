"""Online-vs-offline data-collection comparison (standing-directive target).

At a matched budget of B collected haoo' sequences, train a from-scratch world
model on data gathered by each collection policy, then measure held-out
predictive loss on a SHARED informative eval set (same sequences for every
model -> identical conditioning, so the metric reflects learned dynamics, not
trajectory-distribution match). Expect active (navigate / IG planner) >= passive
(random) up to a budget limit, after which both saturate.

    .venv/bin/python -m nca_wm.active_learning.compare_collection \
        --budgets 64,128,256,512,1024,2048 --updates 1500
"""
from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn

from nca_wm.active_learning import collect as C
from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.data import collate
from nca_wm.active_learning.model import (ModelConfig, TinyTransformerLM,
                                          causal_lm_loss)

_RS_ID = V.STOI[V.RESAMPLE_OBS]


def _resample_mask(ids: list[int]) -> list[bool]:
    """True on tokens belonging to the final RESAMPLE_OBS outcome block."""
    try:
        i = len(ids) - 1 - ids[::-1].index(_RS_ID)
    except ValueError:
        return [False] * len(ids)
    return [j > i for j in range(len(ids))]


def make_eval_set(family, n, seed):
    """Shared held-out informative eval set (navigate policy guarantees triggers)."""
    seqs = C.build_dataset(family, C.navigate_policy, n,
                           min_prefix_steps=6, max_prefix_steps=12, seed=seed)
    return seqs


@torch.no_grad()
def eval_loss(model, seqs, device, batch=128):
    model.eval()
    tot_full, n_full, tot_rs, n_rs = 0.0, 0, 0.0, 0
    for k in range(0, len(seqs), batch):
        chunk = seqs[k:k + batch]
        ids, mask = collate(chunk)
        ids, mask = ids.to(device), mask.to(device)
        logits = model(ids, key_pad_mask=mask)
        # full-sequence loss
        l, n = causal_lm_loss(logits, ids, mask, V.PAD_ID)
        tot_full += l.item() * int(n.item()); n_full += int(n.item())
        # resample-block-only loss (the held-out, mechanism-determined outcome)
        rs = torch.zeros_like(mask)
        for bi, s in enumerate(chunk):
            rm = _resample_mask(s)
            rs[bi, :len(rm)] = torch.tensor(rm, device=device)
        rs = rs & mask
        l2, n2 = causal_lm_loss(logits, ids, rs, V.PAD_ID)
        if int(n2.item()) > 0:
            tot_rs += l2.item() * int(n2.item()); n_rs += int(n2.item())
    return tot_full / max(n_full, 1), tot_rs / max(n_rs, 1)


def train_on_dataset(seqs, device, updates, batch=64, seed=0, max_seq_len=400):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=V.VOCAB_SIZE, d_model=128, n_layer=4, n_head=4,
                      d_ff=384, max_seq_len=max_seq_len, pad_id=V.PAD_ID)
    model = TinyTransformerLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.01)
    rng = random.Random(seed)
    model.train()
    for u in range(updates):
        chunk = [seqs[rng.randrange(len(seqs))] for _ in range(min(batch, len(seqs)))]
        ids, mask = collate(chunk)
        ids, mask = ids.to(device), mask.to(device)
        logits = model(ids, key_pad_mask=mask)
        loss, _ = causal_lm_loss(logits, ids, mask, V.PAD_ID)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


POLICIES = {"random": C.random_policy, "navigate": C.navigate_policy}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=str, default="64,128,256,512,1024,2048")
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--n-layouts", type=int, default=64)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--grid-h", type=int, default=5)
    p.add_argument("--grid-w", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1,
                   help="average over this many seeds (mean+/-std error bands)")
    p.add_argument("--device", type=str,
                   default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)
    budgets = [int(b) for b in args.budgets.split(",")]

    family = W.build_family(n_layouts=args.n_layouts, n_seeds=args.n_seeds,
                            seed=args.seed, form="adjacency", grid_h=args.grid_h,
                            grid_w=args.grid_w, style="far_cluster")
    family.activate_geometry()
    eval_set = make_eval_set(family, args.eval_n, seed=999)  # held-out worlds
    print(f"family {family.mechanisms} geom {args.grid_h}x{args.grid_w} | "
          f"eval_n={len(eval_set)} | budgets={budgets} | updates={args.updates}")
    print(f"{'budget':>7} | {'policy':>8} | {'full_loss':>9} | {'resample_loss':>13}")

    import statistics
    # results[pol]["rs"][i] = list of per-seed resample losses at budgets[i]
    results = {pol: {"full": [[] for _ in budgets], "rs": [[] for _ in budgets]}
               for pol in POLICIES}
    for bi, B in enumerate(budgets):
        for pol, fn in POLICIES.items():
            for s in range(args.seeds):
                data = C.build_dataset(family, fn, B, min_prefix_steps=4,
                                       max_prefix_steps=12, seed=1000 + B + 7919 * s)
                model = train_on_dataset(data, device, args.updates, seed=args.seed + s)
                full, rs = eval_loss(model, eval_set, device)
                results[pol]["full"][bi].append(full)
                results[pol]["rs"][bi].append(rs)
            mrs = statistics.mean(results[pol]["rs"][bi])
            srs = statistics.pstdev(results[pol]["rs"][bi]) if args.seeds > 1 else 0.0
            print(f"{B:>7} | {pol:>8} | resample {mrs:>9.4f} +/- {srs:.4f}", flush=True)

    def mean(pol, key):
        return [statistics.mean(v) for v in results[pol][key]]

    def std(pol, key):
        return [statistics.pstdev(v) if len(v) > 1 else 0.0 for v in results[pol][key]]

    print("\nResample-block loss mean (lower=better):")
    for pol in POLICIES:
        print(f"  {pol:>8}: " + "  ".join(f"{B}:{v:.3f}" for B, v in zip(budgets, mean(pol, 'rs'))))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, key, title in [(axes[0], "full", "Full-sequence loss"),
                               (axes[1], "rs", "Resample-outcome loss")]:
            for pol in POLICIES:
                m, e = mean(pol, key), std(pol, key)
                ax.plot(budgets, m, marker="o", label=pol)
                lo = [max(a - b, 1e-6) for a, b in zip(m, e)]
                hi = [a + b for a, b in zip(m, e)]
                ax.fill_between(budgets, lo, hi, alpha=0.2)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("collection budget (# sequences)")
            ax.set_ylabel("held-out NLL")
            ax.set_title(title)
            ax.legend(); ax.grid(True, alpha=0.3)
        fig.suptitle("Online (navigate) vs offline (random) data collection")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"nca_wm/figures/active_collection_online_vs_offline.{ext}",
                        dpi=140, bbox_inches="tight")
        print("\nsaved nca_wm/figures/active_collection_online_vs_offline.{png,pdf}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
