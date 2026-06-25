"""Train + gate-eval the NCA-belief model on the sokoban-variant family.

Gate (must match the token model): per-variant held-out NLL (deterministic ~0,
chaos floor) and push-IG identification (fresh high, known ~0, chaos-known ~0).

    .venv/bin/python -u -m nca_wm.active_learning.nca_belief_train --updates 6000
"""
from __future__ import annotations

import argparse
import math
import random
import time

import torch

from nca_wm.active_learning import grid_data as G
from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.collect import (is_push_action, make_mixed_policy,
                                            navigate_policy)
from nca_wm.active_learning.nca_belief_model import BeliefConfig, NCABeliefModel


def _usage_reg(logpi):
    mean = logpi.exp().mean(0).clamp_min(1e-8)
    H = -(mean * mean.log()).sum()
    return math.log(logpi.shape[-1]) - H            # 0 when usage uniform


def roll_loss(model, obs, acts, resamp, usage_w):
    B = model.init_belief(obs[:, 0])
    T = acts.shape[1]
    loss = 0.0
    for t in range(T):
        a, onext, op = acts[:, t], obs[:, t + 1], resamp[:, t]
        l0, p0 = model.q0_logits(B, a)
        l1, p1 = model.q1_logits(B, a, onext)
        loss = loss + model.mixture_nll(l0, p0, onext).mean() \
                    + model.mixture_nll(l1, p1, op).mean() \
                    + usage_w * (_usage_reg(p0) + _usage_reg(p1))
        B = model.update_belief(B, onext, a)
    return loss / T


@torch.no_grad()
def per_variant_nll(model, family, device, n=64, n_steps=8, seed=999):
    out = {}
    for m in family.mechanisms:
        single = W.WorldFamily(jsons={m: family.jsons[m]}, n_layouts=family.n_layouts,
                               form=family.form, grid_h=family.grid_h, grid_w=family.grid_w)
        rng = random.Random(seed + hash(m) % 9973)
        o, a, r = G.batch(single, n, n_steps, rng, navigate_policy)
        o, a = o.to(device), a.to(device)
        B = model.init_belief(o[:, 0])
        for t in range(n_steps - 1):                 # roll belief to the final transition
            B = model.update_belief(B, o[:, t + 1], a[:, t])
        l0, p0 = model.q0_logits(B, a[:, -1])        # predict the last transition's outcome
        out[m] = model.mixture_nll(l0, p0, o[:, -1]).mean().item()
    return out


@torch.no_grad()
def push_ig_probe(model, family, device, n_worlds=8, max_steps=10, n_samples=8, seed=123):
    rng = random.Random(seed)
    per = {m: {"fresh": [], "known": []} for m in family.mechanisms}
    for m in family.mechanisms:
        for _ in range(n_worlds):
            eng = W._new_engine(family.jsons[m], rng.randrange(family.n_layouts))
            id2b = W._engine_id_to_canon_bit(eng)
            obs = W.read_obs(eng, id2b)
            B = model.init_belief(torch.from_numpy(G.obs_to_grid(obs))[None].to(device))
            pushed = False
            for _ in range(max_steps):
                a = navigate_policy(obs, rng, history_ids=[], engine=eng)
                ai = torch.tensor([V.ACTIONS.index(a)], device=device)
                if is_push_action(obs, a):
                    ig = model.information_gain(B, ai, n_samples=n_samples)
                    per[m]["known" if pushed else "fresh"].append(ig)
                    pushed = True
                W.step_engine(eng, a, seed=str(rng.getrandbits(40)))
                obs = W.read_obs(eng, id2b)
                og = torch.from_numpy(G.obs_to_grid(obs))[None].to(device)
                B = model.update_belief(B, og, ai)
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    return {m: {"fresh": mean(d["fresh"]), "known": mean(d["known"]),
                "nf": len(d["fresh"]), "nk": len(d["known"])} for m, d in per.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--usage-w", type=float, default=0.02)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--n-layouts", type=int, default=64)
    p.add_argument("--grid-h", type=int, default=7)
    p.add_argument("--grid-w", type=int, default=8)
    p.add_argument("--p-navigate", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)

    family = W.build_family(n_layouts=args.n_layouts, seed=args.seed, form="sokoban",
                            grid_h=args.grid_h, grid_w=args.grid_w, style="box_pushable")
    family.activate_geometry()
    cfg = BeliefConfig(n_obj=V.N_CANON, n_act=len(V.ACTIONS), K=args.K)
    model = NCABeliefModel(cfg).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"family {family.mechanisms} geom {args.grid_h}x{args.grid_w} | "
          f"belief model params {nparam:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    pol = make_mixed_policy(args.p_navigate)
    rng = random.Random(args.seed)
    t0 = time.time()
    model.train()
    for step in range(args.updates):
        o, a, r = G.batch(family, args.batch_size, args.n_steps, rng, pol)
        o, a, r = o.to(device), a.to(device), r.to(device)
        loss = roll_loss(model, o, a, r, args.usage_w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  "
                  f"upd/s {(step+1)/max(time.time()-t0,1e-9):.1f}", flush=True)

    model.eval()
    print("\n=== per-variant held-out NLL (q0 at identified final transition) ===", flush=True)
    for m, v in per_variant_nll(model, family, device).items():
        print(f"  {m:8s}: {v:.4f}", flush=True)
    print("\n=== push-IG identification (fresh vs known) ===", flush=True)
    for m, d in push_ig_probe(model, family, device).items():
        print(f"  {m:8s}: fresh={d['fresh']:+.3f} (n={d['nf']})  "
              f"known={d['known']:+.3f} (n={d['nk']})", flush=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__},
               "nca_wm/active_learning/ckpts/nca_belief_sokoban.pt")
    print("\nsaved ckpts/nca_belief_sokoban.pt", flush=True)


if __name__ == "__main__":
    main()
