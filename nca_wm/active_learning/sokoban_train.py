"""Step 3: train the scaled model on the sokoban-variant family + identify.

Trains a scaled token transformer on haoo' data from the 5-variant sokoban family
(classic/inert/slide/swap/chaos), then reports:
  (a) per-variant held-out predictive NLL (did it learn each push mechanic), and
  (b) a push-IG identification probe: IG of pushing a box when the variant is
      still UNKNOWN (fresh) vs AFTER one push has revealed it (known). Expect
      fresh >> known for all variants, including chaos-known ~0 (not mesmerized).

    .venv/bin/python -u -m nca_wm.active_learning.sokoban_train --updates 8000
"""
from __future__ import annotations

import argparse
import math
import random
import time

import torch
import torch.nn as nn

from nca_wm.active_learning import collect as C
from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.collect import navigate_policy
from nca_wm.active_learning.compare_collection import eval_loss
from nca_wm.active_learning.data import HaooDataset, collate
from nca_wm.active_learning.inference import estimate_information_gain
from nca_wm.active_learning.model import (ModelConfig, TinyTransformerLM,
                                          causal_lm_loss)
from torch.utils.data import DataLoader


def cosine_lr(step, total, base, warmup, min_lr):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


def per_variant_nll(model, family, device, n=160, seed=999):
    """Held-out navigate-collected resample-NLL per variant (lower = learned it)."""
    out = {}
    for mech in family.mechanisms:
        # one-variant family view: reuse build_dataset but force the mechanism by
        # building sequences only from this variant's json.
        seqs = _variant_dataset(family, mech, n, seed)
        _, rs = eval_loss(model, seqs, device)
        out[mech] = rs
    return out


def _variant_dataset(family, mech, n, seed):
    import random as _r
    from nca_wm.active_learning.data import sample_training_tokens
    rng = _r.Random(seed + hash(mech) % 9973)
    single = W.WorldFamily(jsons={mech: family.jsons[mech]}, n_layouts=family.n_layouts,
                           form=family.form, grid_h=family.grid_h, grid_w=family.grid_w)
    seqs = []
    while len(seqs) < n:
        ids = V.encode(sample_training_tokens(single, 6, 12, rng=rng, policy=navigate_policy))
        if len(ids) <= 1024:
            seqs.append(ids)
    return seqs


def push_ig_probe(model, family, device, n_worlds=6, max_steps=10, n_samples=8,
                  seed=123, max_seq_len=1024):
    rng = random.Random(seed)
    # IG scoring appends ~2 obs blocks; leave headroom so it never overruns ctx.
    hist_cap = max_seq_len - 2 * (V.N_CELLS + 2) - 8
    per = {m: {"fresh": [], "known": []} for m in family.mechanisms}
    for m in family.mechanisms:
        for _ in range(n_worlds):
            li = rng.randrange(family.n_layouts)
            eng = W._new_engine(family.jsons[m], li); id2b = W._engine_id_to_canon_bit(eng)
            obs = W.read_obs(eng, id2b)
            hist = V.encode([V.BOS] + V.serialize_obs(obs))
            pushed = False
            for _ in range(max_steps):
                if len(hist) > hist_cap:
                    break
                a = navigate_policy(obs, rng, history_ids=hist, engine=eng)
                if C.is_push_action(obs, a):
                    ig = estimate_information_gain(model, hist, a, device, n_samples=n_samples)
                    per[m]["known" if pushed else "fresh"].append(ig)
                    pushed = True
                W.step_engine(eng, a, seed=str(rng.getrandbits(40)))
                obs = W.read_obs(eng, id2b)
                hist += V.encode(V.serialize_action(a) + V.serialize_obs(obs))
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    return {m: {"fresh": mean(per[m]["fresh"]), "known": mean(per[m]["known"]),
                "nf": len(per[m]["fresh"]), "nk": len(per[m]["known"])}
            for m in family.mechanisms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
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
    print(f"family {family.mechanisms} geom {args.grid_h}x{args.grid_w} vocab {V.VOCAB_SIZE}",
          flush=True)

    ds = HaooDataset(family, min_prefix_steps=4, max_prefix_steps=10,
                     max_seq_len=args.max_seq_len, seed=args.seed,
                     policy=C.make_mixed_policy(args.p_navigate))
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)

    cfg = ModelConfig(vocab_size=V.VOCAB_SIZE, d_model=args.d_model, n_layer=args.n_layer,
                      n_head=args.n_head, d_ff=args.d_ff, max_seq_len=args.max_seq_len,
                      pad_id=V.PAD_ID)
    model = TinyTransformerLM(cfg).to(device)
    print(f"model params: {model.num_params():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    di = iter(loader); model.train(); t0 = time.time()
    for step in range(args.updates):
        try:
            ids, mask = next(di)
        except StopIteration:
            di = iter(loader); ids, mask = next(di)
        ids, mask = ids.to(device), mask.to(device)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.updates, args.lr, 300, 3e-5)
        logits = model(ids, key_pad_mask=mask)
        loss, _ = causal_lm_loss(logits, ids, mask, V.PAD_ID)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  "
                  f"seq/s {(step+1)*args.batch_size/max(time.time()-t0,1e-9):.0f}", flush=True)

    model.eval()
    print("\n=== per-variant held-out predictive NLL (lower = learned the mechanic) ===",
          flush=True)
    nll = per_variant_nll(model, family, device)
    for m, v in nll.items():
        print(f"  {m:8s}: {v:.4f}", flush=True)

    print("\n=== push-IG identification (fresh = variant unknown, known = after a push) ===",
          flush=True)
    probe = push_ig_probe(model, family, device)
    for m, d in probe.items():
        print(f"  {m:8s}: fresh={d['fresh']:+.3f} (n={d['nf']})  "
              f"known={d['known']:+.3f} (n={d['nk']})", flush=True)

    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__},
               "nca_wm/active_learning/ckpts/sokoban_final.pt")
    print("\nsaved ckpts/sokoban_final.pt", flush=True)


if __name__ == "__main__":
    main()
