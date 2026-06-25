"""Train the haoo' sequence model on the engine-backed world family.

Usage:
    .venv/bin/python -m nca_wm.active_learning.train --steps 3000 \
        --log-dir nca_wm/active_learning/runs/probe0
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Iterator

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from nca_wm.active_learning import eval_probes as EP
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.data import (HaooDataset, collate,
                                         sample_training_tokens)
from nca_wm.active_learning.model import (ModelConfig, TinyTransformerLM,
                                          causal_lm_loss)
from nca_wm.active_learning.vocab import PAD_ID, VOCAB_SIZE, encode


def make_eval_batch(family, n, max_len, seed=99):
    rng = random.Random(seed)
    seqs = []
    while len(seqs) < n:
        ids = encode(sample_training_tokens(family, rng=rng))
        if len(ids) <= max_len:
            seqs.append(ids)
    return collate(seqs)


def cosine_lr(step, total, base, warmup, min_lr):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=320)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--probe-every", type=int, default=500)
    p.add_argument("--probe-samples", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-layouts", type=int, default=64)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--form", type=str, default="every_turn",
                   choices=["every_turn", "adjacency"])
    p.add_argument("--grid-h", type=int, default=3)
    p.add_argument("--grid-w", type=int, default=5)
    p.add_argument("--style", type=str, default="scatter",
                   choices=["scatter", "far_cluster"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="nca_wm/active_learning/runs/default")
    p.add_argument("--ckpt-dir", type=str, default="nca_wm/active_learning/ckpts/default")
    p.add_argument("--device", type=str,
                   default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}")

    family = W.build_family(n_layouts=args.n_layouts, n_seeds=args.n_seeds,
                            seed=args.seed, form=args.form, grid_h=args.grid_h,
                            grid_w=args.grid_w, style=args.style)
    EP.set_family(family)
    print(f"family: {family.mechanisms}  layouts={family.n_layouts}")

    ds = HaooDataset(family, min_prefix_steps=1, max_prefix_steps=8,
                     max_seq_len=args.max_seq_len, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=(device.type == "cuda"))
    eval_ids, eval_mask = make_eval_batch(family, 256, args.max_seq_len)
    eval_ids, eval_mask = eval_ids.to(device), eval_mask.to(device)

    cfg = ModelConfig(vocab_size=VOCAB_SIZE, d_model=args.d_model, n_layer=args.n_layer,
                      n_head=args.n_head, d_ff=args.d_ff, max_seq_len=args.max_seq_len,
                      pad_id=PAD_ID)
    model = TinyTransformerLM(cfg).to(device)
    print(f"model params: {model.num_params():,}")

    decay = [pm for pm in model.parameters() if pm.ndim >= 2]
    nodecay = [pm for pm in model.parameters() if pm.ndim < 2]
    opt = AdamW([{"params": decay, "weight_decay": args.weight_decay},
                 {"params": nodecay, "weight_decay": 0.0}],
                lr=args.lr, betas=(0.9, 0.95))

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "metrics.tsv")
    logf = open(log_path, "w")
    logf.write("step\ttrain_loss\teval_loss\t" + "\t".join(n for n, _, _ in EP._PROBES) + "\n")

    data_iter: Iterator = iter(loader)
    model.train()
    t0 = time.time()
    running, running_n, step = 0.0, 0, 0
    while step < args.steps:
        try:
            input_ids, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input_ids, mask = next(data_iter)
        input_ids, mask = input_ids.to(device), mask.to(device)

        lr = cosine_lr(step, args.steps, args.lr, args.warmup, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        logits = model(input_ids, key_pad_mask=mask)
        loss, n_tok = causal_lm_loss(logits, input_ids, mask, PAD_ID)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        running += loss.item() * float(n_tok.item())
        running_n += int(n_tok.item())

        if step % args.log_every == 0:
            avg = running / max(running_n, 1)
            tps = (step + 1) * args.batch_size / max(time.time() - t0, 1e-9)
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  seq/s {tps:.0f}")
            running, running_n = 0.0, 0

        do_probe = step > 0 and step % args.probe_every == 0
        if do_probe:
            model.eval()
            with torch.no_grad():
                ev_logits = model(eval_ids, key_pad_mask=eval_mask)
                ev_loss, _ = causal_lm_loss(ev_logits, eval_ids, eval_mask, PAD_ID)
            ig = EP.evaluate_probes(model, device, n_samples=args.probe_samples)
            print(f"  eval_loss {ev_loss.item():.4f}  probes: "
                  + ", ".join(f"{k}={v:+.3f}" for k, v in ig.items()))
            logf.write(f"{step}\t{avg:.5f}\t{ev_loss.item():.5f}\t"
                       + "\t".join(f"{ig[n]:.5f}" for n, _, _ in EP._PROBES) + "\n")
            logf.flush()
            model.train()
        step += 1

    model.eval()
    ig = EP.evaluate_probes(model, device, n_samples=max(args.probe_samples, 64))
    print("DONE  final probes: " + ", ".join(f"{k}={v:+.3f}" for k, v in ig.items()))
    logf.close()
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                "args": vars(args)}, os.path.join(args.ckpt_dir, "final.pt"))
    print(f"  saved -> {os.path.join(args.ckpt_dir, 'final.pt')}")


if __name__ == "__main__":
    main()
