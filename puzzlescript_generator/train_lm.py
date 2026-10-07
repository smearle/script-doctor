"""Train the PuzzleScript game generator (next-token LM over whole games).

One game per sequence (bos ... eos), truncated to --ctx tokens, never packed with
another game. Games are bucketed by length; each step takes a batch from one bucket
with about --batch-tokens padded tokens, and the loss averages over real tokens.

Resumable: --out/last.pt holds model, optimizer, schedule position, epoch batch
order and RNG state; a restart continues from it. best.pt keeps the lowest-val-loss
weights. status.json carries a heartbeat for monitors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F

from model import GPT, GPTConfig

BUCKETS = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


def load_split(data: Path, split: str, ctx: int):
    arr = np.memmap(data / f"{split}.bin", dtype=np.uint16, mode="r")
    off = np.load(data / f"{split}_offsets.npy")
    lens = np.minimum(np.diff(off), ctx + 1)  # +1: inputs and shifted targets
    return arr, off, lens


def make_batches(lens, ctx, batch_tokens, rng):
    buckets = [b for b in BUCKETS if b < ctx] + [ctx]
    by_bucket = {b: [] for b in buckets}
    for i, n in enumerate(lens):
        by_bucket[next(b for b in buckets if b >= n - 1)].append(i)
    batches = []
    for b, idx in by_bucket.items():
        rng.shuffle(idx)
        bs = max(1, batch_tokens // b)
        batches += [(b, idx[j:j + bs]) for j in range(0, len(idx), bs)]
    rng.shuffle(batches)
    return batches


def collate(arr, off, lens, bucket, idx, pad_id, device):
    x = np.full((len(idx), bucket), pad_id, dtype=np.int64)
    y = np.full((len(idx), bucket), -100, dtype=np.int64)
    for r, i in enumerate(idx):
        seq = np.asarray(arr[off[i]:off[i] + lens[i]], dtype=np.int64)
        x[r, :len(seq) - 1] = seq[:-1]
        y[r, :len(seq) - 1] = seq[1:]
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


@torch.no_grad()
def evaluate(model, arr, off, lens, ctx, pad_id, device, batch_tokens):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for bucket, idx in make_batches(lens, ctx, batch_tokens, random.Random(0)):
        x, y = collate(arr, off, lens, bucket, idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                               ignore_index=-100, reduction="sum")
        tot_loss += loss.item()
        tot_tok += int((y != -100).sum())
    model.train()
    return tot_loss / tot_tok, tot_loss, tot_tok


def lr_at(step, total, warmup, peak, floor_frac):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--batch-tokens", type=int, default=131072)
    ap.add_argument("--epochs", type=float, default=8.0, help="cosine horizon in epochs")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--lr-floor-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--evals-per-epoch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch._dynamo.config.cache_size_limit = 64

    prep = json.loads((args.data / "prep_report.json").read_text())
    pad_id = prep["pad"]
    tr_arr, tr_off, tr_lens = load_split(args.data, "train", args.ctx)
    va_arr, va_off, va_lens = load_split(args.data, "val", args.ctx)

    cfg = GPTConfig(vocab_size=prep["vocab_size"], n_layer=args.n_layer, n_head=args.n_head,
                    d_model=args.d_model, max_seq_len=args.ctx, dropout=args.dropout)
    torch.manual_seed(args.seed)
    model = GPT(cfg).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=device == "cuda")

    steps_per_epoch = len(make_batches(tr_lens, args.ctx, args.batch_tokens, random.Random(0)))
    total_steps = int(args.epochs * steps_per_epoch)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    state = {"step": 0, "epoch": 0, "pos_in_epoch": 0, "best_val": float("inf"),
             "best_step": -1, "history": []}
    rng = random.Random(args.seed)
    last = args.out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        rng.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())  # map_location moved the saved states to GPU
        if ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        epoch_batches = ck["epoch_batches"]
        print(f"resumed at step {state['step']} (epoch {state['epoch']}, "
              f"batch {state['pos_in_epoch']}/{len(epoch_batches)})", flush=True)
    else:
        epoch_batches = make_batches(tr_lens, args.ctx, args.batch_tokens, rng)
        write_json(args.out / "config.json", {"args": {k: str(v) if isinstance(v, Path) else v
                                                       for k, v in vars(args).items()},
                                              "model": cfg.to_dict(), "data_revision": prep["revision"],
                                              "n_params": model.n_params(False),
                                              "n_params_non_embedding": model.n_params(True),
                                              "steps_per_epoch": steps_per_epoch,
                                              "total_steps": total_steps})
    print(f"params {model.n_params(False)/1e6:.1f}M, {steps_per_epoch} steps/epoch, "
          f"{total_steps} total, eval every {eval_every}", flush=True)
    fwd = model if args.no_compile else torch.compile(model)

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                    "py_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
                    "epoch_batches": epoch_batches,
                    "config": cfg.to_dict()}, path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)

    t_log, tok_log, loss_acc, n_acc = time.time(), 0, 0.0, 0
    model.train()
    while state["step"] < total_steps:
        if state["pos_in_epoch"] >= len(epoch_batches):
            state["epoch"] += 1
            state["pos_in_epoch"] = 0
            epoch_batches = make_batches(tr_lens, args.ctx, args.batch_tokens, rng)
        bucket, idx = epoch_batches[state["pos_in_epoch"]]
        x, y = collate(tr_arr, tr_off, tr_lens, bucket, idx, pad_id, device)
        lr = lr_at(state["step"], total_steps, args.warmup, args.lr, args.lr_floor_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                               ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {state['step']}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        state["step"] += 1
        state["pos_in_epoch"] += 1
        ntok = int((y != -100).sum())
        tok_log += ntok
        loss_acc += loss.item() * ntok
        n_acc += ntok

        if state["step"] % 50 == 0:
            dt = time.time() - t_log
            rec = {"step": state["step"], "epoch": state["epoch"] + state["pos_in_epoch"] / len(epoch_batches),
                   "train_loss": loss_acc / n_acc, "lr": lr, "grad_norm": float(gnorm),
                   "tok_per_s": tok_log / dt, "time": time.time()}
            print(json.dumps(rec), flush=True)
            with open(args.out / "train_log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            write_json(args.out / "status.json", {"phase": "train", **rec, "total_steps": total_steps,
                                                  "best_val": state["best_val"],
                                                  "best_step": state["best_step"],
                                                  "eta_s": (total_steps - state["step"]) * dt / 50})
            t_log, tok_log, loss_acc, n_acc = time.time(), 0, 0.0, 0

        if state["step"] % eval_every == 0 or state["step"] == total_steps:
            val, _, _ = evaluate(model, va_arr, va_off, va_lens, args.ctx, pad_id, device,
                                 args.batch_tokens)
            ev = {"step": state["step"], "epoch": state["epoch"] + state["pos_in_epoch"] / len(epoch_batches),
                  "val_loss": val, "time": time.time()}
            state["history"].append(ev)
            if val < state["best_val"]:
                state["best_val"], state["best_step"] = val, state["step"]
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                            "step": state["step"], "val_loss": val}, args.out / "best.pt")
            save(last)
            print(json.dumps({"eval": ev, "best_val": state["best_val"]}), flush=True)
            with open(args.out / "eval_log.jsonl", "a") as f:
                f.write(json.dumps(ev) + "\n")

    write_json(args.out / "status.json", {"phase": "done", "step": state["step"],
                                          "best_val": state["best_val"],
                                          "best_step": state["best_step"], "time": time.time()})
    print("done", flush=True)


if __name__ == "__main__":
    main()
