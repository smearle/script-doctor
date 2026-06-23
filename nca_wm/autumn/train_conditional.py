#!/usr/bin/env python3
"""Train ONE conditional Autumn NCA world model over many games.

The model is conditioned on each game's program (tokenized by
`tokenize_program`) via a Perceiver rule-slot encoder, so a single set of
weights serves the whole training set and can be evaluated on *held-out
programs* — the multi-game generalization upgrade over per-game `AutumnNCA`.

Design mirrors the PuzzleScript conditional pipeline:
  * game-local color channels (each game's collected palette), padded to a
    common C across games; token `COLORi` aligned to channel `i`.
  * per-game partitioned datasets with balanced (uniform-over-games) sampling;
    each minibatch is single-game so the grid size is uniform within the batch.
  * periodic per-game eval on a val split, plus full eval on held-out games.

Usage:
    python -m nca_wm.autumn.train_conditional \
        --games gameOfLife,paint,lights,chomp,lock,coins,charge,disease,egg,magnets,gravity,grow \
        --heldout waterplug,wind --updates 20000 --save_dir nca_wm/autumn/runs/cond_v1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from nca_wm.autumn.model import encode_batch, ACTION_TYPES
from nca_wm.autumn.cond_model import ConditionalAutumnNCA
from nca_wm.autumn.tokenize_program import tokenize_program, PAD
from nca_wm.autumn.train import pick_device, _button_positions, _bucket, fmt_stats

DATA_DIR = "nca_wm/autumn/data"
TESTS = "/home/jupyter-smearle/mara/MARA/domains/autumnbench/Autumn.wasm/tests"


def load_game(name, max_seq_len):
    """Load a game's npz + aligned program tokens. Returns a dict or None."""
    npz = os.path.join(DATA_DIR, f"{name}.npz")
    prog = os.path.join(TESTS, f"{name}.sexp")
    if not (os.path.exists(npz) and os.path.exists(prog)):
        return None
    d = dict(np.load(npz, allow_pickle=True))
    palette = list(d["palette"])
    # Align token COLOR indices to the collected palette channel order.
    toks, info = tokenize_program(open(prog).read(), color_order=palette,
                                  max_len=max_seq_len)
    btn = _button_positions(d, d["states"], palette)
    return dict(
        name=name, states=d["states"], next_states=d["next_states"],
        action_type=d["action_type"], click_x=d["click_x"], click_y=d["click_y"],
        palette=palette, n_colors=len(palette), grid_size=int(d["grid_size"]),
        buttons=btn, tokens=toks, raw=d,
    )


def make_token_tensors(games, max_seq_len, device):
    """Per-game padded token id tensor (1,S) + bool mask (1,S), on device."""
    for g in games:
        t = g["tokens"][:max_seq_len]
        ids = np.full(max_seq_len, PAD, dtype=np.int64)
        ids[:len(t)] = t
        mask = np.zeros(max_seq_len, dtype=bool)
        mask[:len(t)] = True
        g["tok_ids"] = torch.as_tensor(ids, device=device)[None]      # (1,S)
        g["tok_mask"] = torch.as_tensor(mask, device=device)[None]    # (1,S)


@torch.no_grad()
def evaluate_game(model, g, idx, C, device, bs=256):
    """Per-action-bucket exact / changed-cell accuracy for one game."""
    model.eval()
    d = g["raw"]
    states, next_states = d["states"], d["next_states"]
    at, cx, cy = d["action_type"], d["click_x"], d["click_y"]
    buckets = _bucket(at, cx, cy, g["buttons"])
    stats = {}
    cells_ok = cells = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        onehot, atype_oh, click_map, _ = encode_batch(
            states[b], at[b], cx[b], cy[b], C, device)
        nb = len(b)
        logits = model(onehot, atype_oh, click_map,
                       g["tok_ids"], g["tok_mask"])  # (1,S) -> slots broadcast
        pred = logits.argmax(1).cpu().numpy()
        tgt, cur = next_states[b], states[b]
        cells_ok += int((pred == tgt).sum()); cells += pred.size
        for bk in np.unique(buckets[b]):
            sel = buckets[b] == bk
            p, t, c = pred[sel], tgt[sel], cur[sel]
            s = stats.setdefault(bk, dict(n=0, exact=0, ch_cells=0, ch_correct=0))
            s["n"] += int(sel.sum())
            s["exact"] += int((p.reshape(sel.sum(), -1) == t.reshape(sel.sum(), -1)).all(1).sum())
            ch = t != c
            s["ch_cells"] += int(ch.sum()); s["ch_correct"] += int(((p == t) & ch).sum())
    model.train()
    mean_exact = np.mean([stats[k]["exact"] / max(stats[k]["n"], 1) for k in stats]) if stats else 0.0
    ch_cells = sum(s["ch_cells"] for s in stats.values())
    ch_ok = sum(s["ch_correct"] for s in stats.values())
    ch_acc = ch_ok / ch_cells if ch_cells else float("nan")
    return dict(cell_acc=cells_ok / max(cells, 1), mean_exact=float(mean_exact),
                ch_acc=ch_acc, stats=stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="comma-separated train games")
    ap.add_argument("--heldout", default="", help="comma-separated held-out games")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min_lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--n_hid", type=int, default=128)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--copy_skip", type=float, default=5.0)
    ap.add_argument("--n_slots", type=int, default=16)
    ap.add_argument("--d_slot", type=int, default=128)
    ap.add_argument("--max_seq_len", type=int, default=1536)
    ap.add_argument("--no_global_pool", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train_names = [s for s in args.games.split(",") if s]
    held_names = [s for s in args.heldout.split(",") if s]
    train_games, held_games, missing = [], [], []
    for n in train_names:
        g = load_game(n, args.max_seq_len)
        (train_games if g else missing).append(g if g else n)
    for n in held_names:
        g = load_game(n, args.max_seq_len)
        (held_games if g else missing).append(g if g else n)
    if missing:
        print(f"WARNING missing (no npz/program): {missing}")
    assert train_games, "no trainable games loaded"

    all_games = train_games + held_games
    C = max(g["n_colors"] for g in all_games)
    make_token_tensors(all_games, args.max_seq_len, device)
    print(f"device={device} train={[g['name'] for g in train_games]} "
          f"heldout={[g['name'] for g in held_games]} C={C}")

    # Per-game train/val split.
    for g in all_games:
        N = len(g["states"])
        perm = rng.permutation(N)
        nv = int(N * args.val_frac)
        g["val_idx"], g["train_idx"] = perm[:nv], perm[nv:]

    model = ConditionalAutumnNCA(
        C, n_hid=args.n_hid, n_steps=args.n_steps, n_slots=args.n_slots,
        d_slot=args.d_slot, global_pool=not args.no_global_pool,
        copy_skip=args.copy_skip, max_seq_len=args.max_seq_len).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Warmup + cosine decay: stabilizes the early steps (lr=1e-3 diverged on
    # 8 games) and squeezes the last bit of in-distribution accuracy at the end.
    import math
    min_ratio = args.min_lr / args.lr

    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        t = (step - args.warmup) / max(args.updates - args.warmup, 1)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best = -1.0
    for step in range(1, args.updates + 1):
        g = train_games[rng.integers(len(train_games))]
        b = rng.choice(g["train_idx"], size=args.batch_size)
        onehot, atype_oh, click_map, _ = encode_batch(
            g["states"][b], g["action_type"][b], g["click_x"][b], g["click_y"][b],
            C, device)
        tgt = torch.as_tensor(g["next_states"][b], dtype=torch.long, device=device)
        logits = model(onehot, atype_oh, click_map,
                       g["tok_ids"], g["tok_mask"])  # (1,S) -> slots broadcast
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % args.eval_interval == 0 or step == 1:
            tr = {g["name"]: evaluate_game(model, g, g["val_idx"], C, device)
                  for g in train_games}
            ho = {g["name"]: evaluate_game(model, g, np.arange(len(g["states"])),
                                           C, device) for g in held_games}
            tr_me = np.mean([v["mean_exact"] for v in tr.values()])
            tr_ch = np.nanmean([v["ch_acc"] for v in tr.values()])
            worst = min(tr.items(), key=lambda kv: kv[1]["mean_exact"])
            msg = (f"[{step}] loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.1e} "
                   f"train_mean_exact={tr_me:.3f} train_ch_acc={tr_ch:.3f} "
                   f"worst={worst[0]}:{worst[1]['mean_exact']:.3f}")
            if ho:
                ho_me = np.mean([v["mean_exact"] for v in ho.values()])
                ho_ch = np.nanmean([v["ch_acc"] for v in ho.values()])
                msg += f" | HELDOUT mean_exact={ho_me:.3f} ch_acc={ho_ch:.3f}"
            print(msg)
            for nm, v in {**tr, **ho}.items():
                tag = "HO" if nm in ho else "  "
                print(f"   {tag} {nm}: exact={v['mean_exact']:.3f} "
                      f"ch_acc={v['ch_acc']:.3f} cell={v['cell_acc']:.4f}")
            score = tr_me if not ho else 0.5 * (tr_me + np.mean([v["mean_exact"] for v in ho.values()]))
            if score > best:
                best = score
                torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "model_final.pt"))
    cfg = dict(train_games=[g["name"] for g in train_games],
               heldout=[g["name"] for g in held_games], C=C, n_hid=args.n_hid,
               n_steps=args.n_steps, n_slots=args.n_slots, d_slot=args.d_slot,
               max_seq_len=args.max_seq_len, global_pool=not args.no_global_pool,
               palettes={g["name"]: g["palette"] for g in all_games})
    json.dump(cfg, open(os.path.join(args.save_dir, "config.json"), "w"), indent=2)
    print(f"saved -> {args.save_dir} (best score={best:.3f})")


if __name__ == "__main__":
    main()
