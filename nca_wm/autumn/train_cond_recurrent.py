#!/usr/bin/env python3
"""Train ONE conditional RECURRENT Autumn NCA over many games (BPTT).

Combines program-slot conditioning with a recurrent hidden grid carried across
env steps, so a single model can perfectly fit even the hidden-state games
(mario bullet counter, paint currColor, sand clickType, snake direction,
pacman modes) that cap single-frame accuracy.

Data: `*_seq.npz` (ordered episodes; states (E,L+1,H,W), actions (E,L,3)).
Per-game palette -> game-local color channels padded to common C; token COLORi
aligned to channel i. Balanced (uniform-over-games) episode sampling. Reports,
per game: teacher-forced cell / changed-cell / exact-per-step, and an
AUTOREGRESSIVE changed-cell accuracy (feed the model's own prediction) — the
strict test of whether the rollout stays faithful.

Usage:
    python -m nca_wm.autumn.train_cond_recurrent \
        --games mario,sand,paint,snake,pacman,charge,disease,gravity,waterplug \
        --updates 12000 --save_dir nca_wm/autumn/runs/cond_rec_v1
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from nca_wm.autumn.model import _onehot, N_ATYPES, ATYPE_IDX
from nca_wm.autumn.cond_recurrent_model import ConditionalRecurrentAutumnNCA
from nca_wm.autumn.tokenize_program import tokenize_program, PAD
from nca_wm.autumn.train import pick_device

DATA_DIR = "nca_wm/autumn/data"
TESTS = "/home/jupyter-smearle/mara/MARA/domains/autumnbench/Autumn.wasm/tests"
# Prefer a balanced sequence file when one exists (rare-event coverage).
SEQ_OVERRIDE = {"waterplug": "waterplug_balanced.npz"}


def seq_file(game):
    f = SEQ_OVERRIDE.get(game, f"{game}_seq.npz")
    p = os.path.join(DATA_DIR, f)
    return p if os.path.exists(p) else None


def load_game(name, max_seq_len, seq_len):
    npz = seq_file(name)
    prog = os.path.join(TESTS, f"{name}.sexp")
    if not (npz and os.path.exists(prog)):
        return None
    d = dict(np.load(npz, allow_pickle=True))
    palette = list(d["palette"])
    states, actions = d["states"], d["actions"]
    if seq_len and seq_len < states.shape[1] - 1:
        states, actions = states[:, :seq_len + 1], actions[:, :seq_len]
    toks, _ = tokenize_program(open(prog).read(), color_order=palette,
                               max_len=max_seq_len)
    return dict(name=name, states=states, actions=actions, palette=palette,
                n_colors=len(palette), grid_size=int(d["grid_size"]), tokens=toks)


def encode_step(states_t, actions_t, C, device):
    onehot = _onehot(states_t, C, device)
    B, _, H, W = onehot.shape
    at = torch.as_tensor(actions_t[:, 0], dtype=torch.long, device=device)
    atype_oh = F.one_hot(at, N_ATYPES).float()
    click_map = torch.zeros(B, 1, H, W, device=device)
    cx = torch.as_tensor(actions_t[:, 1], dtype=torch.long, device=device)
    cy = torch.as_tensor(actions_t[:, 2], dtype=torch.long, device=device)
    idx = torch.nonzero(at == ATYPE_IDX["click"], as_tuple=True)[0]
    if idx.numel():
        click_map[idx, 0, cy[idx], cx[idx]] = 1.0
    return onehot, atype_oh, click_map


def unroll(model, states, actions, slots, C, device):
    """Teacher-forced unroll. states (B,L+1,H,W) np, actions (B,L,3) np.
    Returns logits (B,L,C,H,W)."""
    B, Lp1, H, W = states.shape
    L = actions.shape[1]
    h = model.init_hidden(B, H, W, device)
    out = []
    for t in range(L):
        oh, at_oh, cm = encode_step(states[:, t], actions[:, t], C, device)
        logits, h = model.step(oh, at_oh, cm, h, slots)
        out.append(logits)
    return torch.stack(out, dim=1)


@torch.no_grad()
def evaluate_game(model, g, idx, C, device, bs=16):
    model.eval()
    states, actions = g["states"], g["actions"]
    slots1 = model.encode(g["tok_ids"], g["tok_mask"])           # (1,K,d)
    tf_cell = tf_tot = tf_ch = tf_ch_ok = tf_exact = tf_steps = 0
    ar_ch = ar_ch_ok = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        s, a = states[b], actions[b]
        B, Lp1, H, W = s.shape
        L = a.shape[1]
        slots = slots1.expand(B, -1, -1)
        # teacher-forced
        logits = unroll(model, s, a, slots, C, device)
        pred = logits.argmax(2).cpu().numpy()                     # (B,L,H,W)
        cur, tgt = s[:, :-1], s[:, 1:]
        tf_cell += int((pred == tgt).sum()); tf_tot += pred.size
        ch = tgt != cur
        tf_ch += int(ch.sum()); tf_ch_ok += int(((pred == tgt) & ch).sum())
        tf_exact += int((pred == tgt).reshape(B, L, -1).all(2).sum()); tf_steps += B * L
        # autoregressive: feed own prediction
        h = model.init_hidden(B, H, W, device)
        cur_frame = s[:, 0]
        for t in range(L):
            oh, at_oh, cm = encode_step(cur_frame, a[:, t], C, device)
            lg, h = model.step(oh, at_oh, cm, h, slots)
            p = lg.argmax(1).cpu().numpy()                        # (B,H,W)
            tg = s[:, t + 1]; cf = cur_frame
            chm = tg != cf
            ar_ch += int(chm.sum()); ar_ch_ok += int(((p == tg) & chm).sum())
            cur_frame = p
    model.train()
    return dict(
        tf_cell=tf_cell / max(tf_tot, 1),
        tf_ch=tf_ch_ok / max(tf_ch, 1),
        tf_exact=tf_exact / max(tf_steps, 1),
        ar_ch=ar_ch_ok / max(ar_ch, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--updates", type=int, default=12000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min_lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--n_hid", type=int, default=128)
    ap.add_argument("--n_micro", type=int, default=6)
    ap.add_argument("--n_slots", type=int, default=24)
    ap.add_argument("--d_slot", type=int, default=160)
    ap.add_argument("--copy_skip", type=float, default=2.5)
    ap.add_argument("--pool", default="meanmax")
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    games, missing = [], []
    for n in [s for s in args.games.split(",") if s]:
        g = load_game(n, args.max_seq_len, args.seq_len)
        (games if g else missing).append(g if g else n)
    if missing:
        print(f"WARNING no seq/program for: {missing}")
    assert games, "no games loaded"
    C = max(g["n_colors"] for g in games)
    for g in games:
        t = g["tokens"][:args.max_seq_len]
        ids = np.full(args.max_seq_len, PAD, dtype=np.int64); ids[:len(t)] = t
        m = np.zeros(args.max_seq_len, dtype=bool); m[:len(t)] = True
        g["tok_ids"] = torch.as_tensor(ids, device=device)[None]
        g["tok_mask"] = torch.as_tensor(m, device=device)[None]
        E = len(g["states"]); perm = rng.permutation(E)
        nv = max(1, int(E * args.val_frac))
        g["val_idx"], g["train_idx"] = perm[:nv], perm[nv:]
    print(f"device={device} games={[g['name'] for g in games]} C={C} "
          f"seq_len={args.seq_len}")

    model = ConditionalRecurrentAutumnNCA(
        C, n_hid=args.n_hid, n_micro=args.n_micro, n_slots=args.n_slots,
        d_slot=args.d_slot, pool=args.pool, copy_skip=args.copy_skip,
        max_seq_len=args.max_seq_len).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    min_ratio = args.min_lr / args.lr

    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        t = (step - args.warmup) / max(args.updates - args.warmup, 1)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best = -1.0
    for step in range(1, args.updates + 1):
        g = games[rng.integers(len(games))]
        b = rng.choice(g["train_idx"], size=min(args.batch_size, len(g["train_idx"])))
        s = g["states"][b]; a = g["actions"][b]
        B, _, H, W = s.shape
        slots = model.encode(g["tok_ids"], g["tok_mask"]).expand(B, -1, -1)
        logits = unroll(model, s, a, slots, C, device)
        L = a.shape[1]
        tgt = torch.as_tensor(s[:, 1:], dtype=torch.long, device=device)
        loss = F.cross_entropy(logits.reshape(-1, C, H, W), tgt.reshape(-1, H, W))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % args.eval_interval == 0 or step == 1:
            ev = {g["name"]: evaluate_game(model, g, g["val_idx"], C, device)
                  for g in games}
            tf = np.mean([v["tf_ch"] for v in ev.values()])
            ar = np.mean([v["ar_ch"] for v in ev.values()])
            worst = min(ev.items(), key=lambda kv: kv[1]["ar_ch"])
            print(f"[{step}] loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.1e} "
                  f"tf_ch={tf:.3f} ar_ch={ar:.3f} worst_ar={worst[0]}:{worst[1]['ar_ch']:.3f}")
            for nm, v in ev.items():
                print(f"   {nm}: tf_ch={v['tf_ch']:.3f} ar_ch={v['ar_ch']:.3f} "
                      f"tf_exact={v['tf_exact']:.3f} cell={v['tf_cell']:.4f}")
            score = 0.5 * (tf + ar)
            if score > best:
                best = score
                torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "model_final.pt"))
    cfg = dict(games=[g["name"] for g in games], C=C, n_hid=args.n_hid,
               n_micro=args.n_micro, n_slots=args.n_slots, d_slot=args.d_slot,
               pool=args.pool, copy_skip=args.copy_skip, recurrent=True,
               conditional=True, seq_len=args.seq_len,
               palettes={g["name"]: g["palette"] for g in games})
    json.dump(cfg, open(os.path.join(args.save_dir, "config.json"), "w"), indent=2)
    print(f"saved -> {args.save_dir} (best={best:.3f})")


if __name__ == "__main__":
    main()
