#!/usr/bin/env python3
"""Train a RecurrentAutumnNCA over ordered episodes with BPTT.

The hidden grid persists across env steps so the model can integrate
unobservable episode-long state (e.g. Mario's bullet counter). Teacher-forced
on the observed frame each step; h reset to zeros at episode start. Reports a
firing-specific metric: on steps where a bullet truly spawns, does the model
predict the spawn?

Usage:
    python -m nca_wm.autumn.train_recurrent --data nca_wm/autumn/data/mario_seq.npz \
        --save_dir nca_wm/autumn/runs/mario_recurrent --updates 4000
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from nca_wm.autumn.model import RecurrentAutumnNCA, _onehot, N_ATYPES, ATYPE_IDX
from nca_wm.autumn.train import pick_device


def encode_step(states_t, actions_t, n_colors, device):
    """states_t (B,H,W) uint8; actions_t (B,3) [atype,cx,cy]."""
    onehot = _onehot(states_t, n_colors, device)
    B, _, H, W = onehot.shape
    at = torch.as_tensor(actions_t[:, 0], dtype=torch.long, device=device)
    atype_oh = F.one_hot(at, N_ATYPES).float()
    click_map = torch.zeros(B, 1, H, W, device=device)
    cx = torch.as_tensor(actions_t[:, 1], dtype=torch.long, device=device)
    cy = torch.as_tensor(actions_t[:, 2], dtype=torch.long, device=device)
    isc = at == ATYPE_IDX["click"]
    idx = torch.nonzero(isc, as_tuple=True)[0]
    if idx.numel():
        click_map[idx, 0, cy[idx], cx[idx]] = 1.0
    return onehot, atype_oh, click_map


def unroll(model, states, actions, n_colors, device):
    """states (B,L+1,H,W), actions (B,L,3). Returns stacked logits (B,L,C,H,W)."""
    B, Lp1, H, W = states.shape
    L = actions.shape[1]
    h = model.init_hidden(B, H, W, device)
    logits_seq = []
    for t in range(L):
        oh, at_oh, cm = encode_step(states[:, t], actions[:, t], n_colors, device)
        logits, h = model.step(oh, at_oh, cm, h)
        logits_seq.append(logits)
    return torch.stack(logits_seq, dim=1)  # (B,L,C,H,W)


def _encode_t(state_long, actions_t, n_colors, device):
    """Like encode_step but the state is already a (B,H,W) long tensor on device."""
    oh = F.one_hot(state_long, n_colors).permute(0, 3, 1, 2).float()
    B, _, H, W = oh.shape
    at = torch.as_tensor(actions_t[:, 0], dtype=torch.long, device=device)
    atype_oh = F.one_hot(at, N_ATYPES).float()
    cm = torch.zeros(B, 1, H, W, device=device)
    cx = torch.as_tensor(actions_t[:, 1], dtype=torch.long, device=device)
    cy = torch.as_tensor(actions_t[:, 2], dtype=torch.long, device=device)
    idx = torch.nonzero(at == ATYPE_IDX["click"], as_tuple=True)[0]
    if idx.numel():
        cm[idx, 0, cy[idx], cx[idx]] = 1.0
    return oh, atype_oh, cm


def unroll_ss(model, states, actions, n_colors, device, p):
    """Scheduled-sampling unroll: with prob p (per step, t>0) feed the model its OWN
    previous prediction instead of the ground-truth frame, so it learns to recover from
    its own errors (fixes autoregressive accumulation + cold-start phase lock-in; loss is
    still vs the ground-truth next frame). See DIVERGENCES.md."""
    B, Lp1, H, W = states.shape
    L = actions.shape[1]
    gt = torch.as_tensor(states, dtype=torch.long, device=device)
    h = model.init_hidden(B, H, W, device)
    logits_seq = []
    prev_pred = None
    for t in range(L):
        if t == 0 or prev_pred is None or float(torch.rand(())) >= p:
            in_state = gt[:, t]
        else:
            in_state = prev_pred           # model's own prediction (detached)
        oh, at_oh, cm = _encode_t(in_state, actions[:, t], n_colors, device)
        logits, h = model.step(oh, at_oh, cm, h)
        prev_pred = logits.argmax(1).detach()
        logits_seq.append(logits)
    return torch.stack(logits_seq, dim=1)


@torch.no_grad()
def evaluate(model, states, actions, n_colors, device, purple_idx, bs=16):
    model.eval()
    tot_c = tot = 0
    ch_cells = ch_correct = 0
    fire_true = fire_hit = 0
    for i in range(0, len(states), bs):
        s = states[i:i + bs]; a = actions[i:i + bs]
        logits = unroll(model, s, a, n_colors, device)
        pred = logits.argmax(2).cpu().numpy()        # (B,L,H,W)
        cur = s[:, :-1]; tgt = s[:, 1:]              # (B,L,H,W)
        tot_c += int((pred == tgt).sum()); tot += pred.size
        # changed-cell accuracy: cells that differ from the current frame
        changed = tgt != cur
        ch_cells += int(changed.sum())
        ch_correct += int(((pred == tgt) & changed).sum())
        if purple_idx >= 0:
            cur_p = (cur == purple_idx).sum(axis=(2, 3))
            nxt_p = (tgt == purple_idx).sum(axis=(2, 3))
            pred_p = (pred == purple_idx).sum(axis=(2, 3))
            spawn = nxt_p > cur_p
            fire_true += int(spawn.sum())
            fire_hit += int(((pred_p > cur_p) & spawn).sum())
    model.train()
    fire_recall = fire_hit / max(fire_true, 1)
    ch_acc = ch_correct / max(ch_cells, 1)
    return tot_c / tot, ch_acc, fire_true, fire_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_hid", type=int, default=96)
    ap.add_argument("--n_micro", type=int, default=4)
    ap.add_argument("--pool", default="meanmax", choices=["none", "mean", "max", "meanmax"],
                    help="grid-wide reduction; max/meanmax preserve sparse global signals "
                         "(e.g. a corner button press) that mean dilutes. meanmax default: fixes "
                         "spatial-locality mode-latch (waterplug, sand) at no cost on globally-pooled games")
    ap.add_argument("--sched_samp", type=float, default=0.0,
                    help="max scheduled-sampling prob (ramped 0->this over training): feed the "
                         "model its own predictions during the unroll so it learns to recover from "
                         "its own errors (reduces autoregressive divergence; see DIVERGENCES.md)")
    ap.add_argument("--seq_len", type=int, default=0,
                    help="truncate episodes to this many steps for cheaper BPTT (0=full)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    d = dict(np.load(args.data, allow_pickle=True))
    palette = list(d["palette"]); n_colors = len(palette)
    gs = int(d["grid_size"])
    states, actions = d["states"], d["actions"]
    if args.seq_len and args.seq_len < states.shape[1] - 1:
        states = states[:, :args.seq_len + 1]
        actions = actions[:, :args.seq_len]
    E, Lp1, H, W = states.shape
    purple = palette.index("mediumpurple") if "mediumpurple" in palette else -1
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(E)
    n_val = max(1, int(E * args.val_frac))
    val, train = perm[:n_val], perm[n_val:]
    print(f"game={d['game']} episodes={E} L={Lp1-1} train={len(train)} val={len(val)} "
          f"colors={palette} purple_idx={purple}")

    model = RecurrentAutumnNCA(n_colors, n_hid=args.n_hid, n_micro=args.n_micro,
                               pool=args.pool).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = -1.0
    for step in range(1, args.updates + 1):
        b = rng.choice(train, size=args.batch_size)
        if args.sched_samp > 0:
            p = args.sched_samp * min(1.0, step / max(1, args.updates // 2))  # ramp 0->max over first half
            logits = unroll_ss(model, states[b], actions[b], n_colors, device, p)
        else:
            logits = unroll(model, states[b], actions[b], n_colors, device)
        tgt = torch.as_tensor(states[b, 1:], dtype=torch.long, device=device)  # (B,L,H,W)
        loss = F.cross_entropy(logits.reshape(-1, n_colors, H, W),
                               tgt.reshape(-1, H, W))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_interval == 0 or step == 1:
            acc, ch_acc, ft, frec = evaluate(model, states[val], actions[val], n_colors, device, purple)
            print(f"[{step}] loss={loss.item():.4f} val_cell_acc={acc:.4f} "
                  f"changed_cell_acc={ch_acc:.3f} fire_events={ft} fire_recall={frec:.3f}")
            # select best by changed-cell acc AND fire_recall combined (not fire_recall alone:
            # that picks a checkpoint that fires well but may have regressed a dense mechanic
            # like coins' agent-behind-coin reappearance, which lives in changed-cell acc)
            score = (ch_acc + frec) if ft > 0 else ch_acc
            if score > best:
                best = score
                torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "model_final.pt"))
    cfg = dict(palette=palette, grid_size=gs, n_colors=n_colors, n_hid=args.n_hid,
               n_micro=args.n_micro, pool=args.pool, recurrent=True, buttons={}, game=str(d["game"]))
    json.dump(cfg, open(os.path.join(args.save_dir, "config.json"), "w"), indent=2)
    print(f"saved -> {args.save_dir} (best fire_recall={best:.3f})")


if __name__ == "__main__":
    main()
