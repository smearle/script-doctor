#!/usr/bin/env python3
"""Per-object-CHANNEL representation (vs the discrete color grid).

`render_all` returns objects keyed by type, so a cell can hold several objects
(Mario *under* a coin). The color grid collapses that to one color (the coin),
making Mario "disappear". Here each object type is its own channel and a cell is
MULTI-HOT, so overlaps are preserved. Multi-hot => sigmoid + BCE (not softmax/CE).

Trains a Mario object-channel WM and verifies the overlap no longer hides Mario.

Usage:
    python -m nca_wm.autumn.objects --game mario --updates 4000
"""
import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from nca_wm.autumn.collect import AutumnGame, _suppress
from nca_wm.autumn.model import AutumnNCA, N_ATYPES, ATYPE_IDX, action_to_fields
from nca_wm.autumn.train import pick_device


def render_objects(env, vocab):
    """Multi-hot (C,H,W) uint8 over object-type channels; grows vocab in place."""
    d = json.loads(env.itp.render_all())
    gs = env.grid_size
    keys = [k for k in d if k != "GRID_SIZE"]
    for k in keys:
        if k not in vocab:
            vocab[k] = len(vocab)
    grid = np.zeros((max(len(vocab), 1), gs, gs), dtype=np.uint8)
    for k in keys:
        for c in d[k]:
            x, y = c["position"]["x"], c["position"]["y"]
            if 0 <= x < gs and 0 <= y < gs:
                grid[vocab[k], y, x] = 1
    return grid


def collect_objects(game, rollouts, length, seed, profile="agent", keep_prev=False):
    rng = np.random.default_rng(seed)
    env = AutumnGame(game, seed=seed)
    gs = env.grid_size
    vocab = {}
    arrow = [("up", -1, -1), ("down", -1, -1), ("left", -1, -1), ("right", -1, -1)]
    aw = {"up": 0.2, "down": 0.05, "left": 0.25, "right": 0.25}

    def act():
        if profile == "agent":
            menu = [("noop", -1, -1)] + arrow + [("click", int(rng.integers(gs)), int(rng.integers(gs)))]
            w = np.array([0.15] + [aw[a[0]] for a in arrow] + [0.20])
        else:
            menu = [("noop", -1, -1), ("click", int(rng.integers(gs)), int(rng.integers(gs)))]
            w = np.array([0.2, 0.8])
        return menu[rng.choice(len(menu), p=w / w.sum())]

    # first pass discovers the full vocab; we pad all states to final C afterwards
    seen = {}
    t0 = time.time()
    for r in range(rollouts):
        with _suppress():
            env.itp.run_script(env.prog, env._stdlib, "", int(rng.integers(1 << 30)))
        s = render_objects(env, vocab)
        prev = s
        for t in range(length):
            a = act()
            env.apply(a)
            ns = render_objects(env, vocab)
            key = (prev.tobytes(), s.tobytes(), a) if keep_prev else (s.tobytes(), a, s.shape[0])
            if key not in seen:
                ai, cx, cy = action_to_fields(a)
                seen[key] = (prev, s, ai, cx, cy, ns)
            prev = s; s = ns
    C = len(vocab)

    def pad(g):
        if g.shape[0] == C:
            return g
        out = np.zeros((C, g.shape[1], g.shape[2]), dtype=np.uint8)
        out[:g.shape[0]] = g
        return out

    prevs = np.stack([pad(v[0]) for v in seen.values()])
    states = np.stack([pad(v[1]) for v in seen.values()])
    ai = np.array([v[2] for v in seen.values()], dtype=np.uint8)
    cx = np.array([v[3] for v in seen.values()], dtype=np.int16)
    cy = np.array([v[4] for v in seen.values()], dtype=np.int16)
    nexts = np.stack([pad(v[5]) for v in seen.values()])
    inv = [None] * C
    for k, i in vocab.items():
        inv[i] = k
    overlap = int((states.sum(1) > 1).sum())
    print(f"[{game}/objects] {len(seen)} transitions in {time.time()-t0:.1f}s | "
          f"channels({C})={inv} | multi-object cells in data: {overlap}")
    out = dict(states=states, next_states=nexts, action_type=ai, click_x=cx,
               click_y=cy, vocab=np.array(inv), grid_size=np.int32(gs), game=np.str_(game))
    if keep_prev:
        out["prev_states"] = prevs
    return out


def collect_object_sequences(game, episodes, length, seed, profile="agent"):
    """Ordered multi-hot episodes for recurrent BPTT. states (E,L+1,C,H,W)."""
    rng = np.random.default_rng(seed)
    env = AutumnGame(game, seed=seed)
    gs = env.grid_size
    vocab = {}
    arrow = [("up", -1, -1), ("down", -1, -1), ("left", -1, -1), ("right", -1, -1)]
    aw = {"up": 0.2, "down": 0.05, "left": 0.25, "right": 0.25}

    def act():
        menu = [("noop", -1, -1)] + arrow + [("click", int(rng.integers(gs)), int(rng.integers(gs)))]
        w = np.array([0.15] + [aw[a[0]] for a in arrow] + [0.20])
        return menu[rng.choice(len(menu), p=w / w.sum())]

    all_s, all_a = [], []
    t0 = time.time()
    for e in range(episodes):
        with _suppress():
            env.itp.run_script(env.prog, env._stdlib, "", int(rng.integers(1 << 30)))
        st = [render_objects(env, vocab)]
        ac = []
        for t in range(length):
            a = act(); env.apply(a)
            st.append(render_objects(env, vocab)); ac.append(list(action_to_fields(a)))
        all_s.append(st); all_a.append(np.array(ac, dtype=np.int16))
    C = len(vocab)

    def pad(g):
        if g.shape[0] == C:
            return g
        o = np.zeros((C, g.shape[1], g.shape[2]), dtype=np.uint8); o[:g.shape[0]] = g; return o

    states = np.stack([np.stack([pad(g) for g in ep]) for ep in all_s]).astype(np.uint8)
    actions = np.stack(all_a)
    inv = [None] * C
    for k, i in vocab.items():
        inv[i] = k
    print(f"[{game}/obj-seq] {episodes}x{length} in {time.time()-t0:.1f}s | channels({C})={inv}")
    return dict(states=states, actions=actions, vocab=np.array(inv),
                grid_size=np.int32(gs), game=np.str_(game))


def encode(states, at, cx, cy, device, hist=None):
    """states (B,C,H,W) multi-hot float; returns (state, atype_oh, click_map, hist|None)."""
    s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
    B, C, H, W = s.shape
    a = torch.as_tensor(np.asarray(at), dtype=torch.long, device=device)
    atype_oh = F.one_hot(a, N_ATYPES).float()
    click = torch.zeros(B, 1, H, W, device=device)
    cxx = torch.as_tensor(np.asarray(cx), dtype=torch.long, device=device)
    cyy = torch.as_tensor(np.asarray(cy), dtype=torch.long, device=device)
    idx = torch.nonzero(a == ATYPE_IDX["click"], as_tuple=True)[0]
    if idx.numel():
        click[idx, 0, cyy[idx], cxx[idx]] = 1.0
    hist_t = None
    if hist is not None:
        hist_t = torch.as_tensor(np.asarray(hist), dtype=torch.float32, device=device)
    return s, atype_oh, click, hist_t


@torch.no_grad()
def evaluate(model, data, idx, device, bs=256):
    model.eval()
    s, ns = data["states"], data["next_states"]
    at, cx, cy = data["action_type"], data["click_x"], data["click_y"]
    prev = data.get("prev_states")
    cell_corr = cell_tot = ch_chg = ch_chg_corr = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        st, ao, cm, ht = encode(s[b], at[b], cx[b], cy[b], device,
                                hist=prev[b] if prev is not None else None)
        pred = (torch.sigmoid(model(st, ao, cm, ht)) > 0.5).float().cpu().numpy().astype(np.uint8)
        tgt = ns[b]
        cell_corr += int((pred == tgt).sum()); cell_tot += pred.size
        chg = tgt != s[b]
        ch_chg += int(chg.sum()); ch_chg_corr += int(((pred == tgt) & chg).sum())
    model.train()
    return cell_corr / cell_tot, ch_chg_corr / max(ch_chg, 1)


def run_recurrent(args, device):
    """Object channels + recurrent hidden grid -> tracks bullet counter too."""
    from nca_wm.autumn.model import RecurrentAutumnNCA
    save_dir = f"nca_wm/autumn/runs/{args.game}_objects_recurrent"
    os.makedirs(save_dir, exist_ok=True)
    d = collect_object_sequences(args.game, args.rollouts, min(args.length, 40), args.seed)
    s, acts = d["states"], d["actions"]
    E, Lp1, C, H, W = s.shape
    L = acts.shape[1]
    vocab = list(d["vocab"]); BUL = vocab.index("bullets") if "bullets" in vocab else -1
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(E); nv = max(1, int(0.1 * E)); val, train = perm[:nv], perm[nv:]
    model = RecurrentAutumnNCA(C, n_hid=args.n_hid, n_micro=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # per-channel positive weight (inverse frequency, capped) so ultra-sparse
    # channels like `bullets` aren't drowned out by always-0 BCE.
    pos = s[:, 1:].reshape(-1, C, H, W).mean((0, 2, 3))  # positive rate per channel
    pw = np.clip((1 - pos) / np.clip(pos, 1e-6, None), 1, 100).astype(np.float32)
    pos_weight = torch.tensor(pw, device=device)[None, None, :, None, None]  # (1,1,C,1,1)
    print(f"obj-recurrent channels={C} E={E} L={L} pos_weight={pw.round(1)}")

    def unroll(bs_states, bs_acts):
        h = model.init_hidden(len(bs_states), H, W, device)
        outs = []
        for t in range(L):
            st, ao, cm, _ = encode(bs_states[:, t], bs_acts[:, t, 0], bs_acts[:, t, 1],
                                   bs_acts[:, t, 2], device)
            logits, h = model.step(st, ao, cm, h)
            outs.append(logits)
        return torch.stack(outs, 1)  # (B,L,C,H,W)

    best = -1
    for step in range(1, args.updates + 1):
        b = rng.choice(train, size=8)
        logits = unroll(s[b], acts[b])
        tgt = torch.as_tensor(s[b, 1:], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                ft = hit = 0
                for i in range(0, len(val), 8):
                    vb = val[i:i + 8]
                    pr = (torch.sigmoid(unroll(s[vb], acts[vb])) > 0.5).cpu().numpy()
                    if BUL >= 0:
                        cur = (s[vb, :-1, BUL] == 1).sum((2, 3)); nxt = (s[vb, 1:, BUL] == 1).sum((2, 3))
                        pn = (pr[:, :, BUL] == 1).sum((2, 3)); sp = nxt > cur
                        ft += int(sp.sum()); hit += int(((pn > cur) & sp).sum())
                frec = hit / max(ft, 1)
            print(f"[{step}] loss={loss.item():.4f} fire_events={ft} bullet_recall={frec:.3f}")
            if frec > best:
                best = frec
                torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
    json.dump(dict(vocab=vocab, grid_size=int(d["grid_size"]), n_colors=C, n_hid=args.n_hid,
                   n_steps=args.n_steps, n_micro=3, global_pool=True, history=0, multihot=True,
                   recurrent=True, game=str(d["game"]), buttons={}),
              open(os.path.join(save_dir, "config.json"), "w"), indent=2)
    print(f"saved -> {save_dir} (best bullet_recall={best:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="mario")
    ap.add_argument("--rollouts", type=int, default=400)
    ap.add_argument("--length", type=int, default=120)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_hid", type=int, default=96)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--history", action="store_true",
                    help="add 1-frame history (object channels + history -> overlap AND enemy fixed)")
    ap.add_argument("--recurrent", action="store_true",
                    help="object channels + recurrent hidden grid (tracks bullet counter)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    if args.recurrent:
        run_recurrent(args, device); return
    tag = "_objects_hist" if args.history else "_objects"
    save_dir = f"nca_wm/autumn/runs/{args.game}{tag}"
    os.makedirs(save_dir, exist_ok=True)

    data = collect_objects(args.game, args.rollouts, args.length, args.seed, keep_prev=args.history)
    C = data["states"].shape[1]
    N = len(data["states"])
    prev = data.get("prev_states")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N); nval = int(0.1 * N)
    val, train = perm[:nval], perm[nval:]
    model = AutumnNCA(C, n_hid=args.n_hid, n_steps=args.n_steps,
                      history=1 if args.history else 0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    s, ns = data["states"], data["next_states"]
    at, cx, cy = data["action_type"], data["click_x"], data["click_y"]
    print(f"channels={C} N={N} history={args.history} params={sum(p.numel() for p in model.parameters()):,}")
    best = -1
    for step in range(1, args.updates + 1):
        b = rng.choice(train, size=args.batch_size)
        st, ao, cm, ht = encode(s[b], at[b], cx[b], cy[b], device,
                                hist=prev[b] if prev is not None else None)
        tgt = torch.as_tensor(ns[b], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(model(st, ao, cm, ht), tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == 1:
            acc, chg = evaluate(model, data, val, device)
            print(f"[{step}] loss={loss.item():.4f} cell_acc={acc:.4f} changed_acc={chg:.3f}")
            if chg > best:
                best = chg
                torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
    json.dump(dict(vocab=list(data["vocab"]), grid_size=int(data["grid_size"]), n_colors=C,
                   n_hid=args.n_hid, n_steps=args.n_steps, global_pool=True,
                   history=1 if args.history else 0,
                   multihot=True, game=str(data["game"]), buttons={}),
              open(os.path.join(save_dir, "config.json"), "w"), indent=2)
    print(f"saved -> {save_dir} (best changed_acc={best:.3f})")


if __name__ == "__main__":
    main()
