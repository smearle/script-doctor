#!/usr/bin/env python3
"""Train a single-game Autumn NCA world model on collected transitions.

Reports per-action-bucket metrics (noop / place / buttonNext / buttonReset) so
a high copy-accuracy cannot mask a failure to learn the actual dynamics.

Usage:
    python -m nca_wm.autumn.train --data nca_wm/autumn/data/gameOfLife.npz \
        --updates 8000 --save_dir nca_wm/autumn/runs/gameOfLife
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from nca_wm.autumn.model import AutumnNCA, encode_batch, ACTION_TYPES


def pick_device(device):
    """Resolve 'auto' to the CUDA device with the most free memory, else CPU."""
    if not torch.cuda.is_available():
        return "cpu"
    if device != "auto":
        return device
    free = []
    for i in range(torch.cuda.device_count()):
        f, _ = torch.cuda.mem_get_info(i)
        free.append((f, i))
    return f"cuda:{max(free)[1]}"


def _button_positions(data, states, palette):
    """Real buttonNext/buttonReset positions from the collector if available
    (game-agnostic); else fall back to the GoL green/silver color heuristic."""
    if "button_pos_json" in data:
        bp = json.loads(str(data["button_pos_json"]))
        return {k: tuple(v) for k, v in bp.items()}
    pal = list(palette)
    pos = {}
    if str(data.get("game", "")) == "gameOfLife":
        for name, key in (("green", "buttonNext"), ("silver", "buttonReset")):
            if name in pal:
                idx = pal.index(name)
                ys, xs = np.where(states[0] == idx)
                if len(xs):
                    pos[key] = (int(xs[0]), int(ys[0]))
    return pos


def _bucket(action_type, click_x, click_y, btn):
    """Per-transition action bucket label array (noop/place/arrows/button names)."""
    out = np.array([ACTION_TYPES[a] for a in action_type], dtype=object)
    out[action_type == 1] = "place"  # click that isn't a named button
    for key, (bx, by) in btn.items():
        sel = (action_type == 1) & (click_x == bx) & (click_y == by)
        out[sel] = key
    return out


@torch.no_grad()
def evaluate(model, data, idx, n_colors, device, btn, bs=512):
    model.eval()
    states, next_states = data["states"], data["next_states"]
    at, cx, cy = data["action_type"], data["click_x"], data["click_y"]
    prev = data.get("prev_states")
    buckets = _bucket(at, cx, cy, btn)
    stats = {}
    tot_cells_correct = tot_cells = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        hist = [prev[b]] if prev is not None else None
        onehot, atype_oh, click_map, hist_oh = encode_batch(
            states[b], at[b], cx[b], cy[b], n_colors, device, hist_states=hist)
        logits = model(onehot, atype_oh, click_map, hist_oh)
        pred = logits.argmax(1).cpu().numpy()  # (B,H,W)
        tgt = next_states[b]
        cur = states[b]
        tot_cells_correct += int((pred == tgt).sum())
        tot_cells += pred.size
        for bk in np.unique(buckets[b]):
            sel = buckets[b] == bk
            p, t, c = pred[sel], tgt[sel], cur[sel]
            d = stats.setdefault(bk, dict(n=0, exact=0, ch_cells=0, ch_correct=0))
            d["n"] += int(sel.sum())
            d["exact"] += int((p.reshape(sel.sum(), -1) == t.reshape(sel.sum(), -1)).all(1).sum())
            changed = t != c
            d["ch_cells"] += int(changed.sum())
            d["ch_correct"] += int(((p == t) & changed).sum())
    model.train()
    cell_acc = tot_cells_correct / max(tot_cells, 1)
    return cell_acc, stats


def fmt_stats(stats):
    rows = []
    for bk in sorted(stats):
        d = stats[bk]
        exact = d["exact"] / max(d["n"], 1)
        chacc = d["ch_correct"] / max(d["ch_cells"], 1) if d["ch_cells"] else float("nan")
        rows.append(f"{bk}: n={d['n']} exact={exact:.3f} chg_cell_acc={chacc:.3f}")
    return " | ".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--updates", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_hid", type=int, default=96)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--no_global_pool", action="store_true")
    ap.add_argument("--no_history", action="store_true",
                    help="ignore prev_states even if present in the data")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    data = dict(np.load(args.data, allow_pickle=True))
    palette = list(data["palette"])
    n_colors = len(palette)
    gs = int(data["grid_size"])
    btn = _button_positions(data, data["states"], palette)
    N = len(data["states"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = int(N * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"game={data['game']} N={N} train={len(train_idx)} val={len(val_idx)} "
          f"colors={palette} grid={gs} buttons={btn}")

    prev = data.get("prev_states")
    history = 1 if (prev is not None and not args.no_history) else 0
    model = AutumnNCA(n_colors, n_hid=args.n_hid, n_steps=args.n_steps,
                      global_pool=not args.no_global_pool, history=history).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,} history={history}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    states, next_states = data["states"], data["next_states"]
    at, cx, cy = data["action_type"], data["click_x"], data["click_y"]
    best = -1.0
    for step in range(1, args.updates + 1):
        b = rng.choice(train_idx, size=args.batch_size)
        hist = [prev[b]] if history else None
        onehot, atype_oh, click_map, hist_oh = encode_batch(
            states[b], at[b], cx[b], cy[b], n_colors, device, hist_states=hist)
        tgt = torch.as_tensor(next_states[b], dtype=torch.long, device=device)
        logits = model(onehot, atype_oh, click_map, hist_oh)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_interval == 0 or step == 1:
            cell_acc, stats = evaluate(model, data, val_idx, n_colors, device, btn)
            mean_exact = np.mean([stats[k]["exact"] / max(stats[k]["n"], 1) for k in stats])
            print(f"[{step}] loss={loss.item():.4f} val_cell_acc={cell_acc:.4f} "
                  f"val_mean_exact={mean_exact:.3f}")
            print(f"        {fmt_stats(stats)}")
            if mean_exact > best:
                best = mean_exact
                torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "model_final.pt"))
    cfg = dict(palette=palette, grid_size=gs, n_colors=n_colors, n_hid=args.n_hid,
               n_steps=args.n_steps, global_pool=not args.no_global_pool,
               history=history, buttons=btn, game=str(data["game"]))
    json.dump(cfg, open(os.path.join(args.save_dir, "config.json"), "w"), indent=2)
    print(f"saved -> {args.save_dir} (best val_mean_exact={best:.3f})")


if __name__ == "__main__":
    main()
