#!/usr/bin/env python3
"""Evaluate a game's best model on the REAL human action distribution.

Replays human action sequences through the engine (human_replay) to get ground-truth
trajectories, then scores the model two ways:
  * teacher-forced  -- feed the real state each step (carry hidden state for recurrent),
                       compare the one-step prediction to the real next frame. This is the
                       faithful "given exactly what the human saw + did, is the WM right?"
                       blindspot probe -- no exposure-bias drift, just genuine error.
  * autoregressive  -- feed the model its own prediction; record the first step it diverges
                       from the human trajectory (what a person sees in the viewer).

Per (game) reports changed-cell accuracy (TF) and AR first-divergence distribution.
"""
import argparse, json, os
import numpy as np
from nca_wm.autumn import infer as I
from nca_wm.autumn.human_replay import collect_human


def step_wm(model, cfg, wm, prev, h, a, rec, mh):
    if mh:   nxt, h = I.object_step(model, cfg, wm, a, prev, h); return nxt, h
    if rec:  nxt, h = I.recurrent_step(model, cfg, wm, a, h);    return nxt, h
    nxt = I.wm_step(model, cfg, wm, a, prev_board=prev);         return nxt, None


def eval_game(game, device="cpu", max_episodes=0, ar_max=60, seq_cap=120):
    best = json.load(open(os.path.join(os.path.dirname(__file__), "best.json")))
    run = best.get(game)
    if not run:
        return None
    model, cfg = I.load_run(f"nca_wm/autumn/runs/{run}", device=device)
    rec, mh = cfg.get("recurrent"), cfg.get("multihot")
    palette = list(cfg["palette"])
    eps, meta = collect_human(game, palette, max_episodes)
    if not eps:
        return dict(game=game, run=run, episodes=0)

    tf_cells = tf_correct = tf_changed = tf_ch_correct = 0
    ar_first = []          # first-divergence step per episode (None -> survived)
    ar_steps_total = 0
    for S, A in eps:
        L = min(A.shape[0], seq_cap) if seq_cap else A.shape[0]
        acts = [tuple(int(x) for x in a) for a in A]
        acts = [("click", a[1], a[2]) if a[0] == 1 else (["noop","click","up","down","left","right"][a[0]],) for a in acts]
        # ---- teacher forced ----
        h = None; prev = S[0]
        for t in range(L):
            pred, h = step_wm(model, cfg, S[t], prev, h, acts[t], rec, mh)
            tgt = S[t + 1]; cur = S[t]
            tf_cells += tgt.size; tf_correct += int((pred == tgt).sum())
            ch = tgt != cur
            tf_changed += int(ch.sum()); tf_ch_correct += int(((pred == tgt) & ch).sum())
            prev = S[t]
        # ---- autoregressive (cap length for cost) ----
        Lar = min(L, ar_max)
        h = None; wm = S[0].copy(); prev = S[0].copy(); first = None
        for t in range(Lar):
            wm2, h = step_wm(model, cfg, wm, prev, h, acts[t], rec, mh)
            prev = wm; wm = wm2
            if first is None and int((wm != S[t + 1]).sum()) > 0:
                first = t
        ar_first.append(first); ar_steps_total += Lar
    tf_acc = tf_correct / max(tf_cells, 1)
    tf_ch = tf_ch_correct / max(tf_changed, 1)
    diverged = [f for f in ar_first if f is not None]
    survived = sum(1 for f in ar_first if f is None)
    med_div = int(np.median(diverged)) if diverged else -1
    return dict(game=game, run=run, episodes=len(eps), steps=meta["n_steps"],
                changing_frac=meta["n_changing"] / max(meta["n_steps"], 1),
                tf_cell_acc=tf_acc, tf_changed_acc=tf_ch,
                ar_diverged=len(diverged), ar_survived=survived, ar_median_div_step=med_div,
                oop=meta["n_oop"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="comma list")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--seq_cap", type=int, default=120,
                    help="cap each human episode to this many steps (long idle-noop tails add cost, not signal)")
    args = ap.parse_args()
    print(f"{'game':16s} {'eps':>4s} {'TFchg%':>7s} {'TFcell%':>8s} {'ARdiv':>6s} {'ARsurv':>7s} {'medDiv':>7s} {'oop':>6s}")
    for g in args.games.split(","):
        g = g.strip()
        try:
            r = eval_game(g, args.device, args.max_episodes, seq_cap=args.seq_cap)
        except Exception as e:
            print(f"{g:16s} ERROR {e}"); continue
        if r is None:
            print(f"{g:16s} (no best run)"); continue
        if r.get("episodes", 0) == 0:
            print(f"{g:16s} (no human episodes)"); continue
        print(f"{g:16s} {r['episodes']:4d} {100*r['tf_changed_acc']:7.2f} {100*r['tf_cell_acc']:8.3f} "
              f"{r['ar_diverged']:6d} {r['ar_survived']:7d} {r['ar_median_div_step']:7d} {r['oop']:6d}")


if __name__ == "__main__":
    main()
