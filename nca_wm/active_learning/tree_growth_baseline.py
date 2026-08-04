"""Offline-collection baseline for tree_growth.py, matched on env-step budget.

Replicates the OLD heroes data regime — solver-based exhaustive collection
(C++ BFS/A* via ``data_collection.collect_unique_transitions``) — but trains
the IDENTICAL ``NCAWorldModel`` with the identical eval as tree_growth.py, so
the only variable is HOW the transitions were chosen:

  * tree_growth:  online, WM-surprisal-prioritized frontier growth
                  (each engine step spent where the model is most surprised)
  * this script:  offline uniform search dump, uniformly subsampled down to
                  the SAME number of transitions the active run spent
                  (``--budget``, read from the active run's metrics.json)

Same levels, same heldout builder + seed (byte-identical heldout set), same
model/optimizer hyperparameters, uniform replay (the standard offline regime).

    .venv/bin/python -m nca_wm.active_learning.tree_growth_baseline \
        --game heroes_of_sokoban --levels 0-13 --budget 40000 \
        --algo bfs --out /tmp/tg_baseline
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm.active_learning.tree_growth import (
    WM, build_heldout, compile_game, enabled_actions, new_engine,
    parse_levels)


def collect_level(json_str, game, li, algo, max_iters, timeout_ms, keep, rng):
    """C++ exhaustive search dump for one level → (S, A, T) uint8 multihot
    arrays at the level's native (n_obj, H, W), subsampled to ``keep`` rows
    BEFORE unpacking (the full float32 union of all dumps is ~100 GB and
    OOMs; packed-row subsampling keeps peak memory per level tiny)."""
    from nca_wm.data_collection import collect_unique_transitions
    d = collect_unique_transitions(json_str, game, level_i=li,
                                   max_iters=max_iters, timeout_ms=timeout_ms,
                                   search_algo=algo)
    W = int(d["W"])
    n = len(d["states"])
    idx = (np.arange(n) if n <= keep
           else np.sort(rng.choice(n, size=keep, replace=False)))
    S = np.unpackbits(d["states"][idx], axis=-1)[..., :W]
    T = np.unpackbits(d["next_states"][idx], axis=-1)[..., :W]
    A = np.asarray(d["actions"], np.int32)[idx]
    return S, A, T, n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="heroes_of_sokoban")
    p.add_argument("--game_json", default=None)
    p.add_argument("--levels", default="0-13")
    p.add_argument("--budget", type=int, required=True,
                   help="total transitions to keep (== active run env_steps)")
    p.add_argument("--algo", default="bfs", choices=["bfs", "astar"])
    p.add_argument("--max_iters", type=int, default=100_000)
    p.add_argument("--timeout_ms", type=int, default=120_000)
    p.add_argument("--n_updates", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_hid", type=int, default=128)
    p.add_argument("--n_nca_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--heldout_levels", type=int, default=8)
    p.add_argument("--heldout_eps", type=int, default=4)
    p.add_argument("--heldout_len", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/tmp/tg_baseline")
    args = p.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.game_json:
        json_str = Path(args.game_json).read_text()
    else:
        json_str = compile_game(args.game)
    acts = enabled_actions(json_str)
    probe = new_engine(json_str, 0)
    n_obj = probe.get_object_count()
    n_levels = probe.get_num_levels()
    levels = parse_levels(args.levels, n_levels)
    hl = [li for li in range(n_levels) if li not in levels][:args.heldout_levels]
    hp = wp = 0
    for li in levels + hl:
        e = new_engine(json_str, li)
        hp = max(hp, e.get_height()); wp = max(wp, e.get_width())
    print(f"[game] {args.game}  train levels {levels}  n_obj={n_obj} "
          f"pad=({hp},{wp})  heldout {hl}  budget={args.budget}", flush=True)

    # ---- offline collection (the old regime), capped to budget ----
    # Two passes: sizes first (cache hits, cheap), then proportional
    # per-level subsampling in PACKED form so peak memory stays bounded.
    # Data is kept per-level as uint8 and batches gather + cast on the fly.
    from nca_wm.data_collection import collect_unique_transitions
    sizes = []
    for li in levels:
        d = collect_unique_transitions(json_str, args.game, level_i=li,
                                       max_iters=args.max_iters,
                                       timeout_ms=args.timeout_ms,
                                       search_algo=args.algo)
        sizes.append(len(d["states"]))
        del d
    total = sum(sizes)
    frac = min(1.0, args.budget / max(total, 1))
    Ss, As, Ts, Ms = [], [], [], []
    for li, n in zip(levels, sizes):
        keep = max(1, int(round(n * frac)))
        S, A, T, n0 = collect_level(json_str, args.game, li, args.algo,
                                    args.max_iters, args.timeout_ms, keep, rng)
        C, H, Wd = S.shape[1], S.shape[2], S.shape[3]
        Sp = np.zeros((len(S), n_obj, hp, wp), np.uint8)
        Tp = np.zeros_like(Sp)
        Sp[:, :C, :H, :Wd] = S; Tp[:, :C, :H, :Wd] = T
        m = np.zeros((n_obj, hp, wp), np.float32)
        m[:, :H, :Wd] = 1.0
        Ss.append(Sp); As.append(A); Ts.append(Tp); Ms.append(m)
        print(f"  L{li}: kept {len(S)}/{n0} transitions ({args.algo})",
              flush=True)
    counts = np.asarray([len(s) for s in Ss], np.float64)
    lvl_p = counts / counts.sum()
    n_data = int(counts.sum())
    print(f"[data] {n_data} transitions after budget cap "
          f"(budget {args.budget}, dumps total {total})", flush=True)

    def sample_batch(bs):
        gis = rng.choice(len(Ss), size=bs, p=lvl_p)
        S = np.empty((bs, n_obj, hp, wp), np.float32)
        T = np.empty_like(S)
        A = np.empty(bs, np.int32)
        M = np.empty((bs, n_obj, hp, wp), np.float32)
        for j, gi in enumerate(gis):
            i = rng.integers(0, len(Ss[gi]))
            S[j] = Ss[gi][i]; T[j] = Ts[gi][i]
            A[j] = As[gi][i]; M[j] = Ms[gi]
        return S, A, T, M

    wm = WM(n_obj, hp, wp, args.n_hid, args.n_nca_steps, args.seed, args.lr)
    heldout = build_heldout(json_str, hl, n_obj, hp, wp, acts,
                            args.heldout_eps, args.heldout_len, args.seed + 1)
    print(f"[heldout] {len(heldout[0])} transitions from {len(hl)} levels",
          flush=True)

    metrics = []
    t0 = time.time()
    loss_ema = None
    for u in range(1, args.n_updates + 1):
        S, A, T, M = sample_batch(args.batch_size)
        loss, _ = wm.train(S, A, T, M, np.ones(args.batch_size, np.float32))
        loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss
        if u % args.log_every == 0:
            print(f"u={u:6d}  loss={loss_ema:.4f}  {time.time()-t0:.0f}s",
                  flush=True)
        if u % args.eval_every == 0:
            err = wm.tf_err(*heldout)
            Se, Ae, Te, Me = sample_batch(512)
            errt = wm.tf_err(Se, Ae, Te, Me)
            print(f"  [eval u={u}] heldout_tf_err={err:.4f} "
                  f"train_tf_err={errt:.4f}", flush=True)
            metrics.append(dict(u=u, heldout_tf_err=err, train_tf_err=errt,
                                n_data=n_data, loss_ema=loss_ema))

    with open(out / "params.pkl", "wb") as f:
        pickle.dump(wm.params, f)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] baseline on {n_data} transitions. Saved to {out}", flush=True)


if __name__ == "__main__":
    main()
