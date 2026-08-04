"""Sliced heldout eval for tree_growth vs offline-baseline checkpoints.

The plain per-cell teacher-forced error saturates (~0.0006) at every budget:
almost all cells are static and almost all transitions are plain walking.
The regimes differ in the RARE-EVENT TAIL, so this evaluator reports:

  1. changed-cell error   — error restricted to cells that actually change
                            (the ~2% of cells that carry the dynamics);
  2. per-rule-family error — heldout transitions labeled by which engine
                            rules fired (push / pull / wizard / conversion /
                            doors), changed-cell error per family;
  3. AR rollout exactness  — K-step autoregressive rollouts on heldout
                            levels, fraction of steps with the full frame
                            exactly right.

Usage (compares any number of runs side by side):

    python -m nca_wm.active_learning.eval_slices \
        --game_json .../heroes_of_sokoban.json --heldout 14-21 \
        --runs active=/path/to/tg_run offline=/path/to/baseline_run
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm.active_learning.tree_growth import (
    WM, enabled_actions, engine_step, new_engine, parse_levels, read_obs)

# rule-index families for heroes_of_sokoban (55 direction-expanded rules;
# grouping by the source-rule blocks: movement rules expand x4 directions)
FAMILIES = {
    "fighter_push": range(0, 8),      # push + push-chain, 4 dirs each
    "thief_pull": range(8, 12),
    "wizard": range(12, 40),          # temp emit/move/land/swap + late cleanup
    "conversion": range(40, 47),
    "doors_switches": range(47, 55),
}


def family_of(rules: set[int]) -> list[str]:
    return [f for f, rr in FAMILIES.items() if any(r in rr for r in rules)] or ["none"]


def build_labeled_heldout(json_str, levels, n_obj, hp, wp, acts,
                          n_eps, ep_len, seed):
    """Random-rollout heldout transitions + fired-rule labels + level id."""
    rng = np.random.default_rng(seed)
    S, A, T, M, R, L = [], [], [], [], [], []
    for li in levels:
        eng = new_engine(json_str, li)
        w, h = eng.get_width(), eng.get_height()
        m = np.zeros((n_obj, hp, wp), np.float32)
        m[:, :min(h, hp), :min(w, wp)] = 1.0
        for _ in range(n_eps):
            eng.restart()
            o = read_obs(eng, n_obj, hp, wp)
            for _ in range(ep_len):
                a = int(rng.choice(acts))
                eng.clear_rules_fired()
                engine_step(eng, a)
                o2 = read_obs(eng, n_obj, hp, wp)
                S.append(o); A.append(a); T.append(o2); M.append(m)
                R.append({int(r) for r in eng.get_rules_fired()})
                L.append(li)
                o = o2
    return (np.stack(S), np.asarray(A, np.int32), np.stack(T), np.stack(M),
            R, np.asarray(L))


def changed_cell_err(wm, S, A, T, M, chunk=256):
    """(err_on_changed_cells, err_on_all_cells, n_changed_cells)."""
    import jax
    wrong_ch = tot_ch = wrong_all = tot_all = 0.0
    for i in range(0, len(S), chunk):
        sl = slice(i, min(i + chunk, len(S)))
        n = sl.stop - sl.start
        (Sc, Ac, Tc, Mc), _ = wm._pad([S[sl], A[sl], T[sl], M[sl]], n)
        logits = wm.model.apply(wm.params, Sc, jax.nn.one_hot(Ac, 6))[0]
        pred = np.asarray(logits > 0)[:n]
        tgt = T[sl] > 0.5
        mask = M[sl] > 0.5
        chg = (S[sl] > 0.5) != tgt          # cells that change this step
        wrong = (pred != tgt) & mask
        wrong_ch += (wrong & chg).sum(); tot_ch += (chg & mask).sum()
        wrong_all += wrong.sum(); tot_all += mask.sum()
    return (wrong_ch / max(tot_ch, 1), wrong_all / max(tot_all, 1),
            int(tot_ch))


def ar_rollout_exact(wm, json_str, levels, n_obj, hp, wp, acts,
                     n_eps, k, seed):
    """K-step autoregressive rollouts: fraction of frames exactly right."""
    import jax
    rng = np.random.default_rng(seed)
    exact = total = 0
    for li in levels:
        eng = new_engine(json_str, li)
        w, h = eng.get_width(), eng.get_height()
        m = np.zeros((n_obj, hp, wp), np.float32)
        m[:, :min(h, hp), :min(w, wp)] = 1.0
        for _ in range(n_eps):
            eng.restart()
            true = read_obs(eng, n_obj, hp, wp)
            pred = true.copy()
            for _ in range(k):
                a = int(rng.choice(acts))
                engine_step(eng, a)
                true = read_obs(eng, n_obj, hp, wp)
                (Sc, Ac), _ = wm._pad(
                    [pred[None].astype(np.float32),
                     np.asarray([a], np.int32)], 1)
                logits = wm.model.apply(wm.params, Sc, jax.nn.one_hot(Ac, 6))[0]
                pred = (np.asarray(logits[0]) > 0).astype(np.float32) * m
                exact += int(np.array_equal(pred > 0.5, (true > 0.5) & (m > 0.5)))
                total += 1
    return exact / max(total, 1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--game_json", required=True)
    p.add_argument("--heldout", default="14-21")
    p.add_argument("--heldout_eps", type=int, default=12)
    p.add_argument("--heldout_len", type=int, default=32)
    p.add_argument("--ar_eps", type=int, default=6)
    p.add_argument("--ar_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--runs", nargs="+", required=True,
                   help="name=/path/to/run_dir (needs params.pkl+config.json)")
    args = p.parse_args(argv)

    json_str = Path(args.game_json).read_text()
    acts = enabled_actions(json_str)
    probe = new_engine(json_str, 0)
    n_obj = probe.get_object_count()
    n_levels = probe.get_num_levels()
    levels = parse_levels(args.heldout, n_levels)

    runs = {}
    hp = wp = 0
    for spec in args.runs:
        name, path = spec.split("=", 1)
        cfg = json.loads((Path(path) / "config.json").read_text())
        runs[name] = (path, cfg)
    # pad must match training pad: recompute exactly as tree_growth did
    # (max over train levels 0-13 + heldout), assuming shared level split
    for li in range(n_levels):
        e = new_engine(json_str, li)
        hp = max(hp, e.get_height()); wp = max(wp, e.get_width())

    S, A, T, M, R, L = build_labeled_heldout(
        json_str, levels, n_obj, hp, wp, acts,
        args.heldout_eps, args.heldout_len, args.seed)
    fam_idx = defaultdict(list)
    for i, rr in enumerate(R):
        for f in family_of(rr):
            fam_idx[f].append(i)
    print(f"labeled heldout: {len(S)} transitions from levels {levels}")
    for f, ii in sorted(fam_idx.items()):
        print(f"  {f}: {len(ii)}")

    for name, (path, cfg) in runs.items():
        wm = WM(n_obj, hp, wp, cfg["n_hid"], cfg["n_nca_steps"],
                cfg.get("seed", 0), cfg["lr"])
        with open(Path(path) / "params.pkl", "rb") as fh:
            wm.params = pickle.load(fh)
        ce, ae, nch = changed_cell_err(wm, S, A, T, M)
        ar = ar_rollout_exact(wm, json_str, levels, n_obj, hp, wp, acts,
                              args.ar_eps, args.ar_k, args.seed + 7)
        print(f"\n=== {name} ({path}) ===")
        print(f"  all-cell err:      {ae:.5f}")
        print(f"  changed-cell err:  {ce:.4f}   (n_changed={nch})")
        print(f"  AR-{args.ar_k} exact-frame: {ar:.3f}")
        for f, ii in sorted(fam_idx.items()):
            if not ii:
                continue
            ii = np.asarray(ii)
            ce_f, _, nch_f = changed_cell_err(wm, S[ii], A[ii], T[ii], M[ii])
            print(f"  {f:16s} changed-cell err: {ce_f:.4f}  (n={len(ii)}, "
                  f"cells={nch_f})")


if __name__ == "__main__":
    main()
