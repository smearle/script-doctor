"""Are the THL rollout-failure transitions in the training data?

A* collection saves ALL transitions encountered during a 200k-iter frontier
search (not just the solution), so an enabled-action trajectory from the same
root is very likely INSIDE the explored set. This checks it empirically.

For each requested (level, algo): replay the cached solution (validated to be a
winning cpp_sols path), run the model teacher-forced from each TRUE state to
find genuine 1-step errors, then check each (s,a,s') against:
  - the L<level> training subsample (exactly what the model trained on, incl.
    the val holdout) — split into train vs held-out val (seed-0 reconstruction
    over the merged per-game array);
  - the FULL A* exploration of that level (collect_transitions_astar direct).
Positive control: env.reset() state must be in the full set AND its enabled-
action edges must be found (validates encoding + action convention).

Run from repo root:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 \
        nca_wm/scripts/thl_check_rollout_membership.py --levels 6,9
"""
from __future__ import annotations
import argparse, json, pickle, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from nca_wm.serve_wm import _build_wm, _unwrap_wm
from nca_wm.train import (
    N_ACTIONS, make_apply_fn, _pad_state_for_model, _enabled_action_count,
    collect_unique_transitions, collect_multigame_dataset,
    _unpack_states, _pack_states, _dats_to_multihot_batch, _cache_dir, _load_npz_dict,
)
from puzzlescript_cpp import CppPuzzleScriptEnv, _build_dedup_maps
from puzzlescript_cpp._puzzlescript_cpp import Engine, collect_transitions_astar
from puzzlescript_jax.utils import init_ps_lark_parser
import jax, jax.numpy as jnp

CFG_DIR = os.path.join(REPO, "nca_wm", "logs", "take_heart_lass", "pool_on_skip_on")
SEED = 0


def _val_idx_for_level(cfg, game_infos, level):
    """Reconstruct train.py's seed-0 per-game val split (over the merged
    per-game concatenation of all levels) and return the set of LEVEL-LOCAL
    indices that are held out for `level`."""
    ps = init_ps_lark_parser()
    ds, _ = collect_multigame_dataset(
        [game_infos[0]["name"]], ps, level_i=cfg.get("level"),
        n_search_steps=cfg["n_search_steps"], search_timeout_ms=cfg["search_timeout_ms"],
        search_algo=cfg["search_algo"],
        max_transitions_per_game=(cfg.get("max_transitions_per_game") or None),
        train_levels=cfg.get("train_levels"))
    n_g = int(ds["per_game_n_transitions"][0])
    tsh = np.asarray(ds["per_game_transition_shapes"][0])  # (N,3) per-transition C,H,W
    n_val = max(0, min(n_g - 1, int(round(n_g * cfg["val_frac"]))))
    sub = np.random.RandomState(SEED * 7919 + 0 + 1)
    val_global = set(sub.choice(n_g, size=n_val, replace=False).tolist())
    # Level blocks are contiguous in level order; identify this level's block by
    # matching the level's (H,W) — sufficient here since levels have distinct sizes
    # within their contiguous runs, and we only need train/val *labels* per
    # global index, which we return directly.
    return ds, val_global, n_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="6")
    ap.add_argument("--algo", default="bfs")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    cfg = json.load(open(os.path.join(CFG_DIR, "config.json")))
    params = _unwrap_wm(pickle.load(open(os.path.join(CFG_DIR, "params.pkl"), "rb")))
    game_infos = pickle.load(open(os.path.join(CFG_DIR, "game_infos.pkl"), "rb"))
    info = game_infos[0]
    n_objs, js, name = info["n_objs"], info["json_str"], info["name"]
    model, max_tok_len = _build_wm(cfg, game_infos)
    apply_fn = make_apply_fn(model)
    max_C = model.n_out
    max_H = max(g["H"] for g in game_infos); max_W = max(g["W"] for g in game_infos)
    tids = info.get("token_ids", [])
    gt = np.zeros(max_tok_len, np.int32); gm = np.zeros(max_tok_len, bool)
    gt[:len(tids)] = tids; gm[:len(tids)] = True
    gtj, gmj = jnp.array(gt[None]), jnp.array(gm[None])

    # Reconstruct merged dataset + val split once (for train/val labelling).
    ds, val_global, n_g = _val_idx_for_level(cfg, game_infos, levels[0])
    PS = ds["per_game_states"][0]; PNS = ds["per_game_next_states"][0]
    PA = np.asarray(ds["per_game_actions"][0]).astype(int)
    TS = np.asarray(ds["per_game_transition_shapes"][0]); Wpad_ds = PS.shape[-1] * 8

    def dkey(packed_s, shp, a, packed_ns):
        _, H, W = (int(x) for x in shp)
        s = _unpack_states(packed_s, Wpad_ds)[:n_objs, :H, :W]
        ns = _unpack_states(packed_ns, Wpad_ds)[:n_objs, :H, :W]
        return (s.tobytes(), int(a), ns.tobytes())

    # Map merged-dataset transition -> 'train'/'val', keyed by (s,a,ns).
    print("Indexing merged training dataset (train/val labels) ...", flush=True)
    split_of = {}
    for i in range(n_g):
        split_of[dkey(PS[i], TS[i], PA[i], PNS[i])] = "val" if i in val_global else "train"

    for level in levels:
        # FULL A* exploration of this level (un-subsampled).
        eng = Engine(); eng.load_from_json(js); eng.load_level(level)
        res = collect_transitions_astar(eng, max_iters=cfg["n_search_steps"],
                                        timeout_ms=cfg["search_timeout_ms"])
        rs = np.asarray(res.states, np.int32); rn = np.asarray(res.next_states, np.int32)
        ra = np.asarray(res.actions, np.int32)
        rno = len(res.id_dict); w, h = res.width, res.height
        _c, r2c = _build_dedup_maps(res.id_dict); nc = len(_c)
        S = _dats_to_multihot_batch(rs, rno, w, h, r2c, nc)
        NS = _dats_to_multihot_batch(rn, rno, w, h, r2c, nc)

        def fkey(s, a, ns):
            return (_pack_states(s).tobytes(), int(a), _pack_states(ns).tobytes())
        full_set = {fkey(S[i], ra[i], NS[i]) for i in range(len(ra))}
        full_states = {_pack_states(S[i]).tobytes() for i in range(len(ra))}

        # positive control
        env = CppPuzzleScriptEnv(js, level_i=level, max_episode_steps=2); s0, _ = env.reset()
        ctrl_state = _pack_states(s0).tobytes() in full_states
        n_act = _enabled_action_count(js)
        edges = 0
        for a in range(n_act):
            e = CppPuzzleScriptEnv(js, level_i=level, max_episode_steps=2); s, _ = e.reset()
            sn, _, _, _, _ = e.step(a)
            edges += fkey(s, a, sn) in full_set
        print(f"\n===== L{level} ({h}x{w}); full A* explored = {len(ra):,} transitions =====")
        print(f"[control] reset state in full set: {ctrl_state};  "
              f"reset edges found: {edges}/{n_act}")

        # replay cached solution
        sol = _load_npz_dict(os.path.join(_cache_dir(name, level),
                                          f"search_{args.algo}_200000_120000.npz"))
        acts = sol["actions"].tolist()
        env = CppPuzzleScriptEnv(js, level_i=level, max_episode_steps=len(acts) + 1)
        obs, _ = env.reset(); real = [obs.copy()]; won = False
        for a in acts:
            obs, _, d, t, i2 = env.step(int(a)); real.append(obs.copy())
            won = bool(i2.get("won", False))
            if won or d or t:
                break
        T = len(real) - 1
        H, W = real[0].shape[1], real[0].shape[2]
        print(f"[{args.algo}] {T} steps, won={won}")
        print(f"{'t':>3} {'a':>2} {'tf_wrong':>8} {'in_full':>7} {'in_subsample':>13}")
        for t in range(T):
            s_t, s_n, a = real[t], real[t + 1], int(acts[t])
            st = _pad_state_for_model(s_t, max_C, max_H, max_W)
            a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a][None])
            logits, _, _ = apply_fn(params, st, a_oh, gtj, gmj)
            pred = np.array(jax.nn.sigmoid(logits[0, :n_objs, :H, :W]) > 0.5, np.uint8)
            wrong = int((pred != s_n).any(axis=0).sum())
            k = fkey(s_t, a, s_n)
            in_full = k in full_set
            # subsample membership: build a key compatible with split_of (which
            # was keyed on padded-then-cropped reals -> identical crop here).
            dk = (s_t.tobytes(), a, s_n.tobytes())
            sub = split_of.get(dk, "absent")
            mark = "  <-- TF ERROR" if wrong > 0 else ""
            print(f"{t:>3} {a:>2} {wrong:>8} {str(in_full):>7} {sub:>13}{mark}")


if __name__ == "__main__":
    main()
