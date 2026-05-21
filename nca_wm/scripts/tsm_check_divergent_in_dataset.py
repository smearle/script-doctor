"""Are the transitions where the model diverges in the training/val dataset?

Convention-validated membership check. For the worst-diverging L11 random
rollout (pool_on_skip_off, ep7):

  1. Re-run the C++ A* collector DIRECTLY on L11 (same budget as the run) to
     get the FULL set of explored transitions (pre-subsample), in the canonical
     dedup encoding the dataset stores.
  2. POSITIVE CONTROLS (must pass or the check is meaningless):
       - env.reset() state is present in the collector's state set;
       - every (s0, a) edge the env produces is present in the collector's
         transition set  -> env encoding AND action convention match the data.
  3. Load the actual TRAINING cache for L11 (the random 8333-subsample the model
     was trained on) and reconstruct the seed-0 val split.
  4. For each step of the L11 ep7 random rollout, run the model teacher-forced
     from the TRUE state to find 1-step (TF) errors, and check membership of the
     real (s,a,s') transition in: the full explored set, and the train/val
     subsample.

Run from repo root:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 \
        nca_wm/scripts/tsm_check_divergent_in_dataset.py
"""
from __future__ import annotations
import json, pickle, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from puzzlescript_jax.utils import init_ps_lark_parser
from nca_wm.serve_wm import _build_wm, _unwrap_wm
from nca_wm.train import (
    N_ACTIONS, make_apply_fn, _pad_state_for_model,
    collect_unique_transitions, _unpack_states, _pack_states,
    _dats_to_multihot_batch,
)
from puzzlescript_cpp import CppPuzzleScriptEnv, _build_dedup_maps
from puzzlescript_cpp._puzzlescript_cpp import Engine, collect_transitions_astar
import jax, jax.numpy as jnp

CFG_DIR = os.path.join(REPO, "nca_wm", "logs", "tsm_pool_diag", "pool_on_skip_off")
LEVEL, EP = 11, 7
N_EPS, MAX_STEPS, SEED = 10, 50, 0


def _val_indices(n_g, g, seed, val_frac):
    n_val = max(0, min(n_g - 1, int(round(n_g * val_frac))))
    if n_val == 0:
        return set()
    sub = np.random.RandomState(seed * 7919 + g + 1)
    return set(sub.choice(n_g, size=n_val, replace=False).tolist())


def main():
    cfg = json.load(open(os.path.join(CFG_DIR, "config.json")))
    params = _unwrap_wm(pickle.load(open(os.path.join(CFG_DIR, "params.pkl"), "rb")))
    game_infos = pickle.load(open(os.path.join(CFG_DIR, "game_infos.pkl"), "rb"))
    info = game_infos[0]
    n_objs, json_str = info["n_objs"], info["json_str"]
    model, max_tok_len = _build_wm(cfg, game_infos)
    apply_fn = make_apply_fn(model)
    max_C = model.n_out
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)

    # --- 1. FULL A* exploration on L11 (bypass cache/subsample) ---
    print(f"Running full A* collector on L{LEVEL} "
          f"(iters={cfg['n_search_steps']}, timeout={cfg['search_timeout_ms']}ms) ...")
    eng = Engine(); eng.load_from_json(json_str); eng.load_level(LEVEL)
    res = collect_transitions_astar(eng, max_iters=cfg["n_search_steps"],
                                    timeout_ms=cfg["search_timeout_ms"])
    raw_states = np.asarray(res.states, dtype=np.int32)
    raw_next = np.asarray(res.next_states, dtype=np.int32)
    acts = np.asarray(res.actions, dtype=np.int32)
    raw_n_objs = len(res.id_dict)
    w, h = res.width, res.height
    _canon, r2c = _build_dedup_maps(res.id_dict)
    nc = len(_canon)
    S = _dats_to_multihot_batch(raw_states, raw_n_objs, w, h, r2c, nc)   # (N,C,h,w)
    NS = _dats_to_multihot_batch(raw_next, raw_n_objs, w, h, r2c, nc)
    print(f"  full explored transitions: {len(acts):,}  (canonical C={nc}, {h}x{w})")

    def key(s, a, ns):
        return (_pack_states(s).tobytes(), int(a), _pack_states(ns).tobytes())
    full_set = {key(S[i], acts[i], NS[i]) for i in range(len(acts))}
    full_states = {_pack_states(S[i]).tobytes() for i in range(len(acts))}

    # --- 2. positive controls: env encoding + action convention ---
    env = CppPuzzleScriptEnv(json_str, level_i=LEVEL, max_episode_steps=MAX_STEPS)
    s0, _ = env.reset()
    print("\n[control] env.reset() state in collector state-set:",
          _pack_states(s0).tobytes() in full_states)
    found = 0
    for a in range(N_ACTIONS):
        e = CppPuzzleScriptEnv(json_str, level_i=LEVEL, max_episode_steps=2)
        s, _ = e.reset(); sn, _, _, _, _ = e.step(a)
        if key(s, a, sn) in full_set:
            found += 1
    print(f"[control] (s0,a)->s' edges found in collector set: {found}/{N_ACTIONS}")

    # --- 3. training subsample (what the model actually trained on) + val ---
    sub = collect_unique_transitions(
        json_str, info["name"], level_i=LEVEL,
        max_iters=cfg["n_search_steps"], timeout_ms=cfg["search_timeout_ms"],
        search_algo=cfg["search_algo"],
        max_transitions=(cfg.get("max_transitions_per_game") or None),
    )
    subW = int(sub["W"])
    subS = _unpack_states(sub["states"], subW)            # (n,C,h,w)
    subNS = _unpack_states(sub["next_states"], subW)
    subA = np.asarray(sub["actions"]).astype(int)
    n_sub = len(subA)
    # NOTE: the per-game val split in train.py is over the MERGED per-game array
    # (all levels concatenated). Here we only need to know, for transitions that
    # ARE in the L11 subsample, whether each is held out. The merged split index
    # differs, so we report train/val membership against the *L11 subsample as a
    # whole* and separately note the per-game held-out fraction.
    sub_set = {key(subS[i], subA[i], subNS[i]) for i in range(n_sub)}
    print(f"  L{LEVEL} training subsample: {n_sub:,} transitions "
          f"(cap={cfg.get('max_transitions_per_game')})")

    # --- 4. roll L11 ep7 random trajectory, TF errors + membership ---
    actions = np.random.default_rng(SEED).integers(
        0, N_ACTIONS, size=(N_EPS, MAX_STEPS), dtype=np.int32)[EP]
    env = CppPuzzleScriptEnv(json_str, level_i=LEVEL, max_episode_steps=MAX_STEPS)
    obs, _ = env.reset()
    H, W = obs.shape[1], obs.shape[2]
    real = [obs.copy()]
    for t in range(MAX_STEPS):
        obs, _, d, tr, _ = env.step(int(actions[t])); real.append(obs.copy())
        if d or tr:
            break
    T = len(real) - 1

    tids = info.get("token_ids", [])
    gt = np.zeros(max_tok_len, np.int32); gm = np.zeros(max_tok_len, bool)
    gt[:len(tids)] = tids; gm[:len(tids)] = True
    gtj, gmj = jnp.array(gt[None]), jnp.array(gm[None])

    print(f"\nL{LEVEL} ep{EP}: {T} real transitions ({H}x{W})")
    print(f"{'t':>3} {'a':>2} {'tf_wrong':>8} {'in_full':>7} {'in_train':>8}")
    in_full = in_train = 0
    tf_err = []
    for t in range(T):
        s_t, s_n, a = real[t], real[t + 1], int(actions[t])
        st = _pad_state_for_model(s_t, max_C, max_H, max_W)
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a][None])
        logits, _, _ = apply_fn(params, st, a_oh, gtj, gmj)
        pred = np.array(jax.nn.sigmoid(logits[0, :n_objs, :H, :W]) > 0.5, np.uint8)
        wrong = int((pred != s_n).any(axis=0).sum())
        k = key(s_t, a, s_n)
        f = k in full_set; tr_ = k in sub_set
        in_full += f; in_train += tr_
        if wrong > 0:
            tf_err.append((t, a, wrong, f, tr_))
        print(f"{t:>3} {a:>2} {wrong:>8} {str(f):>7} {str(tr_):>8}")

    print(f"\n--- Summary (L{LEVEL} ep{EP}, {T} transitions) ---")
    print(f"in FULL A*-explored set : {in_full}/{T}")
    print(f"in TRAINING subsample   : {in_train}/{T}")
    print(f"\n1-step (TF) error steps: {len(tf_err)}")
    for t, a, w, f, tr_ in tf_err:
        print(f"  t={t:>2} a={a} wrong={w:>3}  in_full={f}  in_train={tr_}")
    if tf_err:
        nf = sum(1 for *_x, f, _t in tf_err if not f)
        print(f"\nof {len(tf_err)} TF-error transitions: {nf} absent even from the "
              f"FULL A* exploration, {len(tf_err)-nf} explored-but-maybe-subsampled-out.")


if __name__ == "__main__":
    main()
