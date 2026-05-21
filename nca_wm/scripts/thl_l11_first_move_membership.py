"""Why does EVERY first action on L11 diverge in the interactive server?

The user observed that from the L11 initial (reset) state, any action produces a
step-1 divergence in the cap300k model. This checks whether the root edges are in
the training data at all, and if not, WHERE they were dropped:

  1. full A* exploration (uncapped: collect_transitions_astar) -- the true set;
  2. the per-level CACHE (collect_unique_transitions, capped to
     TRANSITIONS_CACHE_CAP=200k via RandomState(42+level)) -- what reaches the
     merged dataset at all;
  3. the cap300k merged dataset, split into TRAIN vs held-out VAL via the
     dataset's own per_game_val_idx (the new water-filled pipeline).

For each enabled-action root edge it also runs the model teacher-forced to confirm
the step-1 error. Positive control: the reset state itself must be in the full set
and its edges found (validates the encoding + action convention).

Run from repo root:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 \
        nca_wm/scripts/thl_l11_first_move_membership.py --level 11
"""
from __future__ import annotations
import argparse, json, pickle, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from nca_wm.serve_wm import _build_wm, _unwrap_wm
from nca_wm.train import (
    N_ACTIONS, make_apply_fn, _pad_state_for_model, _enabled_action_count,
    collect_unique_transitions, collect_multigame_dataset, _unpack_states,
)
from puzzlescript_cpp import CppPuzzleScriptEnv, _build_dedup_maps
from puzzlescript_cpp._puzzlescript_cpp import Engine, collect_transitions_astar
from puzzlescript_jax.utils import init_ps_lark_parser
import jax, jax.numpy as jnp

CFG_DIR = os.path.join(REPO, "nca_wm", "logs", "take_heart_lass",
                       "pool_on_skip_on_cap300k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=11)
    ap.add_argument("--cfg_dir", default=CFG_DIR)
    args = ap.parse_args()
    level = args.level

    cfg = json.load(open(os.path.join(args.cfg_dir, "config.json")))
    params = _unwrap_wm(pickle.load(open(os.path.join(args.cfg_dir, "params.pkl"), "rb")))
    game_infos = pickle.load(open(os.path.join(args.cfg_dir, "game_infos.pkl"), "rb"))
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

    # ---- The exact cap300k merged dataset (cache hit), with its val carve. ----
    ps = init_ps_lark_parser()
    ds, _gi = collect_multigame_dataset(
        [name], ps, level_i=cfg.get("level"),
        n_search_steps=cfg["n_search_steps"], search_timeout_ms=cfg["search_timeout_ms"],
        search_algo=cfg["search_algo"],
        max_transitions_per_game=(cfg.get("max_transitions_per_game") or None),
        train_levels=cfg.get("train_levels"), val_frac=cfg["val_frac"])
    PS = ds["per_game_states"][0]; PNS = ds["per_game_next_states"][0]
    PA = np.asarray(ds["per_game_actions"][0]).astype(int)
    TS = np.asarray(ds["per_game_transition_shapes"][0])
    val_idx = set(np.asarray(ds["per_game_val_idx"][0]).astype(int).tolist())
    Wpad_ds = PS.shape[-1] * 8
    n_g = len(PS)

    def crop_key(packed_s, shp, a, packed_ns):
        _, H, W = (int(x) for x in shp)
        s = _unpack_states(packed_s, Wpad_ds)[:n_objs, :H, :W]
        ns = _unpack_states(packed_ns, Wpad_ds)[:n_objs, :H, :W]
        return (s.tobytes(), int(a), ns.tobytes())

    print(f"Indexing cap300k merged dataset ({n_g:,} kept; "
          f"{len(val_idx):,} val) ...", flush=True)
    train_keys, val_keys = set(), set()
    for i in range(n_g):
        k = crop_key(PS[i], TS[i], PA[i], PNS[i])
        (val_keys if i in val_idx else train_keys).add(k)

    # ---- Full uncapped A* exploration of this level. ----
    eng = Engine(); eng.load_from_json(js); eng.load_level(level)
    res = collect_transitions_astar(eng, max_iters=cfg["n_search_steps"],
                                    timeout_ms=cfg["search_timeout_ms"])
    rs = np.asarray(res.states, np.int32); rn = np.asarray(res.next_states, np.int32)
    ra = np.asarray(res.actions, np.int32)
    rno = len(res.id_dict); w, h = res.width, res.height
    _c, r2c = _build_dedup_maps(res.id_dict); nc = len(_c)
    from nca_wm.train import _dats_to_multihot_batch
    S = _dats_to_multihot_batch(rs, rno, w, h, r2c, nc)
    NS = _dats_to_multihot_batch(rn, rno, w, h, r2c, nc)

    def raw_key(s, a, ns):
        return (s[:n_objs, :h, :w].tobytes(), int(a), ns[:n_objs, :h, :w].tobytes())
    full_set = {raw_key(S[i], ra[i], NS[i]) for i in range(len(ra))}
    full_states = {S[i][:n_objs, :h, :w].tobytes() for i in range(len(ra))}
    print(f"L{level}: full A* explored = {len(ra):,} transitions ({h}x{w})")

    # ---- The per-level CACHE (capped 200k) that actually feeds the dataset. ----
    cache = collect_unique_transitions(
        js, name, level_i=level, max_iters=cfg["n_search_steps"],
        timeout_ms=cfg["search_timeout_ms"], search_algo=cfg["search_algo"],
        max_transitions=200_000)
    cW = int(cache["W"]); cWpad = cache["states"].shape[-1] * 8
    cached_set = set()
    cs = cache["states"]; cns = cache["next_states"]; ca = np.asarray(cache["actions"]).astype(int)
    for i in range(len(cs)):
        s = _unpack_states(cs[i], cWpad)[:n_objs, :, :cW]
        ns = _unpack_states(cns[i], cWpad)[:n_objs, :, :cW]
        cached_set.add((s.tobytes(), int(ca[i]), ns.tobytes()))
    print(f"L{level}: per-level cache (cap 200k) = {len(cs):,} transitions")

    # ---- Root edges: reset state + each enabled action. ----
    env = CppPuzzleScriptEnv(js, level_i=level, max_episode_steps=2)
    s0, _ = env.reset()
    ctrl = s0[:n_objs, :h, :w].tobytes() in full_states
    n_act = _enabled_action_count(js)
    print(f"\n[control] reset state in full set: {ctrl};  enabled actions: {n_act}")
    print(f"\n{'a':>2} {'tf_wrong':>8} {'in_full':>8} {'in_cache':>9} "
          f"{'in_train':>9} {'in_val':>7}")
    for a in range(n_act):
        e = CppPuzzleScriptEnv(js, level_i=level, max_episode_steps=2)
        s, _ = e.reset(); sn, _, _, _, _ = e.step(a)
        # model teacher-forced from the true root state
        st = _pad_state_for_model(s, max_C, max_H, max_W)
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a][None])
        logits, _, _ = apply_fn(params, st, a_oh, gtj, gmj)
        pred = np.array(jax.nn.sigmoid(logits[0, :n_objs, :s.shape[1], :s.shape[2]]) > 0.5,
                        np.uint8)
        wrong = int((pred != sn[:n_objs]).any(axis=0).sum())
        rk = raw_key(s, a, sn)
        ck = (s[:n_objs].tobytes(), int(a), sn[:n_objs].tobytes())
        in_full = rk in full_set
        in_cache = ck in cached_set
        in_train = ck in train_keys
        in_val = ck in val_keys
        mark = "  <-- TF ERROR" if wrong > 0 else ""
        print(f"{a:>2} {wrong:>8} {str(in_full):>8} {str(in_cache):>9} "
              f"{str(in_train):>9} {str(in_val):>7}{mark}")


if __name__ == "__main__":
    main()
