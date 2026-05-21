"""Why is val ~0 but rollout fails? Measure the model's teacher-forced 1-step
error over a LARGE uniform sample of the FULL A* exploration of L6, vs the tiny
val sample. If the full-set error rate is small-but-nonzero and val is just too
small to contain any erroring transition, that's undersampling (not bias).

Also reports whether erroring transitions are 'rare' (concentrated) by comparing
the despair-cell count distribution of erroring vs non-erroring transitions.
"""
from __future__ import annotations
import json, pickle, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from nca_wm.serve_wm import _build_wm, _unwrap_wm
from nca_wm.train import (
    N_ACTIONS, make_apply_fn, _pad_state_for_model, _dats_to_multihot_batch,
)
from puzzlescript_cpp import _build_dedup_maps
from puzzlescript_cpp._puzzlescript_cpp import Engine, collect_transitions_astar
import jax, jax.numpy as jnp

CFG_DIR = os.path.join(REPO, "nca_wm", "logs", "take_heart_lass", "pool_on_skip_on")
LEVEL = 6
SAMPLE = 30000
BATCH = 256


def main():
    cfg = json.load(open(os.path.join(CFG_DIR, "config.json")))
    params = _unwrap_wm(pickle.load(open(os.path.join(CFG_DIR, "params.pkl"), "rb")))
    gi = pickle.load(open(os.path.join(CFG_DIR, "game_infos.pkl"), "rb"))
    info = gi[0]; n_objs = info["n_objs"]; js = info["json_str"]
    model, mtl = _build_wm(cfg, gi); apply_fn = make_apply_fn(model)
    max_C = model.n_out; max_H = max(g["H"] for g in gi); max_W = max(g["W"] for g in gi)
    tids = info.get("token_ids", []); gt = np.zeros(mtl, np.int32); gm = np.zeros(mtl, bool)
    gt[:len(tids)] = tids; gm[:len(tids)] = True

    print(f"Full A* explore L{LEVEL} ...", flush=True)
    eng = Engine(); eng.load_from_json(js); eng.load_level(LEVEL)
    res = collect_transitions_astar(eng, max_iters=cfg["n_search_steps"],
                                    timeout_ms=cfg["search_timeout_ms"])
    rs = np.asarray(res.states, np.int32); rn = np.asarray(res.next_states, np.int32)
    ra = np.asarray(res.actions, np.int32)
    rno = len(res.id_dict); w, h = res.width, res.height
    _c, r2c = _build_dedup_maps(res.id_dict); nc = len(_c)
    N = len(ra)
    print(f"  full explored = {N:,} transitions ({h}x{w}, C={nc})", flush=True)

    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(SAMPLE, N), replace=False)
    S = _dats_to_multihot_batch(rs[idx], rno, w, h, r2c, nc)    # (n,C,h,w)
    NS = _dats_to_multihot_batch(rn[idx], rno, w, h, r2c, nc)
    A = ra[idx]

    gtj = jnp.broadcast_to(jnp.array(gt[None]), (BATCH, mtl))
    gmj = jnp.broadcast_to(jnp.array(gm[None]), (BATCH, mtl))
    eye = np.eye(N_ACTIONS, dtype=np.float32)

    n = len(idx)
    wrong_cells = np.zeros(n, dtype=np.int64)
    changed_cells = np.zeros(n, dtype=np.int64)
    wrong_changed = np.zeros(n, dtype=np.int64)
    for b0 in range(0, n, BATCH):
        b1 = min(n, b0 + BATCH)
        bs = b1 - b0
        st = np.zeros((bs, max_C, max_H, max_W), np.float32)
        st[:, :n_objs, :h, :w] = S[b0:b1]
        a_oh = eye[A[b0:b1]]
        gtb = gtj[:bs] if bs == BATCH else jnp.broadcast_to(jnp.array(gt[None]), (bs, mtl))
        gmb = gmj[:bs] if bs == BATCH else jnp.broadcast_to(jnp.array(gm[None]), (bs, mtl))
        logits, _, _ = apply_fn(params, jnp.array(st), jnp.array(a_oh), gtb, gmb)
        pred = np.array(jax.nn.sigmoid(logits[:, :n_objs, :h, :w]) > 0.5, np.uint8)
        real_s = S[b0:b1]; real_ns = NS[b0:b1]
        mism = (pred != real_ns).any(axis=1)             # (bs,h,w)
        chg = (real_s != real_ns).any(axis=1)            # cells that change
        wrong_cells[b0:b1] = mism.sum(axis=(1, 2))
        changed_cells[b0:b1] = chg.sum(axis=(1, 2))
        wrong_changed[b0:b1] = (mism & chg).sum(axis=(1, 2))

    err = wrong_cells > 0
    frac = err.mean()
    print(f"\n=== Teacher-forced 1-step error on {n:,} uniform-random full-explored L{LEVEL} transitions ===")
    print(f"transitions with >=1 wrong cell : {err.sum():,} / {n:,}  = {100*frac:.3f}%")
    print(f"mean wrong cells / transition   : {wrong_cells.mean():.4f}")
    tot_changed = changed_cells.sum()
    print(f"change_err (wrong-changed / changed cells) : {wrong_changed.sum()/max(1,tot_changed):.3e}")
    print(f"  (val reported ~1.2e-9 on its 1,584-transition sample)")
    print(f"\nExtrapolated to full {N:,}: ~{int(frac*N):,} erroring transitions")
    print(f"Expected erroring transitions in a 1,584 val sample: {frac*1584:.2f}  "
          f"-> P(val sees zero) = {np.exp(-frac*1584):.2f}")
    # bias check: are erroring transitions 'deeper'/more-despair? compare a proxy
    # = number of changed cells (spread activity) for erroring vs clean.
    if err.any() and (~err).any():
        print(f"\nchanged-cells (spread activity) mean: "
              f"erroring={changed_cells[err].mean():.1f}  clean={changed_cells[~err].mean():.1f}")


if __name__ == "__main__":
    main()
