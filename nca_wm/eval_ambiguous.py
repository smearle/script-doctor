"""Eval the full-data model on the GENUINELY AMBIGUOUS pre-bonk states (identical
all-channel state that breaks in mario_breakable but occurs in mario under UP).

Expectation if the model learned the irreducible uncertainty: P(step present|UP)
~ 0.5 on the ambiguous states, while staying ~0 (confident break) on unambiguous
break states and ~perfect elsewhere.
"""
import glob
import json
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from nca_wm.models import NCAWorldModel, N_ACTIONS
from nca_wm.state_ops import _unpack_states

STEP, UP = 6, 0
CHUNK = 100_000
SAVE = "nca_wm/logs/mario_uncond_fulldata"


def cache(name):
    f = glob.glob(f"rollout_data/{name}/level_0/bfs_transitions_v5_200000_-1_capall.npz")[0]
    return np.load(f, allow_pickle=True)


def main():
    db = cache("mario_breakable")
    Spb, Nspb, Ab = db["states"], db["next_states"], np.asarray(db["actions"], np.int64)
    Wb = int(db["W"])
    brk = {}                                   # packed bytes -> (global_idx, h, w)
    for i in range(0, len(Spb), CHUNK):
        sl = slice(i, i + CHUNK)
        up = Ab[sl] == UP
        S = _unpack_states(Spb[sl], Wb); Ns = _unpack_states(Nspb[sl], Wb)
        broke = (S[:, STEP] > 0) & ~(Ns[:, STEP] > 0)
        for j in np.where(up & broke.reshape(len(S), -1).any(1))[0]:
            h, w = np.argwhere(broke[j])[0]
            brk[Spb[i + j].tobytes()] = (i + j, int(h), int(w))

    dm = cache("mario")
    Spm, Am = dm["states"], np.asarray(dm["actions"], np.int64)
    mup = set()
    for i in range(0, len(Spm), CHUNK):
        up = Am[i:i + CHUNK] == UP
        for j in np.where(up)[0]:
            mup.add(Spm[i + j].tobytes())

    amb = [v for k, v in brk.items() if k in mup]
    unamb = [v for k, v in brk.items() if k not in mup]
    print(f"{len(brk)} break-states: {len(amb)} ambiguous, {len(unamb)} unambiguous")

    params = pickle.load(open(f"{SAVE}/params_best.pkl", "rb"))
    cfg = json.load(open(f"{SAVE}/config.json"))
    gi = pickle.load(open(f"{SAVE}/game_infos.pkl", "rb"))
    max_C = max(g["n_objs"] for g in gi)
    model = NCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
        input_skip=cfg["input_skip"], n_repeats=cfg["n_nca_repeats"],
        history=cfg["history"], axis_pool=cfg["axis_pool"],
        axis_cummax=cfg["axis_cummax"], global_pool=cfg["global_pool"])

    @jax.jit
    def pred(s, a):
        l, _w, _s = model.apply(params, s, jax.nn.one_hot(a, N_ACTIONS))
        return jax.nn.sigmoid(l)

    def probs_at(rows):
        idx = np.array([r[0] for r in rows])
        S = _unpack_states(Spb[idx], Wb).astype(np.float32)
        if S.shape[1] < max_C:
            S = np.pad(S, ((0, 0), (0, max_C - S.shape[1]), (0, 0), (0, 0)))
        A = np.full(len(S), UP, np.int64)
        out = []
        for i in range(0, len(S), 2048):
            out.append(np.asarray(pred(jnp.asarray(S[i:i+2048]), jnp.asarray(A[i:i+2048]))))
        P = np.concatenate(out)
        return np.array([P[k, STEP, r[1], r[2]] for k, r in enumerate(rows)])

    for tag, rows in [("AMBIGUOUS", amb), ("unambiguous", unamb)]:
        if not rows:
            continue
        p = probs_at(rows)
        print(f"\n[{tag}] P(step present|UP) at break cell: mean={p.mean():.3f} "
              f"median={np.median(p):.3f}  n={len(p)}")
        print(f"   histogram[0..1]: {np.histogram(p, bins=np.linspace(0,1,11))[0].tolist()}")
    print("\n  expect AMBIGUOUS ~0.5 (irreducible uncertainty learned), unambiguous ~0 (confident break)")


if __name__ == "__main__":
    main()
