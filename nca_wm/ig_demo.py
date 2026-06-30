"""Demonstrate information gain from the base world model (q0) + adapter (q1).

IG(s,a,o) = log q1(o | s,a,o) - log q0(o | s,a),  o ~ q0.
Expect: large on the genuinely ambiguous pre-bonk states (q0 ~ 0.5 there, but q1
having "observed" o is confident), ~0 on unambiguous / confident transitions.
"""
import glob
import json
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from nca_wm.models import NCAWorldModel, N_ACTIONS
from nca_wm.adapter import AdapterHead
from nca_wm.state_ops import _unpack_states

STEP, UP, CHUNK = 6, 0, 100_000
BASE = "nca_wm/logs/mario_uncond_fulldata"
ADAPT = "nca_wm/logs/adapter_mario_skip/adapter_params.pkl"
EPS = 1e-6


def cache(name):
    f = glob.glob(f"rollout_data/{name}/level_0/bfs_transitions_v5_200000_-1_capall.npz")[0]
    return np.load(f, allow_pickle=True)


def main():
    db = cache("mario_breakable")
    Spb, Nspb, Ab = db["states"], db["next_states"], np.asarray(db["actions"], np.int64)
    Wb = int(db["W"])
    brk = {}
    for i in range(0, len(Spb), CHUNK):
        up = Ab[i:i+CHUNK] == UP
        S = _unpack_states(Spb[i:i+CHUNK], Wb); Ns = _unpack_states(Nspb[i:i+CHUNK], Wb)
        broke = (S[:, STEP] > 0) & ~(Ns[:, STEP] > 0)
        for j in np.where(up & broke.reshape(len(S), -1).any(1))[0]:
            h, w = np.argwhere(broke[j])[0]
            brk[Spb[i+j].tobytes()] = (i + j, int(h), int(w))
    dm = cache("mario")
    Spm, Am = dm["states"], np.asarray(dm["actions"], np.int64)
    mup = set()
    for i in range(0, len(Spm), CHUNK):
        for j in np.where(Am[i:i+CHUNK] == UP)[0]:
            mup.add(Spm[i+j].tobytes())
    amb = [v for k, v in brk.items() if k in mup]
    unamb = [v for k, v in brk.items() if k not in mup]
    print(f"{len(amb)} ambiguous, {len(unamb)} unambiguous break states")

    # base q0
    bp = pickle.load(open(f"{BASE}/params_best.pkl", "rb"))
    cfg = json.load(open(f"{BASE}/config.json"))
    gi = pickle.load(open(f"{BASE}/game_infos.pkl", "rb"))
    C = max(g["n_objs"] for g in gi)
    base = NCAWorldModel(n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=C,
                         input_skip=cfg["input_skip"], n_repeats=cfg["n_nca_repeats"],
                         history=cfg["history"], axis_pool=cfg["axis_pool"],
                         axis_cummax=cfg["axis_cummax"], global_pool=cfg["global_pool"])
    # adapter q1
    ad = pickle.load(open(ADAPT, "rb"))
    adapter = AdapterHead(n_hid=ad["cfg"]["n_hid"], n_steps=ad["cfg"]["n_steps"], n_out=ad["cfg"]["n_out"])

    @jax.jit
    def q0(s, a):
        l, _, _ = base.apply(bp, s, jax.nn.one_hot(a, N_ACTIONS)); return jax.nn.sigmoid(l)

    @jax.jit
    def q1(s, a, o):
        return jax.nn.sigmoid(adapter.apply(ad["params"], s, jax.nn.one_hot(a, N_ACTIONS), o))

    def ig_for(triples, n_samples=16, seed=0):
        idx = np.array([t[0] for t in triples]); cells = [(t[1], t[2]) for t in triples]
        S = _unpack_states(Spb[idx], Wb).astype(np.float32)
        if S.shape[1] < C:
            S = np.pad(S, ((0, 0), (0, C - S.shape[1]), (0, 0), (0, 0)))
        A = np.full(len(S), UP, np.int64)
        Sj, Aj = jnp.asarray(S), jnp.asarray(A)
        P0 = np.asarray(q0(Sj, Aj)).clip(EPS, 1 - EPS)
        m = (S.sum(1, keepdims=True) > 0)
        rng = np.random.default_rng(seed)
        ig_full = np.zeros(len(S)); ig_cell = np.zeros(len(S))
        rows = np.arange(len(S)); hs = np.array([c[0] for c in cells]); ws = np.array([c[1] for c in cells])
        for _ in range(n_samples):
            o = (rng.random(P0.shape) < P0).astype(np.float32)
            P1 = np.asarray(q1(Sj, Aj, jnp.asarray(o))).clip(EPS, 1 - EPS)
            lq0 = o * np.log(P0) + (1 - o) * np.log(1 - P0)
            lq1 = o * np.log(P1) + (1 - o) * np.log(1 - P1)
            ig_full += ((lq1 - lq0) * m).sum(axis=(1, 2, 3))
            ig_cell += (lq1 - lq0)[rows, STEP, hs, ws]
        return ig_full / n_samples, ig_cell / n_samples

    # random transitions (mostly non-bonk) as the "IG should be ~0" baseline
    rng0 = np.random.default_rng(1)
    rnd = [(int(i), 0, 0) for i in rng0.integers(len(Spb), size=2000)]
    for tag, tr in [("AMBIGUOUS bonk", amb[:2000]), ("unambiguous bonk", unamb[:2000]),
                    ("random transition", rnd)]:
        igf, igc = ig_for(tr)
        print(f"[{tag:18s}] IG@full-obs: mean={igf.mean():.3f} median={np.median(igf):.3f} "
              f"p90={np.percentile(igf,90):.3f}  |  IG@break-cell mean={igc.mean():.3f}  n={len(igf)}")
    print("\n  expect AMBIGUOUS bonk IG > 0; unambiguous bonk & random ~0")


if __name__ == "__main__":
    main()
