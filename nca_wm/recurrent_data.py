"""Parser-free, jax-free dataset loading + trajectory infra.

This holds the numpy-only data plumbing shared by the recurrent (jax) world
model and the torch-only belief models. It is deliberately free of jax / optax /
flax / puzzlescript_cpp imports so a pure-torch consumer can ``import`` it (and
the helpers it needs) without dragging in the entire jax + C++ engine stack.
``train_recurrent`` re-exports these names for backward compatibility.

Splitting these out of ``train_recurrent`` is what lets e.g.
``nca_wm.active_learning.mario_nca_belief`` (torch only) run without building the
C++ backend or installing jax.
"""
from __future__ import annotations

import glob as _glob
import json
from pathlib import Path

import numpy as np

from nca_wm.state_ops import _unpack_states
from nca_wm.transition_graph import (
    build_predecessor_adjacency,
    sample_backward_paths,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Number of discrete actions (kept in sync with nca_wm.models.N_ACTIONS).
N_ACTIONS = 6


# ---------------------------------------------------------------------------
# Parser-free dataset loader (read cached transitions directly; no JS parse)
# ---------------------------------------------------------------------------
def load_dataset_from_caches(game_names, max_transitions_per_game, val_frac,
                             ancestor_closed, max_grid_dim, seed):
    """Build the dataset directly from cached A* transition npz files — NO JS
    parsing — for the recurrent (unconditional) model, which needs only the
    transition arrays + grid shape (not game tokens). This can't hang on
    pathological games (the parser-loop we hit) and scales to the whole cached
    corpus; RAM is governed by max_transitions_per_game * len(game_names).

    Per game: merge level caches (zero-pad to the game's max C/H/W — masked
    downstream), drop games whose grid exceeds max_grid_dim, ancestor-closed-
    subsample to the cap, split val_frac. Returns (dataset, game_infos) in the
    same format as collect_multigame_dataset.
    """
    import hashlib
    import pickle as _pickle
    from nca_wm.state_ops import _pack_states, _unpack_states
    from nca_wm.transition_graph import ancestor_closed_subsample
    cap = max_transitions_per_game
    # Disk cache so repeated runs (different k / arms) don't re-pay the slow
    # single-threaded subsample over the whole corpus.
    _ckey = hashlib.sha256(json.dumps({
        "games": sorted(game_names), "cap": cap,
        "val_frac": round(float(val_frac), 6), "ac": bool(ancestor_closed),
        "mgd": max_grid_dim, "seed": int(seed), "v": 1}, sort_keys=True
    ).encode()).hexdigest()[:16]
    _cpath = _REPO_ROOT / "rollout_data" / "_merged" / f"recurrent_ds_{_ckey}.pkl"
    if _cpath.is_file():
        try:
            with open(_cpath, "rb") as _f:
                _d, _gi = _pickle.load(_f)
            print(f"[load_caches] reused {_cpath.name}: {len(_gi)} games")
            return _d, _gi
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    per_states, per_next, per_actions, per_val = [], [], [], []
    game_infos = []
    n_skip_size = n_empty = 0
    for name in game_names:
        fs = sorted(_glob.glob(
            f"rollout_data/{name}/level_*/astar_transitions_*.npz"))
        # First pass: just shapes/W (cheap headers) to find the game's max dims.
        metas = []  # (path, C, H, W, N)
        for f in fs:
            try:
                with np.load(f, allow_pickle=True) as d:
                    sh = d["states"].shape
                    W = int(d["W"])
                if sh[0] > 0:
                    metas.append((f, sh[1], sh[2], W, sh[0]))
            except Exception:
                continue
        if not metas:
            n_empty += 1
            continue
        gC = max(m[1] for m in metas)
        gH = max(m[2] for m in metas)
        gW = max(m[3] for m in metas)
        if max(gH, gW) > max_grid_dim:
            n_skip_size += 1
            continue
        uniform = all((m[1], m[2], m[3]) == (gC, gH, gW) for m in metas)
        # Stay in PACKED space (8x smaller than unpacked) to bound RAM: concat
        # packed level arrays directly when uniform; only unpack-pad-repack the
        # (rare) mismatched levels, pre-capping huge ones to avoid a spike.
        Sp, Np, A = [], [], []
        for f, C, H, W, N in metas:
            with np.load(f, allow_pickle=True) as d:
                s = d["states"]
                ns = d["next_states"]
                a = np.asarray(d["actions"], dtype=np.int64)
            if not uniform:
                if cap and N > cap:
                    sel = rng.choice(N, cap, replace=False)
                    s, ns, a = s[sel], ns[sel], a[sel]
                pad = ((0, 0), (0, gC - C), (0, gH - H), (0, gW - W))
                s = _pack_states(np.pad(_unpack_states(s, W), pad).astype(np.uint8))
                ns = _pack_states(np.pad(_unpack_states(ns, W), pad).astype(np.uint8))
            Sp.append(s)
            Np.append(ns)
            A.append(a)
        Sp = np.concatenate(Sp)
        Np = np.concatenate(Np)
        A = np.concatenate(A)
        if cap and len(Sp) > cap:
            if ancestor_closed:
                keep = ancestor_closed_subsample(Sp, Np, cap, seed)
            else:
                keep = rng.choice(len(Sp), cap, replace=False)
            Sp, Np, A = Sp[keep], Np[keep], A[keep]
        n = len(Sp)
        n_val = int(round(val_frac * n))
        val_idx = np.sort(rng.permutation(n)[:n_val]).astype(np.int64)
        per_states.append(Sp)
        per_next.append(Np)
        per_actions.append(A)
        per_val.append(val_idx)
        game_infos.append({"name": name, "n_objs": gC, "H": gH, "W": gW})
    print(f"[load_caches] {len(game_infos)} games loaded; "
          f"skipped {n_skip_size} oversize, {n_empty} empty/missing")
    dataset = {"per_game_states": per_states,
               "per_game_next_states": per_next,
               "per_game_actions": per_actions,
               "per_game_val_idx": per_val}
    try:
        _cpath.parent.mkdir(parents=True, exist_ok=True)
        with open(_cpath, "wb") as _f:
            _pickle.dump((dataset, game_infos), _f, protocol=4)
        print(f"[load_caches] wrote {_cpath.name}")
    except Exception as _e:
        print(f"[load_caches] cache write skipped: {_e}")
    return dataset, game_infos


# ---------------------------------------------------------------------------
# Per-game trajectory infra
# ---------------------------------------------------------------------------
class GameData:
    """Holds the unpacked (or lazily-unpacked) per-game arrays + predecessor
    adjacency + train/val row partition for trajectory sampling."""

    def __init__(self, g, dataset, info):
        self.g = g
        self.name = info["name"]
        self.n_objs = int(info["n_objs"])
        self.H = int(info["H"])
        self.W = int(info["W"])
        self.states_packed = dataset["per_game_states"][g]
        self.next_packed = dataset["per_game_next_states"][g]
        self.actions = np.asarray(dataset["per_game_actions"][g], dtype=np.int64)
        self.n = len(self.states_packed)
        # Predecessor adjacency over PACKED rows (hashes raw bytes).
        self.pred_lists = build_predecessor_adjacency(
            self.states_packed, self.next_packed)
        # Train/val partition: val_idx are held-out rows.
        val_idx = np.asarray(dataset["per_game_val_idx"][g], dtype=np.int64) \
            if dataset.get("per_game_val_idx") is not None else np.empty(0, np.int64)
        is_val = np.zeros(self.n, dtype=bool)
        is_val[val_idx] = True
        self.val_rows = np.nonzero(is_val)[0]
        self.train_rows = np.nonzero(~is_val)[0]

    def unpack(self, packed_rows: np.ndarray) -> np.ndarray:
        """Unpack a (M, C, H, ceil(W/8)) packed array -> (M, C, H, W) float32."""
        return _unpack_states(packed_rows, self.W).astype(np.float32)


def build_trajectory_batch(gd: GameData, target_rows: np.ndarray, k: int,
                           rng, max_C: int, max_H: int, max_W: int):
    """Construct a teacher-forced trajectory batch ending at ``target_rows``.

    Returns (states, actions_onehot, targets, valid):
      states:        (B, L, max_C, max_H, max_W) float32 input frames
      actions_onehot:(B, L, N_ACTIONS) float32
      targets:       (B, L, max_C, max_H, max_W) float32 next-state targets
      valid:         (B, L) bool tick-validity (False at -1 holes)
    L = k + 1 (k history ticks + the current transition).

    Returns numpy arrays; callers convert to their framework of choice (torch
    for the belief models, jax for train_recurrent — jax.jit ingests numpy
    inputs transparently).
    """
    B = len(target_rows)
    L = k + 1
    hist_rows, _miss = sample_backward_paths(gd.pred_lists, target_rows, k, rng)
    rows_seq = np.concatenate([hist_rows, target_rows[:, None]], axis=1)  # (B, L)
    valid = rows_seq != -1                                                # (B, L)

    states = np.zeros((B, L, max_C, max_H, max_W), dtype=np.float32)
    targets = np.zeros((B, L, max_C, max_H, max_W), dtype=np.float32)
    actions = np.zeros((B, L), dtype=np.int64)

    C, H, W = gd.n_objs, gd.H, gd.W
    # Unpack only the rows we need (flatten unique rows for one unpack call).
    flat_rows = rows_seq.reshape(-1)
    real = flat_rows >= 0
    uniq = np.unique(flat_rows[real])
    s_unp = gd.unpack(gd.states_packed[uniq])      # (U, C, H, W)
    n_unp = gd.unpack(gd.next_packed[uniq])
    row_to_idx = {int(r): i for i, r in enumerate(uniq)}
    for b in range(B):
        for t in range(L):
            r = int(rows_seq[b, t])
            if r < 0:
                continue
            idx = row_to_idx[r]
            states[b, t, :C, :H, :W] = s_unp[idx][:C, :H, :W]
            targets[b, t, :C, :H, :W] = n_unp[idx][:C, :H, :W]
            actions[b, t] = gd.actions[r]

    actions_onehot = np.eye(N_ACTIONS, dtype=np.float32)[actions]   # (B, L, 5)
    actions_onehot = actions_onehot * valid[..., None].astype(np.float32)
    return (np.asarray(states), np.asarray(actions_onehot),
            np.asarray(targets), np.asarray(valid))
