"""Grid (tensor) data adapter for the NCA-belief model.

Reuses the engine + haoo' rollout layer but emits multihot (C,H,W) grids instead
of token sequences, with a per-step i.i.d. resample o'_t of each transition.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W


def obs_to_grid(masks: list[int]) -> np.ndarray:
    """Row-major canonical cell bitmasks -> (C,H,W) float32 multihot."""
    C, H, Wd = V.N_CANON, V.GRID_H, V.GRID_W
    m = np.asarray(masks, dtype=np.int64).reshape(H, Wd)
    bits = (m[None] >> np.arange(C)[:, None, None]) & 1   # (C,H,W)
    return bits.astype(np.float32)


def sample_grid_trajectory(family: W.WorldFamily, n_steps: int, rng: random.Random,
                           policy):
    """One fixed-length trajectory with per-step resamples.

    Returns obs (T+1,C,H,W), acts (T,) int action-index, resamp (T,C,H,W),
    mech (str). The trajectory continues along the FIRST draw; resamp[t] is an
    i.i.d. redraw of transition t from the same pre-action snapshot.
    """
    mech = rng.choice(family.mechanisms)
    li = rng.randrange(family.n_layouts)
    eng = W._new_engine(family.jsons[mech], li)
    id2b = W._engine_id_to_canon_bit(eng)
    obs = W.read_obs(eng, id2b)
    grids = [obs_to_grid(obs)]
    acts, resamps = [], []
    for _ in range(n_steps):
        a = policy(obs, rng, history_ids=[], engine=eng)
        ai = V.ACTIONS.index(a)
        s1, s2 = str(rng.getrandbits(40)), str(rng.getrandbits(40))
        bak = eng.backup_level()
        W.step_engine(eng, a, seed=s1); o1 = W.read_obs(eng, id2b)          # first draw
        eng.restore_level(bak); W.step_engine(eng, a, seed=s2)
        o2 = W.read_obs(eng, id2b)                                          # resample
        eng.restore_level(bak); W.step_engine(eng, a, seed=s1)             # continue on draw 1
        obs = o1
        grids.append(obs_to_grid(o1)); acts.append(ai); resamps.append(obs_to_grid(o2))
    return (np.stack(grids), np.asarray(acts, np.int64), np.stack(resamps), mech)


def batch(family, n, n_steps, rng, policy):
    """Stack n fixed-length trajectories into tensors."""
    O, A, R = [], [], []
    for _ in range(n):
        o, a, r, _ = sample_grid_trajectory(family, n_steps, rng, policy)
        O.append(o); A.append(a); R.append(r)
    return (torch.from_numpy(np.stack(O)), torch.from_numpy(np.stack(A)),
            torch.from_numpy(np.stack(R)))
