"""Pure-numpy transition-graph helpers (predecessor adjacency + backward-path
sampling + ancestor-closed subsampling).

Extracted from ``nca_wm.data_collection`` so torch-only consumers (the belief
models, via ``nca_wm.recurrent_data``) can use them without importing
``data_collection`` and its jax / puzzlescript_cpp / puzzlescript_jax chain.
``data_collection`` re-exports these names for backward compatibility.
"""
from __future__ import annotations

import numpy as np


def build_predecessor_adjacency(states_packed, next_packed):
    """Full predecessor adjacency over a transition set (Phase 1, no cache
    change). Each transition i is an edge ``state[i] --action--> next[i]``;
    its predecessors are the rows j whose ``next[j] == state[i]`` (the
    transitions that produced i's current state).

    Non-injective dynamics mean a state can have several predecessors, so
    this returns a *set* per row (a list), not a single canonical parent —
    backward path sampling later picks among them, which decorrelates the
    history from the current state.

    No-op transitions (``next[j] == state[j]``, e.g. moving into a wall) are
    excluded as predecessor *edges*: chaining through them would make a state
    its own predecessor (an infinite self-loop). They remain valid target
    transitions; they just can't serve as a "previous" step.

    Args:
        states_packed: (N, ...) array; each row hashed by its raw bytes.
        next_packed:   (N, ...) array, same layout as states_packed.

    Returns:
        list of length N; entry i is an int64 ndarray of predecessor row
        indices for transition i (possibly empty — an episode start, or a
        state whose predecessors were dropped by the write-time cache cap).
    """
    n = len(states_packed)
    state_bytes = [states_packed[i].tobytes() for i in range(n)]
    next_bytes = [next_packed[i].tobytes() for i in range(n)]
    by_next: dict[bytes, list[int]] = {}
    for j in range(n):
        if next_bytes[j] == state_bytes[j]:
            continue  # no-op edge — never a meaningful predecessor
        by_next.setdefault(next_bytes[j], []).append(j)
    empty = np.empty(0, dtype=np.int64)
    return [np.asarray(by_next.get(state_bytes[i], empty), dtype=np.int64)
            for i in range(n)]


def sample_backward_paths(pred_lists, target_rows, k, rng):
    """Sample a length-k backward trajectory for each target transition.

    Walks predecessors from each target, picking one uniformly at random at
    each step (fresh every call — this is the augmentation that prevents the
    model from treating history as a state proxy). Stops early when a node has
    no surviving predecessor.

    Returns:
        hist_rows: (B, k) int64. Column k-1 is the immediate predecessor
        (most recent), column 0 the oldest — i.e. oldest→newest as the index
        grows, matching the model's history-channel convention. Missing steps
        (chain shorter than k, or a dead end) are -1, right-aligned so the
        real steps stay adjacent to the current transition.
        miss_mask: (B,) bool, True where the chain hit a dead end before
        filling all k slots (for hole-fraction logging).
    """
    B = len(target_rows)
    hist_rows = np.full((B, k), -1, dtype=np.int64)
    miss = np.zeros(B, dtype=bool)
    for b, t in enumerate(target_rows):
        cur = int(t)
        for step in range(k):
            preds = pred_lists[cur]
            if len(preds) == 0:
                miss[b] = True
                break
            cur = int(preds[rng.integers(len(preds))]) if hasattr(rng, "integers") \
                else int(preds[rng.randint(len(preds))])
            hist_rows[b, k - 1 - step] = cur
    return hist_rows, miss


def ancestor_closed_subsample(states, next_states, budget, seed=0):
    """Subsample a transition set to ~``budget`` rows that are
    ancestor-closed: every kept transition has at least one kept predecessor
    (a transition whose next_state equals this transition's state) all the way
    back to a root (episode-start) state.

    This replaces uniform subsampling for the --history pipeline: uniform
    subsampling drops predecessors and breaks the backward chains the history
    sampler walks, producing "holes" (masked history steps). Keeping whole
    predecessor chains instead guarantees the per-game adjacency built later is
    hole-free except at true episode-starts.

    Strategy: randomly order rows, then greedily keep each row plus one
    predecessor chain to its root until the budget is reached. Connected
    ancestor-closed subtrees, sampled uniformly at their leaves — preserves a
    representative spread of states (not depth-biased), unlike keeping a single
    shallow subtree.

    Args:
        states, next_states: (N, C, H, W) arrays (hashed by row bytes).
        budget: target number of kept rows (may overshoot slightly to finish
            the final chain).
        seed: RNG seed for the row ordering.

    Returns:
        int64 ndarray of kept row indices (sorted), length ~min(N, budget).
    """
    n = len(states)
    if budget is None or n <= budget:
        return np.arange(n, dtype=np.int64)
    sflat = states.reshape(n, -1)
    nflat = next_states.reshape(n, -1)
    sb = [sflat[i].tobytes() for i in range(n)]
    nb = [nflat[i].tobytes() for i in range(n)]
    # First predecessor row per state (deterministic); -1 if none. No-op edges
    # (next == state) are excluded so a state can't be its own predecessor.
    by_next: dict[bytes, int] = {}
    for j in range(n):
        if nb[j] == sb[j]:
            continue
        by_next.setdefault(nb[j], j)
    pred = np.fromiter((by_next.get(sb[i], -1) for i in range(n)),
                       dtype=np.int64, count=n)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    keep = np.zeros(n, dtype=bool)
    n_keep = 0
    for t in order:
        if n_keep >= budget:
            break
        cur = int(t)
        guard = 0
        while cur != -1 and not keep[cur]:
            keep[cur] = True
            n_keep += 1
            cur = int(pred[cur])
            guard += 1
            if guard > n:
                break
    return np.nonzero(keep)[0]
