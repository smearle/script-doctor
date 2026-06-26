"""Transition collection and multi-game dataset assembly for the NCA world model.

Solver-based transition collection (``collect_unique_transitions`` via the C++
backend's A*/BFS search), per-level and merged-dataset disk caches, the
predecessor-graph helpers used for transition-history sampling, and the
sprite-target builder. Pulled out of ``train.py`` so the data pipeline is a
self-contained unit; it imports only the dependency-free ``state_ops`` leaf
helpers, the tokenizer, and the C++ backend, never ``train.py``.
"""
import glob
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from nca_wm.tokenize_game import tokenize_game, get_game_tree_from_js
from nca_wm.state_ops import (
    _pack_states, _unpack_states, _pad_obs, _pad_packed, _dats_to_multihot_batch,
)

# Action-space size (mirrors train.py / models.py; data collection one-hots
# actions without importing the model code). Action ids:
#   0=up 1=left 2=down 3=right 4=action button 5=no-op real-time tick.
# Slot 5 is only ever active for real-time games (`realtime_interval`); for all
# other games it stays zero, so the one-hot width is a fixed 6 everywhere.
N_ACTIONS = 6


TRANSITIONS_CACHE_VERSION = 5

# Bounded per-level transition cache (on disk + in the per-level RAM array).
# Levels are cached up to this many transitions; the per-GAME train/val budgets
# are then water-filled across levels at dataset-assembly time (see
# collect_multigame_dataset). Keeps disk bounded while letting complex levels
# receive far more than an equal split when the per-game budget allows.
TRANSITIONS_CACHE_CAP = 200_000


def _water_fill(counts, budget: int):
    """Max-min fair allocation of an integer ``budget`` across bins with
    capacities ``counts``. Each round splits the remaining budget equally among
    not-yet-full bins, capping at each bin's capacity and recycling the leftover
    from bins that fill up — so a simple level's slack flows to complex levels
    instead of being wasted (unlike ``budget // n_levels``).

    Returns an int array ``alloc`` with ``alloc[i] <= counts[i]`` and
    ``sum(alloc) == min(budget, sum(counts))``.
    """
    counts = np.asarray(counts, dtype=np.int64)
    alloc = np.zeros(len(counts), dtype=np.int64)
    remaining = int(budget)
    active = counts > 0
    while remaining > 0 and active.any():
        share = remaining // int(active.sum())
        if share == 0:
            # Hand out the sub-bin remainder one at a time to bins with room.
            for j in np.where(active)[0]:
                if remaining <= 0:
                    break
                if alloc[j] < counts[j]:
                    alloc[j] += 1
                    remaining -= 1
            break
        progressed = False
        for j in np.where(active)[0]:
            give = min(share, int(counts[j] - alloc[j]))
            if give > 0:
                alloc[j] += give
                remaining -= give
                progressed = True
            if alloc[j] >= counts[j]:
                active[j] = False
        if not progressed:
            break
    return alloc


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


def _rollout_history(hist_buf, k, chw):
    """Build (hist_states, hist_actions) jnp tensors from a rolling list of
    recent (state, action) pairs for autoregressive rollout.

    hist_buf entries are (state ndarray (C,H,W), action int), most recent last.
    Returns (None, None) when k == 0. Otherwise (1, k, C, H, W) / (1, k) with
    the most recent step at column k-1 and zero-padded (masked) leading slots.
    """
    if k <= 0:
        return None, None
    C, H, W = (int(d) for d in chw)
    hs = np.zeros((1, k, C, H, W), dtype=np.float32)
    ha = np.zeros((1, k), dtype=np.int32)
    recent = hist_buf[-k:]
    base = k - len(recent)
    for idx, (st, ac) in enumerate(recent):
        hs[0, base + idx] = st
        ha[0, base + idx] = ac
    return jnp.array(hs), jnp.array(ha)


def _enabled_actions(json_str: str) -> list[int]:
    """Action ids a game actually allows, matching ``actionsForEngine()`` in
    puzzlescript_cpp/src/solver.cpp (the C++ search/transition collector):

      * 0-3 movement (always available),
      * 4 action button (dropped when the game declares ``noaction``),
      * 5 no-op real-time tick (added only for ``realtime_interval`` games,
        whose world advances on its own between key presses).

    Random-action eval samples from exactly this set so it never feeds an
    action the training data was never collected over (a disabled action key,
    or a no-op tick on a turn-based game). The set can be non-contiguous
    (``noaction`` + ``realtime_interval`` gives ``[0,1,2,3,5]``), so callers
    must sample *from this list*, not ``range(len(...))``."""
    try:
        meta = json.loads(json_str).get("metadata", {})
    except Exception:
        meta = {}
    actions = [0, 1, 2, 3]
    if "noaction" not in meta:
        actions.append(4)
    if "realtime_interval" in meta:
        actions.append(5)
    return actions


def _enabled_action_count(json_str: str) -> int:
    """Number of enabled actions; see ``_enabled_actions``. Use the list, not
    this count, when *sampling* — the id set can be non-contiguous."""
    return len(_enabled_actions(json_str))


# ---------------------------------------------------------------------------
# 1. Data collection with per-game cache
# ---------------------------------------------------------------------------
#
# Cache layout:
#   rollout_data/{game}/level_{i}/random.npz
#   rollout_data/{game}/level_{i}/search_{algo}_{budget}_{timeout}.npz
#
# Random cache stores episodes contiguously with an episode boundary index.
# If an experiment needs more episodes than cached, only the deficit is collected
# and appended. Search caches are keyed by (algo, budget, timeout) — if the file
# exists the data is reused as-is.

ROLLOUT_CACHE_DIR = "rollout_data"


def _cache_dir(game_name: str, level_i: int) -> str:
    return os.path.join(ROLLOUT_CACHE_DIR, game_name, f"level_{level_i}")


def _load_npz_dict(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    # Caches may be DEFLATE-compressed (savez_compressed); mmap requires
    # contiguous data so we drop it. Multihot states compress ~100x so
    # caches are usually small enough that mmap was overkill anyway.
    return np.load(path, allow_pickle=True)


def _save_npz_dict(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)


def _solution_from_sol_dir(
    sol_root: str,
    game_name: str,
    level_i: int,
    algos: tuple[str, ...] = ("astar", "bfs", "gbfs", "mcts"),
) -> list[int] | None:
    """Look for a pre-computed winning solution under
    `<sol_root>/<game_name>/<algo>_<budget>-steps_level-<level_i>.json` and
    return its action sequence. By default, preference order is astar > bfs >
    gbfs > mcts; within each algorithm, larger search budgets first. Pass
    `algos=(algo,)` when an evaluation cache is algorithm-specific.

    Both `data/cpp_sols/` and `data/js_sols/` store actions in the C++-backend
    action convention used by eval rollouts, so they are read as-is. (Verified
    empirically 2026-05-21: identity replay wins on 26/26 (game,level) js_sols
    across 14 games; the former JS→JAX remap corrupted already-correct actions.)
    """
    import glob
    import json
    game_dir = os.path.join(sol_root, game_name)
    if not os.path.isdir(game_dir):
        return None
    candidates: list[str] = []
    for prio_algo in algos:
        pattern = os.path.join(
            game_dir, f"{prio_algo}_*-steps_level-{level_i}.json"
        )
        # Sort by largest budget first (longer search ⇒ likelier to have won).
        files = sorted(glob.glob(pattern),
                        key=lambda p: int(re.search(r"_(\d+)-steps_", p).group(1))
                                       if re.search(r"_(\d+)-steps_", p) else 0,
                        reverse=True)
        candidates.extend(files)
    for path in candidates:
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        if not d.get("won"):
            continue
        actions = d.get("actions") or []
        if not actions:
            continue
        return [int(a) for a in actions]
    return None


def _solution_from_transitions_cache(
    game_name: str,
    level_i: int,
    search_algo: str | None = None,
) -> list[int] | None:
    """Look for an existing transitions cache that contains a winning
    trajectory for `(game_name, level_i)` and reconstruct the action sequence
    via the BFS-on-collected-edges helper from `synthetic_levels`. Returns
    None if no cache exists or no winning trajectory is reachable from the
    initial state in the collected edge set.

    Bypasses the per-eval search call when training already explored the
    goal. If `search_algo` is given, only caches produced by that algorithm
    are considered so we do not persist an A* trajectory as a BFS rollout.
    """
    import glob
    cache_dir = _cache_dir(game_name, level_i)
    if not os.path.isdir(cache_dir):
        return None
    candidates = []
    algos = (search_algo,) if search_algo is not None else ("astar", "bfs")
    for prio_algo in algos:
        for prio_cap in ("capall", "cap*"):
            pattern = os.path.join(
                cache_dir,
                f"{prio_algo}_transitions_v{TRANSITIONS_CACHE_VERSION}_*_{prio_cap}.npz",
            )
            candidates.extend(sorted(glob.glob(pattern)))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        d = _load_npz_dict(path)
        if d is None:
            continue
        wons = np.asarray(d.get("wons", []))
        if wons.size == 0 or not bool(wons.any()):
            continue
        # `_extract_bfs_solution` keys states as `tuple(int(x) for x in s)`,
        # which assumes 1-D rows. Cached states are bitpacked-multidim
        # `(N, C, H, packed_W)` — flatten the trailing axes so each "state"
        # becomes one 1-D byte sequence. Hash equality is preserved.
        states = np.asarray(d["states"])
        next_states = np.asarray(d["next_states"])
        actions = np.asarray(d["actions"]).tolist()
        wons_list = wons.tolist()
        n = len(actions)
        if n == 0:
            continue
        states_flat = states.reshape(n, -1)
        next_states_flat = next_states.reshape(n, -1)
        try:
            from nca_wm.synthetic_levels import _extract_bfs_solution
            chain = _extract_bfs_solution(
                list(states_flat), actions, list(next_states_flat), wons_list,
            )
        except Exception:
            chain = []
        if chain:
            return [int(a) for a in chain]
    return None


def collect_unique_transitions(
    json_str: str,
    game_name: str,
    level_i: int = 0,
    max_iters: int = 100_000,
    timeout_ms: int = -1,
    search_algo: str = "astar",
    max_transitions: int | None = None,
    ancestor_closed: bool = False,
) -> dict:
    """Collect unique transitions via C++ state-space exploration.

    Every (state, action, next_state) transition visited during search is returned.
    Uses A* (default) or BFS to explore the state space.

    Cached at rollout_data/{game}/level_{i}/{algo}_transitions_v{TRANSITIONS_CACHE_VERSION}_*.npz.

    `max_transitions` (per-level cap) caps the *cached* output: if the search
    finds more than that many unique transitions, we uniformly subsample down
    to the cap before caching. This bounds disk + in-RAM size when the same
    cache later gets loaded into the merged dataset.

    States are stored bitpacked along W (`np.packbits(axis=-1)`), giving ~8×
    reduction in both compressed and decompressed size. The original W is
    stored alongside so unpack count is known on load.
    """
    from puzzlescript_cpp._puzzlescript_cpp import (
        Engine, collect_transitions_bfs, collect_transitions_astar,
    )

    collector_fn = {
        "bfs": collect_transitions_bfs,
        "astar": collect_transitions_astar,
    }[search_algo]

    # v5: bitpacked states (axis=-1) + per-level write-time transition cap.
    # Ancestor-closed capping gets a distinct "ac" tag so its caches never
    # collide with the uniform-subsample caches the rest of the pipeline uses.
    if max_transitions is None:
        cap_tag = "all"
    elif ancestor_closed:
        cap_tag = f"ac{int(max_transitions)}"
    else:
        cap_tag = str(int(max_transitions))
    cache_dir = _cache_dir(game_name, level_i)
    cache_path = os.path.join(
        cache_dir,
        f"{search_algo}_transitions_v{TRANSITIONS_CACHE_VERSION}_{max_iters}_{timeout_ms}_cap{cap_tag}.npz",
    )
    # On cache lookup, fall back to any prior cache with the same algo / iters /
    # cap but a different `timeout_ms` — what's stored on disk is the resulting
    # transition set, and the cap is what bounds it. This lets you tighten the
    # per-level wallclock timeout without invalidating already-collected data.
    cached = _load_npz_dict(cache_path)
    if cached is None:
        import glob
        glob_pattern = os.path.join(
            cache_dir,
            f"{search_algo}_transitions_v{TRANSITIONS_CACHE_VERSION}_{max_iters}_*_cap{cap_tag}.npz",
        )
        for alt in sorted(glob.glob(glob_pattern)):
            if alt == cache_path:
                continue
            cached = _load_npz_dict(alt)
            if cached is not None and len(cached["states"]) > 0:
                print(f"  reusing cache from differing timeout: {os.path.basename(alt)}")
                break
            cached = None
    # Ancestor-closed reuse: if no ac-cache exists, a uniform cache that was
    # NOT capped (size < max_transitions) is byte-identical to what ac would
    # produce (no subsample happened), so reuse it instead of re-searching.
    # Only genuinely-capped (large) games then need the expensive re-collection.
    if cached is None and ancestor_closed and max_transitions is not None:
        import glob
        uni_pattern = os.path.join(
            cache_dir,
            f"{search_algo}_transitions_v{TRANSITIONS_CACHE_VERSION}_{max_iters}_*_cap*.npz",
        )
        import re as _re
        for alt in sorted(glob.glob(uni_pattern)):
            base = os.path.basename(alt)
            if "_capac" in base:
                continue  # skip other ac-caches
            alt_d = _load_npz_dict(alt)
            if alt_d is None:
                continue
            n_alt = len(alt_d["states"])
            # Only reuse if the alt cache holds the COMPLETE explored set (not
            # itself capped), so it is byte-identical to what ac would produce.
            # Parse the alt's own cap from its filename: "capall" = uncapped;
            # "cap{N}" = capped at N, complete iff it found fewer than N rows.
            m = _re.search(r"_cap(all|\d+)\.npz$", base)
            if m is None:
                continue
            alt_cap = m.group(1)
            complete = (alt_cap == "all") or (n_alt < int(alt_cap))
            # ...and it must fit the current cap (else ac would subsample it).
            if complete and 0 < n_alt <= int(max_transitions):
                cached = alt_d
                print(f"  ancestor-closed: reusing complete uniform cache "
                      f"({n_alt:,} rows, cap={alt_cap}) {base}")
                break
    if cached is not None and len(cached["states"]) > 0:
        n = len(cached["states"])
        print(f"  {search_algo} transitions: {n:,} from cache")
        return {
            "states": cached["states"], "next_states": cached["next_states"],
            "actions": cached["actions"], "wons": cached["wons"],
            "W": int(cached["W"]),
        }

    # Run C++ transition collector
    engine = Engine()
    engine.load_from_json(json_str)
    engine.load_level(level_i)

    result = collector_fn(engine, max_iters=max_iters, timeout_ms=timeout_ms)
    # `result.states` (and friends) are pybind11 property getters that
    # materialize a fresh Python list on every access — each call rebuilds
    # ~3M int objects, costing tens of ms. Pull each parallel array into
    # numpy ONCE so all downstream indexing is O(1).
    all_states = np.asarray(result.states, dtype=np.int32)
    all_next = np.asarray(result.next_states, dtype=np.int32)
    all_actions = np.asarray(result.actions, dtype=np.int32)
    all_wons = np.asarray(result.wons, dtype=np.uint8)
    n_trans = len(all_actions)
    raw_n_objs = len(result.id_dict)
    # Dedup to match CppPuzzleScriptEnv's observation_shape (which is what
    # every other path in the NCAWM pipeline uses). Without this, the
    # transition states report raw_n_objs channels while the probing env
    # reports canonical (deduped) count, producing shape mismatches.
    from puzzlescript_cpp import _build_dedup_maps
    canonical_ids, raw_to_canonical = _build_dedup_maps(result.id_dict)
    n_objs = len(canonical_ids)
    w, h = result.width, result.height

    print(f"  {search_algo} transitions: {n_trans:,} from {result.iterations:,} iterations "
          f"({result.time:.3f}s, timeout={result.timeout})")

    if n_trans == 0:
        empty_packed = _pack_states(np.empty((0, n_objs, h, w), dtype=np.uint8))
        empty = {
            "states": empty_packed,
            "actions": np.empty((0,), dtype=np.int32),
            "next_states": empty_packed,
            "wons": np.empty((0,), dtype=np.uint8),
            "W": np.int32(w),
        }
        _save_npz_dict(cache_path, empty)
        return {**empty, "W": int(w)}

    # Subsample at write-time to cap cache + downstream RAM. We pick the
    # indices once and apply to all parallel arrays (states / next_states /
    # actions / wons) so we never materialize the full multihot tensor for
    # transitions we'll throw away.
    if max_transitions is not None and n_trans > max_transitions:
        if ancestor_closed:
            # Keep whole predecessor chains so history backward-walks never hit
            # a subsample hole. Operates on the raw engine state words (hashed
            # by row bytes) before multihot conversion.
            keep = np.sort(ancestor_closed_subsample(
                all_states, all_next, int(max_transitions), seed=42 + level_i))
            print(f"    ancestor-closed capping {n_trans:,} → {len(keep):,} "
                  f"at write-time")
        else:
            rng = np.random.RandomState(42 + level_i)
            keep = np.sort(rng.choice(n_trans, size=int(max_transitions),
                                      replace=False))
            print(f"    capping {n_trans:,} → {len(keep):,} at write-time")
        keep_states = all_states[keep]
        keep_next = all_next[keep]
        keep_actions = all_actions[keep]
        keep_wons = all_wons[keep]
        n_trans = len(keep)
    else:
        keep_states = all_states
        keep_next = all_next
        keep_actions = all_actions
        keep_wons = all_wons

    # Convert bitpacked-uint32 (engine layout) → multihot uint8, then pack
    # along W. The intermediate multihot is the largest tensor that exists
    # at any point; capping above is what keeps it from blowing up RAM.
    states_mh = _dats_to_multihot_batch(
        keep_states, raw_n_objs, w, h, raw_to_canonical, n_objs)
    next_states_mh = _dats_to_multihot_batch(
        keep_next, raw_n_objs, w, h, raw_to_canonical, n_objs)
    states = _pack_states(states_mh)
    next_states = _pack_states(next_states_mh)
    del states_mh, next_states_mh
    actions = np.array(keep_actions, dtype=np.int32)
    wons = np.array(keep_wons, dtype=np.uint8)

    data = {
        "states": states, "actions": actions, "next_states": next_states,
        "wons": wons, "W": np.int32(w),
    }
    _save_npz_dict(cache_path, data)
    print(f"    Cached -> {cache_path} ({wons.sum():,} winning / {n_trans:,})")
    return {**data, "W": int(w)}


# Merged-dataset cache format version. Bump on any change to dataset
# layout (new keys, packed-array shape, etc.). Encoded into the cache
# filename so old-version files are easy to identify and evict.
DATASET_FORMAT_VERSION = 18  # v18: water-filled per-game budget + val carved from full explored set (per_game_val_idx)

# Skip writing the merged-dataset cache when total transition count
# is below this threshold. Tiny merged caches (sub-MB) save < 1s on
# reload but still pollute _merged/ — for the very small n_per_rule
# experiments (n=1..3) we don't bother.
MERGED_CACHE_MIN_TRANSITIONS = 50_000

# Cap the number of merged-dataset caches kept per host (LRU). Older
# caches are deleted after a fresh write. Keep enough for quick
# diagnostic re-launches across recent experiments without filling
# disk indefinitely.
MERGED_CACHE_KEEP_LAST = 10


def _dataset_cache_key(
    game_names: list[str],
    level_i: int | None,
    search_algo: str,
    n_search_steps: int,
    search_timeout_ms: int,
    encode_sprites: bool = False,
    max_transitions_per_game: int | None = None,
    train_levels: list[int] | None = None,
    val_frac: float = 0.0,
    ancestor_closed: bool = False,
    max_grid_dim: int | None = None,
) -> str:
    """Deterministic hash of all args that affect dataset contents."""
    import hashlib
    key = {
        "format_version": DATASET_FORMAT_VERSION,
        "games": sorted(game_names),
        "level": level_i,
        "train_levels": sorted(train_levels) if train_levels is not None else None,
        "search_algo": search_algo,
        "n_search_steps": n_search_steps,
        "search_timeout_ms": search_timeout_ms,
        "encode_sprites": encode_sprites,
        "max_transitions_per_game": max_transitions_per_game,
        "val_frac": round(float(val_frac), 6),
    }
    # Only perturb the key when ancestor-closed is on, so existing (uniform)
    # caches stay valid for the default path.
    if ancestor_closed:
        key["ancestor_closed"] = True
    if max_grid_dim is not None:
        key["max_grid_dim"] = int(max_grid_dim)
    blob = json.dumps(key, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _merged_cache_paths(merged_cache_dir: str, cache_hash: str
                         ) -> tuple[str, str]:
    """Return (dataset_npz_path, game_infos_pkl_path), both filename-
    prefixed with the current DATASET_FORMAT_VERSION so older versions
    are visible at a glance and easy to evict."""
    v = DATASET_FORMAT_VERSION
    return (
        os.path.join(merged_cache_dir, f"dataset_v{v}_{cache_hash}.npz"),
        os.path.join(merged_cache_dir, f"game_infos_v{v}_{cache_hash}.pkl"),
    )


def _evict_stale_format_caches(merged_cache_dir: str) -> int:
    """Delete merged-cache files that don't match the current
    DATASET_FORMAT_VERSION filename prefix. Returns count deleted.

    Catches both pre-versioning legacy files (`dataset_<hash>.npz` with
    no version prefix) and explicit older versions (`dataset_v16_*`)."""
    import glob
    if not os.path.isdir(merged_cache_dir):
        return 0
    keep_prefix_d = f"dataset_v{DATASET_FORMAT_VERSION}_"
    keep_prefix_i = f"game_infos_v{DATASET_FORMAT_VERSION}_"
    n_evict = 0
    for fp in glob.glob(os.path.join(merged_cache_dir, "dataset_*.npz")):
        if not os.path.basename(fp).startswith(keep_prefix_d):
            try:
                os.remove(fp); n_evict += 1
            except OSError:
                pass
    for fp in glob.glob(os.path.join(merged_cache_dir, "game_infos_*.pkl")):
        if not os.path.basename(fp).startswith(keep_prefix_i):
            try:
                os.remove(fp); n_evict += 1
            except OSError:
                pass
    return n_evict


def _lru_prune_merged_caches(merged_cache_dir: str,
                              keep_last: int = MERGED_CACHE_KEEP_LAST) -> int:
    """Keep only the `keep_last` most-recently-modified merged-cache
    file pairs (dataset_v* + game_infos_v*); delete older ones.
    Returns count deleted."""
    import glob
    if not os.path.isdir(merged_cache_dir):
        return 0
    pairs = []  # list of (mtime, dataset_path, infos_path) for each cache hash
    for fp in glob.glob(os.path.join(merged_cache_dir,
                                      f"dataset_v{DATASET_FORMAT_VERSION}_*.npz")):
        # Recover hash from filename
        base = os.path.basename(fp)
        prefix = f"dataset_v{DATASET_FORMAT_VERSION}_"
        suffix = ".npz"
        if not (base.startswith(prefix) and base.endswith(suffix)):
            continue
        h = base[len(prefix):-len(suffix)]
        infos = os.path.join(merged_cache_dir,
                              f"game_infos_v{DATASET_FORMAT_VERSION}_{h}.pkl")
        try:
            mt = os.path.getmtime(fp)
        except OSError:
            continue
        pairs.append((mt, fp, infos))
    # Sort newest-first; drop everything past keep_last
    pairs.sort(key=lambda x: x[0], reverse=True)
    to_delete = pairs[keep_last:]
    n_evict = 0
    for _mt, ds, infos in to_delete:
        for fp in (ds, infos):
            if os.path.isfile(fp):
                try:
                    os.remove(fp); n_evict += 1
                except OSError:
                    pass
    return n_evict


def collect_multigame_dataset(
    game_names: list[str],
    ps_parser,
    level_i: int | None = None,
    n_search_steps: int = 100_000,
    search_timeout_ms: int = -1,
    search_algo: str = "astar",
    encode_sprites: bool = False,
    max_transitions_per_game: int | None = None,
    train_levels: list[int] | None = None,
    val_frac: float = 0.0,
    history: int = 0,
    ancestor_closed: bool | None = None,
    max_grid_dim: int | None = None,
) -> tuple[dict, list[dict]]:
    """Collect padded transitions from multiple games via search-based unique-transition exploration.

    Args:
        level_i: If None (default), collect from all levels. If int, collect from that level only.
        max_transitions_per_game: per-game TRAIN budget, water-filled across the
            game's levels (a simple level's slack flows to complex levels —
            see ``_water_fill``), not an equal ``// n_levels`` split.
        val_frac: fraction held out for validation, carved from the FULL
            explored set per level (so val is representative of the true
            distribution, not of the capped train subsample). Held-out indices
            are returned as ``per_game_val_idx``; train() consumes them.

    Returns:
        dataset: merged dict with per-game packed states + ``per_game_val_idx``.
        game_infos: list of per-game metadata dicts.
    """
    # Ancestor-closed subsampling preserves history backward-chains (no holes);
    # default on when history>0. It changes the kept transition set, so it must
    # participate in the cache key.
    use_ac = ancestor_closed if ancestor_closed is not None else (history > 0)
    # Check for cached merged dataset (shared across experiments)
    cache_hash = _dataset_cache_key(
        game_names, level_i, search_algo,
        n_search_steps, search_timeout_ms,
        encode_sprites=encode_sprites,
        max_transitions_per_game=max_transitions_per_game,
        train_levels=train_levels,
        val_frac=val_frac,
        ancestor_closed=use_ac,
        max_grid_dim=max_grid_dim,
    )
    merged_cache_dir = os.path.join(ROLLOUT_CACHE_DIR, "_merged")
    # Auto-evict pre-versioning legacy files and explicit older versions.
    # Cheap (one stat per matching file) and prevents stale caches from
    # silently masking format-bump intent.
    n_evict = _evict_stale_format_caches(merged_cache_dir)
    if n_evict > 0:
        print(f"  evicted {n_evict} stale-format-version cache file(s) from {merged_cache_dir}")
    dataset_cache, infos_cache = _merged_cache_paths(merged_cache_dir, cache_hash)
    if os.path.isfile(dataset_cache) and os.path.isfile(infos_cache):
        print(f"Loading cached dataset from {dataset_cache}")
        t0 = time.time()
        data = np.load(dataset_cache, allow_pickle=True)
        # v7 cache stores per-game arrays under prefixed keys (per_game_states_{g})
        # plus shared scalars (max_C/H/W). Reconstruct into the per-game lists.
        raw = {k: data[k] for k in data.files}
        dataset = _unpack_v7_cache(raw)
        with open(infos_cache, "rb") as f:
            game_infos = pickle.load(f)
        n_games = len(game_infos)
        per_game_keys = (
            "per_game_states", "per_game_next_states", "per_game_actions",
            "per_game_wons", "per_game_tokens", "per_game_masks",
            "per_game_sprites", "game_shapes", "per_game_n_transitions",
            "per_game_transition_shapes",
        )
        cache_valid = all(len(dataset.get(k, [])) == n_games for k in per_game_keys)
        if not cache_valid:
            print(
                "  cached dataset/game_infos length mismatch; ignoring merged "
                "cache and rebuilding"
            )
        else:
            n_total = sum(len(s) for s in dataset["per_game_states"])
            print(f"  {n_total:,} transitions loaded in {time.time()-t0:.1f}s")
            return dataset, game_infos

    # First pass: compile all games and get shapes (max across all levels)
    game_infos = []
    skipped_games = []
    for name in game_names:
        print(f"\nCompiling {name}...")
        backend = CppPuzzleScriptBackend()
        try:
            json_str = backend.compile_and_serialize(ps_parser, name)
            env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
        except Exception as e:
            print(f"  SKIP {name}: compile/env init failed ({e})")
            skipped_games.append(name)
            continue
        n_objs = env0.observation_shape[0]
        n_levels = env0.num_levels
        if n_levels < 1:
            print(f"  SKIP {name}: 0 playable levels")
            skipped_games.append(name)
            continue
        # Get max dims across all levels. Some games have different object
        # counts per level (e.g. level-specific sprites unused elsewhere), so
        # track max n_objs too — level-0 alone isn't a reliable ceiling.
        game_max_H, game_max_W = 0, 0
        for li in range(n_levels):
            env_li = CppPuzzleScriptEnv(json_str, level_i=li, max_episode_steps=10)
            lC, lH, lW = env_li.observation_shape
            n_objs = max(n_objs, lC)
            game_max_H = max(game_max_H, lH)
            game_max_W = max(game_max_W, lW)
        # Tokenize game spec
        try:
            tree, canonical_ids = get_game_tree_from_js(ps_parser, name)
            token_ids = tokenize_game(tree, canonical_ids,
                                       encode_sprites=encode_sprites)
        except Exception as e:
            print(f"  WARNING: tokenization failed ({e}), using empty tokens")
            tree, canonical_ids = None, None
            token_ids = []

        # Build per-game target sprite tensor (n_objs, 5, 5, 4) RGBA uint8,
        # used as the target if a sprite-decoder head is active.
        sprite_tensor = None
        if tree is not None and canonical_ids is not None:
            try:
                sprite_tensor = _build_sprite_tensor(tree, canonical_ids)
            except Exception as e:
                print(f"  WARNING: sprite tensor build failed ({e})")

        print(f"  n_objs={n_objs}, max_shape=({game_max_H}, {game_max_W}), "
              f"n_levels={n_levels}, n_tokens={len(token_ids)}")
        game_infos.append({
            "name": name,
            "json_str": json_str,
            "n_objs": n_objs,
            "H": game_max_H,
            "W": game_max_W,
            "n_levels": n_levels,
            "token_ids": token_ids,
            "sprite_tensor": sprite_tensor,   # (n_objs, 5, 5, 4) uint8 or None
        })

    if not game_infos:
        raise RuntimeError(
            "No games compiled successfully; cannot build a multi-game dataset."
        )

    # Preliminary max (will be recomputed after collection, since per-game
    # refinement can push n_objs up when the transition collector reports
    # more objects than the env probe).
    max_C = max(g["n_objs"] for g in game_infos)
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    print(f"\nPreliminary padded shape (may refine during collection): "
          f"({max_C}, {max_H}, {max_W})")

    # Second pass: collect per-game at each game's OWN (game_n_objs, game_H,
    # game_W) — no global padding. Saves ~10x RAM when game sizes vary.
    per_game_states = []
    per_game_actions = []
    per_game_next_states = []
    per_game_wons = []
    per_game_transition_shapes = []
    per_game_val_idx = []   # list of (n_val_g,) int32 — held-out indices per game
    collected_game_infos = []

    for game_id, info in enumerate(game_infos):
        name = info["name"]
        json_str = info["json_str"]
        if train_levels is not None:
            levels = [li for li in train_levels if 0 <= li < info["n_levels"]]
        elif level_i is not None:
            levels = [level_i]
        else:
            levels = list(range(info["n_levels"]))
        print(f"\n[{game_id+1}/{len(game_infos)}] Collecting data for {name} "
              f"({len(levels)} level{'s' if len(levels) != 1 else ''})...")

        g_C = info["n_objs"]
        g_H = info["H"]
        g_W = info["W"]

        game_states, game_actions, game_next_states, game_wons = [], [], [], []
        game_transition_shapes = []
        # Collect once, then refine shapes from what the collector actually
        # returned. Transition collector's id_dict can include more objects
        # than the env-probe reported (objects added by rules/legend not
        # present at level spawn), so we re-pad all levels at the end once
        # we know the true max.
        #
        # Each level is cached up to TRANSITIONS_CACHE_CAP (bounded disk/RAM);
        # the per-GAME train/val budgets are water-filled across levels below,
        # so a complex level can draw far more than an equal split.
        per_level_cap = TRANSITIONS_CACHE_CAP
        raw_levels = []
        for li in levels:
            print(f"  Level {li}:")
            # Per-level cap is applied at write-time inside the cache so the
            # disk file size and the in-RAM packed array are both bounded.
            level_data = collect_unique_transitions(
                json_str, name, level_i=li,
                max_iters=n_search_steps,
                timeout_ms=search_timeout_ms,
                search_algo=search_algo,
                max_transitions=per_level_cap,
                ancestor_closed=use_ac,
            )
            # Refine per-game max shape from actual collected data. Transition
            # collector may report more objects than the env-probe did.
            # `level_data["states"]` is packed (N, lC, lH, ceil(lW/8));
            # the original lW lives in level_data["W"].
            if len(level_data["states"]) > 0:
                _, lC, lH, _ = level_data["states"].shape
                lW = int(level_data["W"])
                g_C = max(g_C, lC)
                g_H = max(g_H, lH)
                g_W = max(g_W, lW)
            raw_levels.append(level_data)

        # Second pass: water-fill the per-game train + val budgets across this
        # game's levels (a simple level's slack flows to complex levels), carve
        # val from the FULL explored set per level, then pad the selected rows
        # to the refined per-game max and concat. Per-game array layout: each
        # level contributes its train rows then its val rows; val positions are
        # recorded (global indices into the concatenated per-game array).
        level_counts = [len(ld["states"]) for ld in raw_levels]
        total_explored = int(sum(level_counts))
        if total_explored == 0:
            print(f"  {name}: skipping — no transitions across any level")
            continue
        # Enforce the grid-size cap on the ACTUAL collected dims (the
        # games_metadata `max_level_area` field is unreliable — it records max
        # level *height*, not area, so wide grids slip past the upstream
        # selection filter). max_grid_dim caps the longest axis (H and W),
        # bounding NCA compute + bucket padding. Skip oversized games here so
        # the cap is actually honored regardless of stale metadata.
        if max_grid_dim is not None and max(g_H, g_W) > max_grid_dim:
            print(f"  {name}: skipping — grid {g_H}x{g_W} exceeds "
                  f"max_grid_dim={max_grid_dim}")
            continue
        train_budget = (max_transitions_per_game if max_transitions_per_game
                        is not None else total_explored)
        rng = np.random.RandomState(42 + game_id)

        # Per-level selection. Two modes:
        #  - uniform (default): hold out val_frac of each level's full explored
        #    set, then water-fill the per-game TRAIN budget over the remainder.
        #  - ancestor-closed (history): water-fill a per-level keep budget over
        #    the FULL level counts, select an ancestor-closed set of that size
        #    (whole predecessor chains, so history backward-walks never hit a
        #    subsample hole), then carve val from within the kept set.
        if use_ac:
            keep_alloc = _water_fill(level_counts, train_budget)
            train_pool_idx = [None] * len(raw_levels)
            val_pick = [None] * len(raw_levels)
            for i, ld in enumerate(raw_levels):
                if level_counts[i] == 0:
                    train_pool_idx[i] = np.empty(0, np.int64)
                    val_pick[i] = np.empty(0, np.int64)
                    continue
                kept = ancestor_closed_subsample(
                    ld["states"], ld["next_states"], int(keep_alloc[i]),
                    seed=42 + game_id + i)
                permk = rng.permutation(len(kept))
                vi = (max(0, min(len(kept) - 1, int(round(len(kept) * val_frac))))
                      if val_frac > 0 else 0)
                val_pick[i] = kept[permk[:vi]]
                train_pool_idx[i] = kept[permk[vi:]]
            train_alloc = np.array([len(p) for p in train_pool_idx], dtype=np.int64)
        else:
            # Hold out val_frac of EACH level's full explored set (a uniform
            # per-level fraction — representative of the true distribution and
            # never consuming a whole small level), then water-fill the per-game
            # TRAIN budget over what remains so complex levels draw more.
            train_pool_idx, val_pick = [], []
            for i in range(len(raw_levels)):
                ci = level_counts[i]
                if ci == 0:
                    train_pool_idx.append(np.empty(0, np.int64))
                    val_pick.append(np.empty(0, np.int64))
                    continue
                vi = max(0, min(ci - 1, int(round(ci * val_frac)))) if val_frac > 0 else 0
                perm = rng.permutation(ci)
                val_pick.append(perm[:vi])
                train_pool_idx.append(perm[vi:])
            train_alloc = _water_fill([len(p) for p in train_pool_idx], train_budget)

        val_positions, offset = [], 0
        for i, ld in enumerate(raw_levels):
            if level_counts[i] == 0:
                continue
            train_sel = train_pool_idx[i][:int(train_alloc[i])]
            val_sel = val_pick[i]
            sel = np.concatenate([train_sel, val_sel]).astype(np.int64)
            if len(sel) == 0:
                continue
            lW = int(ld["W"])
            game_states.append(_pad_packed(ld["states"][sel], lW, g_C, g_H, g_W))
            game_next_states.append(_pad_packed(ld["next_states"][sel], lW, g_C, g_H, g_W))
            game_actions.append(np.asarray(ld["actions"])[sel])
            game_wons.append(np.asarray(ld["wons"], dtype=np.uint8)[sel])
            game_transition_shapes.append(np.tile(
                np.array([[ld["states"].shape[1], ld["states"].shape[2], lW]],
                         dtype=np.int32), (len(sel), 1)))
            # val rows are the tail of this level's block
            val_positions.append(
                offset + len(train_sel) + np.arange(len(val_sel), dtype=np.int64))
            offset += len(sel)

        if not game_states:
            print(f"  {name}: skipping — no transitions selected")
            continue

        # Record refined shape so game_infos is accurate for later code paths.
        info["n_objs"] = g_C
        info["H"] = g_H
        info["W"] = g_W

        states = np.concatenate(game_states)        # packed: (N, g_C, g_H, ceil(g_W/8))
        actions = np.concatenate(game_actions)
        next_states = np.concatenate(game_next_states)
        wons = np.concatenate(game_wons)
        transition_shapes = np.concatenate(game_transition_shapes)
        val_idx_g = (np.concatenate(val_positions) if val_positions
                     else np.empty(0, np.int64)).astype(np.int32)
        n_trans = len(states)

        info["n_transitions"] = n_trans
        info["n_wins"] = int(wons.sum())
        collected_game_infos.append(info)

        per_game_states.append(states)
        per_game_actions.append(actions)
        per_game_next_states.append(next_states)
        per_game_wons.append(wons)
        per_game_transition_shapes.append(transition_shapes)
        per_game_val_idx.append(val_idx_g)

        changed = (states != next_states).any(axis=(1, 2, 3))
        print(f"  {name}: {n_trans:,} kept ({n_trans - len(val_idx_g):,} train + "
              f"{len(val_idx_g):,} val) of {total_explored:,} explored over "
              f"{len(levels)} levels; {changed.sum():,} changed "
              f"({100*changed.mean():.1f}%), {wons.sum():,} winning "
              f"({100*wons.mean():.3f}%), shape=({g_C}, {g_H}, {g_W})")

    game_infos = collected_game_infos
    if not game_infos:
        raise RuntimeError(
            "No transitions collected for any compiled game; cannot build a "
            "multi-game dataset."
        )

    # Recompute global max shape after per-game refinement.
    max_C = max((g.get("n_objs", 0) for g in game_infos), default=1)
    max_H = max((g.get("H", 0) for g in game_infos), default=1)
    max_W = max((g.get("W", 0) for g in game_infos), default=1)
    print(f"\nFinal padded shape after collection: ({max_C}, {max_H}, {max_W})")

    # Pad token sequences to common length and expand to per-transition
    max_tok_len = max(len(g["token_ids"]) for g in game_infos)
    max_tok_len = max(max_tok_len, 1)  # at least 1
    per_game_tokens = []
    per_game_masks = []
    for info in game_infos:
        tids = info["token_ids"]
        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[:len(tids)] = tids
        mask[:len(tids)] = True
        per_game_tokens.append(padded)
        per_game_masks.append(mask)
    per_game_tokens = np.array(per_game_tokens)  # (n_games, max_tok_len)
    per_game_masks = np.array(per_game_masks)    # (n_games, max_tok_len)

    # Per-game sprite tensors padded to (max_C, 5, 5, 4). One tensor per
    # game; stays game-scoped so we gather by game_id when needed at
    # training time (not expanded per-transition).
    per_game_sprites = np.zeros((len(game_infos), max_C, 5, 5, 4), dtype=np.uint8)
    for gi, info in enumerate(game_infos):
        st = info.get("sprite_tensor")
        if st is None:
            continue
        n = min(max_C, st.shape[0])
        per_game_sprites[gi, :n] = st[:n]

    # Per-game native storage. Model consumers pad to (max_C, max_H, max_W)
    # per batch via `_gather_batch` so batch shape stays constant for JIT.
    game_shapes = np.array(
        [(info["n_objs"], info["H"], info["W"]) for info in game_infos],
        dtype=np.int32,
    )
    per_game_n_transitions = np.array(
        [len(s) for s in per_game_states], dtype=np.int64,
    )
    merged = {
        # Per-game lists of packed states: (N_g, C_g, H_g, ceil(W_g/8)) uint8.
        # Unpack with np.unpackbits(..., axis=-1, count=W_g) at sample time.
        "per_game_states": per_game_states,
        "per_game_next_states": per_game_next_states,
        "per_game_actions": per_game_actions,       # list of (N_g,) int32
        "per_game_wons": per_game_wons,             # list of (N_g,) uint8
        "per_game_transition_shapes": per_game_transition_shapes,  # list of (N_g,3) real C,H,W
        # Held-out val indices per game (carved from the FULL explored set, not
        # the capped train subsample) — consumed by train()'s val split.
        "per_game_val_idx": per_game_val_idx,       # list of (n_val_g,) int32
        "per_game_tokens": per_game_tokens,         # (n_games, max_tok_len) int32
        "per_game_masks": per_game_masks,           # (n_games, max_tok_len) bool
        "per_game_sprites": per_game_sprites,       # (n_games, max_C, 5, 5, 4) uint8
        "game_shapes": game_shapes,                 # (n_games, 3) int32
        "per_game_n_transitions": per_game_n_transitions,
        "max_C": int(max_C),
        "max_H": int(max_H),
        "max_W": int(max_W),
        # Expose a flat game_ids for code paths that still want per-transition
        # game_id labels. Concatenated in game order.
        "game_ids": np.concatenate(
            [np.full(len(s), g, dtype=np.int32)
             for g, s in enumerate(per_game_states)],
            axis=0,
        ) if per_game_states else np.empty((0,), dtype=np.int32),
        # Back-compat alias used by eval/summary prints; treat as scalar total.
        # DO NOT index into this — use per_game_states for actual data access.
        "game_tokens": per_game_tokens,   # (n_games, max_tok_len); gather at batch time
        "game_masks": per_game_masks,
    }
    n_total = int(per_game_n_transitions.sum())
    n_wins_tot = int(sum(int(w.sum()) for w in per_game_wons))
    print(f"\nTotal multi-game dataset: {n_total:,} transitions "
          f"from {len(game_infos)} games, per-game native shapes "
          f"(global max=({max_C}, {max_H}, {max_W})), max_tokens={max_tok_len}, "
          f"{n_wins_tot:,} winning ({100*n_wins_tot/max(1,n_total):.3f}%)")

    # Cache the merged dataset (shared across experiments) — but skip
    # the npz write for tiny presets (mostly the n_per_rule_games=1..3
    # case) where the merged file is sub-MB and rebuilding from per-game
    # caches takes <1s anyway. The game_infos pkl is always cached
    # because it captures the slow JS-engine compile step.
    os.makedirs(merged_cache_dir, exist_ok=True)
    if n_total < MERGED_CACHE_MIN_TRANSITIONS:
        print(f"Caching game_infos only (n_total={n_total:,} < "
              f"{MERGED_CACHE_MIN_TRANSITIONS:,}; merged dataset npz "
              f"skipped to keep _merged/ tidy)")
        t0 = time.time()
        with open(infos_cache, "wb") as f:
            pickle.dump(game_infos, f)
        print(f"  Cached in {time.time()-t0:.1f}s")
    else:
        print(f"Caching dataset to {dataset_cache}")
        t0 = time.time()
        np.savez(dataset_cache, **_pack_v7_cache(merged))
        with open(infos_cache, "wb") as f:
            pickle.dump(game_infos, f)
        print(f"  Cached in {time.time()-t0:.1f}s")
        # LRU-prune after write so the disk footprint stays bounded.
        n_pruned = _lru_prune_merged_caches(merged_cache_dir,
                                             keep_last=MERGED_CACHE_KEEP_LAST)
        if n_pruned > 0:
            print(f"  LRU-pruned {n_pruned} old merged-cache file(s) "
                  f"(keeping last {MERGED_CACHE_KEEP_LAST})")

    return merged, game_infos


def collect_multigame_dataset_synthetic(
    game_names: list[str],
    ps_parser,
    *,
    n_levels: int,
    width: int,
    height: int,
    seed: int,
    mode: str,
    require_solvable: bool,
    max_attempts_per_level: int,
    max_iters_search: int,
    timeout_ms_search: int,
    min_states: int,
    evolve_pop_size: int = 64,
    evolve_max_generations: int = 200,
    evolve_n_mutations_min: int = 1,
    evolve_n_mutations_max: int = 3,
    encode_sprites: bool = False,
    kernel_sep: bool = False,
    max_transitions_per_game: int | None = None,
    per_game_size: bool = False,
    multi_grid: bool = False,
    grid_sizes: list[tuple[int, int]] | None = None,
    fallback_dynamics: bool = False,
    no_a_count_max: int = 3,
    track_rules_fired: bool = False,
    rule_coverage_weight: float = 0.0,
    coverage_select_topk: bool = False,
    seed_from_authored: bool = False,
    seed_level_indices: list[int] | None = None,
    selection: str = "fitness",
    nslc_k: int = 5,
    nslc_archive_size: int = 500,
    history: int = 0,
    ancestor_closed: bool | None = None,
) -> tuple[dict, list[dict]]:
    """Synthetic-level variant of collect_multigame_dataset.

    Generates ``n_levels`` valid synthetic levels per game (via
    nca_wm.synthetic_levels.collect_synthetic_dataset) and assembles the
    same per-game merged dataset shape used by the rest of the pipeline.

    Game-spec tokenization and sprite tensors come from the same code
    paths as the authored-data variant, so downstream training is
    identical.
    """
    from nca_wm.synthetic_levels import collect_synthetic_dataset

    game_infos: list[dict] = []
    per_game_states: list[np.ndarray] = []
    per_game_actions: list[np.ndarray] = []
    per_game_next_states: list[np.ndarray] = []
    per_game_wons: list[np.ndarray] = []
    per_game_transition_shapes: list[np.ndarray] = []

    for game_id, name in enumerate(game_names):
        print(f"\n[synth {game_id+1}/{len(game_names)}] {name}")
        # Compile + tokenize game spec (identical to authored path)
        backend = CppPuzzleScriptBackend()
        try:
            json_str = backend.compile_and_serialize(ps_parser, name)
            env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
        except Exception as e:
            print(f"  SKIP {name}: compile/env init failed ({e})")
            continue
        # Per-game grid sizing.
        # Priority: (1) explicit --grid_sizes overrides everything (recommended
        # for multi_grid since authored sizes can be huge — TSP 20x19, kettle
        # 15x15 — and synth at those defeats the purpose of avoiding big-grid
        # BFS). (2) --multi_grid uses unique authored sizes. (3) --per_game_size
        # picks max authored dim. (4) Fallback: global --synthetic_w/h.
        sizes_to_gen: list[tuple[int, int]] = [(int(width), int(height))]
        if grid_sizes:
            sizes_to_gen = list(grid_sizes)
            print(f"  grid_sizes: {name} → {sizes_to_gen} (explicit override)")
        elif per_game_size or multi_grid:
            try:
                authored_dims: set[tuple[int, int]] = set()
                for li in range(int(env0.num_levels)):
                    env_li = CppPuzzleScriptEnv(json_str, level_i=li, max_episode_steps=10)
                    _, lh, lw = env_li.observation_shape
                    authored_dims.add((int(lw), int(lh)))
                if authored_dims:
                    if multi_grid:
                        sizes_to_gen = sorted(authored_dims)
                        print(f"  multi_grid: {name} → {sizes_to_gen} ({len(sizes_to_gen)} unique authored sizes)")
                    else:
                        # per_game_size: pick max-dim
                        max_w = max(w for w, h in authored_dims)
                        max_h = max(h for w, h in authored_dims)
                        sizes_to_gen = [(max_w, max_h)]
                        print(f"  per_game_size: {name} → ({max_w}, {max_h}) (authored max-dim)")
            except Exception as e:
                print(f"  per_game_size detection failed for {name} ({e}); using global ({width}, {height})")
        try:
            tree, canonical_ids = get_game_tree_from_js(ps_parser, name)
            token_ids = tokenize_game(
                tree, canonical_ids,
                encode_sprites=encode_sprites,
            )
        except Exception as e:
            print(f"  WARNING: tokenization failed ({e}), using empty tokens")
            tree, canonical_ids = None, None
            token_ids = []

        sprite_tensor = None
        if tree is not None and canonical_ids is not None:
            try:
                sprite_tensor = _build_sprite_tensor(tree, canonical_ids)
            except Exception as e:
                print(f"  WARNING: sprite tensor build failed ({e})")

        # Collect synthetic transitions at each size (single size when not
        # using --multi_grid). Per-size n_levels is divided across the sizes
        # so total levels-per-game stays at n_levels.
        per_size_n = max(1, n_levels // max(1, len(sizes_to_gen)))
        synth_parts = []
        for s_w, s_h in sizes_to_gen:
            synth = collect_synthetic_dataset(
                game_name=name,
                n_levels=per_size_n, width=s_w, height=s_h,
                seed=seed,
                max_iters_search=max_iters_search,
                timeout_ms_search=timeout_ms_search,
                min_states=min_states,
                mode=mode,
                require_solvable=require_solvable,
                max_attempts_per_level=max_attempts_per_level,
                evolve_pop_size=evolve_pop_size,
                evolve_max_generations=evolve_max_generations,
                evolve_n_mutations_min=evolve_n_mutations_min,
                evolve_n_mutations_max=evolve_n_mutations_max,
                seed_from_authored=seed_from_authored,
                seed_level_indices=seed_level_indices,
                fallback_dynamics=fallback_dynamics,
                no_a_count_max=no_a_count_max,
                track_rules_fired=track_rules_fired,
                rule_coverage_weight=rule_coverage_weight,
                coverage_select_topk=coverage_select_topk,
                selection=selection,
                nslc_k=nslc_k,
                nslc_archive_size=nslc_archive_size,
                verbose=True,
            )
            s_states = np.asarray(synth["states"], dtype=np.uint8)
            if s_states.size == 0:
                continue
            synth_parts.append({
                "states": s_states,
                "next_states": np.asarray(synth["next_states"], dtype=np.uint8),
                "actions": np.asarray(synth["actions"], dtype=np.int32),
                "wons": np.asarray(synth["wons"], dtype=np.uint8),
            })
        if not synth_parts:
            print(f"  SKIP {name}: no transitions collected at any size")
            continue
        # Pad each part's spatial dims to per-game max and concatenate.
        max_C = max(p["states"].shape[1] for p in synth_parts)
        max_H = max(p["states"].shape[2] for p in synth_parts)
        max_W = max(p["states"].shape[3] for p in synth_parts)
        def _pad_part(arr):
            return _pad_obs(arr, max_C, max_H, max_W)
        states = np.concatenate([_pad_part(p["states"]) for p in synth_parts])
        next_states = np.concatenate([_pad_part(p["next_states"]) for p in synth_parts])
        actions = np.concatenate([p["actions"] for p in synth_parts])
        wons = np.concatenate([p["wons"] for p in synth_parts])
        transition_shapes = np.concatenate([
            np.tile(
                np.array([[p["states"].shape[1],
                           p["states"].shape[2],
                           p["states"].shape[3]]], dtype=np.int32),
                (len(p["states"]), 1),
            )
            for p in synth_parts
        ])
        if len(synth_parts) > 1:
            print(f"  multi_grid: {name} merged {len(synth_parts)} sizes "
                  f"→ ({max_C}, {max_H}, {max_W}); {len(states):,} total transitions")

        n_trans = len(states)
        use_ac = ancestor_closed if ancestor_closed is not None else (history > 0)
        if (max_transitions_per_game is not None
                and n_trans > max_transitions_per_game):
            if use_ac:
                # Ancestor-closed subsample so history backward-chains stay
                # intact (no holes). Keeps whole predecessor chains rather than
                # uniformly-dropped transitions.
                idx = ancestor_closed_subsample(
                    states, next_states, max_transitions_per_game,
                    seed=42 + game_id)
            else:
                rng = np.random.RandomState(42 + game_id)
                idx = rng.choice(n_trans, size=max_transitions_per_game,
                                 replace=False)
            states = states[idx]; actions = actions[idx]
            next_states = next_states[idx]; wons = wons[idx]
            transition_shapes = transition_shapes[idx]
            n_trans = len(states)

        _, g_C, g_H, g_W = states.shape
        per_game_states.append(_pack_states(states))
        per_game_actions.append(actions)
        per_game_next_states.append(_pack_states(next_states))
        per_game_wons.append(wons)
        per_game_transition_shapes.append(transition_shapes)

        # n_levels is the engine's authored-level count, not the synthetic
        # count: evaluation rolls out via the real engine and uses these
        # indices, so synthetic-trained models are evaluated on authored
        # levels as a held-out test (which is the whole point of this path).
        try:
            n_authored_levels = int(env0.num_levels)
        except Exception:
            n_authored_levels = 1
        info = {
            "name": name,
            "json_str": json_str,
            "n_objs": int(g_C),
            "H": int(g_H),
            "W": int(g_W),
            "n_levels": n_authored_levels,
            "n_synth_levels": int(n_levels),
            "token_ids": token_ids,
            "sprite_tensor": sprite_tensor,
            "n_transitions": int(n_trans),
            "n_wins": int(wons.sum()),
        }
        game_infos.append(info)
        print(f"  {name}: {n_trans:,} transitions, {int(wons.sum()):,} winning, "
              f"shape=({g_C}, {g_H}, {g_W})")

    if not game_infos:
        raise RuntimeError("Synthetic data collection produced no games. Check params.")

    max_C = max(g["n_objs"] for g in game_infos)
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)

    # Token padding (mirrors collect_multigame_dataset)
    max_tok_len = max(len(g["token_ids"]) for g in game_infos)
    max_tok_len = max(max_tok_len, 1)
    per_game_tokens = []
    per_game_masks = []
    for info in game_infos:
        tids = info["token_ids"]
        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[:len(tids)] = tids
        mask[:len(tids)] = True
        per_game_tokens.append(padded)
        per_game_masks.append(mask)
    per_game_tokens = np.array(per_game_tokens)
    per_game_masks = np.array(per_game_masks)

    per_game_sprites = np.zeros((len(game_infos), max_C, 5, 5, 4), dtype=np.uint8)
    for gi, info in enumerate(game_infos):
        st = info.get("sprite_tensor")
        if st is None:
            continue
        n = min(max_C, st.shape[0])
        per_game_sprites[gi, :n] = st[:n]

    game_shapes = np.array(
        [(info["n_objs"], info["H"], info["W"]) for info in game_infos],
        dtype=np.int32,
    )
    per_game_n_transitions = np.array(
        [len(s) for s in per_game_states], dtype=np.int64,
    )
    merged = {
        "per_game_states": per_game_states,
        "per_game_next_states": per_game_next_states,
        "per_game_actions": per_game_actions,
        "per_game_wons": per_game_wons,
        "per_game_transition_shapes": per_game_transition_shapes,
        "per_game_tokens": per_game_tokens,
        "per_game_masks": per_game_masks,
        "per_game_sprites": per_game_sprites,
        "game_shapes": game_shapes,
        "per_game_n_transitions": per_game_n_transitions,
        "max_C": int(max_C),
        "max_H": int(max_H),
        "max_W": int(max_W),
        "game_ids": np.concatenate(
            [np.full(len(s), g, dtype=np.int32)
             for g, s in enumerate(per_game_states)],
            axis=0,
        ) if per_game_states else np.empty((0,), dtype=np.int32),
        "game_tokens": per_game_tokens,
        "game_masks": per_game_masks,
    }
    n_total = int(per_game_n_transitions.sum())
    n_wins_tot = int(sum(int(w.sum()) for w in per_game_wons))
    print(f"\n[synth] Total: {n_total:,} transitions from {len(game_infos)} games, "
          f"per-game native shapes (global max=({max_C}, {max_H}, {max_W})), "
          f"max_tokens={max_tok_len}, "
          f"{n_wins_tot:,} winning ({100*n_wins_tot/max(1,n_total):.3f}%)")
    return merged, game_infos


# --------- v7 per-game cache pack/unpack helpers ---------
PER_GAME_LIST_KEYS = (
    "per_game_states", "per_game_next_states",
    "per_game_actions", "per_game_wons", "per_game_transition_shapes",
    "per_game_val_idx",
)


def _pack_v7_cache(merged: dict) -> dict:
    """Flatten per-game list-of-ndarray entries into prefixed keys for npz."""
    out = {}
    for k, v in merged.items():
        if k in PER_GAME_LIST_KEYS:
            out[f"_n_{k}"] = np.array(len(v), dtype=np.int32)
            for g, arr in enumerate(v):
                out[f"{k}__{g}"] = arr
        else:
            out[k] = v
    return out


def _unpack_v7_cache(raw: dict) -> dict:
    """Reverse of _pack_v7_cache: reconstruct per-game lists from prefixed keys."""
    out = {}
    for k, v in raw.items():
        if k.startswith("_n_") or "__" in k:
            continue
        out[k] = v
    for k in PER_GAME_LIST_KEYS:
        n_key = f"_n_{k}"
        if n_key in raw:
            n = int(raw[n_key])
            out[k] = [raw[f"{k}__{g}"] for g in range(n)]
    return out


def _build_sprite_tensor(tree, canonical_ids,
                          sprite_h: int = 5, sprite_w: int = 5) -> np.ndarray:
    """Resolve each canonical object's declared palette + sprite grid into
    an RGBA pixel tensor. Returns shape ``(n_objs, sprite_h, sprite_w, 4)``
    dtype uint8, where the 4th channel is alpha (0=transparent).

    Used as the training target when a sprite-decoder head is active.
    Missing or un-sprited objects get an all-zero (fully transparent) entry.
    """
    from puzzlescript_jax.colors import resolve_color_to_rgb
    n_objs = len(canonical_ids)
    out = np.zeros((n_objs, sprite_h, sprite_w, 4), dtype=np.uint8)
    for ch_i, name in enumerate(canonical_ids):
        obj = tree.objects.get(name) or tree.objects.get(name.lower())
        if obj is None:
            continue
        colors = obj.colors if obj.colors is not None else []
        sprite = obj.sprite if obj.sprite is not None else []
        if len(sprite) == 0 or len(colors) == 0:
            continue
        # Resolve each palette slot to RGBA (alpha=0 for transparent)
        palette_rgba = []
        for c in colors:
            rgb = resolve_color_to_rgb(c)
            if rgb is None:
                palette_rgba.append((0, 0, 0, 0))
            else:
                palette_rgba.append((rgb[0], rgb[1], rgb[2], 255))
        for r in range(min(sprite_h, len(sprite))):
            row = sprite[r]
            for c in range(min(sprite_w, len(row))):
                cell = str(row[c]).strip()
                if not cell or cell == "." or not cell.isdigit():
                    continue  # stays transparent
                d = int(cell)
                if 0 <= d < len(palette_rgba):
                    out[ch_i, r, c] = palette_rgba[d]
    return out
