"""Train an NCA world model on a PuzzleScript game.

The NCA learns to predict the next game state given the current state and
player action: f(state_t, action_t) -> state_{t+1}.

Trajectories are collected from random rollouts and search (BFS/A*) via the
C++ PuzzleScript backend.

Usage (run from repo root):
    python nca_wm/train.py --game pipe_bend
    python nca_wm/train.py --games small --conditional --n_hid 128
"""
import argparse
import base64
import io
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so we can import the top-level backends
# (puzzlescript_cpp, puzzlescript_jax) while living under nca_wm/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import flax.linen as nn
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax

import wandb

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from puzzlescript_jax.utils import init_ps_lark_parser
from nca_wm.tokenize_game import (
    tokenize_game, get_game_tree_from_js,
    VOCAB_SIZE_EXT,
)

N_ACTIONS = 5


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


def _pack_states(arr: np.ndarray) -> np.ndarray:
    """Bitpack a (..., H, W) uint8 multihot along W → (..., H, ceil(W/8)) uint8.

    Pads the last axis up to a multiple of 8 with zeros (`np.packbits` does
    this implicitly; we just need to remember the original W to unpack).
    Yields ~8x reduction in both on-disk and in-RAM size for the per-game
    state arrays — needed to keep a gallery-scale dataset in RAM without
    OOMing the worker.
    """
    return np.packbits(np.ascontiguousarray(arr, dtype=np.uint8), axis=-1)


def _unpack_states(packed: np.ndarray, W: int) -> np.ndarray:
    """Reverse of `_pack_states` — returns (..., H, W) uint8."""
    return np.unpackbits(packed, axis=-1, count=W)


def _solution_from_sol_dir(sol_root: str, game_name: str, level_i: int,
                            translate_js_to_jax: bool = False) -> list[int] | None:
    """Look for a pre-computed winning solution under
    `<sol_root>/<game_name>/<algo>_<budget>-steps_level-<level_i>.json` and
    return its action sequence. Preference order: astar > bfs > gbfs > mcts;
    within each algorithm, larger search budgets first.

    Solutions under `data/cpp_sols/` use the same C++-backend action IDs as
    eval rollouts (no translation). Solutions under `data/js_sols/` use the
    JS-engine convention; pass `translate_js_to_jax=True` to remap to the
    JAX/CPP convention via the table in
    `.claude/projects/.../memory/reference_action_mappings.md`.
    """
    import glob
    import json
    game_dir = os.path.join(sol_root, game_name)
    if not os.path.isdir(game_dir):
        return None
    candidates: list[str] = []
    for prio_algo in ("astar", "bfs", "gbfs", "mcts"):
        pattern = os.path.join(
            game_dir, f"{prio_algo}_*-steps_level-{level_i}.json"
        )
        # Sort by largest budget first (longer search ⇒ likelier to have won).
        files = sorted(glob.glob(pattern),
                        key=lambda p: int(re.search(r"_(\d+)-steps_", p).group(1))
                                       if re.search(r"_(\d+)-steps_", p) else 0,
                        reverse=True)
        candidates.extend(files)
    # JS → JAX action-ID remap: js[0,1,2,3,4]=L,R,U,D,A → jax[0,2,3,1,4]=L,D,R,U,A.
    JS2JAX = [0, 2, 3, 1, 4]
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
        if translate_js_to_jax:
            try:
                actions = [JS2JAX[int(a)] for a in actions]
            except (IndexError, ValueError):
                continue
        return [int(a) for a in actions]
    return None


def _solution_from_transitions_cache(game_name: str, level_i: int) -> list[int] | None:
    """Look for an existing transitions cache that contains a winning
    trajectory for `(game_name, level_i)` and reconstruct the action sequence
    via the BFS-on-collected-edges helper from `synthetic_levels`. Returns
    None if no cache exists or no winning trajectory is reachable from the
    initial state in the collected edge set.

    Bypasses the per-eval search call when training already explored the
    goal — typically saves tens of seconds per level on hard games where
    eval search would otherwise hit the wallclock timeout.
    """
    import glob
    cache_dir = _cache_dir(game_name, level_i)
    if not os.path.isdir(cache_dir):
        return None
    candidates = []
    # Prefer astar (heuristic-guided ⇒ likelier to have hit a goal sooner)
    # over bfs; prefer non-capped over capped (full edge set survives).
    for prio_algo in ("astar", "bfs"):
        for prio_cap in ("capall", "cap*"):
            pattern = os.path.join(
                cache_dir,
                f"{prio_algo}_transitions_v5_*_{prio_cap}.npz",
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


def _dat_to_multihot(
    dat: list[int], n_objs: int, width: int, height: int,
    raw_to_canonical: dict[int, int] | None = None,
    n_canonical: int | None = None,
) -> np.ndarray:
    """Convert bitpacked int32 state vector to (n_objs, H, W) uint8 multihot.

    If raw_to_canonical is given, collapses raw object indices → canonical ones
    (same dedup as CppPuzzleScriptEnv.observation_shape). Output channels are
    then n_canonical, and the stride is still computed over the raw n_objs
    since that's the bitpacking layout used by the engine.

    For converting many states at once, use `_dats_to_multihot_batch`, which
    vectorizes over states (~100x faster on big collections).
    """
    return _dats_to_multihot_batch(
        [dat], n_objs, width, height, raw_to_canonical, n_canonical,
    )[0]


def _dats_to_multihot_batch(
    dats, n_objs: int, width: int, height: int,
    raw_to_canonical: dict[int, int] | None = None,
    n_canonical: int | None = None,
) -> np.ndarray:
    """Batch convert bitpacked states to (n_states, out_C, H, W) uint8 multihot.

    Replaces a Python triple-loop over (state, cell, object) with a single
    vectorized pass per raw object index. Numpy does the per-cell bit-test
    across all states in C, so a workload that previously took hours of
    Python time finishes in seconds.
    """
    stride_obj = (n_objs + 31) // 32
    if raw_to_canonical is not None:
        assert n_canonical is not None
        out_C = n_canonical
    else:
        out_C = n_objs

    n_states = len(dats)
    if n_states == 0:
        return np.zeros((0, out_C, height, width), dtype=np.uint8)

    # Stack & reshape into (n_states, width, height, stride_obj) uint32.
    # The flat layout matches `(x * height + y) * stride_obj + word` from the
    # original loop, so reshape((-1, w, h, stride_obj)) lines up directly.
    # The C++ backend returns words as signed int32 — np.asarray(..., dtype=
    # np.uint32) rejects those with the high bit set. Cast to int32 first
    # (which fits the negative range), then reinterpret-view as uint32 so the
    # bit pattern is preserved.
    arr = np.asarray(dats, dtype=np.int32).view(np.uint32).reshape(
        n_states, width, height, stride_obj
    )

    out = np.zeros((n_states, out_C, height, width), dtype=np.uint8)
    for raw_i in range(n_objs):
        c = raw_to_canonical[raw_i] if raw_to_canonical is not None else raw_i
        word = raw_i // 32
        bit_mask = np.uint32(1 << (raw_i % 32))
        # mask shape: (n_states, width, height) bool. Transpose to (..., h, w)
        # to match the original `obs[c, y, x] = 1` ordering, then OR-merge
        # bits into the canonical channel.
        mask = (arr[..., word] & bit_mask) != 0
        out[:, c] |= mask.transpose(0, 2, 1).astype(np.uint8)
    return out


def collect_unique_transitions(
    json_str: str,
    game_name: str,
    level_i: int = 0,
    max_iters: int = 100_000,
    timeout_ms: int = -1,
    search_algo: str = "astar",
    max_transitions: int | None = None,
) -> dict:
    """Collect unique transitions via C++ state-space exploration.

    Every (state, action, next_state) transition visited during search is returned.
    Uses A* (default) or BFS to explore the state space.

    Cached at rollout_data/{game}/level_{i}/{algo}_transitions_v5_*.npz.

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
    cap_tag = "all" if max_transitions is None else str(int(max_transitions))
    cache_dir = _cache_dir(game_name, level_i)
    cache_path = os.path.join(
        cache_dir,
        f"{search_algo}_transitions_v5_{max_iters}_{timeout_ms}_cap{cap_tag}.npz",
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
            f"{search_algo}_transitions_v5_{max_iters}_*_cap{cap_tag}.npz",
        )
        for alt in sorted(glob.glob(glob_pattern)):
            if alt == cache_path:
                continue
            cached = _load_npz_dict(alt)
            if cached is not None and len(cached["states"]) > 0:
                print(f"  reusing cache from differing timeout: {os.path.basename(alt)}")
                break
            cached = None
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
        rng = np.random.RandomState(42 + level_i)
        keep = np.sort(rng.choice(n_trans, size=int(max_transitions), replace=False))
        keep_states = all_states[keep]
        keep_next = all_next[keep]
        keep_actions = all_actions[keep]
        keep_wons = all_wons[keep]
        print(f"    capping {n_trans:,} → {len(keep):,} at write-time")
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




# Preset game sets for multi-game training
MULTI_GAME_PRESETS = {
    "synthetic": [
        "push_sokoban_synthetic",    # standard push
        "swap_sokoban_synthetic",    # player-box swap
        "vanish_sokoban_synthetic",  # box vanishes on contact
    ],
    "small": [
        "nekopuzzle",              # 3 objs,  7x8
        "notsnake",                # 3 objs,  5x8
        "blocks",                  # 4 objs, 11x13
        "sokoban_basic",           # 5 objs,  7x6
        "sokoban_match3",          # 5 objs,  7x9
        "Zen_Puzzle_Garden",       # 6 objs, 12x12
        "Multi-word_Dictionary_Game",  # 7 objs,  7x9
        "kettle",                  # 8 objs, 13x15
        "Travelling_salesman",     # 9 objs,  5x5
    ],
    # Nested subsets of "small" for dataset-scaling experiments. Each larger
    # set is a superset of the previous; games are ordered to include a mix of
    # "well-fit" (sokoban_basic, blocks) and "poorly-fit" (nekopuzzle,
    # Travelling_salesman) from the existing cond_vs_uncond run.
    "scaling_1": ["sokoban_basic"],
    "scaling_1_neko": ["nekopuzzle"],  # sanity-check: can the model fit nekopuzzle alone?
    "scaling_2": ["sokoban_basic", "nekopuzzle"],
    # Singletons of games used in the global-rules architecture experiment.
    # Each tests one rule type in isolation.
    "global_neko": ["nekopuzzle"],          # `...` only
    "global_mazezam": ["MazezaM"],           # `...` only (longer rows)
    "global_constellationz": ["constellationz"],  # `[X][Y]` only
    "global_clearing": ["Clearing_Space"],   # `[X][Y]` only
    "global_nirvana": ["Nirvana"],           # both `...` and `[X][Y]` (separate rules)
    "global_n_step_punt": ["N_Step_Punt"],   # both on the SAME rule line
    "global_sokoban_ctrl": ["sokoban_basic"],  # control: no global rules
    # Bottleneck singletons from scaling_6 multi-game experiment:
    "global_kettle": ["kettle"],
    "global_zen": ["Zen_Puzzle_Garden"],
    "global_travelling_salesman": ["Travelling_salesman"],
    "scaling_4": ["sokoban_basic", "nekopuzzle", "blocks", "Travelling_salesman"],
    "scaling_6": ["sokoban_basic", "nekopuzzle", "blocks", "Travelling_salesman",
                  "Zen_Puzzle_Garden", "kettle"],
    # scaling_14 = "small" (9) + 5 representative mid-sized additions.
    # Binary-search between small (9, no collapse) and scaling_large (19,
    # total collapse). Excludes the biggest additions (constellationz,
    # the_undertaking) and the tiniest ones (sumo=300, wrappingrecipe=125,
    # rigidfail1=3K) so the balanced-sampling pools stay healthy.
    "scaling_14": [
        # original "small" 9:
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman",
        # +5 mid-sized gallery additions:
        "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
        "scriptcross", "Modality",
    ],
    # scaling_large: "small" plus simple gallery additions (1-4 rules each),
    # picked for low complexity first so the model has a path to grow.
    "scaling_large": [
        # original "small" 9:
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman",
        # simple gallery additions (11 more):
        "blank", "sumo", "the_undertaking", "wrappingrecipe",
        "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
        "rigidfail1", "scriptcross", "Modality", "constellationz",
    ],
    # scaling_gallery_v1: scaling_large (20) + ~20 extra gallery games filtered
    # for moderate complexity (n_rules ≤ 8, n_objects ≤ 12, max_level_area ≤ 20,
    # 1 ≤ n_levels ≤ 20). Sokoban_basic/Microban-style variants and lower-cased
    # duplicates are deduped, capitalized form preferred where both exist.
    "scaling_gallery_v1": [
        # scaling_large (20):
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman",
        "blank", "sumo", "the_undertaking", "wrappingrecipe",
        "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
        "rigidfail1", "scriptcross", "Modality", "constellationz",
        # +20 more gallery games (sorted by rules, then objects):
        "randomrobots",                # 1r,3o,1L
        "againexample",                # 1r,5o,1L
        "Microban",                    # 1r,5o,10L (capitalized)
        "naughtysprite",               # 2r,6o,1L
        "randomspawner",               # 2r,6o,1L
        "twolittlecrates1",            # 2r,6o,1L
        "rigid_11",                    # 3r,5o,1L
        "Long_Haul_Space_Flight",      # 3r,9o,13L
        "leftrightnpcs",               # 4r,5o,1L
        "twolittlecrates2",            # 5r,6o,1L
        "twolittlecrates3",            # 5r,6o,1L
        "twolittlecrates4",            # 5r,6o,1L
        "octat",                       # 5r,6o,8L
        "lunar_lockout",               # 5r,7o,4L
        "Stairways",                   # 6r,5o,3L
        "the_art_of_cloning",          # 6r,9o,1L
        "rigid_scott1",                # 7r,7o,1L
        "rigid_one_unlimited",         # 7r,8o,1L
        "Some_lines_were_meant_to_be_crossed",  # 7r,8o,7L
        "blockfaker",                  # 7r,11o,5L
    ],
    # scaling_gallery_v2: scaling_gallery_v1 (40) + 20 more gallery games at
    # higher rule/object counts. Filter: ≤20 rules, ≤20 objects, ≤30 max
    # level area, ≤30 levels. sokoban_basic_*-style pixel variants and
    # lowercase duplicates of existing canonical names omitted.
    "scaling_gallery_v2": [
        # scaling_gallery_v1 (40):
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman",
        "blank", "sumo", "the_undertaking", "wrappingrecipe",
        "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
        "rigidfail1", "scriptcross", "Modality", "constellationz",
        "randomrobots", "againexample", "Microban",
        "naughtysprite", "randomspawner", "twolittlecrates1",
        "rigid_11", "Long_Haul_Space_Flight", "leftrightnpcs",
        "twolittlecrates2", "twolittlecrates3", "twolittlecrates4",
        "octat", "lunar_lockout", "Stairways", "the_art_of_cloning",
        "rigid_scott1", "rigid_one_unlimited",
        "Some_lines_were_meant_to_be_crossed", "blockfaker",
        # +20 more games (sorted by complexity):
        "Pushing_It",                  # 7r,15o,1L
        "2D_Whale_World",              # 8r,7o,8L
        "MazezaM",                     # 8r,13o,30L
        "Ebony_&_Ivory",               # 9r,6o,1L
        "Singleton_Traffic",           # 10r,4o,6L
        "mazetest",                    # 10r,11o,1L
        "Slidings",                    # 10r,12o,11L
        "Lime_Rick",                   # 11r,11o,11L
        "riverpuzzle",                 # 11r,13o,1L
        "Midas",                       # 15r,13o,15L
        "rigid_parallel_many",         # 17r,10o,1L
        "Take_Heart_Lass",             # 17r,14o,12L
        "rigid_many_broken",           # 18r,11o,2L
        "Lightdown",                   # 18r,14o,8L
        "The_observer's_paradox",      # 18r,19o,6L
        "rigid_parallel_unlimited",    # 19r,9o,1L
        "MC_Escher's_Equestrian_Armageddon",  # 19r,14o,4L
        "Smother",                     # 19r,15o,16L
        "It_Dies_In_The_Light",        # 19r,16o,4L
        "Pushcat_Jr",                  # 19r,19o,8L
    ],
}


def _pad_offsets(src: int, dst: int) -> tuple[int, int]:
    """(before, after) amounts to center ``src`` within ``dst`` (extra pixel goes after)."""
    delta = max(0, dst - src)
    before = delta // 2
    after = delta - before
    return before, after


def _pad_obs(obs: np.ndarray, target_C: int, target_H: int, target_W: int) -> np.ndarray:
    """Pad (N, C, H, W) observations to (N, target_C, target_H, target_W) with zeros.

    Channels are top-aligned (semantic, not spatial). Spatial dims are centered.
    """
    N, C, H, W = obs.shape
    if C == target_C and H == target_H and W == target_W:
        return obs
    oy, _ = _pad_offsets(H, target_H)
    ox, _ = _pad_offsets(W, target_W)
    padded = np.zeros((N, target_C, target_H, target_W), dtype=obs.dtype)
    padded[:, :C, oy:oy+H, ox:ox+W] = obs
    return padded


def _pad_packed(
    packed: np.ndarray, src_W: int, target_C: int, target_H: int, target_W: int,
) -> np.ndarray:
    """Pad bitpacked observations (N, C, H, ceil(src_W/8)) to packed
    (N, target_C, target_H, ceil(target_W/8)).

    Bit-alignment makes raw byte padding ambiguous (an offset that isn't a
    multiple of 8 would split a packed byte across cells), so we unpack →
    pad → repack. This is per-level, so the temporary unpacked tensor is
    bounded by the per-level transition count, never the per-game total.
    """
    N = packed.shape[0]
    if N == 0:
        return _pack_states(
            np.zeros((0, target_C, target_H, target_W), dtype=np.uint8)
        )
    unpacked = _unpack_states(packed, src_W)  # (N, C, H, src_W)
    padded = _pad_obs(unpacked, target_C, target_H, target_W)
    return _pack_states(padded)


def _dataset_cache_key(
    game_names: list[str],
    level_i: int | None,
    search_algo: str,
    n_search_steps: int,
    search_timeout_ms: int,
    encode_sprites: bool = False,
    max_transitions_per_game: int | None = None,
    train_levels: list[int] | None = None,
) -> str:
    """Deterministic hash of all args that affect dataset contents."""
    import hashlib
    blob = json.dumps({
        "format_version": 14,  # v14: per-game state arrays bitpacked along W
        "games": sorted(game_names),
        "level": level_i,
        "train_levels": sorted(train_levels) if train_levels is not None else None,
        "search_algo": search_algo,
        "n_search_steps": n_search_steps,
        "search_timeout_ms": search_timeout_ms,
        "encode_sprites": encode_sprites,
        "max_transitions_per_game": max_transitions_per_game,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


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
) -> tuple[dict, list[dict]]:
    """Collect padded transitions from multiple games via search-based unique-transition exploration.

    Args:
        level_i: If None (default), collect from all levels. If int, collect from that level only.

    Returns:
        dataset: merged dict with keys "states", "actions", "next_states", "game_ids"
            all padded to (max_n_objs, max_H, max_W).
        game_infos: list of per-game metadata dicts.
    """
    # Check for cached merged dataset (shared across experiments)
    cache_hash = _dataset_cache_key(
        game_names, level_i, search_algo,
        n_search_steps, search_timeout_ms,
        encode_sprites=encode_sprites,
        max_transitions_per_game=max_transitions_per_game,
        train_levels=train_levels,
    )
    merged_cache_dir = os.path.join(ROLLOUT_CACHE_DIR, "_merged")
    dataset_cache = os.path.join(merged_cache_dir, f"dataset_{cache_hash}.npz")
    infos_cache = os.path.join(merged_cache_dir, f"game_infos_{cache_hash}.pkl")
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
        # Collect once, then refine shapes from what the collector actually
        # returned. Transition collector's id_dict can include more objects
        # than the env-probe reported (objects added by rules/legend not
        # present at level spawn), so we re-pad all levels at the end once
        # we know the true max.
        per_level_cap = (
            max(1, max_transitions_per_game // max(1, len(levels)))
            if max_transitions_per_game is not None else None
        )
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

        # Second pass: pad all levels to the refined per-game max and concat.
        # Each pad happens in unpacked space (per-level RAM cost ≤ per-level
        # cap × g_C × g_H × g_W bytes), then immediately repacks.
        for level_data in raw_levels:
            if len(level_data["states"]) == 0:
                continue
            lW = int(level_data["W"])
            game_states.append(
                _pad_packed(level_data["states"], lW, g_C, g_H, g_W))
            game_actions.append(level_data["actions"])
            game_next_states.append(
                _pad_packed(level_data["next_states"], lW, g_C, g_H, g_W))
            game_wons.append(np.asarray(level_data["wons"], dtype=np.uint8))

        if not game_states:
            # No data collected for any level of this game (e.g. all empty).
            print(f"  {name}: skipping — no transitions across any level")
            continue

        # Record refined shape so game_infos is accurate for later code paths.
        info["n_objs"] = g_C
        info["H"] = g_H
        info["W"] = g_W

        states = np.concatenate(game_states)        # packed: (N, g_C, g_H, ceil(g_W/8))
        actions = np.concatenate(game_actions)
        next_states = np.concatenate(game_next_states)
        wons = np.concatenate(game_wons)
        n_trans = len(states)

        # Safety-net cap after concat in case per-level caps didn't quite tally.
        if (max_transitions_per_game is not None
                and n_trans > max_transitions_per_game):
            rng = np.random.RandomState(42 + game_id)
            idx = rng.choice(n_trans, size=max_transitions_per_game, replace=False)
            states = states[idx]; actions = actions[idx]
            next_states = next_states[idx]; wons = wons[idx]
            n_trans = len(states)

        info["n_transitions"] = n_trans
        info["n_wins"] = int(wons.sum())

        per_game_states.append(states)
        per_game_actions.append(actions)
        per_game_next_states.append(next_states)
        per_game_wons.append(wons)

        changed = (states != next_states).any(axis=(1, 2, 3))
        print(f"  {name}: {n_trans:,} transitions, {changed.sum():,} with state change "
              f"({100*changed.mean():.1f}%), {wons.sum():,} winning "
              f"({100*wons.mean():.3f}%), "
              f"shape=({g_C}, {g_H}, {g_W})")

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

    # Cache the merged dataset (shared across experiments)
    os.makedirs(merged_cache_dir, exist_ok=True)
    print(f"Caching dataset to {dataset_cache}")
    t0 = time.time()
    np.savez(dataset_cache, **_pack_v7_cache(merged))
    with open(infos_cache, "wb") as f:
        pickle.dump(game_infos, f)
    print(f"  Cached in {time.time()-t0:.1f}s")

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
                encode_sprites=encode_sprites, kernel_sep=kernel_sep,
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
                fallback_dynamics=fallback_dynamics,
                no_a_count_max=no_a_count_max,
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
        if len(synth_parts) > 1:
            print(f"  multi_grid: {name} merged {len(synth_parts)} sizes "
                  f"→ ({max_C}, {max_H}, {max_W}); {len(states):,} total transitions")

        n_trans = len(states)
        if (max_transitions_per_game is not None
                and n_trans > max_transitions_per_game):
            rng = np.random.RandomState(42 + game_id)
            idx = rng.choice(n_trans, size=max_transitions_per_game, replace=False)
            states = states[idx]; actions = actions[idx]
            next_states = next_states[idx]; wons = wons[idx]
            n_trans = len(states)

        _, g_C, g_H, g_W = states.shape
        per_game_states.append(states)
        per_game_actions.append(actions)
        per_game_next_states.append(next_states)
        per_game_wons.append(wons)

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
    "per_game_actions", "per_game_wons",
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


# ---------------------------------------------------------------------------
# 2. NCA world model (Flax/JAX)
# ---------------------------------------------------------------------------

def _pool_features(h, *, axis_pool: bool, axis_cummax: bool, global_pool: bool):
    """Augment NHWC hidden state with global-context features for use as
    extra input channels to the next NCA conv.

    All operations preserve the channel dim (one summary value per channel)
    and broadcast back to (B, H, W, C). No new parameters.

    - axis_pool   (2 features): max over W, max over H. Each cell sees
                  "max of channel c anywhere in my row / my column".
                  Needed for rules like `[ X | ... | Y ]` (X exists in row).
    - axis_cummax (4 features): prefix max from L→R, R→L, T→B, B→T along
                  each axis. Each cell sees "max of channel c to my left /
                  right / above / below". Encodes directional info that plain
                  axis_pool loses.
    - global_pool (1 feature): max over (H, W). Each cell sees "max of
                  channel c anywhere in the grid". Needed for multi-bracket
                  rules like `[X] [Y]` (X and Y both exist somewhere).
    """
    feats = []
    if axis_pool:
        row_max = jnp.max(h, axis=2, keepdims=True)  # (B, H, 1, C)
        col_max = jnp.max(h, axis=1, keepdims=True)  # (B, 1, W, C)
        feats.append(jnp.broadcast_to(row_max, h.shape))
        feats.append(jnp.broadcast_to(col_max, h.shape))
    if axis_cummax:
        feats.append(jax.lax.cummax(h, axis=2))                # L→R along W
        feats.append(jax.lax.cummax(h, axis=2, reverse=True))  # R→L
        feats.append(jax.lax.cummax(h, axis=1))                # T→B along H
        feats.append(jax.lax.cummax(h, axis=1, reverse=True))  # B→T
    if global_pool:
        gmax = jnp.max(h, axis=(1, 2), keepdims=True)          # (B, 1, 1, C)
        feats.append(jnp.broadcast_to(gmax, h.shape))
    if not feats:
        return None
    return jnp.concatenate(feats, axis=-1)


class NCAWorldModel(nn.Module):
    """Neural Cellular Automaton world model.

    Given multihot state (C, H, W) and a one-hot action (5,), predicts the
    next multihot state. The action is broadcast spatially and concatenated
    as extra input channels.

    The NCA applies `n_steps` shared-weight local update rules (3x3 conv),
    with skip connections from the input at each step. Optional global-
    context flags (axis_pool, axis_cummax, global_pool) inject pooled hidden
    features as extra conv inputs at each step (see ``_pool_features``).
    """
    n_hid: int = 128
    n_steps: int = 4
    n_out: int = 1  # set to n_objs at init time
    return_intermediates: bool = False
    axis_pool: bool = False
    axis_cummax: bool = False
    global_pool: bool = False
    # LayerNorm on h between NCA steps. Off by default to keep existing
    # checkpoints loadable and runs consistent; enable when running deeper
    # (large n_nca_steps) models to stabilize training.
    use_layernorm: bool = False

    @nn.compact
    def __call__(self, state, action_onehot):
        """
        Args:
            state: (B, C, H, W) float32 multihot level.
            action_onehot: (B, 5) float32 one-hot action.
        Returns:
            If return_intermediates is False:
                (logits, win_logit) where
                    logits: (B, C, H, W) next-state logits
                    win_logit: (B,) scalar logit for P(next_state is winning)
            If return_intermediates is True:
                (logits, win_logit, intermediates) with per-step hidden / readouts.
        """
        B, C, H, W = state.shape
        # NHWC for Flax convolutions
        x = state.transpose(0, 2, 3, 1)  # (B, H, W, C)

        # Broadcast action to spatial dims: (B, 5) -> (B, H, W, 5)
        act = action_onehot[:, None, None, :]
        act = jnp.broadcast_to(act, (B, H, W, N_ACTIONS))

        # Input = state channels + action channels
        inp = jnp.concatenate([x, act], axis=-1)  # (B, H, W, C+5)

        # Embed to hidden
        h = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="embed")(inp)
        h = nn.relu(h)

        # Shared-weight NCA update steps
        nca_conv = nn.Conv(self.n_hid, (3, 3), padding="SAME", name="nca_conv")
        nca_gate = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="nca_gate")
        readout_conv = nn.Conv(self.n_out, (1, 1), padding="SAME", name="readout")
        # LayerNorm is shared across NCA steps (one weight set); applied to h
        # *after* the residual update. Standard trick for stabilising deep
        # shared-weight RNN-style models.
        nca_norm = nn.LayerNorm(name="nca_norm") if self.use_layernorm else None

        hidden_steps = []
        readout_steps = []

        for _ in range(self.n_steps):
            parts = [h, inp]  # skip connection
            pool_feats = _pool_features(
                h,
                axis_pool=self.axis_pool,
                axis_cummax=self.axis_cummax,
                global_pool=self.global_pool,
            )
            if pool_feats is not None:
                parts.append(pool_feats)
            h_in = jnp.concatenate(parts, axis=-1)
            dh = nca_conv(h_in)
            dh = nn.relu(dh)
            dh = nca_gate(dh)
            h = h + dh  # residual update
            h = nn.relu(h)
            if nca_norm is not None:
                h = nca_norm(h)

            if self.return_intermediates:
                hidden_steps.append(h)
                step_logits = readout_conv(h).transpose(0, 3, 1, 2)
                readout_steps.append(step_logits)

        # Final state readout
        logits = readout_conv(h)
        logits = logits.transpose(0, 3, 1, 2)

        # Win-condition head: 1x1 conv -> global mean pool -> MLP -> 1 logit
        win_feat = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="win_conv")(h)
        win_feat = nn.relu(win_feat)
        win_feat = jnp.mean(win_feat, axis=(1, 2))  # (B, n_hid)
        win_feat = nn.Dense(self.n_hid, name="win_dense")(win_feat)
        win_feat = nn.relu(win_feat)
        win_logit = nn.Dense(1, name="win_out")(win_feat)[:, 0]  # (B,)

        # No sprite decoder on the unconditional model (no z to condition on).
        # Return a zeros placeholder to keep the output tuple shape symmetric
        # with ConditionalNCAWorldModel.
        sprite_logits = jnp.zeros((B, self.n_out, 5, 5, 4), dtype=jnp.float32)

        if self.return_intermediates:
            return logits, win_logit, sprite_logits, {"hidden": hidden_steps, "readouts": readout_steps}
        return logits, win_logit, sprite_logits


# ---------------------------------------------------------------------------
# 2a. Conditional NCA world model (game-spec encoder + FiLM)
# ---------------------------------------------------------------------------

class GameSpecEncoder(nn.Module):
    """Transformer encoder: game token sequence -> latent z.

    Prepends a learnable [CLS] token. Output is the CLS representation
    projected to d_z dimensions.
    """
    vocab_size: int = 142       # VOCAB_SIZE + 1 for CLS
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_z: int = 64
    max_seq_len: int = 192      # max tokens + 1 for CLS
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, token_ids, mask, deterministic=True):
        """
        Args:
            token_ids: (B, S) int32 token IDs (without CLS, PAD=0).
            mask: (B, S) bool, True for real tokens.
        Returns:
            z: (B, d_z) float32 latent vector.
        """
        B, S = token_ids.shape
        # Token + positional embeddings
        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="tok_embed")
        pos_emb = nn.Embed(self.max_seq_len, self.d_model, name="pos_embed")

        # CLS token (position 0)
        cls_tok = jnp.full((B, 1), self.vocab_size - 1, dtype=jnp.int32)  # CLS token ID
        cls_mask = jnp.ones((B, 1), dtype=jnp.bool_)

        # Prepend CLS
        all_tokens = jnp.concatenate([cls_tok, token_ids], axis=1)  # (B, 1+S)
        all_mask = jnp.concatenate([cls_mask, mask], axis=1)        # (B, 1+S)

        L = all_tokens.shape[1]
        positions = jnp.arange(L)[None, :]  # (1, L)
        x = tok_emb(all_tokens) + pos_emb(positions)  # (B, L, d_model)

        # Attention mask: Flax's MHA treats mask as BOOLEAN (True=keep,
        # False=mask-out) via `jnp.where(mask, attn_weights, big_neg)`.
        # Prior version passed a float mask (0.0=keep, -1e9=mask-out), which
        # jnp.where interprets via truthy/falsy semantics — 0.0 is falsy, so
        # real tokens got masked OUT and padding got KEPT (inverted). This
        # collapsed all games to near-identical z's. Bool mask fixes it.
        attn_mask = all_mask[:, None, None, :]  # (B, 1, 1, L) bool

        # Transformer encoder layers
        for i in range(self.n_layers):
            # Pre-norm self-attention
            y = nn.LayerNorm(name=f"ln1_{i}")(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                name=f"attn_{i}",
            )(y, y, mask=attn_mask, deterministic=deterministic)
            x = x + y
            # Pre-norm FFN
            y = nn.LayerNorm(name=f"ln2_{i}")(x)
            y = nn.Dense(self.d_model * 4, name=f"ff1_{i}")(y)
            y = nn.gelu(y)
            y = nn.Dense(self.d_model, name=f"ff2_{i}")(y)
            x = x + y

        x = nn.LayerNorm(name="ln_final")(x)

        # CLS output -> z
        cls_out = x[:, 0, :]  # (B, d_model)
        z = nn.Dense(self.d_z, name="z_proj")(cls_out)
        return z


class ConditionalNCAWorldModel(nn.Module):
    """NCA world model conditioned on a game specification via FiLM.

    The game spec (token sequence) is encoded by a transformer into a latent z.
    At each NCA step, z modulates the hidden state update via
    FiLM: dh = gamma(z) * dh + beta(z).

    Optional global-context flags (axis_pool, axis_cummax, global_pool) are
    identical to NCAWorldModel and inject pooled hidden features as extra
    conv inputs at each step.
    """
    # NCA params
    n_hid: int = 128
    n_steps: int = 4
    n_out: int = 1
    return_intermediates: bool = False
    axis_pool: bool = False
    axis_cummax: bool = False
    global_pool: bool = False
    use_layernorm: bool = False
    # Sprite decoder: small Dense head on z → (n_out, 5, 5, 4) RGBA kernel
    # per object (the "lookup table" per-game sprite set). When False,
    # sprite_logits output is zeros.
    sprite_decoder: bool = False
    sprite_hid: int = 128
    # Encoder params
    vocab_size: int = 142
    d_model: int = 64
    n_heads: int = 4
    n_enc_layers: int = 2
    d_z: int = 64
    max_seq_len: int = 192

    @nn.compact
    def __call__(self, state, action_onehot, game_tokens, game_mask,
                 z_override=None):
        """
        Args:
            state: (B, C, H, W) float32 multihot level.
            action_onehot: (B, 5) float32 one-hot action.
            game_tokens: (B, S) int32 tokenized game spec.
            game_mask: (B, S) bool mask (True for real tokens).
            z_override: optional (B, d_z) latent to substitute for the
                encoder's output. When provided, the FiLM/NCA path uses
                this z; the encoder is still invoked (with the supplied
                tokens) so its params are exercised, then its output is
                discarded. Used by interpolation/sampling tools to roll
                out under custom latents without rebuilding the module.
        Returns:
            (logits, win_logit) or (logits, win_logit, intermediates).
        """
        B, C, H, W = state.shape

        # --- Encode game spec → z ---
        z = GameSpecEncoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_enc_layers,
            d_z=self.d_z,
            max_seq_len=self.max_seq_len,
            name="game_encoder",
        )(game_tokens, game_mask)  # (B, d_z)
        if z_override is not None:
            z = z_override

        # --- FiLM parameters from z (shared across NCA steps) ---
        # Initialize gamma near 1, beta near 0 for identity-like start
        gamma = nn.Dense(
            self.n_hid, name="film_gamma",
            kernel_init=nn.initializers.zeros,
        )(z) + 1.0  # (B, n_hid)
        beta = nn.Dense(
            self.n_hid, name="film_beta",
            kernel_init=nn.initializers.zeros,
        )(z)  # (B, n_hid)

        # Broadcast for spatial dims: (B, 1, 1, n_hid)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]

        # --- NCA forward (same as NCAWorldModel, with FiLM) ---
        x = state.transpose(0, 2, 3, 1)  # (B, H, W, C)

        act = action_onehot[:, None, None, :]
        act = jnp.broadcast_to(act, (B, H, W, N_ACTIONS))
        inp = jnp.concatenate([x, act], axis=-1)

        h = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="embed")(inp)
        h = nn.relu(h)

        nca_conv = nn.Conv(self.n_hid, (3, 3), padding="SAME", name="nca_conv")
        nca_gate = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="nca_gate")
        readout_conv = nn.Conv(self.n_out, (1, 1), padding="SAME", name="readout")
        nca_norm = nn.LayerNorm(name="nca_norm") if self.use_layernorm else None

        hidden_steps = []
        readout_steps = []

        for _ in range(self.n_steps):
            parts = [h, inp]
            pool_feats = _pool_features(
                h,
                axis_pool=self.axis_pool,
                axis_cummax=self.axis_cummax,
                global_pool=self.global_pool,
            )
            if pool_feats is not None:
                parts.append(pool_feats)
            h_in = jnp.concatenate(parts, axis=-1)
            dh = nca_conv(h_in)
            dh = nn.relu(dh)
            dh = nca_gate(dh)
            # FiLM modulation on the update
            dh = gamma * dh + beta
            h = h + dh
            h = nn.relu(h)
            if nca_norm is not None:
                h = nca_norm(h)

            if self.return_intermediates:
                hidden_steps.append(h)
                step_logits = readout_conv(h).transpose(0, 3, 1, 2)
                readout_steps.append(step_logits)

        logits = readout_conv(h)
        logits = logits.transpose(0, 3, 1, 2)

        # Win head: pool NCA features, concat z (game-spec latent), MLP -> 1 logit
        win_feat = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="win_conv")(h)
        win_feat = nn.relu(win_feat)
        win_feat = jnp.mean(win_feat, axis=(1, 2))      # (B, n_hid)
        win_feat = jnp.concatenate([win_feat, z], axis=-1)  # (B, n_hid + d_z)
        win_feat = nn.Dense(self.n_hid, name="win_dense")(win_feat)
        win_feat = nn.relu(win_feat)
        win_logit = nn.Dense(1, name="win_out")(win_feat)[:, 0]  # (B,)

        # Sprite decoder head: z → per-object 5x5x4 RGBA kernel.
        # Output shape (B, n_out, 5, 5, 4). Sigmoid applied downstream where
        # a normalized [0,1] value is required (e.g. for MSE vs target).
        # This is a pure per-game lookup table (no cross-channel weights),
        # implemented as a Dense head whose output is reshaped — equivalent
        # to a learned (n_out, 5, 5, 4) tensor produced from z.
        if self.sprite_decoder:
            sh = nn.Dense(self.sprite_hid, name="sprite_hid")(z)
            sh = nn.relu(sh)
            sprite_flat = nn.Dense(
                self.n_out * 5 * 5 * 4, name="sprite_out"
            )(sh)  # (B, n_out*5*5*4)
            sprite_logits = sprite_flat.reshape(B, self.n_out, 5, 5, 4)
        else:
            sprite_logits = jnp.zeros((B, self.n_out, 5, 5, 4), dtype=jnp.float32)

        if self.return_intermediates:
            return logits, win_logit, sprite_logits, {"hidden": hidden_steps, "readouts": readout_steps}
        return logits, win_logit, sprite_logits


# ---------------------------------------------------------------------------
# 2b. Activation visualization
# ---------------------------------------------------------------------------

def _make_channel_grid(activations, ncols=None, pad=1, normalize=True, scale=3):
    """Arrange (H, W, C) activations into a single image grid.

    Each channel cell is upscaled by `scale` for visibility.
    Returns an (grid_H, grid_W) float array suitable for colormap application.
    """
    H, W, C = activations.shape
    sH, sW = H * scale, W * scale
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(C)))
    nrows = int(np.ceil(C / ncols))
    grid = np.zeros((nrows * (sH + pad) - pad, ncols * (sW + pad) - pad), dtype=np.float32)

    for i in range(C):
        r, c = divmod(i, ncols)
        y0 = r * (sH + pad)
        x0 = c * (sW + pad)
        ch = np.array(activations[:, :, i], dtype=np.float32)
        if normalize:
            lo, hi = ch.min(), ch.max()
            ch = (ch - lo) / (hi - lo + 1e-8)
        # Upscale with nearest neighbor
        ch = np.repeat(np.repeat(ch, scale, axis=0), scale, axis=1)
        grid[y0:y0 + sH, x0:x0 + sW] = ch

    return grid


def _apply_colormap(gray, cmap_name="viridis"):
    """Convert (H, W) float in [0,1] to (H, W, 3) uint8 via matplotlib colormap."""
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(gray)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _labeled_channel_grid(logits_chw, obj_names, pad=2, scale=4):
    """Render per-object output channels as a labeled grid.

    Always shows ALL channels in order, labeled with object names.

    Args:
        logits_chw: (C, H, W) logits or probabilities.
        obj_names: list of C object name strings.
        scale: upscale each cell by this factor for readability.
    Returns:
        (grid_H, grid_W, 3) uint8 image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    C, H, W = logits_chw.shape
    logits_f = np.clip(np.array(logits_chw, dtype=np.float32), -50, 50)
    probs = 1.0 / (1.0 + np.exp(-logits_f))

    # Always show all channels in order
    ncols = min(10, C)
    nrows = int(np.ceil(C / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.5),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for idx in range(C):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.imshow(probs[idx], vmin=0, vmax=1, cmap="magma",
                  interpolation="nearest", aspect="equal")
        name = obj_names[idx] if idx < len(obj_names) else f"ch{idx}"
        # Truncate long names
        if len(name) > 14:
            name = name[:12] + ".."
        ax.set_title(name, fontsize=5, pad=2)

    fig.tight_layout(pad=0.3)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


def visualize_nca_step(
    model: "NCAWorldModel",
    params,
    state_obs: np.ndarray,
    action: int,
    obj_names: list[str],
    backend: "CppPuzzleScriptBackend",
    grid_w: int,
    grid_h: int,
):
    """Run one NCA step with intermediates and return a visualization image.

    Returns a tall image with rows:
      - Input state (rendered)
      - For each NCA step: hidden activation grid + discrete output channels
      - Final predicted state (rendered)
    """
    # Create model variant that returns intermediates
    model_viz = NCAWorldModel(n_hid=model.n_hid, n_steps=model.n_steps,
                              n_out=model.n_out, return_intermediates=True)
    state_jnp = jnp.array(state_obs[None], dtype=jnp.float32)
    a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])
    logits, _win_logit, _sprite_logits, intermediates = model_viz.apply(params, state_jnp, a_oh)

    sections = []

    # Input state rendered
    input_frame = backend.render_frame_from_objects(
        _multihot_to_objects(state_obs), grid_w, grid_h
    )
    sections.append(input_frame)

    # Per-step hidden activations + readouts
    for step_i, (h, readout) in enumerate(
        zip(intermediates["hidden"], intermediates["readouts"])
    ):
        h_np = np.array(h[0])   # (H, W, n_hid)
        r_np = np.array(readout[0])  # (C, H, W)

        # Hidden channel grid
        hid_grid = _make_channel_grid(h_np)
        hid_img = _apply_colormap(hid_grid)

        # Labeled output channels
        out_img = _labeled_channel_grid(r_np, obj_names)

        # Match widths for stacking
        target_w = max(hid_img.shape[1], out_img.shape[1], input_frame.shape[1])
        hid_img = _pad_to_width(hid_img, target_w)
        out_img = _pad_to_width(out_img, target_w)

        sections.append(hid_img)
        sections.append(out_img)

    # Final predicted state rendered
    pred_obs = np.array((jax.nn.sigmoid(logits[0]) > 0.5), dtype=np.uint8)
    pred_frame = backend.render_frame_from_objects(
        _multihot_to_objects(pred_obs), grid_w, grid_h
    )
    sections.append(pred_frame)

    # Match all widths
    max_w = max(s.shape[1] for s in sections)
    sections = [_pad_to_width(s, max_w) for s in sections]

    return np.concatenate(sections, axis=0)


def _pad_to_width(img, target_w):
    """Pad or resize image to target width, preserving aspect ratio."""
    h, w = img.shape[:2]
    if w == target_w:
        return img
    if w < target_w:
        pad = np.zeros((h, target_w - w, 3), dtype=img.dtype)
        return np.concatenate([img, pad], axis=1)
    # Resize down
    import PIL.Image
    pil = PIL.Image.fromarray(img)
    new_h = int(h * target_w / w)
    pil = pil.resize((target_w, new_h), PIL.Image.NEAREST)
    return np.array(pil)


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def _weighted_bce_with_logits(logit, label, pos_weight):
    """Numerically stable BCE-with-logits that upweights the positive class.

    loss = pos_weight * label * softplus(-logit) + (1 - label) * softplus(logit)
    """
    return pos_weight * label * jax.nn.softplus(-logit) + (1.0 - label) * jax.nn.softplus(logit)


def _wm_p(params):
    """Extract WM params from a possibly-joint params tree.

    When joint token-decoder training is enabled, ``params`` is shaped
    ``{"wm": <wm_params>, "dec": <decoder_params>}``. All eval / render code
    only needs the WM half — this helper unwraps both shapes uniformly.
    """
    if isinstance(params, dict) and set(params.keys()) >= {"wm", "dec"}:
        return params["wm"]
    return params


def make_train_step(model, optimizer, conditional=False,
                    win_loss_weight: float = 1.0, win_pos_weight: float = 1.0,
                    sprite_loss_weight: float = 0.0,
                    change_loss_weight: float = 0.0,
                    decoder=None,
                    token_decoder_loss_weight: float = 0.0,
                    use_vq: bool = False,
                    vq_commitment_weight: float = 0.25,
                    vq_loss_weight: float = 1.0,
                    adaptive_halt: bool = False,
                    halt_prior_p: float = 0.1,
                    halt_kl_weight: float = 0.01,
                    halt_mode: str = "ponder"):
    """Returns a JIT-compiled train step with a win-prediction head.

    If ``sprite_loss_weight > 0`` (and conditional), also optimizes a sprite-
    decoder MSE loss: sigmoid(sprite_logits) vs target_sprites normalized to
    [0, 1]. ``target_sprites`` is a (B, n_out, 5, 5, 4) uint8 tensor gathered
    from per-game sprite tensors by game_id.
    """

    def _heads_loss(logits, win_logit, sprite_logits, states, next_states, wons,
                     target_sprites=None):
        bce = optax.sigmoid_binary_cross_entropy(logits, next_states)
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        acc = (preds == next_states).mean()
        changed = (states != next_states)
        n_changed = changed.sum()
        changed_correct = ((preds == next_states) & changed).sum()
        change_acc = jnp.where(n_changed > 0, changed_correct / n_changed, 1.0)
        # Optionally upweight BCE on cells that actually changed (anti
        # identity-collapse escape hatch). weight = 1 on unchanged cells,
        # (1 + change_loss_weight) on changed cells. At c_l_w=0, identical
        # to uniform mean. At c_l_w=10, changed cells count 11× as much.
        if change_loss_weight > 0:
            weight = 1.0 + change_loss_weight * changed.astype(bce.dtype)
            state_loss = (bce * weight).sum() / weight.sum()
        else:
            state_loss = bce.mean()
        # Win-head BCE with pos_weight to counter class imbalance
        wons_f = wons.astype(jnp.float32)
        win_bce = _weighted_bce_with_logits(win_logit, wons_f, win_pos_weight).mean()
        win_preds = (jax.nn.sigmoid(win_logit) > 0.5).astype(jnp.float32)
        win_acc = (win_preds == wons_f).mean()
        n_win = wons_f.sum()
        win_tp = ((win_preds == 1.0) & (wons_f == 1.0)).sum()
        win_recall = jnp.where(n_win > 0, win_tp / n_win, 1.0)
        total = state_loss + win_loss_weight * win_bce

        # Sprite-decoder loss (optional). Sprite logits → sigmoid → [0,1] RGBA;
        # target is uint8 normalized to [0,1]. MSE over the whole per-object
        # kernel tensor (including alpha channel).
        if sprite_loss_weight > 0 and target_sprites is not None:
            sprite_pred = jax.nn.sigmoid(sprite_logits)
            sprite_tgt = target_sprites.astype(jnp.float32) / 255.0
            sprite_mse = ((sprite_pred - sprite_tgt) ** 2).mean()
            total = total + sprite_loss_weight * sprite_mse
        else:
            sprite_mse = jnp.asarray(0.0, dtype=jnp.float32)
        return total, (state_loss, acc, change_acc, win_bce, win_acc, win_recall,
                        sprite_mse)

    def _ponder_loss(per_step_logits, per_step_win, per_step_halt_logits,
                      states, next_states, wons):
        """PonderNet-style adaptive-halting loss.

        per_step_logits: (T, B, n_out, H, W) — logits at each NCA step.
        per_step_win:    (T, B)              — win logits at each step.
        per_step_halt_logits: (T, B)         — halt logits at each step.

        Halt distribution: λ_k = sigmoid(halt_logits_k) for k < T; the last
        step is forced halt (λ_T := 1) so Σ_k p_k = 1 exactly.
            p_k = λ_k · Π_{j<k}(1 - λ_j)

        Loss = Σ_k p_k · L_k + halt_kl_weight · KL(p || Geom(halt_prior_p))
        where L_k is the same _heads_loss-style state+win loss at step k.

        Returns (total_loss, aux) where aux mirrors _heads_loss's aux but
        with metrics computed at the *expected* step (Σ_k p_k · metric_k)
        plus an extra (expected_steps, kl) pair.
        """
        T = per_step_logits.shape[0]
        # Halt probs: λ ∈ (0, 1)^{T, B}, with the last row forced to 1.
        lam = jax.nn.sigmoid(per_step_halt_logits)              # (T, B)
        # Build cumulative survival Π_{j<k}(1 - λ_j) along T.
        # surv[k] = Π_{j<k} (1 - λ_j); surv[0] = 1.
        log_one_minus = jnp.log(jnp.clip(1.0 - lam, 1e-6, 1.0))  # (T, B)
        surv = jnp.exp(jnp.concatenate([
            jnp.zeros((1, log_one_minus.shape[1])),
            jnp.cumsum(log_one_minus, axis=0)[:-1],
        ], axis=0))                                             # (T, B)
        # p_k for k<T uses λ_k * surv_k; p_T uses surv_T (forced halt).
        p = lam * surv                                          # (T, B)
        # Replace the last row with the forced-halt mass.
        last_surv = jnp.exp(jnp.sum(log_one_minus[:-1], axis=0))  # (B,)
        p = p.at[-1].set(last_surv)                             # (T, B), Σ_k p_k = 1

        # Per-step, per-batch-element state loss. Critically we keep the
        # batch axis until *after* multiplying by p, so each batch element's
        # halt distribution can pair with its own per-step loss — which is
        # what enables per-input adaptive halting. (The previous version
        # averaged loss + p over batch independently and then took their
        # outer product; that lost the per-element coupling that PonderNet
        # depends on.)
        next_b = next_states[None]                              # (1, B, n_out, H, W)
        states_b = states[None]
        bce = optax.sigmoid_binary_cross_entropy(per_step_logits, jnp.broadcast_to(next_b, per_step_logits.shape))
        if change_loss_weight > 0:
            changed = (states_b != next_b).astype(bce.dtype)
            changed = jnp.broadcast_to(changed, per_step_logits.shape)
            weight = 1.0 + change_loss_weight * changed
            # (T, B): weighted-mean BCE per (step, batch-element).
            state_loss_per_step_per_b = (bce * weight).sum(axis=(2, 3, 4)) / weight.sum(axis=(2, 3, 4))
        else:
            state_loss_per_step_per_b = bce.mean(axis=(2, 3, 4))  # (T, B)

        # Per-step, per-batch-element win BCE.
        wons_f = wons.astype(jnp.float32)
        # _weighted_bce_with_logits returns (B,) for each step's (B,) logits.
        win_bce_per_step_per_b = jax.vmap(
            lambda wl: _weighted_bce_with_logits(wl, wons_f, win_pos_weight)
        )(per_step_win)                                          # (T, B)

        # Per-batch-element total head loss at each step.
        L_per_step_per_b = state_loss_per_step_per_b + win_loss_weight * win_bce_per_step_per_b  # (T, B)
        # Loss aggregation across the T (per-step) axis. Three modes:
        #   ponder    — PonderNet-style: weight by learned halt distribution p.
        #               L = E_b[Σ_k p_k(b) · L_k(b)]. Each example chooses its
        #               own halt step via the halt head; KL regularizer pulls
        #               p toward a geometric prior. Body gets gradient at
        #               every k, weighted — rewards shortcut predictions.
        #   uniform   — mean over k (treats every step's prediction as equally
        #               important). Body must make readout good at *every*
        #               depth — prerequisite for convergence-based stopping at
        #               inference but actively rewards shortcuts even more.
        #   argmax_st — Straight-through argmax: forward computes L only at
        #               k* = argmax_k p_k(b) (one selected step per batch
        #               element); backward gradient on halt logits flows via
        #               soft p (so halt can still learn). Body sees gradient
        #               only through L_{k*}, so it isn't penalized for being
        #               wrong at unselected k's. Combined with the KL prior
        #               this should let the model learn depth-specialised
        #               predictions per instance without shortcut pressure.
        if halt_mode == "uniform":
            L_rec = L_per_step_per_b.mean()  # mean over (T, B)
        elif halt_mode == "argmax_st":
            # Hard one-hot mask of argmax in forward; soft p in backward.
            k_star = jnp.argmax(p, axis=0)                        # (B,)
            mask_hard = jax.nn.one_hot(k_star, T, axis=0)          # (T, B)
            mask = mask_hard + p - jax.lax.stop_gradient(p)
            L_rec = (mask * L_per_step_per_b).sum(axis=0).mean()
        elif halt_mode == "convergence_st":
            # Convergence-based selection. k*(b) = first k where the
            # discrete prediction has converged (fraction of cells whose
            # binary readout flipped between k-1 and k is below
            # halt_prior_p). If never converges within T, falls back to T.
            # No halt-head gradient is meaningful here (the head is unused
            # for selection); body gradient flows only through L_{k*}.
            B_ax = per_step_logits.shape[1]
            preds = (jax.nn.sigmoid(per_step_logits) > 0.5).astype(jnp.float32)
            n_cells = preds.shape[2] * preds.shape[3] * preds.shape[4]
            # diff[k] = fraction of cells changing between step k and k-1
            # for k=1..T-1.
            diff = (preds[1:] != preds[:-1]).astype(jnp.float32).sum(
                axis=(2, 3, 4)) / n_cells                          # (T-1, B)
            converged = diff < halt_prior_p                        # (T-1, B)
            # Stack a sentinel "always converged" row at the end so argmax
            # finds the latest step if no earlier convergence happened.
            converged_full = jnp.concatenate(
                [converged, jnp.ones((1, B_ax), dtype=bool)], axis=0)  # (T, B)
            k_star_idx = jnp.argmax(converged_full.astype(jnp.int32), axis=0)
            k_star = jnp.minimum(k_star_idx + 1, T - 1)             # (B,) in 0..T-1
            mask_hard = jax.nn.one_hot(k_star, T, axis=0)           # (T, B)
            L_rec = (mask_hard * L_per_step_per_b).sum(axis=0).mean()
        else:  # "ponder" (default)
            L_rec = (p * L_per_step_per_b).sum(axis=0).mean()
        # Batch-marginal helpers for the reporting metrics below.
        state_loss_per_step = state_loss_per_step_per_b.mean(axis=1)  # (T,)
        win_bce_k = win_bce_per_step_per_b.mean(axis=1)               # (T,)
        p_marginal = p.mean(axis=1)                                    # (T,)

        # KL(p || Geometric(halt_prior_p)) per batch element, mean.
        # prior_k = (1 - halt_prior_p)^(k-1) * halt_prior_p for k < T;
        # prior_T = (1 - halt_prior_p)^(T-1)  (truncation mass).
        ks = jnp.arange(T)
        log_prior = jnp.where(
            ks < T - 1,
            ks * jnp.log1p(-halt_prior_p) + jnp.log(halt_prior_p),
            (T - 1) * jnp.log1p(-halt_prior_p),
        )                                                        # (T,)
        # KL = Σ_k p_k log(p_k / prior_k), averaged over batch.
        log_p = jnp.log(jnp.clip(p, 1e-8, 1.0))                  # (T, B)
        kl_per_b = (p * (log_p - log_prior[:, None])).sum(axis=0)  # (B,)
        kl = kl_per_b.mean()

        total = L_rec + halt_kl_weight * kl

        # Reporting metrics: use the expected step (E[k]+1, since k is
        # 0-indexed) and the expected per-step state metrics.
        exp_step = (p_marginal * (jnp.arange(T) + 1).astype(jnp.float32)).sum()

        # For aux compatibility with _heads_loss, compute expected-state-loss
        # (= L_rec but state-only), plus expected acc and change_acc using
        # the per-step argmax predictions weighted by p.
        preds = (jax.nn.sigmoid(per_step_logits) > 0.5).astype(jnp.float32)
        next_b_full = jnp.broadcast_to(next_b, per_step_logits.shape)
        states_b_full = jnp.broadcast_to(states_b, per_step_logits.shape)
        correct = (preds == next_b_full).astype(jnp.float32)
        acc_k = correct.mean(axis=(1, 2, 3, 4))                  # (T,)
        changed_full = (states_b_full != next_b_full).astype(jnp.float32)
        changed_correct_k = (correct * changed_full).sum(axis=(1, 2, 3, 4))
        n_changed_k = changed_full.sum(axis=(1, 2, 3, 4))
        change_acc_k = jnp.where(n_changed_k > 0, changed_correct_k / n_changed_k, 1.0)
        acc = (p_marginal * acc_k).sum()
        change_acc = (p_marginal * change_acc_k).sum()
        # Win metrics from the expected-step.
        win_preds = (jax.nn.sigmoid(per_step_win) > 0.5).astype(jnp.float32)
        win_acc_k = (win_preds == wons_f[None]).astype(jnp.float32).mean(axis=1)
        win_acc = (p_marginal * win_acc_k).sum()
        n_win = wons_f.sum()
        win_tp_k = ((win_preds == 1.0) & (wons_f[None] == 1.0)).astype(jnp.float32).sum(axis=1)
        win_recall_k = jnp.where(n_win > 0, win_tp_k / n_win, 1.0)
        win_recall = (p_marginal * win_recall_k).sum()

        sprite_mse = jnp.asarray(0.0, dtype=jnp.float32)
        state_loss_exp = (p_marginal * state_loss_per_step).sum()
        win_bce_exp = (p_marginal * win_bce_k).sum()

        return total, (state_loss_exp, acc, change_acc, win_bce_exp,
                       win_acc, win_recall, sprite_mse), (exp_step, kl)

    joint = decoder is not None and token_decoder_loss_weight > 0

    def _vq_utilization(vq_indices):
        counts = jnp.bincount(
            vq_indices.reshape(-1),
            length=getattr(model, "vq_codebook_size", 1),
        )
        return (counts > 0).sum().astype(jnp.float32)

    # All three train_step variants append (vq_cb_loss, vq_commit_loss,
    # vq_utilization) to aux as the last positions, regardless of whether VQ
    # is enabled. When VQ is off they are zeros — keeps the loop unpack shape
    # stable.
    if conditional and joint:
        from nca_wm.token_decoder import shift_right, decoder_loss as _dec_loss

        @jax.jit
        def train_step(params, opt_state, states, action_onehots, next_states, wons,
                        game_tokens, game_masks, target_sprites=None,
                        target_tokens=None, target_token_masks=None):
            def loss_fn(params):
                wm_p, dec_p = params["wm"], params["dec"]
                if use_vq:
                    logits, win_logit, sprite_logits, all_slots, vq_aux = model.apply(
                        wm_p, states, action_onehots, game_tokens, game_masks,
                        return_slots=True, return_vq_aux=True,
                    )
                    vq_cb_loss, vq_commit_loss, vq_indices = vq_aux
                    vq_util = _vq_utilization(vq_indices)
                else:
                    logits, win_logit, sprite_logits, all_slots = model.apply(
                        wm_p, states, action_onehots, game_tokens, game_masks,
                        return_slots=True,
                    )
                    z = jnp.asarray(0.0, dtype=jnp.float32)
                    vq_cb_loss, vq_commit_loss, vq_util = z, z, z
                heads_total, aux = _heads_loss(
                    logits, win_logit, sprite_logits,
                    states, next_states, wons,
                    target_sprites=target_sprites,
                )
                # Decoder forward — teacher-forced (shift target right by one).
                shifted = shift_right(target_tokens)
                dec_logits = decoder.apply(dec_p, shifted, all_slots)
                dec_loss_v, dec_acc_v = _dec_loss(
                    dec_logits, target_tokens, target_token_masks,
                )
                total = heads_total + token_decoder_loss_weight * dec_loss_v
                if use_vq:
                    total = total + vq_loss_weight * (
                        vq_cb_loss + vq_commitment_weight * vq_commit_loss
                    )
                return total, aux + (
                    dec_loss_v, dec_acc_v, vq_cb_loss, vq_commit_loss, vq_util,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return (params_new, opt_state_new, loss) + aux
    elif conditional:
        @jax.jit
        def train_step(params, opt_state, states, action_onehots, next_states, wons,
                        game_tokens, game_masks, target_sprites=None):
            def loss_fn(params):
                if use_vq:
                    logits, win_logit, sprite_logits, vq_aux = model.apply(
                        params, states, action_onehots, game_tokens, game_masks,
                        return_vq_aux=True,
                    )
                    vq_cb_loss, vq_commit_loss, vq_indices = vq_aux
                    vq_util = _vq_utilization(vq_indices)
                elif adaptive_halt:
                    logits, win_logit, sprite_logits, halt_aux = model.apply(
                        params, states, action_onehots, game_tokens, game_masks,
                    )
                    z = jnp.asarray(0.0, dtype=jnp.float32)
                    vq_cb_loss, vq_commit_loss, vq_util = z, z, z
                else:
                    logits, win_logit, sprite_logits = model.apply(
                        params, states, action_onehots, game_tokens, game_masks
                    )
                    z = jnp.asarray(0.0, dtype=jnp.float32)
                    vq_cb_loss, vq_commit_loss, vq_util = z, z, z
                if adaptive_halt:
                    per_step_logits, per_step_win, per_step_halt = halt_aux
                    heads_total, aux, _ = _ponder_loss(
                        per_step_logits, per_step_win, per_step_halt,
                        states, next_states, wons,
                    )
                else:
                    heads_total, aux = _heads_loss(
                        logits, win_logit, sprite_logits,
                        states, next_states, wons,
                        target_sprites=target_sprites,
                    )
                total = heads_total
                if use_vq:
                    total = total + vq_loss_weight * (
                        vq_cb_loss + vq_commitment_weight * vq_commit_loss
                    )
                # Pad dec losses with zeros so the loop unpack is independent
                # of the joint-decoder branch.
                z = jnp.asarray(0.0, dtype=jnp.float32)
                return total, aux + (z, z, vq_cb_loss, vq_commit_loss, vq_util)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return (params_new, opt_state_new, loss) + aux
    else:
        @jax.jit
        def train_step(params, opt_state, states, action_onehots, next_states, wons):
            def loss_fn(params):
                logits, win_logit, sprite_logits = model.apply(
                    params, states, action_onehots
                )
                heads_total, aux = _heads_loss(
                    logits, win_logit, sprite_logits,
                    states, next_states, wons,
                    target_sprites=None,
                )
                z = jnp.asarray(0.0, dtype=jnp.float32)
                return heads_total, aux + (z, z, z, z, z)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return (params_new, opt_state_new, loss) + aux

    return train_step


def make_apply_fn(model):
    """Jitted model.apply that always returns a (logits, win, sprite) 3-tuple.

    The model can return additional trailing aux tensors (slots, vq_aux,
    halt_aux) under various flags, but the eval / rollout / rendering
    code paths only need the first three. This wrapper hides the
    branching so those callers stay simple.
    """
    _jit = jax.jit(model.apply)
    def apply_fn(*args, **kwargs):
        out = _jit(*args, **kwargs)
        return out[0], out[1], out[2]
    return apply_fn


def make_eval_forward(model, conditional: bool):
    """Forward-only (no grad) eval for per-game diagnostics during training."""
    def _metrics(logits, states, next_states):
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        acc = (preds == next_states).mean()
        changed = (states != next_states)
        n_changed = changed.sum()
        changed_correct = ((preds == next_states) & changed).sum()
        change_acc = jnp.where(n_changed > 0, changed_correct / n_changed, 1.0)
        bce = optax.sigmoid_binary_cross_entropy(logits, next_states).mean()
        return bce, acc, change_acc

    if conditional:
        @jax.jit
        def _eval_forward(wm_params, states, action_onehots, next_states,
                          game_tokens, game_masks):
            logits, _, _sprite_logits = model.apply(wm_params, states, action_onehots,
                                    game_tokens, game_masks)
            return _metrics(logits, states, next_states)

        def eval_forward(params, states, action_onehots, next_states,
                         game_tokens, game_masks):
            return _eval_forward(_wm_p(params), states, action_onehots,
                                 next_states, game_tokens, game_masks)
    else:
        @jax.jit
        def _eval_forward(wm_params, states, action_onehots, next_states):
            logits, _, _sprite_logits = model.apply(wm_params, states, action_onehots)
            return _metrics(logits, states, next_states)

        def eval_forward(params, states, action_onehots, next_states):
            return _eval_forward(_wm_p(params), states, action_onehots, next_states)
    return eval_forward


def _atomic_save_checkpoint(save_dir: str, params, total_steps: int,
                             early_stopped: bool = None, n_updates_requested: int = None):
    """Write params.pkl and train_meta.json atomically via tmp-file + os.replace.

    Safe for a concurrent reader (e.g. a --render_only process) to load without
    catching a half-written file.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "params.pkl")
    tmp_ckpt = ckpt_path + ".tmp"
    with open(tmp_ckpt, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    os.replace(tmp_ckpt, ckpt_path)

    meta_path = os.path.join(save_dir, "train_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta["total_steps"] = total_steps
    if early_stopped is not None:
        meta["early_stopped"] = bool(early_stopped)
    if n_updates_requested is not None:
        meta["n_updates_requested"] = int(n_updates_requested)
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta, meta_path)


def train(
    model: NCAWorldModel,
    dataset: dict,
    lr: float = 1e-3,
    n_updates: int = 5000,
    batch_size: int = 64,
    seed: int = 0,
    log_interval: int = 100,
    save_dir: str = "nca_wm/logs",
    init_params=None,
    start_step: int = 0,
    patience: int = 0,
    min_delta: float = 1e-4,
    win_loss_weight: float = 1.0,
    win_pos_weight: float = 1.0,
    ckpt_interval: int = 1000,
    game_names: list[str] | None = None,
    per_game_eval_interval: int = 1000,
    per_game_eval_size: int = 256,
    balanced_sampling: bool = False,
    game_infos: list[dict] | None = None,
    gif_interval: int = 0,
    gif_n_steps: int = 15,
    max_padded_shape: tuple[int, int, int] | None = None,
    ps_parser=None,
    grad_clip: float = 0.0,  # 0 disables; >0 clips by global-norm to this value
    sprite_loss_weight: float = 0.0,
    change_loss_weight: float = 0.0,
    lr_schedule: str = "constant",  # "constant" or "cosine"
    lr_min: float = 1e-6,
    token_decoder_loss_weight: float = 0.0,
    decoder_d_model: int = 128,
    decoder_n_layers: int = 4,
    decoder_n_heads: int = 4,
    use_vq: bool = False,
    vq_commitment_weight: float = 0.25,
    vq_loss_weight: float = 1.0,
    halt_prior_p: float = 0.1,
    halt_kl_weight: float = 0.01,
    halt_mode: str = "ponder",
):
    """Train (or resume training) the NCA world model.

    If init_params is provided, resumes from those weights instead of
    initializing from scratch. start_step offsets the step counter for logging.
    Supports both conditional (ConditionalNCAWorldModel) and unconditional models.

    Early stopping (enabled when patience > 0):
        Monitors smoothed change_acc (window = patience * log_interval steps).
        Stops when no improvement of at least min_delta for ``patience``
        consecutive evaluation windows.
    """
    # Any encoder-based model (FiLM or rule-attention) counts as "conditional"
    # for dataset/init-signature purposes.
    from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
    conditional = "game_tokens" in dataset and isinstance(
        model, (ConditionalNCAWorldModel, RuleAttnNCAWorldModel)
    )

    # Single-game callers (`collect_unique_transitions` directly into train())
    # hand us a flat dict with packed `states` / `next_states`, plus the
    # original W. Wrap into the per-game layout so the rest of train() has
    # one code path.
    if "per_game_states" not in dataset and "states" in dataset:
        s_flat = dataset["states"]                # packed (N, C, H, ceil(W/8))
        ns_flat = dataset["next_states"]
        a_flat = dataset["actions"]
        w_flat = dataset.get("wons", np.zeros(len(s_flat), dtype=np.uint8))
        # Single-game path always carries the real W in the dict; if missing
        # (shouldn't happen) fall back to inferring an upper bound.
        real_W = int(dataset.get("W", s_flat.shape[-1] * 8))
        C, H = int(s_flat.shape[1]), int(s_flat.shape[2])
        dataset = dict(dataset)  # shallow copy; don't mutate caller's dict
        dataset["per_game_states"] = [s_flat]
        dataset["per_game_next_states"] = [ns_flat]
        dataset["per_game_actions"] = [a_flat]
        dataset["per_game_wons"] = [np.asarray(w_flat, dtype=np.uint8)]
        dataset["game_shapes"] = np.array([[C, H, real_W]], dtype=np.int32)
        dataset["per_game_n_transitions"] = np.array([len(s_flat)], dtype=np.int64)
        dataset["max_C"] = C
        dataset["max_H"] = H
        dataset["max_W"] = real_W
        # Conditional not expected for single-game path; guard anyway.
        if "per_game_tokens" not in dataset:
            dataset["per_game_tokens"] = np.zeros((1, 1), dtype=np.int32)
            dataset["per_game_masks"] = np.zeros((1, 1), dtype=np.bool_)

    rng = jax.random.PRNGKey(seed)
    # v7 per-game native storage. No flat (N, C_max, H_max, W_max) arrays.
    per_game_states = dataset["per_game_states"]          # list of (N_g, C_g, H_g, W_g)
    per_game_next_states = dataset["per_game_next_states"]
    per_game_actions_np = dataset["per_game_actions"]     # list of (N_g,) int32
    per_game_wons_np = dataset["per_game_wons"]           # list of (N_g,) uint8
    n_games = len(per_game_states)
    # Global max for batch buffer shape (kept constant for JIT)
    max_C = int(dataset["max_C"])
    max_H = int(dataset["max_H"])
    max_W = int(dataset["max_W"])
    # Per-game native (C, H, W)
    game_CHW = dataset["game_shapes"]   # (n_games, 3) int32
    per_game_n = dataset["per_game_n_transitions"]  # (n_games,) int64
    if conditional:
        tokens_np = dataset["per_game_tokens"]   # (n_games, max_tok_len) int32
        masks_np = dataset["per_game_masks"]     # (n_games, max_tok_len) bool

    # Pre-sample per-game eval subsets (per-game-LOCAL indices into
    # per_game_states[g]) for periodic diagnostics.
    per_game_eval: dict[int, np.ndarray] | None = None
    per_game_train_pool_sizes: list[int] | None = None
    if game_names is not None and len(game_names) > 1:
        per_game_eval = {}
        per_game_train_pool_sizes = []
        for g in range(n_games):
            n_g = int(per_game_n[g])
            if n_g == 0:
                per_game_train_pool_sizes.append(0)
                continue
            sub_rng = np.random.RandomState(seed + 1000 + g)
            size = min(n_g, per_game_eval_size)
            per_game_eval[g] = sub_rng.choice(n_g, size=size, replace=False)
            per_game_train_pool_sizes.append(n_g)
    # The old global-balanced sampler tracked per-game per-batch sizes here.
    # The new size-bucketed sampler computes its own per-bucket sizes below
    # (see `bucket_sizes_per_game`), so this block only handles the no-data
    # edge case.
    if balanced_sampling and sum(1 for g in range(n_games) if int(per_game_n[g]) > 0) < 2:
        print("  [balanced_sampling] Disabled: need >=2 games with data. Using uniform.")
        balanced_sampling = False

    n_data = int(per_game_n.sum())
    mode_str = "conditional" if conditional else "unconditional"
    n_wins = int(sum(int(w.sum()) for w in per_game_wons_np))
    print(f"Training ({mode_str}) on {n_data:,} transitions "
          f"(per-game native shapes, batch-padded to max ({max_C},{max_H},{max_W})), "
          f"batch_size={batch_size}, lr={lr}, wins={n_wins:,}/{n_data:,} "
          f"({100*n_wins/max(1,n_data):.3f}%)")
    print(f"  win_loss_weight={win_loss_weight}, win_pos_weight={win_pos_weight}")

    # === Size-bucketed batching ===
    # Quantize each game's native (H, W) up to the nearest power-of-2 anchor,
    # then group games sharing the same anchor into a bucket. Each batch is
    # padded only to its bucket's anchor shape, not the global max. Quantizing
    # (vs exact-shape buckets) caps the number of distinct batch shapes JAX
    # must JIT-compile train_step for — the difference between "warmup takes
    # minutes" and "warmup takes hours" on game sets with many distinct shapes
    # (gallery has 80+ unique (H, W) pairs).
    #
    # Power-of-2 anchors (min 8) cover all gallery games in 4-8 active buckets.
    # max_C stays global because the model's n_out is fixed at construction
    # time and the C-dim padding is "predict 0" — easy auxiliary task.
    #
    # Per-game expected exposure stays uniform (matches old balanced_sampling
    # semantics): bucket b is sampled with prob k_b/sum_k where k_b = #games in
    # bucket b, and within a bucket each game gets batch_size/k_b rows.
    def _next_pow2(x: int, min_val: int = 8) -> int:
        v = max(min_val, int(x))
        # round up to nearest power of 2
        p = 1
        while p < v:
            p <<= 1
        return p

    _hw_to_games: dict[tuple[int, int], list[int]] = {}
    for g in range(n_games):
        if int(per_game_n[g]) == 0:
            continue
        H_g = int(game_CHW[g, 1]); W_g = int(game_CHW[g, 2])
        key = (_next_pow2(H_g), _next_pow2(W_g))
        _hw_to_games.setdefault(key, []).append(g)
    bucket_hw = sorted(_hw_to_games.keys())
    bucket_games_lists = [_hw_to_games[k] for k in bucket_hw]
    bucket_n_games_arr = np.array([len(gs) for gs in bucket_games_lists], dtype=np.float64)
    bucket_n_trans_arr = np.array(
        [sum(int(per_game_n[g]) for g in gs) for gs in bucket_games_lists],
        dtype=np.float64,
    )
    # Bucket selection prob:
    #   balanced  -> proportional to game-count  (uniform per-game exposure)
    #   uniform   -> proportional to transition-count (uniform per-transition)
    if balanced_sampling:
        _bucket_probs = bucket_n_games_arr / bucket_n_games_arr.sum()
    else:
        _bucket_probs = bucket_n_trans_arr / bucket_n_trans_arr.sum()

    # Per-bucket per-game per-batch row counts (only for balanced sampling).
    bucket_sizes_per_game: list[list[int]] = []
    for gs in bucket_games_lists:
        n_in = len(gs)
        if balanced_sampling and n_in >= 1:
            base = batch_size // n_in
            rem = batch_size - base * n_in
            bucket_sizes_per_game.append(
                [base + (1 if i < rem else 0) for i in range(n_in)]
            )
        else:
            bucket_sizes_per_game.append([])  # uniform path doesn't use this

    # Pre-allocate per-bucket scratch buffers (sized to bucket H, W; global C).
    bucket_buffers = [
        {"s":  np.zeros((batch_size, max_C, H_b, W_b), dtype=np.uint8),
         "ns": np.zeros((batch_size, max_C, H_b, W_b), dtype=np.uint8)}
        for (H_b, W_b) in bucket_hw
    ]
    _batch_a = np.zeros((batch_size,), dtype=np.int32)
    _batch_w = np.zeros((batch_size,), dtype=np.uint8)
    _batch_g = np.zeros((batch_size,), dtype=np.int32)

    bucket_summary = ", ".join(
        f"({h}x{w}):{int(k)}g/{int(n):,}t"
        for (h, w), k, n in zip(bucket_hw, bucket_n_games_arr, bucket_n_trans_arr)
    )
    print(f"  [size_buckets] {len(bucket_hw)} (H,W) bucket(s): {bucket_summary}")

    # Cumulative transitions for uniform-within-bucket sampling.
    _bucket_cum_n = []
    for gs in bucket_games_lists:
        ns = np.array([int(per_game_n[g]) for g in gs], dtype=np.int64)
        _bucket_cum_n.append(np.concatenate([[0], np.cumsum(ns)]))

    # Init or resume model
    rng, init_rng = jax.random.split(rng)
    dummy_state = jnp.zeros((1, max_C, max_H, max_W), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, N_ACTIONS), dtype=jnp.float32)

    # Decoder is wired in only for rule_attn + token_decoder_loss_weight > 0.
    joint_decoder = None
    is_rule_attn = isinstance(model, RuleAttnNCAWorldModel)
    if conditional and is_rule_attn and token_decoder_loss_weight > 0:
        from nca_wm.token_decoder import SlotTokenDecoder
        max_tok_len = tokens_np.shape[1]
        joint_decoder = SlotTokenDecoder(
            vocab_size=model.vocab_size,
            max_seq_len=max_tok_len,
            d_model=decoder_d_model,
            n_heads=decoder_n_heads,
            n_layers=decoder_n_layers,
            d_slot=model.d_slot,
        )

    if init_params is not None:
        params = init_params
        print(f"Resuming from step {start_step:,}")
    else:
        if conditional:
            max_tok_len = tokens_np.shape[1]
            dummy_tokens = jnp.zeros((1, max_tok_len), dtype=jnp.int32)
            dummy_mask = jnp.zeros((1, max_tok_len), dtype=jnp.bool_)
            wm_params = model.init(init_rng, dummy_state, dummy_action,
                                   dummy_tokens, dummy_mask)
            if joint_decoder is not None:
                rng, dec_init_rng = jax.random.split(rng)
                dummy_slots = jnp.zeros((1, model.n_slots, model.d_slot),
                                        dtype=jnp.float32)
                dec_params = joint_decoder.init(dec_init_rng, dummy_tokens, dummy_slots)
                params = {"wm": wm_params, "dec": dec_params}
            else:
                params = wm_params
        else:
            params = model.init(init_rng, dummy_state, dummy_action)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model params: {n_params:,}")
    if joint_decoder is not None:
        n_wm = sum(p.size for p in jax.tree.leaves(_wm_p(params)))
        n_dec = n_params - n_wm
        print(f"  [joint decoder] wm={n_wm:,}  decoder={n_dec:,}  "
              f"(token_decoder_loss_weight={token_decoder_loss_weight})")

    # Optional LR schedule: constant by default (lr stays at the --lr value
    # the whole run). With --lr_schedule cosine, anneals lr from --lr down
    # to --lr_min over n_updates steps. Useful for sharp-minimum cases
    # where the model walks out of optima without decay.
    if lr_schedule == "cosine":
        lr_fn = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=n_updates, alpha=lr_min / max(lr, 1e-12)
        )
    else:
        lr_fn = lr

    if grad_clip > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(lr_fn),
        )
    else:
        optimizer = optax.adam(lr_fn)
    opt_state = optimizer.init(params)
    if change_loss_weight > 0:
        print(f"  [change_loss_weight={change_loss_weight}] changed-cell BCE upweighted "
              f"— each changed cell counts {1+change_loss_weight:.1f}× vs unchanged.")
    train_step = make_train_step(
        model, optimizer, conditional=conditional,
        win_loss_weight=win_loss_weight, win_pos_weight=win_pos_weight,
        sprite_loss_weight=sprite_loss_weight,
        change_loss_weight=change_loss_weight,
        decoder=joint_decoder,
        token_decoder_loss_weight=token_decoder_loss_weight,
        use_vq=use_vq,
        vq_commitment_weight=vq_commitment_weight,
        vq_loss_weight=vq_loss_weight,
        adaptive_halt=getattr(model, "adaptive_halt", False),
        halt_prior_p=halt_prior_p,
        halt_kl_weight=halt_kl_weight,
        halt_mode=halt_mode,
    )
    per_game_sprites_np = dataset.get("per_game_sprites")
    eval_forward = make_eval_forward(model, conditional=conditional)
    apply_fn = make_apply_fn(model)  # reused by intermittent gif rendering

    os.makedirs(save_dir, exist_ok=True)
    # Note: RUNNING.pid lock is written earlier in main() before dataset load,
    # so parallel launchers see the lock before they decide to launch.

    losses, accs, change_accs = [], [], []
    state_losses, win_losses, win_accs, win_recalls = [], [], [], []
    sprite_mses: list[float] = []
    dec_losses: list[float] = []
    dec_accs: list[float] = []
    vq_cb_losses: list[float] = []
    vq_commit_losses: list[float] = []
    vq_utils: list[float] = []
    # per_game_log[game_id] = list of dicts (step, loss, acc, change_acc)
    per_game_log: dict[int, list[dict]] = {g: [] for g in (per_game_eval or {})}
    t0 = time.time()
    np_rng = np.random.RandomState(seed + start_step)

    gif_enabled = (
        gif_interval > 0 and game_infos is not None
        and max_padded_shape is not None
        and ps_parser is not None
    )

    # Pre-compile renderer once (heavy: loads JS engine + sprite data).
    _gif_backend = None
    if gif_enabled:
        try:
            _gif_backend = CppPuzzleScriptBackend()
            _gif_backend.compile_game(ps_parser, game_infos[0]["name"])
        except Exception as e:
            print(f"  [gif] backend compile failed: {type(e).__name__}: {e}")
            gif_enabled = False

    def _maybe_render_gif(global_step: int, avg_cerr: float | None = None):
        if not gif_enabled:
            return
        g = game_infos[0]
        gt, gm = None, None
        if conditional:
            tlen = tokens_np.shape[1]
            tids = g.get("token_ids", [])
            padded = np.zeros(tlen, dtype=np.int32)
            mask = np.zeros(tlen, dtype=np.bool_)
            padded[:len(tids)] = tids
            mask[:len(tids)] = True
            gt, gm = padded, mask
        mC, mH, mW = max_padded_shape
        # Step is in filename; banner only needs the rollout-step counter
        # (added inside _render_training_gif).
        banner = ""
        path = os.path.join(save_dir, "train_gifs",
                             f"{g['name']}_step{global_step:08d}.gif")
        try:
            _render_training_gif(
                apply_fn, _wm_p(params), g,
                max_C=mC, max_H=mH, max_W=mW,
                save_path=path, n_steps=gif_n_steps,
                seed=global_step,
                conditional=conditional,
                game_tokens=gt, game_mask=gm,
                banner_text=banner,
                backend_render=_gif_backend,
            )
        except Exception as e:
            print(f"  [gif] render failed: {type(e).__name__}: {e}")

    # Early stopping state (monitors smoothed loss — less noisy than change_acc)
    best_loss = float("inf")
    patience_ref_loss = float("inf")  # reset when avg_loss improves by ≥ min_delta
    patience_counter = 0
    early_stopped = False

    # Render a GIF at step 0 (untrained baseline) if enabled.
    if gif_enabled and start_step == 0:
        _maybe_render_gif(start_step)

    def _sample_bucket_batch():
        """Pick a bucket, sample batch_size rows from games in that bucket,
        and populate the bucket's preallocated buffers. Returns:
            (b_idx, batch_s, batch_ns)
        plus mutates the global _batch_a / _batch_w / _batch_g.
        batch_s, batch_ns have shape (batch_size, max_C, H_b, W_b)."""
        b_idx = int(np_rng.choice(len(bucket_hw), p=_bucket_probs))
        games_b = bucket_games_lists[b_idx]
        sizes_b = bucket_sizes_per_game[b_idx]
        s_buf = bucket_buffers[b_idx]["s"]
        ns_buf = bucket_buffers[b_idx]["ns"]
        s_buf.fill(0); ns_buf.fill(0)

        # Build game_ids and local_idx for this batch
        if balanced_sampling and sizes_b:
            game_ids = np.empty(batch_size, dtype=np.int32)
            local_idx = np.empty(batch_size, dtype=np.int32)
            offset = 0
            for i, g in enumerate(games_b):
                sz = sizes_b[i]
                if sz == 0: continue
                game_ids[offset:offset + sz] = g
                local_idx[offset:offset + sz] = np_rng.randint(
                    0, int(per_game_n[g]), size=sz
                )
                offset += sz
        else:
            # Uniform within bucket: sample over flattened (g, j) pairs
            cum = _bucket_cum_n[b_idx]
            n_total_b = int(cum[-1])
            gi = np_rng.randint(0, n_total_b, size=batch_size)
            within_b_idx = np.searchsorted(cum[1:], gi, side="right")
            game_ids = np.array([games_b[k] for k in within_b_idx], dtype=np.int32)
            local_idx = (gi - cum[within_b_idx]).astype(np.int32)

        # Fill bucket-shaped buffers. per_game_states[g] is bitpacked along W
        # — unpack just the sampled row and write into the (H_g, W_g) corner
        # of the bucket buffer; surrounding cells stay zero from `s_buf.fill(0)`
        # above. Per-row unpack allocates ~C_g·H_g·W_g uint8s; the merged
        # in-RAM dataset stays packed.
        for i in range(batch_size):
            g = int(game_ids[i]); j = int(local_idx[i])
            C_g = int(game_CHW[g, 0]); H_g = int(game_CHW[g, 1]); W_g = int(game_CHW[g, 2])
            s_buf[i, :C_g, :H_g, :W_g]  = _unpack_states(per_game_states[g][j], W_g)
            ns_buf[i, :C_g, :H_g, :W_g] = _unpack_states(per_game_next_states[g][j], W_g)
            _batch_a[i] = per_game_actions_np[g][j]
            _batch_w[i] = per_game_wons_np[g][j]
            _batch_g[i] = g
        return b_idx, s_buf, ns_buf

    for step in range(n_updates):
        b_idx, _bs, _bns = _sample_bucket_batch()
        game_ids_batch = _batch_g
        s = jnp.array(_bs, dtype=jnp.float32)
        a_int = _batch_a
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a_int])
        ns = jnp.array(_bns, dtype=jnp.float32)
        w = jnp.array(_batch_w, dtype=jnp.float32)

        if conditional:
            gt = jnp.array(tokens_np[game_ids_batch])
            gm = jnp.array(masks_np[game_ids_batch])
            # Gather per-batch target sprites by game_id when sprite loss is on.
            target_sprites = None
            if sprite_loss_weight > 0 and per_game_sprites_np is not None:
                target_sprites = jnp.array(per_game_sprites_np[game_ids_batch])
            if joint_decoder is not None:
                # Recon target = the same token sequence the encoder consumed.
                tgt_tokens = gt
                tgt_mask = gm
                (params, opt_state, loss, state_loss, acc, change_acc,
                 win_loss, win_acc, win_recall, sprite_mse,
                 dec_loss, dec_acc,
                 vq_cb_loss, vq_commit_loss, vq_util) = train_step(
                    params, opt_state, s, a_oh, ns, w, gt, gm,
                    target_sprites, tgt_tokens, tgt_mask,
                )
            else:
                (params, opt_state, loss, state_loss, acc, change_acc,
                 win_loss, win_acc, win_recall, sprite_mse,
                 dec_loss, dec_acc,
                 vq_cb_loss, vq_commit_loss, vq_util) = train_step(
                    params, opt_state, s, a_oh, ns, w, gt, gm, target_sprites
                )
        else:
            (params, opt_state, loss, state_loss, acc, change_acc,
             win_loss, win_acc, win_recall, sprite_mse,
             dec_loss, dec_acc,
             vq_cb_loss, vq_commit_loss, vq_util) = train_step(
                params, opt_state, s, a_oh, ns, w
            )
        losses.append(float(loss))
        accs.append(float(acc))
        change_accs.append(float(change_acc))
        state_losses.append(float(state_loss))
        win_losses.append(float(win_loss))
        win_accs.append(float(win_acc))
        win_recalls.append(float(win_recall))
        sprite_mses.append(float(sprite_mse))
        dec_losses.append(float(dec_loss))
        dec_accs.append(float(dec_acc))
        vq_cb_losses.append(float(vq_cb_loss))
        vq_commit_losses.append(float(vq_commit_loss))
        vq_utils.append(float(vq_util))

        global_step = start_step + step + 1
        # Atomic periodic checkpoint so a concurrent --render_only process can load
        if ckpt_interval > 0 and global_step % ckpt_interval == 0:
            _atomic_save_checkpoint(save_dir, params, global_step)
        if global_step % log_interval == 0:
            avg_loss = np.mean(losses[-log_interval:])
            avg_state_loss = np.mean(state_losses[-log_interval:])
            avg_err = 1.0 - np.mean(accs[-log_interval:])
            avg_cerr = 1.0 - np.mean(change_accs[-log_interval:])
            avg_win_loss = np.mean(win_losses[-log_interval:])
            avg_win_err = 1.0 - np.mean(win_accs[-log_interval:])
            avg_win_recall = np.mean(win_recalls[-log_interval:])
            avg_sprite_mse = np.mean(sprite_mses[-log_interval:]) if sprite_mses else 0.0
            avg_dec_loss = np.mean(dec_losses[-log_interval:]) if dec_losses else 0.0
            avg_dec_acc = np.mean(dec_accs[-log_interval:]) if dec_accs else 0.0
            avg_vq_cb = np.mean(vq_cb_losses[-log_interval:]) if vq_cb_losses else 0.0
            avg_vq_commit = np.mean(vq_commit_losses[-log_interval:]) if vq_commit_losses else 0.0
            avg_vq_util = np.mean(vq_utils[-log_interval:]) if vq_utils else 0.0
            elapsed = time.time() - t0
            sprite_bit = f"  sprite_mse={avg_sprite_mse:.4e}" if sprite_loss_weight > 0 else ""
            dec_bit = (f"  dec_loss={avg_dec_loss:.4e}  dec_acc={avg_dec_acc:.3f}"
                       if joint_decoder is not None else "")
            vq_bit = (f"  vq_cb={avg_vq_cb:.4e}  vq_commit={avg_vq_commit:.4e}"
                      f"  vq_util={avg_vq_util:.1f}"
                      if use_vq else "")
            print(f"  step {global_step:,}/{start_step + n_updates:,}  loss={avg_loss:.4e}  "
                  f"state_loss={avg_state_loss:.4e}  err={avg_err:.4e}  "
                  f"change_err={avg_cerr:.4e}  win_loss={avg_win_loss:.4e}  "
                  f"win_err={avg_win_err:.4e}  win_recall={avg_win_recall:.3f}"
                  f"{sprite_bit}{dec_bit}{vq_bit}  ({elapsed:.1f}s)")
            if wandb.run is not None:
                wandb_log = {
                    "train/loss": avg_loss,
                    "train/state_loss": avg_state_loss,
                    "train/err": avg_err,
                    "train/change_err": avg_cerr,
                    "train/win_loss": avg_win_loss,
                    "train/win_err": avg_win_err,
                    "train/win_recall": avg_win_recall,
                }
                if sprite_loss_weight > 0:
                    wandb_log["train/sprite_mse"] = avg_sprite_mse
                if use_vq:
                    wandb_log["vq/codebook_loss"] = avg_vq_cb
                    wandb_log["vq/commit_loss"] = avg_vq_commit
                    wandb_log["vq/codebook_utilization"] = avg_vq_util
                wandb.log(wandb_log, step=global_step)

            # Per-game diagnostic pass (multi-game only, with game_names known).
            if per_game_eval is not None and global_step % per_game_eval_interval == 0:
                worst = ("", 1.0, 0.0)  # (name, change_acc, loss) — lowest change_acc
                for g, idx_g in per_game_eval.items():
                    # idx_g = LOCAL indices into per_game_states[g]; pad to global max.
                    N_eval = len(idx_g)
                    C_g = int(game_CHW[g, 0]); H_g = int(game_CHW[g, 1]); W_g = int(game_CHW[g, 2])
                    s_eval = np.zeros((N_eval, max_C, max_H, max_W), dtype=np.uint8)
                    ns_eval = np.zeros((N_eval, max_C, max_H, max_W), dtype=np.uint8)
                    s_eval[:, :C_g, :H_g, :W_g]  = _unpack_states(per_game_states[g][idx_g], W_g)
                    ns_eval[:, :C_g, :H_g, :W_g] = _unpack_states(per_game_next_states[g][idx_g], W_g)
                    s_g = jnp.array(s_eval, dtype=jnp.float32)
                    a_int_g = per_game_actions_np[g][idx_g]
                    a_oh_g = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a_int_g])
                    ns_g = jnp.array(ns_eval, dtype=jnp.float32)
                    if conditional:
                        gt_g = jnp.array(np.broadcast_to(tokens_np[g], (N_eval,) + tokens_np.shape[1:]))
                        gm_g = jnp.array(np.broadcast_to(masks_np[g], (N_eval,) + masks_np.shape[1:]))
                        pg_loss, pg_acc, pg_cacc = eval_forward(
                            params, s_g, a_oh_g, ns_g, gt_g, gm_g
                        )
                    else:
                        pg_loss, pg_acc, pg_cacc = eval_forward(
                            params, s_g, a_oh_g, ns_g
                        )
                    pg_loss, pg_acc, pg_cacc = float(pg_loss), float(pg_acc), float(pg_cacc)
                    per_game_log[g].append({
                        "step": global_step, "loss": pg_loss,
                        "acc": pg_acc, "change_acc": pg_cacc,
                    })
                    if pg_cacc < worst[1]:
                        worst = (game_names[g], pg_cacc, pg_loss)
                    if wandb.run is not None:
                        name = game_names[g]
                        wandb.log({
                            f"train/per_game/{name}/loss": pg_loss,
                            f"train/per_game/{name}/err": 1.0 - pg_acc,
                            f"train/per_game/{name}/change_err": 1.0 - pg_cacc,
                        }, step=global_step)
                if worst[0]:
                    print(f"    worst-fit game: {worst[0]}  "
                          f"change_err={1-worst[1]:.3e}  loss={worst[2]:.3e}")

            # Intermittent rollout GIF (true vs pred side-by-side)
            if gif_enabled and global_step % gif_interval == 0 and global_step > 0:
                _maybe_render_gif(global_step, avg_cerr=avg_cerr)

            # Periodically save curves so plot mode can pick up in-progress runs
            if global_step % 10_000 == 0:
                save_data = {
                    "losses": np.array(losses),
                    "accs": np.array(accs),
                    "change_accs": np.array(change_accs),
                    "vq_cb_losses": np.array(vq_cb_losses),
                    "vq_commit_losses": np.array(vq_commit_losses),
                    "vq_utils": np.array(vq_utils),
                }
                if per_game_log:
                    for g, rows in per_game_log.items():
                        if not rows:
                            continue
                        name = game_names[g]
                        save_data[f"per_game_{name}_step"] = np.array([r["step"] for r in rows])
                        save_data[f"per_game_{name}_loss"] = np.array([r["loss"] for r in rows])
                        save_data[f"per_game_{name}_acc"] = np.array([r["acc"] for r in rows])
                        save_data[f"per_game_{name}_change_acc"] = np.array(
                            [r["change_acc"] for r in rows])
                np.savez(
                    os.path.join(save_dir, f"curves_step{global_step}.npz"),
                    **save_data,
                )

            # Best-loss checkpoint (saved separately from the "latest"
            # checkpoint, so we don't lose the best params when the model
            # walks out of a sharp minimum). Save on ANY improvement.
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(save_dir, "params_best.pkl")
                tmp = best_path + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump(jax.device_get(params), f)
                os.replace(tmp, best_path)
                meta_path = os.path.join(save_dir, "train_meta.json")
                meta = {}
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                    except Exception:
                        meta = {}
                meta["best_loss"] = float(best_loss)
                meta["best_step"] = int(global_step)
                tmp_meta = meta_path + ".tmp"
                with open(tmp_meta, "w") as f:
                    json.dump(meta, f, indent=2)
                os.replace(tmp_meta, meta_path)

            # Early stopping: patience ticks when loss hasn't improved by
            # min_delta vs the last "reset" reference. This keeps patience
            # independent of best_loss tracking (which updates on any
            # improvement).
            if patience > 0:
                if avg_loss < patience_ref_loss - min_delta:
                    patience_ref_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at step {global_step:,}: loss "
                          f"has not improved by {min_delta} for "
                          f"{patience} eval windows ({patience * log_interval:,} steps). "
                          f"Best loss={best_loss:.4e} @ step ~{meta.get('best_step', '?')}")
                    early_stopped = True
                    break

    # Final atomic checkpoint
    final_step = start_step + (step + 1) if early_stopped else start_step + n_updates
    _atomic_save_checkpoint(save_dir, params, final_step,
                             early_stopped=early_stopped,
                             n_updates_requested=start_step + n_updates)
    ckpt_path = os.path.join(save_dir, "params.pkl")
    print(f"Saved params to {ckpt_path} (total steps: {final_step:,})")

    # Final curves save, including per-game arrays (the main() caller's save is
    # dropped in favor of this to avoid clobbering per-game data).
    save_data = {
        "losses": np.array(losses),
        "accs": np.array(accs),
        "change_accs": np.array(change_accs),
        "vq_cb_losses": np.array(vq_cb_losses),
        "vq_commit_losses": np.array(vq_commit_losses),
        "vq_utils": np.array(vq_utils),
    }
    if per_game_log:
        for g, rows in per_game_log.items():
            if not rows:
                continue
            name = game_names[g]
            save_data[f"per_game_{name}_step"] = np.array([r["step"] for r in rows])
            save_data[f"per_game_{name}_loss"] = np.array([r["loss"] for r in rows])
            save_data[f"per_game_{name}_acc"] = np.array([r["acc"] for r in rows])
            save_data[f"per_game_{name}_change_acc"] = np.array(
                [r["change_acc"] for r in rows])
    np.savez(os.path.join(save_dir, f"curves_step{final_step}.npz"), **save_data)

    return params, losses, accs, change_accs, final_step


# ---------------------------------------------------------------------------
# 4. Evaluation — multi-step rollout with the learned world model
# ---------------------------------------------------------------------------

def evaluate_world_model(
    model: NCAWorldModel,
    params,
    json_str: str,
    level_i: int = 0,
    n_episodes: int = 10,
    max_steps: int = 50,
    save_dir: str | None = None,
):
    """Roll out the world model alongside the real env and measure divergence."""
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_steps)
    apply_fn = make_apply_fn(model)

    all_l1_errors = []
    for ep_i in range(n_episodes):
        real_obs, _ = env.reset()
        pred_state = jnp.array(real_obs[None], dtype=jnp.float32)
        ep_errors = []

        for t in range(max_steps):
            action = np.random.randint(N_ACTIONS)
            a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

            # World model prediction
            logits, _win_logit, _sprite_logits = apply_fn(params, pred_state, a_oh)
            pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

            # Real env step
            real_obs, _, done, truncated, _ = env.step(action)
            real = jnp.array(real_obs[None], dtype=jnp.float32)

            l1 = float(jnp.abs(pred_state - real).sum())
            ep_errors.append(l1)
            if done or truncated:
                break

        all_l1_errors.append(ep_errors)

    # Report per-step average divergence
    max_len = max(len(e) for e in all_l1_errors)
    padded = np.full((n_episodes, max_len), np.nan)
    for i, e in enumerate(all_l1_errors):
        padded[i, :len(e)] = e
    mean_per_step = np.nanmean(padded, axis=0)
    print(f"Eval ({n_episodes} eps): step-1 L1={mean_per_step[0]:.1f}, "
          f"step-5 L1={mean_per_step[min(4, len(mean_per_step)-1)]:.1f}, "
          f"step-20 L1={mean_per_step[min(19, len(mean_per_step)-1)]:.1f}")

    if save_dir:
        np.savez(os.path.join(save_dir, "eval_divergence.npz"),
                 mean_per_step=mean_per_step, all_errors=padded)

    return mean_per_step


def _run_eval_rollout(
    apply_fn, params, json_str: str,
    level_i: int, n_objs: int,
    max_C: int, max_H: int, max_W: int,
    actions: list[int] | None = None,
    max_steps: int = 50,
    game_tokens: np.ndarray | None = None,
    game_mask: np.ndarray | None = None,
    teacher_forced: bool = False,
) -> dict:
    """Run a single eval rollout and return per-step metrics.

    Autoregressive (default): model feeds its own prediction back each step.
    Teacher-forced: model is fed the real (env) state each step. Isolates
    single-step prediction error from compounding drift.

    Returns dict with:
        wrong_tiles: (T,) int — per-bit mismatches per step
        wrong_cells: (T,) int — cells where ANY object-channel bit is wrong
        tile_error_rate: (T,) float — wrong_tiles / total_bits per step
        cell_error_rate: (T,) float — wrong_cells / (H*W) per step
        first_div_step: int — first t with wrong_cells > 0 (-1 if never)
        total_tiles: int — n_objs * H * W
        total_cells: int — H * W
    """
    conditional = game_tokens is not None
    if conditional:
        gt = jnp.array(game_tokens[None])  # (1, S)
        gm = jnp.array(game_mask[None])    # (1, S)

    env = CppPuzzleScriptEnv(json_str, level_i=level_i,
                             max_episode_steps=max_steps if actions is None else len(actions))
    real_obs, _ = env.reset()
    _, H, W = real_obs.shape
    total_tiles = n_objs * H * W
    total_cells = H * W
    pred_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)

    n_steps = len(actions) if actions else max_steps
    wrong_tiles = []
    wrong_cells = []
    first_div = -1
    for t in range(n_steps):
        action = actions[t] if actions else np.random.randint(N_ACTIONS)
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        if conditional:
            logits, _win_logit, _sprite_logits = apply_fn(params, pred_state, a_oh, gt, gm)
        else:
            logits, _win_logit, _sprite_logits = apply_fn(params, pred_state, a_oh)
        pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        real_obs, _, done, truncated, _ = env.step(action)

        # Slice predicted and real to the real (centered) extent
        _, pad_H, pad_W = pred_next.shape[1:]
        oy_c, _ = _pad_offsets(H, int(pad_H))
        ox_c, _ = _pad_offsets(W, int(pad_W))
        pred_binary = np.array(
            pred_next[0, :n_objs, oy_c:oy_c+H, ox_c:ox_c+W] > 0.5, dtype=np.uint8,
        )
        mismatch = (pred_binary != real_obs)
        n_wrong_bits = int(mismatch.sum())
        n_wrong_cells = int(mismatch.any(axis=0).sum())
        wrong_tiles.append(n_wrong_bits)
        wrong_cells.append(n_wrong_cells)
        if first_div == -1 and n_wrong_cells > 0:
            first_div = t

        # Next input: model's own prediction (autoregressive) or re-padded real
        # state (teacher-forced).
        if teacher_forced:
            pred_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)
        else:
            pred_state = pred_next

        if done or truncated:
            break

    wrong_tiles = np.array(wrong_tiles)
    wrong_cells = np.array(wrong_cells)
    return {
        "wrong_tiles": wrong_tiles,
        "wrong_cells": wrong_cells,
        "tile_error_rate": wrong_tiles / total_tiles,
        "cell_error_rate": wrong_cells / total_cells,
        "first_div_step": first_div,
        "total_tiles": total_tiles,
        "total_cells": total_cells,
    }


def evaluate_multigame(
    model: NCAWorldModel,
    params,
    game_infos: list[dict],
    ps_parser=None,
    n_random_episodes: int = 10,
    max_steps: int = 50,
    search_algos: list[str] = ("bfs", "astar"),
    search_n_steps: int = 100_000,
    search_timeout_ms: int = -1,
    save_dir: str | None = None,
):
    """Evaluate per game, per level, per rollout type (random + search).

    Reports tile discrepancy counts and error rates.
    """
    max_C = model.n_out
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
    conditional = isinstance(model, (ConditionalNCAWorldModel, RuleAttnNCAWorldModel))
    apply_fn = make_apply_fn(model)

    # Prepare padded token arrays for conditional eval
    if conditional:
        max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
        max_tok_len = max(max_tok_len, 1)

    def _get_token_data(info):
        if not conditional:
            return {}, {}
        tids = info.get("token_ids", [])
        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[:len(tids)] = tids
        mask[:len(tids)] = True
        return padded, mask

    results = {}  # results[game][level_i][rollout_type] = dict of metrics

    for info in game_infos:
        name = info["name"]
        json_str = info["json_str"]
        n_objs = info["n_objs"]
        n_levels = info["n_levels"]
        game_tokens, game_mask = _get_token_data(info)
        cond_kwargs = {}
        if conditional:
            cond_kwargs = {"game_tokens": game_tokens, "game_mask": game_mask}
        game_results = {}

        for level_i in range(n_levels):
            level_results = {}

            # --- Random rollouts (autoregressive + teacher-forced) ---
            for mode_name, tf in [("random", False), ("random_tf", True)]:
                all_bits = []
                all_cells = []
                first_divs = []
                for _ in range(n_random_episodes):
                    r = _run_eval_rollout(
                        apply_fn, params, json_str, level_i, n_objs,
                        max_C, max_H, max_W, max_steps=max_steps,
                        teacher_forced=tf, **cond_kwargs,
                    )
                    all_bits.append(r["wrong_tiles"])
                    all_cells.append(r["wrong_cells"])
                    first_divs.append(r["first_div_step"])

                max_len = max(len(e) for e in all_bits)
                bits_p = np.full((n_random_episodes, max_len), np.nan)
                cells_p = np.full((n_random_episodes, max_len), np.nan)
                for i, (b, c) in enumerate(zip(all_bits, all_cells)):
                    bits_p[i, :len(b)] = b
                    cells_p[i, :len(c)] = c
                mean_bits = np.nanmean(bits_p, axis=0)
                mean_cells = np.nanmean(cells_p, axis=0)
                # First-divergence: treat -1 (no divergence) as max_len (best case)
                fd = np.array([max_len if x < 0 else x for x in first_divs])
                level_results[mode_name] = {
                    "mean_error_rate": mean_bits / r["total_tiles"],
                    "mean_cell_error_rate": mean_cells / r["total_cells"],
                    "mean_wrong_tiles": mean_bits,
                    "mean_wrong_cells": mean_cells,
                    "mean_first_div": float(fd.mean()),
                    "total_tiles": r["total_tiles"],
                    "total_cells": r["total_cells"],
                }

            # --- Search rollouts ---
            backend_search = CppPuzzleScriptBackend()
            backend_search.load_from_json(json_str)
            for algo in search_algos:
                # Try cache first, then training-transitions extraction, then
                # actually re-run search. Both shortcuts yield action IDs in the
                # C++-backend convention, which is what `_run_eval_rollout`
                # consumes — so they're drop-in interchangeable.
                cache_path = os.path.join(
                    _cache_dir(name, level_i),
                    f"search_{algo}_{search_n_steps}_{search_timeout_ms}.npz"
                )
                cached = _load_npz_dict(cache_path)
                sol_actions = None
                if cached is not None and len(cached["actions"]) > 0:
                    sol_actions = cached["actions"].tolist()
                if sol_actions is None:
                    # Reuse the winning trajectory the training-time collector
                    # already explored. Saves up to `search_timeout_ms` of
                    # wall-clock per (game, level, algo) on hard games where
                    # search would otherwise time out at eval.
                    sol_actions = _solution_from_transitions_cache(name, level_i)
                if sol_actions is None:
                    # Pre-computed solutions from prior search_cpp / search_nodejs
                    # runs. cpp_sols use the same C++-backend action IDs as eval.
                    sol_actions = _solution_from_sol_dir(
                        os.path.join(_REPO_ROOT, "data", "cpp_sols"),
                        name, level_i, translate_js_to_jax=False,
                    )
                if sol_actions is None:
                    # js_sols use the JS-engine action convention; remap to JAX/CPP.
                    sol_actions = _solution_from_sol_dir(
                        os.path.join(_REPO_ROOT, "data", "js_sols"),
                        name, level_i, translate_js_to_jax=True,
                    )
                if sol_actions is None:
                    try:
                        backend_search.load_level("", level_i)
                        result = backend_search.run_search(
                            algo, game_text="", level_i=level_i,
                            n_steps=search_n_steps, timeout_ms=search_timeout_ms,
                        )
                        if not result.actions:
                            continue
                        sol_actions = list(result.actions)
                    except Exception:
                        continue
                # Persist whatever we ended up with so the next eval run
                # (this run or any other model trained on the same game) is
                # entirely search-free for this (algo, budget, timeout).
                if cached is None:
                    _save_npz_dict(cache_path, {
                        "actions": np.asarray(sol_actions, dtype=np.int32),
                    })

                r = _run_eval_rollout(
                    apply_fn, params, json_str, level_i, n_objs,
                    max_C, max_H, max_W, actions=sol_actions,
                    **cond_kwargs,
                )
                level_results[algo] = {
                    "error_rate": r["tile_error_rate"],
                    "cell_error_rate": r["cell_error_rate"],
                    "wrong_tiles": r["wrong_tiles"],
                    "wrong_cells": r["wrong_cells"],
                    "first_div_step": r["first_div_step"],
                    "total_tiles": r["total_tiles"],
                    "total_cells": r["total_cells"],
                    "n_steps": len(sol_actions),
                }

            game_results[level_i] = level_results

        results[name] = game_results

        # Print summary for this game
        for level_i, level_results in game_results.items():
            for rtype, metrics in level_results.items():
                wt = metrics.get("mean_wrong_tiles", metrics.get("wrong_tiles"))
                if wt is None:
                    continue
                total = metrics["total_tiles"]
                n = len(wt)
                w_mean = int(round(wt.mean())) if n > 0 else 0
                w_max = int(round(wt.max())) if n > 0 else 0
                fd = metrics.get("mean_first_div", metrics.get("first_div_step"))
                fd_str = f"  first_div={fd:.1f}" if fd is not None else ""
                print(f"  {name} L{level_i} {rtype:<10} "
                      f"wrong: mean={w_mean}  max={w_max}  "
                      f"({n} steps, {total} tiles){fd_str}")

    # Log to wandb
    if wandb.run is not None:
        for name, game_results in results.items():
            for level_i, level_results in game_results.items():
                for rtype, metrics in level_results.items():
                    wt = metrics.get("mean_wrong_tiles", metrics.get("wrong_tiles"))
                    if wt is not None and len(wt) > 0:
                        wandb.log({
                            f"eval/{name}/L{level_i}/{rtype}/mean_wrong": float(wt.mean()),
                            f"eval/{name}/L{level_i}/{rtype}/max_wrong": float(wt.max()),
                        }, commit=False)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Flatten to saveable arrays
        save_dict = {}
        for name, game_results in results.items():
            for level_i, level_results in game_results.items():
                for rtype, metrics in level_results.items():
                    key = f"{name}_L{level_i}_{rtype}"
                    er = metrics.get("mean_error_rate", metrics.get("error_rate"))
                    if er is not None:
                        save_dict[f"{key}_error_rate"] = er
                    cer = metrics.get("mean_cell_error_rate",
                                      metrics.get("cell_error_rate"))
                    if cer is not None:
                        save_dict[f"{key}_cell_error_rate"] = cer
                    if "wrong_tiles" in metrics:
                        save_dict[f"{key}_wrong_tiles"] = metrics["wrong_tiles"]
                    if "wrong_cells" in metrics:
                        save_dict[f"{key}_wrong_cells"] = metrics["wrong_cells"]
                    if "mean_wrong_cells" in metrics:
                        save_dict[f"{key}_mean_wrong_cells"] = metrics["mean_wrong_cells"]
                    if "mean_first_div" in metrics:
                        save_dict[f"{key}_mean_first_div"] = np.array(
                            metrics["mean_first_div"])
                    if "first_div_step" in metrics:
                        save_dict[f"{key}_first_div_step"] = np.array(
                            metrics["first_div_step"])
        np.savez(os.path.join(save_dir, "eval_multigame.npz"), **save_dict)

    return results


def _render_training_gif(
    apply_fn, params, game_info,
    *,
    max_C: int, max_H: int, max_W: int,
    save_path: str,
    backend_render,
    n_steps: int = 15,
    seed: int = 0,
    conditional: bool,
    game_tokens=None, game_mask=None,
    banner_text: str = "",
):
    """Render a short (real | predicted) side-by-side rollout GIF.

    Designed to be fast enough to run intermittently during training — a
    single short rollout (default 15 steps) rendered at native resolution.

    Uses the game's declared sprites for BOTH real and predicted states;
    the "prediction" is the model's autoregressive rollout, thresholded to
    multihot, with each cell rendered by its top set channel.
    """
    import imageio.v2 as imageio
    from puzzlescript_jax.font import (
        draw_text as _ps_draw_text, GLYPH_H_COMPACT,
    )

    rng = np.random.RandomState(seed)
    json_str = game_info["json_str"]
    n_objs = game_info["n_objs"]

    env = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=n_steps)
    real_obs, _ = env.reset()
    _, H, W = real_obs.shape

    # Load this level into the provided renderer (sprite data was pre-loaded
    # by the caller via compile_game).
    backend_render.load_level(game_text="", level_i=0)

    # Fetch learned sprite kernels once for the whole rollout. Only active
    # when the underlying model has sprite_decoder=True (otherwise the
    # sprite_logits slot is zeros and we fall back to declared sprites).
    learned_sprites_u8 = None
    if conditional and game_tokens is not None:
        gt0 = jnp.array(game_tokens[None])
        gm0 = jnp.array(game_mask[None])
        dummy_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)
        dummy_action = jnp.zeros((1, N_ACTIONS), dtype=jnp.float32)
        _, _, sprite_logits = apply_fn(params, dummy_state, dummy_action, gt0, gm0)
        sprite_probs = np.asarray(jax.nn.sigmoid(sprite_logits[0]))  # (n_out, 5, 5, 4)
        # Anything non-trivially non-zero? Treat zeros as "decoder off".
        if float(np.abs(sprite_probs).max()) > 1e-4:
            learned_sprites_u8 = (sprite_probs[:n_objs] * 255.0).clip(0, 255).astype(np.uint8)

    action_names = {0: "up", 1: "left", 2: "down", 3: "right", 4: "action"}

    def _render_multihot_declared(mh: np.ndarray) -> np.ndarray:
        """(n_objs, H, W) binary → (5H, 5W, 3) uint8, using the game's
        declared sprites via the C++ backend renderer."""
        return backend_render.render_frame_from_objects(
            _multihot_to_objects(mh), W, H
        )[..., :3]

    def _render_multihot_learned(mh: np.ndarray) -> np.ndarray:
        """(n_objs, H, W) binary → image using learned sprite kernels."""
        return _render_obs_with_sprite_kernels(
            mh.astype(np.float32), learned_sprites_u8
        )

    def _render_soft_declared(logits_np: np.ndarray, n_objs: int) -> np.ndarray:
        """Per-channel sigmoid-weighted alpha composite of the game's
        declared sprites. Used when no learned decoder is active."""
        probs = 1.0 / (1.0 + np.exp(-logits_np))
        out = np.zeros((H * 5, W * 5, 3), dtype=np.float32)
        for c in range(n_objs):
            channel_mh = np.zeros((n_objs, H, W), dtype=np.uint8)
            channel_mh[c] = 1
            sprite_frame = backend_render.render_frame_from_objects(
                _multihot_to_objects(channel_mh), W, H
            )[..., :3].astype(np.float32)
            p = probs[c]
            p_up = np.kron(p, np.ones((5, 5), dtype=np.float32))[..., None]
            out = out * (1.0 - p_up) + sprite_frame * p_up
        return np.clip(out, 0, 255).astype(np.uint8)

    def _render_soft_learned(logits_np: np.ndarray, n_objs: int) -> np.ndarray:
        """Soft render using learned sprite kernels and per-channel sigmoid
        weights. Pure numpy lookup-table compositing."""
        probs = 1.0 / (1.0 + np.exp(-logits_np))   # (n_objs, H, W)
        dummy_mh = np.ones_like(probs)             # obs ignored (soft_probs wins)
        return _render_obs_with_sprite_kernels(
            dummy_mh, learned_sprites_u8, soft_probs=probs,
        )

    # Select renderer: learned sprites if decoder is active, declared otherwise.
    if learned_sprites_u8 is not None:
        _render_pred_hard = _render_multihot_learned
        _render_pred_soft = _render_soft_learned
    else:
        _render_pred_hard = _render_multihot_declared
        _render_pred_soft = _render_soft_declared

    def render_triptych(real_obs_raw, pred_logits_np, pred_hard, step_i, action):
        # Real panel always uses the game's declared sprites (the ground truth).
        # Pred panels use learned sprites when the decoder is active — shows the
        # model's visual output, not the backend's.
        real_img = _render_multihot_declared(real_obs_raw)
        soft_img = _render_pred_soft(pred_logits_np, n_objs)
        hard_img = _render_pred_hard(pred_hard)

        gap = 4
        banner_h = GLYPH_H_COMPACT + 4
        hR, wR = real_img.shape[:2]
        H_total = hR + banner_h
        W_total = wR * 3 + gap * 2
        canvas = np.zeros((H_total, W_total, 3), dtype=np.uint8)
        canvas[banner_h:banner_h + hR, :wR] = real_img
        canvas[banner_h:banner_h + hR, wR + gap: 2 * wR + gap] = soft_img
        canvas[banner_h:banner_h + hR, 2 * (wR + gap):] = hard_img
        # Short action codes that render readable in the PuzzleScript font.
        # Lowercase 'v' is drawn as a left-leaning slash here, so use 'V'.
        short_act = {0: "^", 1: "<", 2: "V", 3: ">", 4: "x"}
        a_str = short_act.get(action, "-") if step_i > 0 else "r"
        line = f"t{step_i:02d} {a_str} {banner_text}"
        _ps_draw_text(canvas, line, x=2, y=2, color=(255, 255, 255),
                      compact=True)
        return canvas

    # Pad real_obs to model shape for the prediction stream
    pred_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)

    # At t=0 the "prediction" is just a copy of real (no model call yet);
    # render soft version from a dummy logits that will resolve to the real
    # state (all ones/zeros).
    dummy_logits = np.log(np.where(real_obs, 1e6, 1e-6)).astype(np.float32)
    frames = [render_triptych(real_obs, dummy_logits, real_obs, 0, -1)]
    if conditional:
        gt = jnp.array(game_tokens[None])
        gm = jnp.array(game_mask[None])

    for t in range(n_steps):
        action = int(rng.randint(N_ACTIONS))
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])
        if conditional:
            logits, _, _sprite_logits = apply_fn(params, pred_state, a_oh, gt, gm)
        else:
            logits, _, _sprite_logits = apply_fn(params, pred_state, a_oh)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        real_obs, _, done, truncated, _ = env.step(action)
        # Slice pred back to real extent
        _, pad_H, pad_W = pred_state.shape[1:]
        oy_c, _ = _pad_offsets(H, int(pad_H))
        ox_c, _ = _pad_offsets(W, int(pad_W))
        logits_np = np.array(
            logits[0, :n_objs, oy_c:oy_c + H, ox_c:ox_c + W]
        )
        pred_crop = np.array(
            pred_state[0, :n_objs, oy_c:oy_c + H, ox_c:ox_c + W] > 0.5,
            dtype=np.uint8,
        )
        frames.append(render_triptych(real_obs, logits_np, pred_crop,
                                       t + 1, action))
        if done or truncated:
            break

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.15, loop=0)
    return save_path


def _render_rollout_frames(
    model, params, apply_fn, json_str, backend_render,
    level_i, n_objs, grid_h, grid_w, max_H, max_W,
    n_steps, actions=None, label="", obj_names=None,
    game_tokens=None, game_mask=None,
):
    """Run a rollout and return labeled frames.

    Each frame is composed vertically:
      - Banner with label
      - Game renders: real | NCA prediction
      - Hidden activation grid (final NCA step)
      - Per-object output channel predictions
    """
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont

    conditional = game_tokens is not None
    if conditional:
        gt = jnp.array(game_tokens[None])
        gm = jnp.array(game_mask[None])

    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=n_steps)
    real_obs, _ = env.reset()
    pred_state = _pad_state_for_model(real_obs, model.n_out, max_H, max_W)
    frames = []

    # Model variant with intermediates
    pool_kwargs = dict(
        axis_pool=model.axis_pool, axis_cummax=model.axis_cummax,
        global_pool=model.global_pool,
    )
    if conditional:
        model_viz = ConditionalNCAWorldModel(
            n_hid=model.n_hid, n_steps=model.n_steps, n_out=model.n_out,
            vocab_size=model.vocab_size, d_model=model.d_model,
            n_heads=model.n_heads, n_enc_layers=model.n_enc_layers,
            d_z=model.d_z, max_seq_len=model.max_seq_len,
            return_intermediates=True, **pool_kwargs,
        )
    else:
        model_viz = NCAWorldModel(n_hid=model.n_hid, n_steps=model.n_steps,
                                  n_out=model.n_out, return_intermediates=True,
                                  **pool_kwargs)
    viz_fn = jax.jit(model_viz.apply)

    # Object names for labeling channels (pad with generic names for extra channels)
    if obj_names is None:
        obj_names = [f"ch{i}" for i in range(model.n_out)]
    while len(obj_names) < model.n_out:
        obj_names.append(f"pad{len(obj_names)}")

    try:
        font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except OSError:
        font = PIL.ImageFont.load_default()

    action_names = {0: "up", 1: "left", 2: "down", 3: "right", 4: "action"}
    # Per-frame durations: NCA intermediate steps are fast, game steps hold longer
    durations = []

    def _compose_frame(game_img, hid_img, banner_text):
        """Stack banner + game render + activation grid vertically."""
        sections = [game_img, hid_img]
        max_w = max(s.shape[1] for s in sections)
        text_bbox = font.getbbox(banner_text)
        text_w = text_bbox[2] - text_bbox[0] + 8
        bh = 18
        bw = max(max_w, text_w)
        banner = PIL.Image.new("RGB", (bw, bh), (0, 0, 0))
        draw = PIL.ImageDraw.Draw(banner)
        draw.text((4, 2), banner_text, fill=(255, 255, 255), font=font)
        final_w = max(bw, max_w)
        parts = [np.array(banner)]
        for s in sections:
            if s.shape[1] < final_w:
                p = np.zeros((s.shape[0], final_w - s.shape[1], 3), dtype=np.uint8)
                s = np.concatenate([s, p], axis=1)
            parts.append(s)
        return np.concatenate(parts, axis=0)

    max_steps = len(actions) if actions else n_steps
    last_obj_img = None
    for t in range(max_steps):
        action = actions[t] if actions else np.random.randint(N_ACTIONS)
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        # --- Game renders ---
        real_frame = backend_render.render_frame_from_objects(
            _multihot_to_objects(real_obs), grid_w, grid_h
        )
        pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
        pred_frame = backend_render.render_frame_from_objects(
            _multihot_to_objects(pred_obs), grid_w, grid_h
        )
        game_row = np.concatenate([real_frame, pred_frame], axis=1)

        # --- NCA internals ---
        if conditional:
            logits, win_logit, _sprite_logits, intermediates = viz_fn(params, pred_state, a_oh, gt, gm)
        else:
            logits, win_logit, _sprite_logits, intermediates = viz_fn(params, pred_state, a_oh)

        pred_won_prob = float(jax.nn.sigmoid(win_logit)[0])

        # Build per-NCA-step output channel images (only game object channels,
        # cropped from the centered pad).
        n_nca_steps = len(intermediates["readouts"])
        obj_imgs = []
        for readout in intermediates["readouts"]:
            pad_H, pad_W = int(readout.shape[2]), int(readout.shape[3])
            oy_c, _ = _pad_offsets(grid_h, pad_H)
            ox_c, _ = _pad_offsets(grid_w, pad_W)
            r_np = np.array(readout[0, :n_objs,
                                    oy_c:oy_c+grid_h, ox_c:ox_c+grid_w])
            obj_img = _labeled_channel_grid(r_np, obj_names[:n_objs])
            obj_imgs.append(obj_img)
        last_obj_img = obj_imgs[-1]

        # Scale up game renders to match the height of one object grid
        target_game_h = obj_imgs[0].shape[0]
        if game_row.shape[0] < target_game_h:
            scale_factor = target_game_h / game_row.shape[0]
            new_w = int(game_row.shape[1] * scale_factor)
            game_pil = PIL.Image.fromarray(game_row).resize(
                (new_w, target_game_h), PIL.Image.NEAREST
            )
            game_row = np.array(game_pil)

        # Emit one frame per NCA intermediate step (animated quickly)
        for nca_i, oimg in enumerate(obj_imgs):
            text = (f"{label} t={t} {action_names.get(action, '?')}  (real | NCA)  "
                    f"NCA {nca_i}/{n_nca_steps}  pred_win={pred_won_prob:.2f}")
            frame = _compose_frame(game_row, oimg, text)
            frames.append(frame)
            durations.append(0.05)
        durations[-1] = 0.3

        # Step NCA + env
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        real_obs, _, done, truncated, info = env.step(action)
        real_won = bool(info.get("won", False))

        if done or truncated:
            # Render the terminal (post-action) state with win annotations
            real_frame_f = backend_render.render_frame_from_objects(
                _multihot_to_objects(real_obs), grid_w, grid_h
            )
            pred_obs_f = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
            pred_frame_f = backend_render.render_frame_from_objects(
                _multihot_to_objects(pred_obs_f), grid_w, grid_h
            )
            game_row_f = np.concatenate([real_frame_f, pred_frame_f], axis=1)
            if game_row_f.shape[0] < target_game_h:
                new_w = int(game_row_f.shape[1] * (target_game_h / game_row_f.shape[0]))
                game_pil = PIL.Image.fromarray(game_row_f).resize(
                    (new_w, target_game_h), PIL.Image.NEAREST
                )
                game_row_f = np.array(game_pil)
            blank_act = np.zeros_like(last_obj_img)
            reason = "win" if real_won else ("done" if done else "truncated")
            text = (f"{label} terminal ({reason})  real_win={int(real_won)}  "
                    f"pred_win={pred_won_prob:.2f}")
            frame = _compose_frame(game_row_f, blank_act, text)
            frames.append(frame)
            durations.append(1.5)
            break

    return frames, durations


def render_multigame_gifs(
    model: NCAWorldModel,
    params,
    game_infos: list[dict],
    ps_parser,
    n_steps_per_game: int = 30,
    save_dir: str = ".",
    step_label: int | None = None,
    search_algos: list[str] = ("bfs", "astar"),
    search_n_steps: int = 100_000,
    search_timeout_ms: int = -1,
):
    """Render a single combined GIF: for each game and level, random rollout then search rollout.

    The GIF filename includes the training step count for easy comparison across checkpoints.
    """
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    conditional = isinstance(model, ConditionalNCAWorldModel)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare padded token arrays for conditional rendering
    if conditional:
        max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
        max_tok_len = max(max_tok_len, 1)

    def _get_token_kwargs(info):
        if not conditional:
            return {}
        tids = info.get("token_ids", [])
        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[:len(tids)] = tids
        mask[:len(tids)] = True
        return {"game_tokens": padded, "game_mask": mask}

    apply_fn = make_apply_fn(model)
    tag = f"_step{step_label}" if step_label is not None else ""

    for info in game_infos:
        name = info["name"]
        json_str = info["json_str"]
        n_objs = info["n_objs"]
        n_levels = info["n_levels"]
        cond_kwargs = _get_token_kwargs(info)

        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(ps_parser, name)

        # Get object names for this game
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
        obj_names = getattr(env0, "_canonical_ids", None)

        game_frames = []
        game_durations = []

        for level_i in range(n_levels):
            env_li = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=10)
            _, grid_h, grid_w = env_li.observation_shape

            # Random rollout
            label = f"{name} L{level_i} random"
            print(f"  {label}")
            frames, durs = _render_rollout_frames(
                model, params, apply_fn, json_str, backend_render,
                level_i, n_objs, grid_h, grid_w, max_H, max_W,
                n_steps_per_game, label=label, obj_names=obj_names,
                **cond_kwargs,
            )
            game_frames.extend(frames)
            game_durations.extend(durs)

            # Search rollout(s)
            backend_search = CppPuzzleScriptBackend()
            backend_search.load_from_json(json_str)
            for algo in search_algos:
                try:
                    backend_search.load_level("", level_i)
                    result = backend_search.run_search(
                        algo, game_text="", level_i=level_i,
                        n_steps=search_n_steps, timeout_ms=search_timeout_ms,
                    )
                    if not result.actions:
                        continue
                    label = f"{name} L{level_i} {algo} ({'win' if result.solved else 'no win'})"
                    print(f"  {label} ({len(result.actions)} steps)")
                    frames, durs = _render_rollout_frames(
                        model, params, apply_fn, json_str, backend_render,
                        level_i, n_objs, grid_h, grid_w, max_H, max_W,
                        n_steps=len(result.actions), actions=result.actions,
                        label=label, obj_names=obj_names,
                        **cond_kwargs,
                    )
                    game_frames.extend(frames)
                    game_durations.extend(durs)
                except Exception as e:
                    print(f"    {algo} L{level_i} failed: {e}")

        if game_frames:
            # Pad frames for this game to uniform size
            max_fh = max(f.shape[0] for f in game_frames)
            max_fw = max(f.shape[1] for f in game_frames)
            padded_frames = []
            for f in game_frames:
                pf = np.zeros((max_fh, max_fw, 3), dtype=np.uint8)
                pf[:f.shape[0], :f.shape[1]] = f
                padded_frames.append(pf)

            safe_name = name.replace(" ", "_")
            gif_path = os.path.join(save_dir, f"rollout_{safe_name}{tag}.gif")
            imageio.mimsave(gif_path, padded_frames, duration=game_durations, loop=0)
            print(f"Saved {name} rollout GIF ({len(padded_frames)} frames) to {gif_path}")
            if wandb.run is not None:
                wandb.log({f"eval/rollout_{safe_name}": wandb.Video(gif_path, fps=5, format="gif")})


def _render_obs_with_sprite_kernels(
    obs: np.ndarray,
    sprite_kernels: np.ndarray,
    *,
    soft_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Pure-numpy lookup-table renderer: (n_objs, H, W) + (n_objs, 5, 5, 4) → (5H, 5W, 3).

    - ``obs``: multihot float, used as the per-channel alpha multiplier.
      Typically 0/1 (discrete) but any [0,1] value works for soft rendering.
    - ``sprite_kernels``: (n_objs, 5, 5, 4) uint8 or float. Alpha = channel 4.
    - ``soft_probs`` (optional): (n_objs, H, W) in [0,1]; if provided, overrides
      obs for alpha weighting (lets us render the continuous sigmoid of model
      logits, not the thresholded state).

    Channels are painted in index order — later channels composited over earlier.
    No cross-channel weights: pure per-channel lookup, exactly the user's
    lookup-table formulation.
    """
    n_objs, H, W = obs.shape
    sh = sprite_kernels.shape[1]
    sw = sprite_kernels.shape[2]
    # Normalize kernels to float [0,1]
    if sprite_kernels.dtype == np.uint8:
        kernels_f = sprite_kernels.astype(np.float32) / 255.0
    else:
        kernels_f = np.asarray(sprite_kernels, dtype=np.float32)
    out = np.zeros((H * sh, W * sw, 3), dtype=np.float32)
    weights = obs if soft_probs is None else soft_probs
    for c in range(n_objs):
        rgb = kernels_f[c, ..., :3]           # (5, 5, 3)
        alpha = kernels_f[c, ..., 3:4]        # (5, 5, 1)
        for y in range(H):
            for x in range(W):
                w_c = float(weights[c, y, x])
                if w_c <= 0:
                    continue
                eff_alpha = alpha * w_c        # (5, 5, 1)
                patch = out[y * sh:(y + 1) * sh, x * sw:(x + 1) * sw]
                out[y * sh:(y + 1) * sh, x * sw:(x + 1) * sw] = (
                    patch * (1.0 - eff_alpha) + rgb * eff_alpha
                )
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


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


def _multihot_to_objects(obs: np.ndarray) -> np.ndarray:
    """Convert (n_objs, H, W) multihot to flat objects array for CPP renderer."""
    n_objs, grid_h, grid_w = obs.shape
    stride_obj = (n_objs + 31) // 32
    # Use uint32 for bitwise ops, then view as int32 for C++ compatibility
    objects = np.zeros(grid_w * grid_h * stride_obj, dtype=np.uint32)
    for x in range(grid_w):
        for y in range(grid_h):
            flat_idx = (x * grid_h + y) * stride_obj
            for obj_i in range(n_objs):
                if obs[obj_i, y, x]:
                    word = obj_i // 32
                    bit = obj_i % 32
                    objects[flat_idx + word] |= np.uint32(1 << bit)
    return objects.view(np.int32)


def play_world_model(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    level_i: int = 0,
    save_dir: str = "nca_wm/logs/play",
):
    """Interactive play: step through the NCA world model with keyboard input.

    Controls: w=up, a=left, s=down, d=right, x=action, r=restart, q=quit.
    Each step renders the predicted state as an image and saves a GIF at the end.
    """
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=10000)
    apply_fn = make_apply_fn(model)
    n_objs, grid_h, grid_w = env.observation_shape

    real_obs, _ = env.reset()
    pred_state = jnp.array(real_obs[None], dtype=jnp.float32)
    frames = []
    os.makedirs(save_dir, exist_ok=True)

    key_to_action = {"w": 0, "a": 1, "s": 2, "d": 3, "x": 4}
    action_names = {0: "up", 1: "left", 2: "down", 3: "right", 4: "action"}
    step_i = 0

    def _render_and_save(pred_state, real_obs, step_i):
        """Render predicted vs real side by side."""
        pred_obs = np.array(pred_state[0] > 0.5, dtype=np.uint8)
        try:
            # Real: step the backend env to match, render from engine
            real_frame = backend.render_frame()
            # Predicted: convert multihot to objects array and render
            pred_objects = _multihot_to_objects(pred_obs)
            pred_frame = backend.render_frame_from_objects(pred_objects, grid_w, grid_h)
            # Side by side: real | predicted
            combined = np.concatenate([real_frame, pred_frame], axis=1)
            frames.append(combined)
            frame_path = os.path.join(save_dir, f"step_{step_i:04d}.png")
            imageio.imwrite(frame_path, combined)
            return frame_path
        except Exception as e:
            print(f"  (render error: {e})")
            return None

    # Keep backend engine in sync for rendering
    backend.load_level("", level_i)

    print("\n--- NCA World Model: Interactive Play ---")
    print("Controls: w=up, a=left, s=down, d=right, x=action, r=restart, q=quit")
    print("Left side = real engine, Right side = NCA prediction\n")

    frame_path = _render_and_save(pred_state, real_obs, step_i)
    if frame_path:
        print(f"  Step {step_i}: initial state -> {frame_path}")

    while True:
        try:
            key = input(f"Step {step_i}> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if key == "q":
            break
        elif key == "r":
            real_obs, _ = env.reset()
            pred_state = jnp.array(real_obs[None], dtype=jnp.float32)
            backend.load_level("", level_i)
            step_i = 0
            frame_path = _render_and_save(pred_state, real_obs, step_i)
            print(f"  Restarted -> {frame_path}")
            continue
        elif key not in key_to_action:
            print(f"  Unknown key '{key}'. Use w/a/s/d/x/r/q.")
            continue

        action = key_to_action[key]
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        # NCA world model step
        logits, _win_logit, _sprite_logits = apply_fn(params, pred_state, a_oh)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        # Real env step (both gym env and backend engine for rendering)
        real_obs, _, done, _, info = env.step(action)
        backend.process_input(action)
        while backend.againing:
            backend.process_input(-1)
        step_i += 1

        # Divergence
        real_f = jnp.array(real_obs[None], dtype=jnp.float32)
        l1 = float(jnp.abs(pred_state - real_f).sum())

        frame_path = _render_and_save(pred_state, real_obs, step_i)
        status = f"  Step {step_i}: {action_names[action]}  L1={l1:.0f}"
        if info.get("won"):
            status += "  WIN!"
        if frame_path:
            status += f"  -> {frame_path}"
        print(status)

        if done:
            print("  Level complete!")

    # Save session as GIF
    if frames:
        gif_path = os.path.join(save_dir, "play_session.gif")
        imageio.mimsave(gif_path, frames, duration=0.3, loop=0)
        print(f"\nSaved play session GIF to {gif_path}")


def _pad_state_for_model(obs: np.ndarray, target_C: int,
                         target_H: int | None = None,
                         target_W: int | None = None) -> jnp.ndarray:
    """Pad a (C, H, W) observation to (1, target_C, target_H, target_W) for the model.

    Level is spatially centered inside the padded canvas.
    """
    C, H, W = obs.shape
    tH = target_H or H
    tW = target_W or W
    if C == target_C and H == tH and W == tW:
        return jnp.array(obs[None], dtype=jnp.float32)
    oy, _ = _pad_offsets(H, tH)
    ox, _ = _pad_offsets(W, tW)
    padded = np.zeros((1, target_C, tH, tW), dtype=np.float32)
    padded[0, :C, oy:oy+H, ox:ox+W] = obs
    return jnp.array(padded)


def _unpad_pred(pred_state: jnp.ndarray, n_objs: int,
                H: int | None = None, W: int | None = None) -> np.ndarray:
    """Extract (n_objs, H, W) uint8 from padded (1, model_n_out, pad_H, pad_W) prediction,
    cropping from the spatial center to match centered-pad convention."""
    cropped = pred_state[0, :n_objs]
    pad_H = int(cropped.shape[1])
    pad_W = int(cropped.shape[2])
    if H is not None and H < pad_H:
        oy, _ = _pad_offsets(H, pad_H)
        cropped = cropped[:, oy:oy+H, :]
    if W is not None and W < pad_W:
        ox, _ = _pad_offsets(W, pad_W)
        cropped = cropped[:, :, ox:ox+W]
    return np.array(cropped > 0.5, dtype=np.uint8)


def render_rollout_comparison(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    level_i: int = 0,
    n_steps: int = 30,
    actions: list[int] | None = None,
    save_path: str = "nca_wm_rollout.gif",
    pad_H: int | None = None,
    pad_W: int | None = None,
):
    """Render a side-by-side GIF: real env (top) vs world model prediction (bottom).

    If `actions` is provided, replays that sequence. Otherwise uses random actions.
    pad_H, pad_W: if set, pad observations spatially to these dims for the model.
    """
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont

    max_steps = len(actions) if actions else n_steps
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_steps)
    apply_fn = make_apply_fn(model)
    n_objs, grid_h, grid_w = env.observation_shape

    real_obs, _ = env.reset()
    pred_state = _pad_state_for_model(real_obs, model.n_out, pad_H, pad_W)
    frames = []

    try:
        font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except OSError:
        font = PIL.ImageFont.load_default()

    def _with_banner(img: np.ndarray, text: str) -> np.ndarray:
        w = img.shape[1]
        bh = 18
        banner = PIL.Image.new("RGB", (w, bh), (0, 0, 0))
        PIL.ImageDraw.Draw(banner).text((4, 2), text, fill=(255, 255, 255), font=font)
        return np.concatenate([np.array(banner), img], axis=0)

    pred_won_prob = 0.0
    for t in range(max_steps):
        action = actions[t] if actions else np.random.randint(N_ACTIONS)
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        real_frame = backend.render_frame_from_objects(
            _multihot_to_objects(real_obs), grid_w, grid_h
        )
        pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
        pred_frame = backend.render_frame_from_objects(
            _multihot_to_objects(pred_obs), grid_w, grid_h
        )
        combined = np.concatenate([real_frame, pred_frame], axis=0)
        text = f"t={t} action={action}  pred_win={pred_won_prob:.2f}  (real / NCA)"
        frames.append(_with_banner(combined, text))

        logits, win_logit, _sprite_logits = apply_fn(params, pred_state, a_oh)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        pred_won_prob = float(jax.nn.sigmoid(win_logit)[0])

        real_obs, _, done, truncated, info = env.step(action)
        real_won = bool(info.get("won", False))
        if done or truncated:
            real_frame = backend.render_frame_from_objects(
                _multihot_to_objects(real_obs), grid_w, grid_h
            )
            pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
            pred_frame = backend.render_frame_from_objects(
                _multihot_to_objects(pred_obs), grid_w, grid_h
            )
            combined = np.concatenate([real_frame, pred_frame], axis=0)
            reason = "win" if real_won else ("done" if done else "truncated")
            text = (f"terminal ({reason})  real_win={int(real_won)}  "
                    f"pred_win={pred_won_prob:.2f}")
            frames.append(_with_banner(combined, text))
            break

    if frames:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        imageio.mimsave(save_path, frames, duration=0.2, loop=0)
        print(f"Saved rollout GIF ({len(frames)} frames) to {save_path}")


def render_post_training_gifs(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    search_data: dict | None,
    level_i: int = 0,
    save_dir: str = ".",
    n_random_steps: int = 50,
):
    """Render comparison GIFs after training: one random rollout + one per search algo."""
    # Random rollout
    print("Rendering random rollout comparison GIF...")
    render_rollout_comparison(
        model, params, json_str, backend, level_i=level_i,
        n_steps=n_random_steps,
        save_path=os.path.join(save_dir, "random_rollout.gif"),
    )

    # Search trajectories
    if search_data is not None and len(search_data["states"]) > 0:
        print("Rendering search trajectory comparison GIF...")
        search_actions = search_data["actions"].tolist()
        render_rollout_comparison(
            model, params, json_str, backend, level_i=level_i,
            actions=search_actions,
            save_path=os.path.join(save_dir, "search_rollout.gif"),
        )


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train NCA world model on a PuzzleScript game")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--game", help="Single game name (e.g. pipe_bend, sokoban_basic)")
    g.add_argument("--games", help="Comma-separated game names, or a preset name (e.g. 'small')")
    p.add_argument("--level", type=int, default=None,
                   help="Train on a single level index. Default: all levels.")
    p.add_argument("--train_levels", default=None,
                   help="Comma-separated level indices to train on (e.g. '0,1,8'). "
                        "Overrides --level when set. Eval still runs over all "
                        "levels per-game so held-out levels are reported as "
                        "out-of-distribution metrics.")
    # Data collection (search-driven unique-transition exploration; see
    # collect_unique_transitions). --search_algos (plural) is for evaluation
    # rollouts only, not training data collection.
    p.add_argument("--n_search_steps", type=int, default=100_000)
    p.add_argument("--search_timeout_ms", type=int, default=-1)
    p.add_argument("--search_algos", nargs="+", default=["bfs", "astar"],
                   help="Algorithms used during evaluation/GIF rollouts.")
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--search_algo", default="astar", choices=["astar", "bfs"],
                   help="Search algorithm for training-data collection (default: astar)")
    # Architecture
    p.add_argument("--n_nca_steps", type=int, default=4,
                   help="Number of NCA update steps per forward pass")
    p.add_argument("--n_hid", type=int, default=256)
    # Conditional model
    p.add_argument("--conditional", action=argparse.BooleanOptionalAction, default=True,
                   help="Use ConditionalNCAWorldModel with game-spec encoder. "
                        "Pass --no-conditional for the unconditional baseline.")
    p.add_argument("--d_z", type=int, default=64, help="Latent dimension for game encoder")
    p.add_argument("--d_model", type=int, default=64, help="Transformer hidden dim")
    p.add_argument("--n_enc_layers", type=int, default=2, help="Transformer encoder layers")
    p.add_argument("--n_heads", type=int, default=4, help="Transformer attention heads")
    p.add_argument("--architecture", type=str, default="rule_attn",
                   choices=["film", "rule_attn"],
                   help="Conditioning architecture. 'rule_attn' (default) = K "
                        "slot vectors from perceiver-style encoder; cells "
                        "cross-attend to slots each NCA step. 'film' = "
                        "pooled-z FiLM on NCA.")
    p.add_argument("--n_slots", type=int, default=16,
                   help="Number of rule slots for --architecture rule_attn.")
    p.add_argument("--d_slot", type=int, default=64,
                   help="Per-slot dim for --architecture rule_attn.")
    p.add_argument("--n_app_slots", type=int, default=1,
                   help="Of the --n_slots, how many are appearance-only "
                        "(decoder sees, NCA does not). Encourages dynamics "
                        "and visual info to occupy disjoint slot subsets.")
    # VQ-VAE-style codebook on the encoder slots. Off by default — enabling
    # adds a `slot_vq/codebook` param and two VQ losses; default-off path is
    # parameter-identical to the pre-VQ model so existing checkpoints load.
    p.add_argument("--vq_codebook", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="If set, quantize encoder slots to a learned shared "
                        "codebook (VQ-VAE style) before they enter the NCA. "
                        "Only valid with --architecture rule_attn.")
    p.add_argument("--vq_codebook_size", type=int, default=512,
                   help="Number of entries in the slot codebook.")
    p.add_argument("--vq_commitment_weight", type=float, default=0.25,
                   help="Beta in vq_total = codebook_loss + beta*commitment_loss.")
    p.add_argument("--vq_loss_weight", type=float, default=1.0,
                   help="Multiplier on vq_total when added to the training loss.")
    # Joint token-decoder training (encoder is shared with WM; decoder
    # cross-attends to ALL slots, while NCA only sees the dyn slots).
    p.add_argument("--token_decoder_loss_weight", type=float, default=0.0,
                   help="If >0, co-train an AR token decoder (recovering the "
                        "PuzzleScript source from the encoder slots) and add "
                        "this scaled cross-entropy to the WM loss.")
    p.add_argument("--decoder_d_model", type=int, default=128)
    p.add_argument("--decoder_n_layers", type=int, default=4)
    p.add_argument("--decoder_n_heads", type=int, default=4)
    # Training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_updates", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--patience", type=int, default=300,
                   help="Early stopping patience (in eval windows of log_interval steps). "
                        "0 disables early stopping.")
    p.add_argument("--min_delta", type=float, default=1e-8,
                   help="Minimum change_acc improvement to reset patience counter")
    p.add_argument("--win_loss_weight", type=float, default=1.0,
                   help="Scalar multiplier on the win-prediction BCE term (0 disables the head's contribution to grads)")
    p.add_argument("--win_pos_weight", type=float, default=1.0,
                   help="Positive-class weight for the win BCE (raise to counter class imbalance; "
                        "a good rule of thumb is ~(#negatives / #positives))")
    p.add_argument("--ckpt_interval", type=int, default=1000,
                   help="Steps between periodic atomic saves of params.pkl during training. "
                        "Lower means a --render_only process sees fresher weights.")
    p.add_argument("--balanced_sampling", action=argparse.BooleanOptionalAction, default=True,
                   help="Multi-game only: draw each batch with equal share per game "
                        "(replaces uniform-over-transitions sampling). Compensates "
                        "for per-game dataset-size imbalance. Pass --no-balanced_sampling "
                        "to fall back to uniform-over-transitions.")
    p.add_argument("--encode_sprites", action="store_true",
                   help="Include each object's palette + 5x5 sprite in the "
                        "game-spec token sequence (uses VOCAB_SIZE_EXT and a "
                        "larger max_seq_len). Required to decode visual games "
                        "from the latent space.")
    p.add_argument("--sprite_loss_weight", type=float, default=0.0,
                   help="Weight on a sprite-decoder MSE loss term. When >0, "
                        "adds a Dense head on z predicting each channel's "
                        "5x5x4 RGBA kernel; loss = MSE vs target sprite "
                        "from OBJECTS section. (Decoder head itself lands in "
                        "a follow-up; flag is plumbed for dataset-side prep.)")
    p.add_argument("--change_loss_weight", type=float, default=5.0,
                   help="Extra weight on changed cells in state BCE. 0 = uniform "
                        "mean. >0 = each changed cell counts "
                        "(1 + change_loss_weight)x vs unchanged. Default 5.0; "
                        "needed to escape identity collapse on multi-game sets "
                        "where most cells don't change between t and t+1.")
    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["constant", "cosine"],
                   help="LR schedule. 'cosine' (default) anneals from --lr down "
                        "to --lr_min over n_updates steps. 'constant' keeps --lr.")
    p.add_argument("--lr_min", type=float, default=1e-7,
                   help="Floor LR for cosine schedule. Ignored for constant.")
    p.add_argument("--gif_interval", type=int, default=0,
                   help="Render an intermittent (real | pred) rollout GIF "
                        "every N training steps (also at step 0). 0 disables.")
    p.add_argument("--gif_n_steps", type=int, default=15,
                   help="Length of each intermittent training GIF rollout.")
    p.add_argument("--grad_clip", type=float, default=0.5,
                   help="Clip gradients by global norm to this value (0 = off).")
    p.add_argument("--use_layernorm", action="store_true",
                   help="Apply a shared LayerNorm on h between NCA steps "
                        "(post-step on FiLM/uncond, pre-step on rule_attn). "
                        "Stabilizes deep unrolls (large n_nca_steps).")
    p.add_argument("--input_skip", action="store_true",
                   help="rule_attn-only: re-inject the embedded (state, "
                        "action) input into the conv at every NCA step. "
                        "Mirrors the input-skip in NCAWorldModel; helps "
                        "deep unrolls keep contact with the original "
                        "observation. No-op for FiLM/uncond (those already "
                        "have the input skip).")
    p.add_argument("--adaptive_halt", action="store_true",
                   help="rule_attn-only: enable PonderNet-style adaptive "
                        "halting. The model emits a halting probability at "
                        "every NCA step from a small head over pooled `h`. "
                        "Loss is the expected per-step heads-loss under the "
                        "induced halt distribution, plus a KL-to-geometric "
                        "prior regularizer (--halt_prior_p / --halt_kl_weight). "
                        "Requires --n_nca_repeats=n_nca_steps for the "
                        "halt-at-step-k semantic to be coherent (one shared "
                        "rule layer applied k times); otherwise different "
                        "halts pick between distinct per-step rule layers. "
                        "Currently supported only in the conditional, "
                        "non-VQ, non-joint-decoder training path.")
    p.add_argument("--halt_prior_p", type=float, default=0.1,
                   help="Geometric-prior parameter for adaptive halt. "
                        "Smaller → encourages later halt (longer rollouts). "
                        "Default 0.1 → expected ~10 steps under the prior.")
    p.add_argument("--halt_kl_weight", type=float, default=0.01,
                   help="Weight on KL(p || Geom(halt_prior_p)) in the "
                        "ponder loss. Larger → stronger pressure toward the "
                        "geometric prior; smaller → halt distribution is "
                        "fit to data with less regularization. Ignored "
                        "when --halt_mode=uniform.")
    p.add_argument("--halt_mode",
                   choices=["ponder", "uniform", "argmax_st", "convergence_st"],
                   default="ponder",
                   help="Per-step loss aggregation under --adaptive_halt. "
                        "'ponder' (default) is PonderNet-style with the "
                        "learned halt distribution + KL prior. 'uniform' "
                        "weights every step equally — pre-requisite for "
                        "convergence-based stopping at inference, since "
                        "the body has to be good at every depth (not just "
                        "at the expected halt step). Set --halt_kl_weight=0 "
                        "with uniform. 'argmax_st' uses straight-through "
                        "estimator on the argmax of p: forward computes L "
                        "only at the single selected step, backward updates "
                        "halt logits via soft p. Removes the shortcut "
                        "pressure of weighing every k, but body specialises "
                        "at one depth per example — keep KL prior on to "
                        "prevent halt collapse to k=1. 'convergence_st' "
                        "selects the first k at which the binary readout "
                        "has converged (fraction of cells flipping below "
                        "halt_prior_p), then computes L only at that step. "
                        "No halt-head signal is used; body gradient flows "
                        "through k* iterations only. Training and inference "
                        "share the same halting rule mechanically.")
    p.add_argument("--n_nca_repeats", type=int, default=1,
                   help="rule_attn-only: factor n_nca_steps into a "
                        "(n_layers × n_repeats) hierarchy mirroring the "
                        "PuzzleScript engine's two loop levels. Inner block "
                        "of n_layers = n_nca_steps // n_nca_repeats distinct "
                        "rule-application layers (each with its own weights, "
                        "analogous to one ordered rule list). Outer loop "
                        "applies the inner block n_nca_repeats times, sharing "
                        "weights across repeats — analogous to the engine's "
                        "`again` loop. Defaults (n_nca_repeats=1) reproduce "
                        "the historical per-step body bit-identically; "
                        "n_nca_repeats=n_nca_steps is fully shared (one rule "
                        "layer applied n_steps times). Constraint: "
                        "n_nca_steps must be divisible by n_nca_repeats. "
                        "No-op for FiLM/uncond.")
    p.add_argument("--max_transitions_per_game", type=int, default=200_000,
                   help="Cap per-game transition count (uniformly subsample). "
                        "0 disables. Essential for scaling to many games with "
                        "disparate sizes — prevents dataset OOM.")
    # Architectural pool flags (see _pool_features). Independent booleans;
    # any combination may be active.
    p.add_argument("--axis_pool", action=argparse.BooleanOptionalAction, default=True,
                   help="Inject row-max + col-max pooled features at each NCA step "
                        "(handles `[ X | ... | Y ]` style rules — X/Y in same row/col).")
    p.add_argument("--axis_cummax", action=argparse.BooleanOptionalAction, default=True,
                   help="Inject directional cumulative-max (L→R, R→L, T→B, B→T) at "
                        "each NCA step (more expressive variant of axis_pool).")
    p.add_argument("--global_pool", action=argparse.BooleanOptionalAction, default=True,
                   help="Inject grid-global max-pool features at each NCA step "
                        "(handles `[X] [Y]` multi-bracket rules — X and Y both "
                        "exist somewhere on the level).")
    # Output
    # Synthetic-level generation. When --synthetic_levels > 0, replace the
    # authored levels of each requested game with N procedurally-generated
    # valid levels. The generator is game-agnostic: tile patterns are sampled
    # from the empirical distribution observed in the game's authored levels,
    # then exactly one player is forced. Validity = (not already winning,
    # solvable within BFS budget, ≥ min_states reachable, no timeout).
    p.add_argument("--synthetic_levels", type=int, default=0,
                   help="If >0, generate N synthetic levels instead of using authored levels")
    p.add_argument("--synthetic_w", type=int, default=7, help="Width of synthetic levels")
    p.add_argument("--synthetic_h", type=int, default=7, help="Height of synthetic levels")
    p.add_argument("--synthetic_seed", type=int, default=0, help="Seed for synthetic level generation")
    p.add_argument("--synthetic_min_states", type=int, default=20,
                   help="Min reachable BFS states for a synthetic level to be accepted")
    p.add_argument("--synthetic_mode", type=str, default="tile_pattern_empirical",
                   choices=["tile_pattern_empirical", "tile_pattern_uniform", "evolve"],
                   help="Level-finding strategy. 'tile_pattern_*' = rejection sampling "
                        "from the per-tile pattern distribution; 'evolve' = population GA "
                        "with BFS-iterations fitness (use for harder games where rejection "
                        "sampling has very low acceptance).")
    p.add_argument("--synthetic_per_game_size", action="store_true",
                   help="Auto-detect per-game synth grid size from each game's authored max-dim. "
                        "Empirically (see RUNNING_REPORT) only synth at the authored max-dim "
                        "transfers cleanly to authored levels — smaller misses rule structure, "
                        "bigger learns position-padding artifacts. Overrides --synthetic_w/h.")
    p.add_argument("--synthetic_multi_grid", action="store_true",
                   help="Generate at multiple grid sizes per game and merge via spatial padding. "
                        "Closes the residual gap on games with mixed authored sizes. Combined with "
                        "--synthetic_grid_sizes for explicit size list, or default uses authored sizes.")
    p.add_argument("--synthetic_grid_sizes", type=str, default=None,
                   help='Explicit comma-separated grid sizes for multi_grid: "5x5,7x7,9x9". '
                        'Overrides authored-size detection — recommended since authored sizes '
                        'can be very large (TSP 20x19, kettle 15x15) and synth at those defeats '
                        'the purpose of avoiding big-grid BFS. Hardcoded small sizes train at '
                        'tractable BFS depths and rely on size-up generalization to bigger eval levels.')
    p.add_argument("--synthetic_fallback_dynamics", action="store_true",
                   help="If a game produces 0 levels with require_solvable=True (e.g. Zen at "
                        "small grids), retry once with require_solvable=False so the multi-game "
                        "pipeline still gets dynamics-only data for that game.")
    p.add_argument("--synthetic_no_a_count_max", type=int, default=3,
                   help="For 'no A' (num=-1) win conditions, cap count(A) at start to this value "
                        "so BFS reachable-state-space stays tractable. Default 3 enables Zen-class "
                        "synth gen; sweep over {3,5,8,12} to balance solvability vs distribution match.")
    p.add_argument("--synthetic_evolve_pop_size", type=int, default=64)
    p.add_argument("--synthetic_evolve_max_generations", type=int, default=200)
    p.add_argument("--synthetic_evolve_n_mutations_min", type=int, default=1)
    p.add_argument("--synthetic_evolve_n_mutations_max", type=int, default=3)
    p.add_argument("--synthetic_require_solvable", action=argparse.BooleanOptionalAction, default=True,
                   help="Reject levels with no winning transition observed within BFS budget. "
                        "Default True so the wons head sees positives; pass --no-synthetic_require_solvable "
                        "to keep all valid-dynamics levels (will produce wons=0 always, head will collapse).")
    p.add_argument("--synthetic_max_attempts_per_level", type=int, default=1000,
                   help="Total attempt budget = n_levels * this")
    p.add_argument("--synthetic_max_iters_search", type=int, default=5000,
                   help="BFS budget per synthetic level during validity check + transition collection")
    p.add_argument("--synthetic_timeout_ms_search", type=int, default=2000)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--render_gif", action="store_true", help="Render comparison GIF after training")
    p.add_argument("--render_only", action="store_true",
                   help="Skip data collection and training; load the latest checkpoint "
                        "from save_dir (or --load) and run eval + rendering only. "
                        "Use this alongside the same training args to inspect a "
                        "partially-trained model without disturbing the training run.")
    p.add_argument("--load", default=None, metavar="DIR",
                   help="Load params from this dir instead of the default save_dir")
    p.add_argument("--play", action="store_true",
                   help="Interactive play mode (w/a/s/d/x keys)")
    # Logging
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", default="nca-world-model", help="wandb project name")
    p.add_argument("--wandb_name", default=None, help="wandb run name (auto-generated if not set)")
    p.add_argument("--sweep_name", default=None,
                   help="Tag for grouping runs in sweep_nca_wm.py cross-evaluation")
    args = p.parse_args()
    parsed_train_levels = None
    if args.train_levels:
        parsed_train_levels = [int(x) for x in args.train_levels.split(",") if x.strip()]

    if args.vq_codebook and args.architecture != "rule_attn":
        p.error("--vq_codebook is only supported with --architecture rule_attn")
    if args.vq_codebook and not args.conditional:
        p.error("--vq_codebook requires --conditional (slots come from the "
                "game encoder, which only exists in conditional mode)")
    if args.adaptive_halt:
        if args.architecture != "rule_attn":
            p.error("--adaptive_halt is only supported with --architecture rule_attn")
        if not args.conditional:
            p.error("--adaptive_halt requires --conditional (the v1 ponder loss "
                    "is wired only into the conditional training path)")
        if args.vq_codebook:
            p.error("--adaptive_halt + --vq_codebook is not yet supported")
        if args.token_decoder_loss_weight > 0:
            p.error("--adaptive_halt + joint token decoder is not yet supported")
        if args.n_nca_repeats != args.n_nca_steps:
            p.error("--adaptive_halt requires --n_nca_repeats == --n_nca_steps "
                    f"(one shared rule layer applied n_nca_steps times); got "
                    f"n_nca_repeats={args.n_nca_repeats} vs n_nca_steps={args.n_nca_steps}")

    ps_parser = init_ps_lark_parser()
    multigame = args.games is not None

    if multigame:
        # --- Multi-game path ---
        if args.games == "gallery":
            # Full PuzzleScript gallery dataset via the shared helper.
            # Also add NCAWM-specific extras (games we reference a lot in
            # research but which aren't in games_dat.js or PRIORITY_GAMES).
            from puzzlescript_jax.utils import get_list_of_games_for_testing
            NCAWM_EXTRAS = ["nekopuzzle"]
            game_names = list(get_list_of_games_for_testing(dataset="gallery"))
            for g in NCAWM_EXTRAS:
                if g not in game_names:
                    game_names.append(g)
            preset_tag = "gallery"
        elif args.games in MULTI_GAME_PRESETS:
            game_names = MULTI_GAME_PRESETS[args.games]
            preset_tag = args.games
        else:
            game_names = [g.strip() for g in args.games.split(",")]
            preset_tag = f"{len(game_names)}games"

        # Tag only deviations from the canonical recipe; a default run gets a clean name.
        parts = []
        if not args.conditional: parts.append("uncond")
        if not args.balanced_sampling: parts.append("uniform")
        if not args.axis_pool: parts.append("no-ap")
        if not args.axis_cummax: parts.append("no-ac")
        if not args.global_pool: parts.append("no-gp")
        if args.encode_sprites: parts.append("spr")
        if args.change_loss_weight != 5.0: parts.append(f"clw{args.change_loss_weight:g}")
        if args.architecture != "rule_attn": parts.append(f"arch-{args.architecture}")
        if args.n_nca_repeats != 1: parts.append(f"rep{args.n_nca_repeats}")
        if args.adaptive_halt:
            if args.halt_mode == "uniform":
                parts.append("halt-uniform")
            else:
                parts.append(f"halt-p{args.halt_prior_p:g}-kl{args.halt_kl_weight:g}")
        if args.vq_codebook:
            parts.append(f"vq{args.vq_codebook_size}")
        if args.lr_schedule != "cosine": parts.append(f"lr-{args.lr_schedule}")
        if args.grad_clip != 0.5: parts.append(f"gc{args.grad_clip:g}")
        if args.synthetic_levels > 0:
            parts.append(
                f"synth{args.synthetic_levels}-{args.synthetic_w}x{args.synthetic_h}"
                f"-{args.synthetic_mode.replace('tile_pattern_', 'tp-')}"
                + ("-solv" if args.synthetic_require_solvable else "-any")
                + f"-s{args.synthetic_seed}"
            )
        recipe_tag = ("_" + "_".join(parts)) if parts else ""
        patience_tag = f"_pat-{args.patience}" if args.patience != 300 else ""
        save_dir = (args.save_dir or
                    f"nca_wm/logs/multi_{preset_tag}{recipe_tag}_level-{args.level}"
                    f"_nca-{args.n_nca_steps}_hid-{args.n_hid}_lr-{args.lr}"
                    f"{patience_tag}_s-{args.seed}")

        # Write config.json early so monitoring tools can see in-flight runs.
        # Overwritten verbatim at end of training (same content).
        # Also write RUNNING.pid early (before dataset load) so parallel
        # launchers' skip-check sees the lock before the next process
        # decides to launch. atexit cleanup removes it on exit.
        if not args.render_only:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            lock_path_early = os.path.join(save_dir, "RUNNING.pid")
            try:
                with open(lock_path_early, "w") as f:
                    f.write(str(os.getpid()))
                import atexit
                atexit.register(lambda: os.path.isfile(lock_path_early) and os.remove(lock_path_early))
            except Exception:
                pass

        if args.wandb:
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_name or None,
                    config=vars(args),
                    dir=save_dir,
                    resume="allow",
                    settings=wandb.Settings(init_timeout=180),
                )
            except Exception as e:
                # Don't crash a multi-hour training run on a transient wandb
                # connection issue. Fall back to local-only logging.
                print(f"[wandb] init failed: {type(e).__name__}: {e}\n"
                      f"[wandb] continuing without wandb logging.")

        load_dir = args.load or save_dir
        if args.load is not None:
            load_cfg_path = os.path.join(load_dir, "config.json")
            if os.path.isfile(load_cfg_path):
                with open(load_cfg_path) as f:
                    load_cfg = json.load(f)
                load_vq = bool(load_cfg.get("vq_codebook", False))
                load_vq_size = int(load_cfg.get("vq_codebook_size", 512))
                if load_vq != bool(args.vq_codebook):
                    raise RuntimeError(
                        f"--load points to a run with vq_codebook={load_vq}, "
                        f"but this invocation has vq_codebook={args.vq_codebook}. "
                        "Pass matching VQ flags or use the original run command."
                    )
                if load_vq and load_vq_size != int(args.vq_codebook_size):
                    raise RuntimeError(
                        f"--load points to a VQ run with vq_codebook_size={load_vq_size}, "
                        f"but this invocation has vq_codebook_size={args.vq_codebook_size}."
                    )
        ckpt_path = os.path.join(load_dir, "params.pkl")
        infos_path = os.path.join(load_dir, "game_infos.pkl")

        # Load existing checkpoint + game_infos
        init_params = None
        start_step = 0
        if os.path.isfile(infos_path):
            with open(infos_path, "rb") as f:
                game_infos = pickle.load(f)
        else:
            game_infos = None

        if os.path.isfile(ckpt_path):
            print(f"Loading params from {ckpt_path}")
            with open(ckpt_path, "rb") as f:
                init_params = pickle.load(f)
            meta_path = os.path.join(load_dir, "train_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    start_step = json.load(f).get("total_steps", 0)
            print(f"  Resuming from step {start_step:,}")

        # Collect/load dataset only if we need to train
        remaining = args.n_updates - start_step
        needs_training = remaining > 0
        if args.render_only:
            if init_params is None:
                raise RuntimeError(
                    f"--render_only: no checkpoint at {ckpt_path}. "
                    f"Wait for the training run to emit its first --ckpt_interval save."
                )
            if game_infos is None:
                raise RuntimeError(
                    f"--render_only: no game_infos.pkl at {infos_path}. "
                    f"The training run must have started and saved game_infos first."
                )
            needs_training = False
            print(f"--render_only: loaded checkpoint @ step {start_step:,}; "
                  f"skipping dataset collection and training.")

        if needs_training:
            names_to_collect = [g["name"] for g in game_infos] if game_infos else game_names
            os.makedirs(save_dir, exist_ok=True)
            if args.synthetic_levels > 0:
                dataset, game_infos = collect_multigame_dataset_synthetic(
                    names_to_collect, ps_parser,
                    n_levels=args.synthetic_levels,
                    width=args.synthetic_w,
                    height=args.synthetic_h,
                    seed=args.synthetic_seed,
                    mode=args.synthetic_mode,
                    require_solvable=args.synthetic_require_solvable,
                    max_attempts_per_level=args.synthetic_max_attempts_per_level,
                    max_iters_search=args.synthetic_max_iters_search,
                    timeout_ms_search=args.synthetic_timeout_ms_search,
                    min_states=args.synthetic_min_states,
                    per_game_size=args.synthetic_per_game_size,
                    multi_grid=args.synthetic_multi_grid,
                    grid_sizes=(
                        [tuple(int(x) for x in s.split("x"))
                         for s in args.synthetic_grid_sizes.split(",")]
                        if args.synthetic_grid_sizes else None
                    ),
                    fallback_dynamics=args.synthetic_fallback_dynamics,
                    no_a_count_max=args.synthetic_no_a_count_max,
                    evolve_pop_size=args.synthetic_evolve_pop_size,
                    evolve_max_generations=args.synthetic_evolve_max_generations,
                    evolve_n_mutations_min=args.synthetic_evolve_n_mutations_min,
                    evolve_n_mutations_max=args.synthetic_evolve_n_mutations_max,
                    encode_sprites=args.encode_sprites,
                    kernel_sep=getattr(args, "kernel_sep", False),
                    max_transitions_per_game=(args.max_transitions_per_game or None),
                )
            else:
                dataset, game_infos = collect_multigame_dataset(
                    names_to_collect, ps_parser,
                    level_i=args.level,
                    n_search_steps=args.n_search_steps,
                    search_timeout_ms=args.search_timeout_ms,
                    search_algo=args.search_algo,
                    encode_sprites=args.encode_sprites,
                    max_transitions_per_game=(args.max_transitions_per_game or None),
                    train_levels=parsed_train_levels,
                )
            with open(infos_path, "wb") as f:
                pickle.dump(game_infos, f)
        elif game_infos is None:
            raise RuntimeError(
                f"No game_infos.pkl found at {infos_path} and no training to do. "
                "Run training first or provide --load pointing to a trained checkpoint."
            )

        max_C = max(g["n_objs"] for g in game_infos)
        pool_kwargs = dict(
            axis_pool=args.axis_pool,
            axis_cummax=args.axis_cummax,
            global_pool=args.global_pool,
            use_layernorm=args.use_layernorm,
        )
        use_sprite_dec = args.sprite_loss_weight > 0.0
        if args.conditional:
            max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
            max_tok_len = max(max_tok_len, 1)
            # Auto-size the vocab to cover only the tokens actually emitted
            # across the training set. V2 vocab IDs run up to VOCAB_SIZE_EXT_V2,
            # but games using fewer features only emit IDs in the lower range —
            # embedding those extra slots would just waste params. Floor at
            # VOCAB_SIZE_EXT + 1 since KERNEL_SEP (always emitted for multi-
            # kernel rules) sits at index VOCAB_SIZE_EXT.
            max_token_id_used = 0
            for g in game_infos:
                tids = g.get("token_ids") or []
                if tids:
                    max_token_id_used = max(max_token_id_used, int(max(tids)))
            vocab_size = max(max_token_id_used + 1, VOCAB_SIZE_EXT + 1)
            # Stash on args so subsequent vars(args) writes of config.json
            # carry vocab_size through to the saved config (loaders depend
            # on this field to rebuild the embedding).
            args.vocab_size = vocab_size
            if args.architecture == "rule_attn":
                from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
                model = RuleAttnNCAWorldModel(
                    n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=max_C,
                    vocab_size=vocab_size + 1,
                    enc_d_model=args.d_model, enc_n_self_layers=args.n_enc_layers,
                    n_slots=args.n_slots, n_app_slots=args.n_app_slots,
                    d_slot=args.d_slot,
                    n_attn_heads=args.n_heads,
                    max_seq_len=max_tok_len + 1,
                    axis_pool=pool_kwargs.get("axis_pool", False),
                    axis_cummax=pool_kwargs.get("axis_cummax", False),
                    global_pool=pool_kwargs.get("global_pool", False),
                    use_vq=args.vq_codebook,
                    vq_codebook_size=args.vq_codebook_size,
                    vq_commitment_weight=args.vq_commitment_weight,
                    use_layernorm=args.use_layernorm,
                    input_skip=args.input_skip,
                    n_repeats=args.n_nca_repeats,
                    adaptive_halt=args.adaptive_halt,
                )
            else:  # default: film
                model = ConditionalNCAWorldModel(
                    n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=max_C,
                    vocab_size=vocab_size + 1,  # +1 for CLS
                    d_model=args.d_model, n_heads=args.n_heads,
                    n_enc_layers=args.n_enc_layers, d_z=args.d_z,
                    max_seq_len=max_tok_len + 1,  # +1 for CLS
                    sprite_decoder=use_sprite_dec,
                    **pool_kwargs,
                )
        else:
            model = NCAWorldModel(n_hid=args.n_hid, n_steps=args.n_nca_steps,
                                   n_out=max_C, **pool_kwargs)

        if needs_training:
            params, losses, accs, change_accs, final_step = train(
                model, dataset,
                lr=args.lr,
                n_updates=remaining,
                batch_size=args.batch_size,
                seed=args.seed,
                log_interval=args.log_interval,
                save_dir=save_dir,
                init_params=init_params,
                start_step=start_step,
                patience=args.patience,
                min_delta=args.min_delta,
                win_loss_weight=args.win_loss_weight,
                win_pos_weight=args.win_pos_weight,
                ckpt_interval=args.ckpt_interval,
                game_names=[g["name"] for g in game_infos],
                balanced_sampling=args.balanced_sampling,
                game_infos=game_infos,
                gif_interval=args.gif_interval,
                gif_n_steps=args.gif_n_steps,
                max_padded_shape=(
                    max_C,
                    max(g["H"] for g in game_infos),
                    max(g["W"] for g in game_infos),
                ),
                ps_parser=ps_parser,
                grad_clip=args.grad_clip,
                sprite_loss_weight=args.sprite_loss_weight,
                change_loss_weight=args.change_loss_weight,
                lr_schedule=args.lr_schedule,
                lr_min=args.lr_min,
                token_decoder_loss_weight=args.token_decoder_loss_weight,
                decoder_d_model=args.decoder_d_model,
                decoder_n_layers=args.decoder_n_layers,
                decoder_n_heads=args.decoder_n_heads,
                use_vq=args.vq_codebook,
                vq_commitment_weight=args.vq_commitment_weight,
                vq_loss_weight=args.vq_loss_weight,
                halt_prior_p=args.halt_prior_p,
                halt_kl_weight=args.halt_kl_weight,
                halt_mode=args.halt_mode,
            )
            # (curves saved inside train() with per-game arrays)
            os.makedirs(save_dir, exist_ok=True)
        else:
            if not args.render_only:
                print(f"Already at {start_step:,} steps (target {args.n_updates:,}), skipping training.")
            params = init_params
            final_step = start_step

        # Don't clobber config.json if another process is actively training
        if not args.render_only:
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)

        # Downstream code expects WM-only params (eval/render don't use the
        # token decoder); unwrap if joint training produced a {wm,dec} dict.
        wm_params = _wm_p(params)

        # Per-game evaluation
        print("\nEvaluating per-game (autoregressive rollout)...")
        evaluate_multigame(
            model, wm_params, game_infos, ps_parser,
            search_algos=args.search_algos,
            search_n_steps=args.n_search_steps,
            search_timeout_ms=args.search_timeout_ms,
            save_dir=save_dir,
        )

        # Per-game GIFs (random + search, all levels). Gated on --render_gif
        # because rendering can take longer than training and blocks any sweep
        # that's queueing the next config. Render later with
        # `python train_nca_world_model.py ... --render_only --render_gif --load <dir>`.
        if args.render_gif:
            print("\nRendering per-game comparison GIFs...")
            render_multigame_gifs(
                model, wm_params, game_infos, ps_parser,
                save_dir=save_dir, step_label=final_step,
                search_algos=args.search_algos,
                search_n_steps=args.n_search_steps,
                search_timeout_ms=args.search_timeout_ms,
            )
        else:
            print("\nSkipping GIF rendering (pass --render_gif to render).")

        if wandb.run is not None:
            wandb.finish()
        print("Done!")
        return

    # --- Single-game path (original) ---
    # Single-game path defaults to level 0 for backwards compat
    if args.level is None:
        args.level = 0
    save_dir = (args.save_dir or
                f"nca_wm/logs/{args.game}_level-{args.level}_nca-{args.n_nca_steps}"
                f"_hid-{args.n_hid}_lr-{args.lr}_s-{args.seed}")

    # Compile game
    print(f"Compiling {args.game}...")
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(ps_parser, args.game)
    env = CppPuzzleScriptEnv(json_str, level_i=args.level, max_episode_steps=args.max_episode_steps)
    n_objs, H, W = env.observation_shape
    print(f"  obs_shape=({n_objs}, {H}, {W}), num_levels={env.num_levels}")

    model = NCAWorldModel(n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=n_objs,
                           axis_pool=args.axis_pool, axis_cummax=args.axis_cummax,
                           global_pool=args.global_pool)

    # Load existing checkpoint if available, otherwise train
    load_dir = args.load or save_dir
    ckpt_path = os.path.join(load_dir, "params.pkl")

    if args.render_only:
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(
                f"--render_only: no checkpoint at {ckpt_path}. "
                f"Wait for the training run to emit its first --ckpt_interval save."
            )
        print(f"--render_only: loading params from {ckpt_path} "
              f"(skipping dataset collection and training)")
        with open(ckpt_path, "rb") as f:
            params = pickle.load(f)
        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(ps_parser, args.game)
        print("\nEvaluating world model (autoregressive rollout)...")
        evaluate_world_model(model, params, json_str, level_i=args.level, save_dir=save_dir)
        print("\nRendering comparison GIFs...")
        render_post_training_gifs(
            model, params, json_str, backend_render,
            search_data=None, level_i=args.level,
            save_dir=save_dir,
        )
        print("Done!")
        return

    if os.path.isfile(ckpt_path):
        print(f"Loading params from {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            params = pickle.load(f)
    else:
        print(f"Collecting unique transitions ({args.search_algo}, "
              f"{args.n_search_steps:,} iters / {args.search_timeout_ms:,}ms)...")
        dataset = collect_unique_transitions(
            json_str, args.game, level_i=args.level,
            max_iters=args.n_search_steps,
            timeout_ms=args.search_timeout_ms,
            search_algo=args.search_algo,
        )
        print(f"Total dataset: {len(dataset['states']):,} transitions")

        # How many transitions actually involve a state change?
        changed = (dataset["states"] != dataset["next_states"]).any(axis=(1, 2, 3))
        print(f"  Transitions with state change: {changed.sum():,}/{len(changed):,} "
              f"({100*changed.mean():.1f}%)")

        params, losses, accs, change_accs, _ = train(
            model, dataset,
            lr=args.lr,
            n_updates=args.n_updates,
            batch_size=args.batch_size,
            seed=args.seed,
            log_interval=args.log_interval,
            save_dir=save_dir,
            patience=args.patience,
            min_delta=args.min_delta,
            win_loss_weight=args.win_loss_weight,
            win_pos_weight=args.win_pos_weight,
            ckpt_interval=args.ckpt_interval,
        )

        # Save training curves
        np.savez(
            os.path.join(save_dir, "curves.npz"),
            losses=np.array(losses),
            accs=np.array(accs),
            change_accs=np.array(change_accs),
        )

        # Save config
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        # Evaluate
        print("\nEvaluating world model (autoregressive rollout)...")
        evaluate_world_model(model, params, json_str, level_i=args.level, save_dir=save_dir)

        # Render post-training comparison GIFs
        print("\nRendering comparison GIFs...")
        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(ps_parser, args.game)
        render_post_training_gifs(
            model, params, json_str, backend_render,
            search_data=search_data, level_i=args.level,
            save_dir=save_dir,
        )

    # Object names for activation visualization
    obj_names = env._canonical_ids

    # Modes that need the renderer
    need_renderer = args.play or args.render_gif
    if need_renderer:
        # May already exist from post-training GIF rendering; create if not
        try:
            backend_render
        except NameError:
            backend_render = CppPuzzleScriptBackend()
            backend_render.compile_game(ps_parser, args.game)

    if args.play:
        play_dir = os.path.join(save_dir, "play")
        play_world_model(
            model, params, json_str, backend_render,
            level_i=args.level, save_dir=play_dir,
        )
    elif args.render_gif:
        gif_path = os.path.join(save_dir, "rollout_comparison.gif")
        render_rollout_comparison(
            model, params, json_str, backend_render,
            level_i=args.level, save_path=gif_path,
        )

    print("Done!")


if __name__ == "__main__":
    main()
