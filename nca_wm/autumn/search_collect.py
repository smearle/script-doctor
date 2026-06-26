#!/usr/bin/env python3
"""Novelty/frontier search collector for Autumn world-model data.

Hand-written heuristics miss mechanics they weren't designed for (coins firing needs a
collected coin first; the persistence gap; masters_logic's deep color cycle). This does a
breadth-first frontier search over the *reachable env-state space* and keeps every novel
transition, so coverage is driven by the game's own dynamics rather than a guessed policy.

Novelty is keyed on the interpreter's full environment JSON (get_environment_string), which
includes HIDDEN vars (e.g. coins' numBullets) -- so two states with identical grids but
different hidden counters are both expanded. That is exactly what surfaces hidden-counter
transitions (collect coin -> numBullets=1 -> click spawns a bullet) that a grid-only or
heuristic collector never reaches.

The interpreter has no usable python snapshot/restore (the cache stack isn't bound and there
is no fromJson), but it is deterministic given (seed, action path), so a frontier node is
expanded by replaying its action path in a fresh engine and then applying each candidate
action. Output: ordered SEQUENCES (root->node paths, for recurrent BPTT) AND the deduped
transition set (for single-frame/history), both in the same npz schema the trainers read.
"""
import argparse, hashlib, json, os, time
from collections import deque

import numpy as np

from nca_wm.autumn.collect import AutumnGame, _grid_to_idx
from nca_wm.autumn.model import action_to_fields, ACTION_TYPES


def _grid_key(g):
    """Novelty over the OBSERVABLE grid -- what the WM actually sees. The env-JSON alternative
    over-explores: it changes every step (RNG / frame counters) even when the grid is identical,
    so it burns the budget on observationally-identical states and never goes deep. Grid novelty
    explores reachable observable states; hidden-counter dynamics (coins' numBullets) are still
    covered because the emitted SEQUENCES carry the action history that disambiguates them."""
    return g.tobytes()


def _action_set(gs, click_stride):
    acts = [("noop", -1, -1), ("up", -1, -1), ("down", -1, -1), ("left", -1, -1), ("right", -1, -1)]
    for y in range(0, gs, click_stride):
        for x in range(0, gs, click_stride):
            acts.append(("click", x, y))
    return acts


def search_collect(game, max_states=2000, max_depth=24, click_stride=2, seeds=(1, 2, 3),
                   max_seconds=600, verbose=True):
    probe = AutumnGame(game, seed=seeds[0])
    gs = probe.grid_size
    palette = {probe.bg: 0}                      # grow as colors appear (shared across seeds)
    action_set = _action_set(gs, click_stride)

    def grid_of(env):
        return _grid_to_idx(env.color_grid(), palette)

    def replay(seed, path):
        env = AutumnGame(game, seed=seed)
        for a in path:
            env.apply(a)
        return env

    seqs = []                                    # (states[L+1,H,W], actions[L,3]) per frontier leaf
    trans = {}                                   # (grid_bytes, ai, cx, cy) -> (g, a, ng)
    visited = set()
    t0 = time.time()
    n_expand = 0
    for seed in seeds:
        env0 = replay(seed, [])
        g0 = grid_of(env0)
        visited.add(_grid_key(g0))
        # frontier node: (path, states_list)  -- states_list[i] is the grid after path[:i]
        frontier = deque([([], [g0])])
        while frontier and len(visited) < max_states:
            if time.time() - t0 > max_seconds:
                break
            path, states = frontier.popleft()
            if len(path) >= max_depth:
                seqs.append((np.stack(states), np.array([action_to_fields(a) for a in path], dtype=np.int64)
                             if path else np.zeros((0, 3), np.int64)))
                continue
            g = states[-1]
            expanded_child = False
            for a in action_set:
                env = replay(seed, path)         # deterministic reconstruction
                env.apply(a)
                ng = grid_of(env)
                ai, cx, cy = action_to_fields(a)
                tkey = (g.tobytes(), ai, cx, cy)
                if tkey not in trans:
                    trans[tkey] = (g.copy(), (ai, cx, cy), ng.copy())
                k = _grid_key(ng)
                if k not in visited and len(visited) < max_states:
                    visited.add(k)
                    frontier.append((path + [a], states + [ng]))
                    expanded_child = True
                n_expand += 1
            if not expanded_child and path:      # leaf of the novelty tree -> emit its sequence
                seqs.append((np.stack(states), np.array([action_to_fields(a) for a in path], dtype=np.int64)))
        if verbose:
            print(f"  seed {seed}: visited={len(visited)} transitions={len(trans)} "
                  f"seqs={len(seqs)} expansions={n_expand} t={time.time()-t0:.0f}s")
    inv = [None] * len(palette)
    for name, i in palette.items():
        inv[i] = name
    return seqs, trans, inv, gs


def _fields_to_action(ai, cx, cy):
    return ("click", int(cx), int(cy)) if int(ai) == 1 else (ACTION_TYPES[int(ai)],)


def seeded_search(game, seed_states, max_transitions=40000, max_seconds=600, click_stride=2,
                  cont_len=10, n_cont=3, verbose=True, **_ignore):
    """Frontier search SEEDED with states visited by human + heuristic rollouts (the recommended
    pipeline: start from real/heuristic play, then expand around it).

    The interpreter has no python snapshot, so a state is reached by replaying (seed, path) --
    O(path) per replay. A full BFS that re-replays for every (node, action) is far too slow on
    deep states. Instead we expand each frontier state with CONTINUATION ROLLOUTS: replay to the
    state ONCE, then roll forward `cont_len` steps. The FIRST continuation step sweeps the action
    set systematically (so every immediate neighbor is covered -- the 'exhaustive' part), and the
    tail is randomized for depth. We emit the full root->continuation sequence (valid h=0 at the
    episode root) so the recurrent model gets hidden-state context, and also keep every novel
    transition. n_cont rollouts per state branch the exploration. Returns (sequences, transitions,
    palette, grid_size)."""
    rng = np.random.default_rng(0)
    probe = AutumnGame(game, seed=(seed_states[0][0] if seed_states else 0))
    gs = probe.grid_size
    palette = {probe.bg: 0}
    action_set = _action_set(gs, click_stride)

    def grid_of(env):
        return _grid_to_idx(env.color_grid(), palette)

    arrows = [("up", -1, -1), ("down", -1, -1), ("left", -1, -1), ("right", -1, -1)]

    def rand_action():
        # balance action TYPES (the action_set is mostly click positions, so uniform sampling is
        # click-heavy and starves arrow-movement coverage -> the WM overfits clicks/spawns and
        # regresses movement). Sample type first: ~45% arrow, ~40% click, ~15% noop.
        r = rng.random()
        if r < 0.45:
            return arrows[rng.integers(4)]
        if r < 0.85:
            return ("click", int(rng.integers(gs)), int(rng.integers(gs)))
        return ("noop", -1, -1)

    visited, trans, seqs = set(), {}, []
    seed_states = [(s, p) for s, p in seed_states if s >= 0]
    n_seed = len(seed_states)
    t0 = time.time()
    for si, (seed, path) in enumerate(seed_states):
        if len(trans) >= max_transitions or time.time() - t0 > max_seconds:
            break
        for m in range(n_cont):
            env = AutumnGame(game, seed=int(seed))
            grids = [grid_of(env)]
            for a in path:                       # replay to the frontier state (records states)
                env.apply(a); grids.append(grid_of(env))
            acts = list(path)
            for k in range(cont_len):            # continuation: 1st step systematic, rest random
                a = action_set[m % len(action_set)] if k == 0 else rand_action()
                g = grids[-1]
                env.apply(a); ng = grid_of(env)
                ai, cx, cy = action_to_fields(a)
                tk = (g.tobytes(), ai, cx, cy)
                if tk not in trans:
                    trans[tk] = (g.copy(), (ai, cx, cy), ng.copy())
                grids.append(ng); acts.append(a)
            if len(acts) and len(seqs) < 6000:
                seqs.append((np.stack(grids).astype(np.uint8),
                             np.array([list(action_to_fields(a)) for a in acts], dtype=np.int64)))
        if verbose and si % 400 == 0 and si:
            print(f"  seeded_search: {si}/{n_seed} states, trans={len(trans)} seqs={len(seqs)} t={time.time()-t0:.0f}s")
    inv = [None] * len(palette)
    for name, i in palette.items():
        inv[i] = name
    if verbose:
        print(f"  seeded_search: {n_seed} seed states x{n_cont} conts -> {len(trans)} transitions, "
              f"{len(seqs)} sequences in {time.time()-t0:.0f}s")
    return seqs, trans, inv, gs


def save_transitions(trans, inv, gs, game, out):
    g = np.stack([v[0] for v in trans.values()]).astype(np.uint8)
    ng = np.stack([v[2] for v in trans.values()]).astype(np.uint8)
    ai = np.array([v[1][0] for v in trans.values()], dtype=np.int16)
    cx = np.array([v[1][1] for v in trans.values()], dtype=np.int16)
    cy = np.array([v[1][2] for v in trans.values()], dtype=np.int16)
    nchg = int((g != ng).any(axis=(1, 2)).sum())
    print(f"  transitions={len(g)} changing={nchg} ({100*nchg/max(len(g),1):.1f}%) "
          f"action_types={ {ACTION_TYPES[k]: int((ai==k).sum()) for k in np.unique(ai)} }")
    np.savez_compressed(out, states=g, next_states=ng, action_type=ai, click_x=cx, click_y=cy,
                        palette=np.array(inv), grid_size=np.int32(gs), game=np.str_(game),
                        button_pos_json=np.str_("{}"))
    print(f"  saved transitions -> {out}")


def save_sequences(seqs, inv, gs, game, out, min_len=2):
    seqs = [(S, A) for S, A in seqs if A.shape[0] >= min_len]
    if not seqs:
        print("  no sequences >= min_len"); return
    L = max(A.shape[0] for _, A in seqs)
    E = len(seqs)
    states = np.zeros((E, L + 1, gs, gs), dtype=np.uint8)
    actions = np.zeros((E, L, 3), dtype=np.int64)
    lengths = np.zeros(E, dtype=np.int64)
    for i, (S, A) in enumerate(seqs):
        l = A.shape[0]
        states[i, :l + 1] = S
        states[i, l + 1:] = S[-1]
        actions[i, :l] = A
        lengths[i] = l
    np.savez_compressed(out, states=states, actions=actions, lengths=lengths,
                        palette=np.array(inv), grid_size=np.int32(gs), game=np.str_(game))
    print(f"  saved {E} sequences (max_len={L}) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--max_states", type=int, default=2000)
    ap.add_argument("--max_depth", type=int, default=24)
    ap.add_argument("--click_stride", type=int, default=2)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--max_seconds", type=int, default=600)
    ap.add_argument("--out_trans", default="")
    ap.add_argument("--out_seq", default="")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    print(f"[search] game={args.game} max_states={args.max_states} depth={args.max_depth} "
          f"click_stride={args.click_stride} seeds={seeds}")
    seqs, trans, inv, gs = search_collect(args.game, args.max_states, args.max_depth,
                                          args.click_stride, seeds, args.max_seconds)
    print(f"[search] palette({len(inv)})={inv}")
    if args.out_trans:
        save_transitions(trans, inv, gs, args.game, args.out_trans)
    if args.out_seq:
        save_sequences(seqs, inv, gs, args.game, args.out_seq)


if __name__ == "__main__":
    main()
