#!/usr/bin/env python3
"""Unified per-game training-dataset builder (standing pipeline).

Every NCA-autumn WM retrain folds in THREE sequence sources, palette-aligned + padded into one
`{game}_train_seq.npz`:
  1. heuristic/agent sequences  (collect.collect_sequences)        -- persistence/occlusion coverage
  2. grid-novelty search        (search_collect.search_collect)    -- mechanics heuristics miss
  3. human playtraces (TRAIN split) (human_replay.collect_human)   -- the real distribution

Human users are split train/val by user index; the VAL users are written to
`{game}_human_val_seq.npz` and excluded from training, so the human-distribution metric is not
leaked. Games absent from the human data (e.g. coins) use sources 1+2 only.

All sources are re-encoded to a single UNION palette (so channel indices align across sources
and match the model). See feedback_data_pipeline_human_search.
"""
import argparse, os
import numpy as np

from nca_wm.autumn.collect import collect_sequences
from nca_wm.autumn.search_collect import seeded_search
from nca_wm.autumn import human_replay

# per-game heuristic profile if one exists (else "agent")
HEUR_PROFILE = {
    "coins": "coins_heuristic", "mario": "mario_heuristic", "snake": "snake_heuristic",
    "waterplug": "waterplug_heuristic", "masters_logic": "masters_logic_heuristic",
    "paint": "paint_heuristic",
}


def _remap(states_idx, src_palette, union):
    """Re-encode an (E,L+1,H,W) index array from src_palette (list) to union (list) by name."""
    lut = np.array([union.index(c) for c in src_palette], dtype=np.uint8)
    return lut[states_idx]


def _pad_concat(seqlist, H):
    """seqlist: list of (states (E,l+1,H,W), actions (E,l,3)). Pad to common L, concat."""
    L = max(s.shape[1] - 1 for s, _ in seqlist)
    outS, outA = [], []
    for S, A in seqlist:
        E, lp1 = S.shape[0], S.shape[1]; l = lp1 - 1
        s2 = np.zeros((E, L + 1, H, H), np.uint8); a2 = np.zeros((E, L, 3), np.int64)
        s2[:, :l + 1] = S; s2[:, l + 1:] = S[:, l:l + 1]; a2[:, :l] = A
        outS.append(s2); outA.append(a2)
    return np.concatenate(outS), np.concatenate(outA)


def _cap(S, A, cap):
    """Truncate episodes to the first `cap` steps from their TRUE start (so a recurrent model's
    h=0 stays valid; mid-episode chunks would start with unknowable hidden state). Long human
    idle-noop tails carry no signal anyway."""
    if cap and S.shape[1] - 1 > cap:
        return S[:, :cap + 1], A[:, :cap]
    return S, A


def _seed_states_from(states, actions, seeds, cap, rng):
    """Extract (seed, action_path) for each UNIQUE grid state visited by a rollout set, so the
    search can re-reach and expand around it. Dedup by grid; sample down to `cap`."""
    from nca_wm.autumn.search_collect import _fields_to_action
    seen, out = set(), []
    L = actions.shape[1]
    for e in range(len(states)):
        if int(seeds[e]) < 0:                  # not pure-seed-replayable -> can't seed search
            continue
        path = []
        for t in range(L + 1):
            k = states[e, t].tobytes()
            if k not in seen:
                seen.add(k); out.append((int(seeds[e]), list(path)))
            if t < L:
                path.append(_fields_to_action(*actions[e, t]))
    if len(out) > cap:
        out = [out[i] for i in rng.choice(len(out), cap, replace=False)]
    return out


def build(game, n_heur=200, heur_len=60, max_transitions=40000, search_stride=2,
          search_seconds=420, expand_depth=2, seed_cap=2500, human_val_frac=0.0,
          seq_len=80, out_dir="nca_wm/autumn/data"):
    # PIPELINE (per directive): human + heuristic rollouts -> their visited states seed an
    # exhaustive frontier search (seeded_search) -> generate transitions up to max_transitions ->
    # train on all of it. human_val_frac=0.0: train on ALL human data (human-in-viewer is judge).
    rng = np.random.default_rng(0)
    parts = []          # (states_idx_in_own_palette, actions, palette_list, tag)

    # 1. heuristic / agent rollouts (with per-episode seeds)
    prof = HEUR_PROFILE.get(game, "agent")
    h = collect_sequences(game, n_heur, heur_len, seed=0, profile=prof)
    parts.append((h["states"], h["actions"], [str(x) for x in h["palette"]], f"heur:{prof}"))
    seed_states = _seed_states_from(h["states"], h["actions"], h["seeds"], seed_cap, rng)

    # 2. human rollouts (all of them; seed-correct replay) -> add to training AND to the frontier
    human_present = game in {human_replay.base_game(t) for u in human_replay.json.load(
        open(human_replay.HUMAN))["users"] for t in u["tasks"]} if os.path.exists(human_replay.HUMAN) else False
    if human_present:
        union0 = [str(x) for x in h["palette"]]
        eps, meta = human_replay.collect_human(game, union0)
        hseeds = meta["seeds"]
        if eps:
            L = max(a.shape[0] for _, a in eps); E = len(eps)
            S = np.zeros((E, L + 1, meta["grid_size"], meta["grid_size"]), np.uint8)
            A = np.zeros((E, L, 3), np.int64)
            for i, (s, a) in enumerate(eps):
                l = a.shape[0]; S[i, :l + 1] = s; S[i, l + 1:] = s[-1]; A[i, :l] = a
            S, A = _cap(S, A, seq_len)
            parts.append((S, A, union0, "human"))
            seed_states += _seed_states_from(S, A, np.array(hseeds), seed_cap, rng)

    # 3. EXHAUSTIVE seeded search from the human+heuristic frontier
    print(f"  seeding search with {len(seed_states)} unique frontier states")
    seqs, _trans, inv, gs = seeded_search(game, seed_states, max_transitions=max_transitions,
                                          max_seconds=search_seconds, click_stride=search_stride,
                                          expand_depth=expand_depth, verbose=True)
    if seqs:
        L = max(A.shape[0] for _, A in seqs); E = len(seqs)
        Ss = np.zeros((E, L + 1, gs, gs), np.uint8); As = np.zeros((E, L, 3), np.int64)
        for i, (s, a) in enumerate(seqs):
            l = a.shape[0]; Ss[i, :l + 1] = s; Ss[i, l + 1:] = s[-1]; As[i, :l] = a
        Ss, As = _cap(Ss, As, seq_len)
        parts.append((Ss, As, [str(x) for x in inv], "seeded_search"))

    # union palette (order: first source, then any new colors)
    union = []
    for _, _, pal, _ in parts:
        for c in pal:
            if c not in union:
                union.append(c)
    human_val = None
    H = parts[0][0].shape[2]
    # remap every part to union, then pad+concat
    remapped = []
    for S, A, pal, tag in parts:
        Su = S if pal == union else _remap(S, pal, union)
        remapped.append((Su, A))
        print(f"  source {tag:16s}: {S.shape[0]} episodes")
    states, actions = _pad_concat(remapped, H)
    out = os.path.join(out_dir, f"{game}_train_seq.npz")
    np.savez_compressed(out, states=states, actions=actions, palette=np.array(union),
                        grid_size=np.int32(H), game=np.str_(game))
    print(f"  UNION palette({len(union)})={union}")
    print(f"  -> {out}  ({states.shape[0]} episodes, max_len={states.shape[1]-1})")

    if human_val:
        L = max(a.shape[0] for _, a in human_val); E = len(human_val)
        S = np.zeros((E, L + 1, H, H), np.uint8); A = np.zeros((E, L, 3), np.int64)
        lens = np.zeros(E, np.int64)
        for i, (s, a) in enumerate(human_val):
            l = a.shape[0]; S[i, :l + 1] = s; S[i, l + 1:] = s[-1]; A[i, :l] = a; lens[i] = l
        S, A = _cap(S, A, seq_len); lens = np.minimum(lens, seq_len)
        vout = os.path.join(out_dir, f"{game}_human_val_seq.npz")
        np.savez_compressed(vout, states=S, actions=A, lengths=lens, palette=np.array(union),
                            grid_size=np.int32(H), game=np.str_(game))
        print(f"  held-out human VAL: {E} episodes -> {vout}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--n_heur", type=int, default=200)
    ap.add_argument("--max_transitions", type=int, default=40000)
    ap.add_argument("--search_seconds", type=int, default=420)
    ap.add_argument("--expand_depth", type=int, default=2)
    ap.add_argument("--seed_cap", type=int, default=2500)
    args = ap.parse_args()
    print(f"[build_dataset] {args.game}")
    build(args.game, n_heur=args.n_heur, max_transitions=args.max_transitions,
          search_seconds=args.search_seconds, expand_depth=args.expand_depth,
          seed_cap=args.seed_cap)


if __name__ == "__main__":
    main()
