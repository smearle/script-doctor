"""Propose a scaling_v3 game preset for multi-game NCA WM training.

Reads `data/dedup_candidates_v2.json` (the token-deduped pool produced by
`dedup_games.py`) and selects:

1. The current gallery_v2 preset, deduplicated against itself by token-hash
   (so we drop the redundant `Microban` ≡ `sokoban_basic` ≡ `blank` group
   and `twolittlecrates2` ≡ `twolittlecrates4`).
2. N additional games from the deduped pool, none of which collide on token-
   hash with anything already in step (1). Stratified across complexity
   buckets so the new preset spans low-to-mid rule counts (skipping the
   highest-complexity tail).

Writes the resulting preset list to JSON, ready to paste into `train.py`
MULTI_GAME_PRESETS as a new entry. Also reports per-game expected scale (rules, objs,
levels, area) so the user can sanity-check before launching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


GALLERY_V2_PRESET = [
    "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
    "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
    "Travelling_salesman", "blank", "sumo", "the_undertaking", "wrappingrecipe",
    "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
    "rigidfail1", "scriptcross", "Modality", "constellationz",
    "randomrobots", "againexample", "Microban", "naughtysprite",
    "randomspawner", "twolittlecrates1", "rigid_11", "Long_Haul_Space_Flight",
    "leftrightnpcs", "twolittlecrates2", "twolittlecrates3",
    "twolittlecrates4", "octat", "lunar_lockout", "Stairways",
    "the_art_of_cloning", "rigid_scott1", "rigid_one_unlimited",
    "Some_lines_were_meant_to_be_crossed", "blockfaker", "Pushing_It",
    "2D_Whale_World", "MazezaM", "Ebony_&_Ivory", "Singleton_Traffic",
    "mazetest", "Slidings", "Lime_Rick", "riverpuzzle", "Midas",
    "rigid_parallel_many", "Take_Heart_Lass", "rigid_many_broken",
    "Lightdown", "The_observer's_paradox", "rigid_parallel_unlimited",
    "MC_Escher's_Equestrian_Armageddon", "Smother", "It_Dies_In_The_Light",
    "Pushcat_Jr",
]


def get_token_hash(name: str) -> str | None:
    from puzzlescript_jax.gen_tree import GenPSTree
    from nca_wm.tokenize_game import tokenize_game

    fp = os.path.join(REPO, "data", "game_trees", name + ".pkl")
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, "rb") as f:
            lt = pickle.load(f)
        ps = GenPSTree().transform(lt)
        objs = ps.objects
        cids = (list(objs.keys()) if isinstance(objs, dict)
                else [o.name for o in objs])
        toks = tokenize_game(ps, cids, encode_sprites=False)
        return hashlib.sha256(
            ",".join(str(t) for t in toks).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        return None


def stratified_pick(remaining, n, buckets):
    """Pick n games across buckets, equal share each. `buckets` is a list of
    (label, predicate, target_share). Returns picked list and updates seen.
    """
    picked = []
    for label, pred, share in buckets:
        n_this = int(round(n * share))
        in_bucket = [g for g in remaining if pred(g)]
        # Sort within bucket by total complexity score so we get a spread.
        in_bucket.sort(key=lambda g: (g["n_rules"], g["n_objects"],
                                       g["n_levels"]))
        # Stride-sample so we cover the bucket evenly
        if len(in_bucket) <= n_this:
            chosen = in_bucket
        else:
            stride = len(in_bucket) / n_this
            idxs = sorted({int(stride * i) for i in range(n_this)})[:n_this]
            chosen = [in_bucket[i] for i in idxs]
        picked.extend(chosen)
        print(f"  bucket {label}: {len(chosen)}/{n_this} (pool={len(in_bucket)})",
              file=sys.stderr)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup_file", default="data/dedup_candidates_v2.json")
    ap.add_argument("--n_extra", type=int, default=40,
                    help="How many games to add on top of gallery_v2.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.dedup_file) as f:
        dedup = json.load(f)
    pool = dedup["candidates"]
    pool_by_name = {c["name"]: c for c in pool}
    pool_by_hash = {c["token_hash"]: c for c in pool if c.get("token_hash")}

    # Step 1: hash gallery_v2; drop preset members redundant against each
    # other (e.g. sokoban_basic ≡ blank ≡ Microban).
    print("Step 1: dedup gallery_v2 internally", file=sys.stderr)
    gv2_keep: list[str] = []
    gv2_seen_hashes: set[str] = set()
    gv2_dropped: list[tuple[str, str]] = []
    for name in GALLERY_V2_PRESET:
        h = get_token_hash(name)
        if h is None:
            print(f"  WARN: {name} has no token hash (cached tree missing?). "
                  f"Keeping anyway.", file=sys.stderr)
            gv2_keep.append(name)
            continue
        if h in gv2_seen_hashes:
            # Find which prior name occupies this hash
            prior = next(p for p in gv2_keep if get_token_hash(p) == h)
            gv2_dropped.append((name, prior))
            continue
        gv2_seen_hashes.add(h)
        gv2_keep.append(name)
    print(f"  kept {len(gv2_keep)} of {len(GALLERY_V2_PRESET)}; dropped:",
          file=sys.stderr)
    for name, prior in gv2_dropped:
        print(f"    {name} (≡ {prior})", file=sys.stderr)

    # Step 2: from deduped pool, exclude any candidate whose hash is already
    # in gv2_seen_hashes.
    available = [c for c in pool
                 if c["name"] not in gv2_keep
                 and c.get("token_hash")
                 and c["token_hash"] not in gv2_seen_hashes]
    # And exclude obvious template/tutorial games by name pattern.
    SPAM_PATTERNS = [
        "by_unknown_author",
        "_by_increpare",   # mostly templates / dev tests
    ]
    SPAM_NAMES = {
        # The 445-game "default sokoban template" cluster's canonical was
        # already in gallery_v2 (sokoban_basic), so it's excluded by hash.
        # But guard against future template-shaped games.
    }
    available = [c for c in available
                 if c["name"] not in SPAM_NAMES]
    print(f"\nStep 2: {len(available)} pool candidates not in gallery_v2 + not "
          f"hash-redundant", file=sys.stderr)

    # Step 3: stratified pick across complexity buckets
    print(f"\nStep 3: pick {args.n_extra} stratified additions", file=sys.stderr)
    buckets = [
        ("low (1-3 rules, ≤8 objs)",
         lambda c: c["n_rules"] <= 3 and c["n_objects"] <= 8, 0.30),
        ("mid (4-8 rules, ≤12 objs)",
         lambda c: 4 <= c["n_rules"] <= 8 and c["n_objects"] <= 12, 0.40),
        ("high (9-15 rules, ≤16 objs)",
         lambda c: 9 <= c["n_rules"] <= 15 and c["n_objects"] <= 16, 0.20),
        ("xhigh (16-20 rules)",
         lambda c: 16 <= c["n_rules"] <= 20, 0.10),
    ]
    picks = stratified_pick(available, args.n_extra, buckets)

    # Filter to keep multi-level games preferentially: a single-level game has
    # less data to learn from. Prefer ≥2 levels when available.
    multi_level_picks = [p for p in picks if p["n_levels"] >= 2]
    single_level_picks = [p for p in picks if p["n_levels"] < 2]
    if len(multi_level_picks) >= args.n_extra:
        picks = multi_level_picks[:args.n_extra]
    else:
        picks = multi_level_picks + single_level_picks[
            : args.n_extra - len(multi_level_picks)]

    # Step 4: emit preset
    new_preset = list(gv2_keep) + [p["name"] for p in picks]
    out = {
        "preset_name": f"scaling_gallery_v3_n{args.n_extra}",
        "n_total": len(new_preset),
        "gallery_v2_kept": gv2_keep,
        "gallery_v2_dropped_redundant": [
            {"dropped": d, "redundant_with": r} for d, r in gv2_dropped
        ],
        "added_from_dedup_pool": [
            {
                "name": p["name"],
                "source": p["source"],
                "n_rules": p["n_rules"],
                "n_objects": p["n_objects"],
                "n_levels": p["n_levels"],
                "max_level_area": p["max_level_area"],
            }
            for p in picks
        ],
        "preset_for_train_py": new_preset,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Print summary table
    print(f"\nWrote {args.out} with {len(new_preset)} games "
          f"({len(gv2_keep)} from gallery_v2 + {len(picks)} new).")
    print(f"\nPreview of additions (sorted by n_rules):")
    sorted_adds = sorted(picks, key=lambda c: (c["n_rules"], c["n_objects"]))
    for p in sorted_adds:
        print(f"  {p['name']:55s} src={p['source']:9s} "
              f"r={p['n_rules']:2d} o={p['n_objects']:2d} "
              f"L={p['n_levels']:2d} A={p['max_level_area']:2d}")


if __name__ == "__main__":
    main()
