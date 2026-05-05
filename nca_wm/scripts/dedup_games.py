"""Dedup the scraped + increpare game corpora using token-sequence hashes.

The `tokenize_game` function in `nca_wm/tokenize_game.py` produces a
name-invariant integer sequence covering layers, legend groups, rules, win
conditions, and levels. Two games with identical mechanics modulo cosmetic
naming (object names, color palettes, sprite art with encode_sprites=False)
produce the same token sequence — so SHA-256 of the token stream is the
right structural fingerprint.

Pipeline:
1. Collect game stems across {custom, gallery, scraped, increpare} dirs in
   priority order; keep the highest-priority copy of each lowercased stem.
2. Filter by `data/games_metadata.json` complexity caps
   (gallery_v1 / v2 / wide; or "none").
3. For each surviving stem, load the cached Lark tree from
   `data/game_trees/{stem}.pkl`, transform to PSGameTree via GenPSTree, and
   tokenize with encode_sprites=False.
4. Hash the tuple of token IDs.
5. Within each token-hash group, pick the canonical name (highest-priority
   dir, then shortest stem, then lex order).

Usage:
    .venv/bin/python3 -m nca_wm.scripts.dedup_games \
        --filter v2 --out data/dedup_candidates_v2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import traceback
from collections import defaultdict


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

DIRS_IN_PRIORITY_ORDER = [
    ("custom", os.path.join(REPO, "custom_games")),
    ("gallery", os.path.join(REPO, "gallery_games")),
    ("scraped", os.path.join(REPO, "data", "scraped_games")),
    ("increpare", os.path.join(REPO, "data", "scraped_games_increpare")),
]
METADATA_JSON = os.path.join(REPO, "data", "games_metadata.json")
GAME_TREES_DIR = os.path.join(REPO, "data", "game_trees")


def passes_filter(meta: dict, name: str) -> bool:
    if not isinstance(meta, dict):
        return False
    n_rules = meta.get("n_rules")
    n_objects = meta.get("n_objects")
    n_levels = meta.get("n_levels")
    max_area = meta.get("max_level_area")
    if (n_rules is None or n_objects is None or n_levels is None
            or max_area is None):
        return False
    if name == "v2":
        return (1 <= n_rules <= 20 and n_objects <= 20
                and 1 <= n_levels <= 30 and max_area <= 30)
    if name == "v1":
        return (1 <= n_rules <= 8 and n_objects <= 12
                and 1 <= n_levels <= 20 and max_area <= 20)
    if name == "wide":
        return (1 <= n_rules <= 30 and n_objects <= 30
                and 1 <= n_levels <= 50 and max_area <= 50)
    raise ValueError(name)


def tokenize_stem(stem: str, gen_tree, tokenize_game):
    """Return (token_hash_hex16, n_tokens, n_objects) or None on failure.

    Uses cached Lark tree at data/game_trees/{stem}.pkl, transforms via
    GenPSTree, tokenizes with encode_sprites=False.
    """
    pkl_path = os.path.join(GAME_TREES_DIR, stem + ".pkl")
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            lark_tree = pickle.load(f)
        ps_tree = gen_tree.transform(lark_tree)
        objs = ps_tree.objects
        if isinstance(objs, dict):
            canonical_ids = list(objs.keys())
        else:
            canonical_ids = [o.name for o in objs]
        tokens = tokenize_game(ps_tree, canonical_ids, encode_sprites=False)
    except Exception:
        return None
    h = hashlib.sha256(
        ",".join(str(t) for t in tokens).encode("utf-8")
    ).hexdigest()[:16]
    return h, len(tokens), len(canonical_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="v2", choices=["v1", "v2", "wide", "none"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many files (debug).")
    args = ap.parse_args()

    # Lazy imports so --help doesn't pay the JAX startup cost
    from puzzlescript_jax.gen_tree import GenPSTree
    from nca_wm.tokenize_game import tokenize_game
    gen_tree = GenPSTree()

    # Stage 1: dir-priority dedup by lowercased stem.
    # When the same lowercased stem appears multiple times within the highest-
    # priority directory (case duplicates like Modality.txt + modality.txt),
    # prefer the capitalized variant (it's typically the canonical name).
    # Across dirs, earlier-priority dir wins regardless of casing.
    by_stem: dict[str, tuple[int, str, str, str]] = {}
    n_files_total = 0
    for prio, (label, d) in enumerate(DIRS_IN_PRIORITY_ORDER):
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".txt"):
                continue
            n_files_total += 1
            stem = fn[:-4]
            key = stem.lower()
            if key in by_stem:
                cur_prio, _, _, cur_stem = by_stem[key]
                if cur_prio < prio:
                    continue  # higher-priority dir already won
                # same dir: prefer name with any uppercase letter, else lex
                cur_has_upper = any(c.isupper() for c in cur_stem)
                new_has_upper = any(c.isupper() for c in stem)
                if cur_has_upper and not new_has_upper:
                    continue
                if new_has_upper and not cur_has_upper:
                    by_stem[key] = (prio, label, d, stem)
                    continue
                # both same caseness — keep first seen
                continue
            by_stem[key] = (prio, label, d, stem)
    print(f"Stage 1 (dir priority): {len(by_stem)} unique stems / "
          f"{n_files_total} total files", file=sys.stderr)

    # Stage 2: metadata filter
    with open(METADATA_JSON) as f:
        metadata = json.load(f)
    candidates = []
    n_filtered = 0
    n_no_meta = 0
    items = sorted(by_stem.items())
    if args.limit:
        items = items[:args.limit]
    for key, (prio, label, d, stem) in items:
        meta = metadata.get(stem + ".txt")
        if meta is None:
            n_no_meta += 1
            continue
        if args.filter != "none" and not passes_filter(meta, args.filter):
            n_filtered += 1
            continue
        candidates.append({
            "name": stem,
            "source": label,
            "path": os.path.join(d, stem + ".txt"),
            **{k: meta.get(k) for k in
               ("n_rules", "n_objects", "n_levels", "max_level_area",
                "mean_level_area", "n_collision_layers", "has_randomness")},
        })
    print(f"Stage 2 (metadata filter={args.filter}): {len(candidates)} kept, "
          f"{n_filtered} filtered, {n_no_meta} no metadata", file=sys.stderr)

    # Stage 3: tokenize + hash
    n_no_tree = 0
    n_tok_fail = 0
    for i, c in enumerate(candidates):
        if i % 500 == 0:
            print(f"  tokenizing {i}/{len(candidates)}...", file=sys.stderr)
        res = tokenize_stem(c["name"], gen_tree, tokenize_game)
        if res is None:
            pkl_path = os.path.join(GAME_TREES_DIR, c["name"] + ".pkl")
            if not os.path.exists(pkl_path):
                n_no_tree += 1
            else:
                n_tok_fail += 1
            c["token_hash"] = None
            c["n_tokens"] = None
            c["n_objects_canonical"] = None
        else:
            c["token_hash"], c["n_tokens"], c["n_objects_canonical"] = res
    print(f"Stage 3 (tokenize): {n_no_tree} have no cached tree, "
          f"{n_tok_fail} failed tokenization", file=sys.stderr)

    # Names already in human-curated training presets — when a dup group
    # contains one of these, prefer it as canonical so the gallery_v2 list
    # in train.py keeps resolving to the same on-disk game.
    GALLERY_V2_NAMES = {
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman", "blank", "sumo", "the_undertaking",
        "wrappingrecipe", "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
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
    }

    # Stage 4: dedup by token_hash. Games without a hash survive individually
    # (we keep them as-is rather than dropping; they may still be useful for
    # training but can't be deduped without a re-parse).
    def pick_canonical(group):
        prio_order = ["custom", "gallery", "scraped", "increpare"]
        # Sort key: 1) is in curated preset (preferred), 2) source priority,
        # 3) lacks "_by_" suffix (increpare-style), 4) shorter, 5) lex.
        def k(g):
            return (
                0 if g["name"] in GALLERY_V2_NAMES else 1,
                prio_order.index(g["source"]),
                0 if "_by_" not in g["name"] else 1,
                len(g["name"]),
                g["name"],
            )
        return min(group, key=k)

    by_hash = defaultdict(list)
    no_hash = []
    for c in candidates:
        if c.get("token_hash"):
            by_hash[c["token_hash"]].append(c)
        else:
            no_hash.append(c)

    print(f"Stage 4 (token-hash dedup): {len(by_hash)} unique hashes, "
          f"{sum(1 for v in by_hash.values() if len(v) > 1)} have ≥2 members; "
          f"{len(no_hash)} kept (no hash)", file=sys.stderr)

    deduped = [pick_canonical(g) for g in by_hash.values()] + no_hash

    collision_groups = {h: sorted([c["name"] for c in v])
                        for h, v in by_hash.items() if len(v) > 1}
    # Sort by group size descending
    sorted_groups = sorted(collision_groups.items(),
                            key=lambda kv: -len(kv[1]))

    out = {
        "filter": args.filter,
        "n_total_after_dir_priority": len(by_stem),
        "n_after_metadata_filter": len(candidates),
        "n_with_token_hash": sum(1 for c in candidates if c.get("token_hash")),
        "n_after_token_dedup": len(deduped),
        "candidates": sorted(deduped, key=lambda c: c["name"].lower()),
        "collision_examples": dict(sorted_groups[:50]),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {args.out} with {len(deduped)} candidates "
          f"({len(candidates)} → {len(deduped)} after dedup; "
          f"cut {len(candidates) - len(deduped)})")
    if sorted_groups:
        biggest = sorted_groups[0]
        print(f"  Largest dup group ({len(biggest[1])} games): "
              f"{biggest[1][:5]}{'...' if len(biggest[1]) > 5 else ''}")


if __name__ == "__main__":
    main()
