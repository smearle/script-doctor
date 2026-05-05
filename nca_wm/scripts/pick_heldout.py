"""Pick a stratified held-out set of games disjoint from a training preset.

Held-out games are selected from the deduped pool such that:
1. No name overlap with the training preset.
2. No token-hash overlap (so we don't pick a structural duplicate of a
   training game).
3. Stratified across the same complexity buckets as the training adds.

The result is a list of (name, n_rules, n_objects, n_levels) ready to feed
into latent overlay scripts and per-game eval scripts.

Usage:
    .venv/bin/python3 -m nca_wm.scripts.pick_heldout \
        --train_preset scaling_gallery_v4 \
        --n_heldout 30 \
        --out data/heldout_v4_n30.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import random
import sys


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def load_preset(preset_name: str) -> list[str]:
    """Read MULTI_GAME_PRESETS[preset_name] from train.py via AST (no JAX import)."""
    with open(os.path.join(REPO, "nca_wm", "train.py")) as f:
        src = f.read()
    tree = ast.parse(src)
    found: list[str] | None = None

    def walk(node):
        nonlocal found
        if hasattr(node, "_fields"):
            for f in node._fields:
                v = getattr(node, f, None)
                if isinstance(v, list):
                    for x in v:
                        if isinstance(x, ast.AST):
                            walk(x)
                elif isinstance(v, ast.AST):
                    walk(v)
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == preset_name:
                    found = [el.value for el in v.elts]

    walk(tree)
    if found is None:
        raise ValueError(f"preset {preset_name!r} not found in train.py MULTI_GAME_PRESETS")
    return found


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_preset", required=True,
                    help="Name in train.py MULTI_GAME_PRESETS to exclude.")
    ap.add_argument("--dedup_file", default="data/dedup_candidates_v2.json")
    ap.add_argument("--n_heldout", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_names = load_preset(args.train_preset)
    print(f"Training preset {args.train_preset!r}: {len(train_names)} games",
          file=sys.stderr)

    with open(args.dedup_file) as f:
        dedup = json.load(f)
    pool = dedup["candidates"]
    print(f"Deduped pool: {len(pool)} candidates", file=sys.stderr)

    # Compute token hashes for the training set
    print("Hashing training preset…", file=sys.stderr)
    train_hashes: set[str] = set()
    for g in train_names:
        h = get_token_hash(g)
        if h:
            train_hashes.add(h)

    # Filter pool: must not be in train preset by name, and must not have a
    # token hash matching any training game.
    train_names_set = set(train_names)
    available = [c for c in pool
                 if c["name"] not in train_names_set
                 and c.get("token_hash")
                 and c["token_hash"] not in train_hashes]
    print(f"After exclusion: {len(available)} eligible heldout candidates",
          file=sys.stderr)

    # Stratify across complexity buckets (same shape as v3 adds).
    buckets = [
        ("low", lambda c: c["n_rules"] <= 3 and c["n_objects"] <= 8, 0.30),
        ("mid", lambda c: 4 <= c["n_rules"] <= 8 and c["n_objects"] <= 12, 0.40),
        ("high", lambda c: 9 <= c["n_rules"] <= 15 and c["n_objects"] <= 16, 0.20),
        ("xhigh", lambda c: 16 <= c["n_rules"] <= 20, 0.10),
    ]
    rng = random.Random(args.seed)
    picked = []
    for label, pred, share in buckets:
        n_this = int(round(args.n_heldout * share))
        in_bucket = [g for g in available if pred(g)]
        rng.shuffle(in_bucket)
        # Prefer games with >=2 levels for richer eval; fall back to 1-level
        multi = [g for g in in_bucket if g["n_levels"] >= 2]
        single = [g for g in in_bucket if g["n_levels"] < 2]
        chosen = (multi[:n_this] if len(multi) >= n_this
                  else multi + single[:n_this - len(multi)])
        picked.extend(chosen)
        print(f"  bucket {label}: {len(chosen)}/{n_this} "
              f"(pool={len(in_bucket)})", file=sys.stderr)

    out = {
        "train_preset": args.train_preset,
        "n_train_games": len(train_names),
        "n_heldout": len(picked),
        "seed": args.seed,
        "heldout": [
            {
                "name": p["name"],
                "source": p["source"],
                "n_rules": p["n_rules"],
                "n_objects": p["n_objects"],
                "n_levels": p["n_levels"],
                "max_level_area": p["max_level_area"],
                "token_hash": p["token_hash"],
            }
            for p in picked
        ],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}: {len(picked)} held-out games "
          f"(target {args.n_heldout}).")
    for p in sorted(picked, key=lambda c: (c["n_rules"], c["n_objects"])):
        print(f"  {p['name']:55s} src={p['source']:9s} "
              f"r={p['n_rules']:2d} o={p['n_objects']:2d} "
              f"L={p['n_levels']:2d}")


if __name__ == "__main__":
    main()
