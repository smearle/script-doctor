"""CPU-only A* cache warmer for a list of games.

Calls `collect_multigame_dataset` to populate the per-level
`rollout_data/{game}/level_{i}/astar_transitions_v5_*.npz` caches without
training. When the matching multi-game training run is later launched, its
data-collection step finds the per-level caches and skips A* search.

Use this to parallelize CPU search work alongside an in-progress training
run on a different process. Run with `CUDA_VISIBLE_DEVICES=` so JAX
doesn't grab GPU memory.

Usage:
    CUDA_VISIBLE_DEVICES= .venv/bin/python3 -m nca_wm.scripts.warm_caches \
        --games scaling_gallery_v4_minus_v3 \
        --search_timeout_ms 60000 \
        --max_transitions_per_game 200000
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import time


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def load_preset(preset_name: str) -> list[str]:
    """Read MULTI_GAME_PRESETS[preset_name] from train.py via AST."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True,
                    help="Either a preset name from MULTI_GAME_PRESETS, or "
                         "'<preset>_minus_<other_preset>' for set difference, "
                         "or comma list 'g1,g2,...'.")
    ap.add_argument("--n_search_steps", type=int, default=100_000)
    ap.add_argument("--search_timeout_ms", type=int, default=60_000)
    ap.add_argument("--search_algo", default="astar")
    ap.add_argument("--max_transitions_per_game", type=int, default=200_000)
    ap.add_argument("--encode_sprites", action="store_true")
    args = ap.parse_args()

    # Resolve game list
    if "_minus_" in args.games:
        a, b = args.games.split("_minus_", 1)
        ga = set(load_preset(a)) if a in [
            *_existing_presets()] else set(a.split(","))
        gb = set(load_preset(b)) if b in [
            *_existing_presets()] else set(b.split(","))
        games = sorted(ga - gb, key=lambda s: s.lower())
    elif "," in args.games:
        games = args.games.split(",")
    else:
        games = load_preset(args.games)
    print(f"Resolved {len(games)} games to warm")
    for g in games[:5]:
        print(f"  {g}")
    if len(games) > 5:
        print(f"  ... ({len(games) - 5} more)")

    # Lazy import — pulls JAX (no GPU thanks to CUDA_VISIBLE_DEVICES env)
    print("\nImporting train module (this loads JAX)...")
    from nca_wm.train import collect_multigame_dataset
    from puzzlescript_jax.utils import init_ps_lark_parser

    print("Initializing PuzzleScript parser...")
    ps_parser = init_ps_lark_parser()

    t0 = time.time()
    print(f"\nWarming caches for {len(games)} games "
          f"(search={args.search_algo}, timeout={args.search_timeout_ms}ms, "
          f"cap={args.max_transitions_per_game})...")
    dataset, infos = collect_multigame_dataset(
        game_names=games,
        ps_parser=ps_parser,
        n_search_steps=args.n_search_steps,
        search_timeout_ms=args.search_timeout_ms,
        search_algo=args.search_algo,
        encode_sprites=args.encode_sprites,
        max_transitions_per_game=args.max_transitions_per_game,
    )
    elapsed = time.time() - t0
    n_total = sum(len(s) for s in dataset["per_game_states"])
    print(f"\nDONE in {elapsed/60:.1f}m. "
          f"{len(infos)} games, {n_total:,} total transitions.")


def _existing_presets() -> list[str]:
    """Return preset names from MULTI_GAME_PRESETS (lazy)."""
    with open(os.path.join(REPO, "nca_wm", "train.py")) as f:
        src = f.read()
    tree = ast.parse(src)
    presets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "MULTI_GAME_PRESETS":
                    if isinstance(node.value, ast.Dict):
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant):
                                presets.append(k.value)
    return presets


if __name__ == "__main__":
    main()
