"""JS-grounded round-trip test for tokenize / detokenize.

Drops a detokenized game source into custom_games/ under a unique name,
then re-parses it via the *same* `get_game_tree_from_js` path that
train.py uses (PS preprocessor -> JS engine compile -> parsed_state_to_tree),
and compares the resulting token stream against the original.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_jax.globals import CUSTOM_GAMES_DIR, SIMPLIFIED_GAMES_DIR

from nca_wm.tokenize_game import (
    tokenize_game, get_game_tree_from_js, INV_VOCAB, VOCAB,
)
from nca_wm.detokenize_game import detokenize


def _parse_orig(ps_parser, game_name):
    return get_game_tree_from_js(ps_parser, game_name)


def _parse_recon(ps_parser, source_text: str, tag: str):
    """Compile a raw source string via the JS engine and return (tree, canon_ids)."""
    name = f"_rt_{tag}"
    path = os.path.join(CUSTOM_GAMES_DIR, f"{name}.txt")
    simp_path = os.path.join(SIMPLIFIED_GAMES_DIR, f"{name}_simplified.txt")
    os.makedirs(CUSTOM_GAMES_DIR, exist_ok=True)
    try:
        # Bust the simplified-file cache so the JS engine recompiles fresh.
        if os.path.isfile(simp_path):
            os.remove(simp_path)
        with open(path, "w") as f:
            f.write(source_text)
        return get_game_tree_from_js(ps_parser, name)
    finally:
        for p in (path, simp_path):
            try:
                os.remove(p)
            except OSError:
                pass


def round_trip(ps_parser, game_name: str, encode_sprites: bool):
    """Return (toks_orig, toks_recon, success)."""
    tree_a, canon_a = _parse_orig(ps_parser, game_name)
    toks_a = tokenize_game(tree_a, canon_a, encode_sprites=encode_sprites)
    src = detokenize(toks_a, title=game_name, generate_placeholder_level=True)
    tag = "".join(c if c.isalnum() else "_" for c in game_name)[:40] + f"_{int(time.time()*1000) % 100000}"
    tree_b, canon_b = _parse_recon(ps_parser, src, tag)
    toks_b = tokenize_game(tree_b, canon_b, encode_sprites=encode_sprites)
    return toks_a, toks_b, list(toks_a) == list(toks_b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True,
                   help="Game names (looked up via the standard search hierarchy).")
    p.add_argument("--encode_sprites", action="store_true")
    p.add_argument("--diff_window", type=int, default=8)
    args = p.parse_args()

    ps_parser = init_ps_lark_parser()
    n_ok = n_bad = n_err = 0
    for g in args.games:
        try:
            a, b, same = round_trip(
                ps_parser, g,
                encode_sprites=args.encode_sprites,
            )
        except Exception as e:
            n_err += 1
            print(f"  {g:30s}  ERR  {type(e).__name__}: {str(e)[:120]}")
            continue
        if same:
            n_ok += 1
            print(f"  {g:30s}  OK   ({len(a)} tokens)")
        else:
            n_bad += 1
            # Find first divergence
            for i in range(min(len(a), len(b))):
                if a[i] != b[i]:
                    w = args.diff_window
                    print(f"  {g:30s}  DIFF orig={len(a)} recon={len(b)}  first @ {i}: "
                          f"{INV_VOCAB.get(a[i])} vs {INV_VOCAB.get(b[i])}")
                    print(f"     orig  ...{' '.join(INV_VOCAB.get(t,'?') for t in a[max(0,i-w):i+w])}...")
                    print(f"     recon ...{' '.join(INV_VOCAB.get(t,'?') for t in b[max(0,i-w):i+w])}...")
                    break
            else:
                print(f"  {g:30s}  DIFF orig={len(a)} recon={len(b)}  (length only; prefix matches)")

    total = n_ok + n_bad + n_err
    print(f"\nbit-exact: {n_ok}/{total}  diff: {n_bad}  error: {n_err}")


if __name__ == "__main__":
    main()
