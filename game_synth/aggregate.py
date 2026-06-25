"""Track C: aggregate + validate + dedup candidate environments into a dataset.

Sources:
  - GP (Track B): game_synth/gp_*/manifest.jsonl  (pre-validated: compile + BFS dynamics)
  - generated_games/  (Track A / prior ELM runs, LLM-authored)
  - any --extra dirs (e.g. a fresh ELM run's games/)

Validates LLM-authored games by compiling them, dedups by a normalized mechanics
signature, and curates up to --target unique valid environments (LLM-authored
first for novelty, then GP for volume) into game_synth/dataset/ + manifest.jsonl.

    .venv/bin/python -u -m game_synth.aggregate --target 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from game_synth.dedup import token_key


def mechanics_sig(code: str) -> str:
    """Canonical token-based mechanics key (Lark->tokenize, names/colors/sprites
    canonicalized away; falls back to normalized/content hash if unparseable).
    Same approach as the puzzlescript-gists / dedup_master dedup."""
    return token_key(code)[0]


def collect_candidates(extra_dirs):
    cands = []  # (source, name, path, meta)
    # GP (trusted-valid via manifest)
    for mf in sorted(_REPO.glob("game_synth/gp_*/manifest.jsonl")):
        gdir = mf.parent / "games"
        for line in mf.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            p = gdir / f"{r['name']}.txt"
            if p.exists():
                cands.append(("gp", r["name"], p, {"solvable": r.get("all_solvable", False),
                                                   "validated": True}))
    # LLM-authored corpora (need compile validation)
    for d in ["generated_games", *extra_dirs]:
        dd = _REPO / d
        for p in sorted(dd.glob("*.txt")) if dd.exists() else []:
            cands.append(("llm", p.stem, p, {"validated": False}))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--out", default="game_synth/dataset")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--validate-llm", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    out = _REPO / args.out
    gdir = out / "games"
    gdir.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(out / "_scratch")
    (out / "_scratch").mkdir(exist_ok=True)

    cands = collect_candidates(args.extra)
    n_gp = sum(1 for c in cands if c[0] == "gp")
    n_llm = sum(1 for c in cands if c[0] == "llm")
    print(f"candidates: {len(cands)} (gp {n_gp}, llm {n_llm})", flush=True)

    parser = None
    if args.validate_llm:
        from puzzlescript_jax.utils import init_ps_lark_parser
        parser = init_ps_lark_parser()
        from puzzlescript_cpp import CppPuzzleScriptBackend

    # Order: LLM-authored first (novelty), then GP (volume).
    cands.sort(key=lambda c: 0 if c[0] == "llm" else 1)
    seen_sig = set()
    chosen = []
    n_dup = n_badcompile = n_checked = 0
    t0 = time.time()
    for source, name, path, meta in cands:
        if len(chosen) >= args.target:
            break
        try:
            code = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        sig = mechanics_sig(code)
        if sig in seen_sig:
            n_dup += 1
            continue
        if source == "llm" and args.validate_llm:
            n_checked += 1
            try:
                gc._materialize_game(name, code)
                from puzzlescript_cpp import CppPuzzleScriptBackend
                CppPuzzleScriptBackend().compile_game(parser, name)
            except Exception:
                n_badcompile += 1
                continue
            meta["validated"] = True
        seen_sig.add(sig)
        chosen.append((source, name, path, meta, sig))
        if len(chosen) % 100 == 0:
            print(f"  chosen {len(chosen)}/{args.target} | dup {n_dup} "
                  f"badcompile {n_badcompile} | {time.time()-t0:.0f}s", flush=True)

    # Write dataset
    man = (out / "manifest.jsonl").open("w", encoding="utf-8")
    n_solv = 0
    for i, (source, name, path, meta, sig) in enumerate(chosen):
        dest = gdir / f"env_{i:04d}_{source}_{name[:40]}.txt"
        dest.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        n_solv += int(meta.get("solvable", False))
        man.write(json.dumps({"id": i, "file": dest.name, "source": source,
                              "orig_name": name, "mech_sig": sig, **meta}) + "\n")
    man.close()
    print(f"\nDATASET: {len(chosen)} envs -> {gdir}", flush=True)
    print(f"  sources: llm {sum(1 for c in chosen if c[0]=='llm')}, "
          f"gp {sum(1 for c in chosen if c[0]=='gp')} | solvable(gp) {n_solv} | "
          f"dups dropped {n_dup} | llm badcompile {n_badcompile}", flush=True)


if __name__ == "__main__":
    main()
