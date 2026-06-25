"""Compile + filter the puzzlescript-gists corpus into a trainable multi-game set.

Heterogeneous human games: filter to compile-able, bounded-object, bounded-grid
games and cache their serialized engine JSON + metadata. Channel convention: the
engine id_dict orders background=0, player=1, then game objects, so those roles
are pinned for free; the rest are game-specific slots the belief infers in-context.

    .venv/bin/python -u -m nca_wm.active_learning.multigame_build --n 450 --cmax 24
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import signal
import time
from pathlib import Path

_GISTS = Path.home() / "puzzlescript-gists"
_OUT = Path(__file__).resolve().parent / "_multigame"


class _TO(Exception):
    pass


def _alarm(sec):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_TO()))
    signal.alarm(sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--cmax", type=int, default=24)
    ap.add_argument("--hmax", type=int, default=16)
    ap.add_argument("--wmax", type=int, default=20)
    ap.add_argument("--scan-limit", type=int, default=4000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from nca_wm import game_curriculum as gc
    from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
    from puzzlescript_jax.utils import init_ps_lark_parser
    from game_synth.dedup import token_key

    _OUT.mkdir(exist_ok=True)
    (_OUT / "json").mkdir(exist_ok=True)
    gc._set_materialize_dir(_OUT / "_scratch")
    (_OUT / "_scratch").mkdir(exist_ok=True)
    parser = init_ps_lark_parser()

    files = sorted(glob.glob(str(_GISTS / "*.txt")))
    random.Random(args.seed).shuffle(files)
    files = files[args.shard::args.nshards][: args.scan_limit]   # disjoint shard

    man_name = "manifest.jsonl" if args.nshards == 1 else f"manifest_{args.shard}.jsonl"
    man = (_OUT / man_name).open("w")
    seen_sig = set()
    kept = 0
    n_fail = n_size = n_dup = 0
    t0 = time.time()
    for f in files:
        if kept >= args.n:
            break
        name = Path(f).stem
        try:
            code = Path(f).read_text(errors="ignore")
            sig = token_key(code)[0]
            if sig in seen_sig:
                n_dup += 1
                continue
            _alarm(20)
            gc._materialize_game(name, code)
            js = CppPuzzleScriptBackend().compile_and_serialize(parser, name)
            env = CppPuzzleScriptEnv(js, level_i=0, max_episode_steps=5)
            C, H, Wd = (int(x) for x in env.observation_shape)
            nlev = int(env.num_levels)
            signal.alarm(0)
        except Exception:
            signal.alarm(0)
            n_fail += 1
            continue
        if C > args.cmax or H > args.hmax or Wd > args.wmax:
            n_size += 1
            continue
        seen_sig.add(sig)
        (_OUT / "json" / f"{name}.json").write_text(js)
        man.write(json.dumps({"gist": name, "n_obj": C, "H": H, "W": Wd,
                              "n_levels": nlev, "sig": sig}) + "\n")
        man.flush()
        kept += 1
        if kept % 25 == 0:
            print(f"  kept {kept}/{args.n} | fail {n_fail} size {n_size} dup {n_dup} "
                  f"| {time.time()-t0:.0f}s", flush=True)
    man.close()
    print(f"\nDONE kept {kept} | fail {n_fail} oversize {n_size} dup {n_dup} "
          f"| {time.time()-t0:.0f}s -> {_OUT}", flush=True)


if __name__ == "__main__":
    main()
