"""Token-based dedup signature (reuses dedup_master / puzzlescript-gists pipeline).

`token_key(code)` returns a canonical group key: "T:<mech_hash>:<lev_hash>" when the
game parses (Lark -> Strip -> GenPSTree -> tokenize_game, so object names / colors /
sprites are canonicalized away), else a normalized/content-hash fallback. Two games
with the same key are mechanically identical up to naming/cosmetics.

    .venv/bin/python -u -m game_synth.dedup --audit game_synth/dataset
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import re

from nca_wm.scripts import dedup_master as DM

_HEADERS = {"OBJECTS", "LEGEND", "SOUNDS", "COLLISIONLAYERS", "RULES",
            "WINCONDITIONS", "LEVELS"}


def _is_noop_rule(line: str) -> bool:
    """A rule whose bracket pattern is identical LHS<->RHS and has no command/sound
    (e.g. `[ Player | ObjA ] -> [ Player | ObjA ]`, `random [A]->[A]`) — does nothing."""
    if line.count("->") != 1:
        return False
    lhs, rhs = line.split("->")
    if "[" not in lhs or "]" not in rhs:
        return False
    lb = lhs[lhs.find("["):lhs.rfind("]") + 1]
    rlast = rhs.rfind("]")
    if rhs[rlast + 1:].strip():            # trailing command/sound (again/win/sfx/...) -> has effect
        return False
    rb = rhs[rhs.find("["):rlast + 1]
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    return bool(lb.strip()) and norm(lb) == norm(rb)


def strip_noop_rules(code: str) -> str:
    """Remove identity/no-op rules from the RULES section (behavior-preserving)."""
    out, in_rules = [], False
    for line in code.split("\n"):
        u = line.strip().upper()
        if u in _HEADERS:
            in_rules = (u == "RULES")
            out.append(line)
            continue
        if in_rules and "->" in line and not line.lstrip().startswith("(") \
                and _is_noop_rule(line):
            continue                       # drop the no-op rule
        out.append(line)
    return "\n".join(out)


def _drop_noop_tree_rules(rules):
    """Filter no-op rules from a GenPSTree rule list (recurses into rule groups).
    A leaf rule is a no-op iff LHS kernels == RHS kernels and it has no command;
    a group is dropped if all its children are."""
    out = []
    for r in rules:
        sub = getattr(r, "rules", None)
        if sub:                                       # startloop/endloop group
            r.rules = _drop_noop_tree_rules(sub)
            if r.rules:
                out.append(r)
        elif getattr(r, "command", None) or r.left_kernels != r.right_kernels:
            out.append(r)                             # has effect -> keep
    return out


def token_key(code: str, parse_timeout: int = 15) -> tuple[str, str]:
    """Canonical mechanics key with NO-OP RULES STRIPPED on the PARSED tree
    (structural left_kernels==right_kernels, no command). Falls back to a regex
    no-op strip + normalized/content hash if the game does not parse."""
    if not DM._W:
        DM._worker_init()
    W = DM._W
    try:
        pre = W["preprocess_ps"](code)
        func_sig = DM._functional_prelude_sig(pre)
        with DM._Timeout(parse_timeout):
            tree = W["Gen"]().transform(W["Strip"]().transform(W["parser"].parse(pre)))
        tree.rules = _drop_noop_tree_rules(tree.rules)
        objs = tree.objects
        ids = list(objs.keys()) if isinstance(objs, dict) else [o.name for o in objs]
        mech = W["tokenize_game"](tree, ids, encode_sprites=False, include_levels=False)
        return "T:" + DM._sha(",".join(map(str, mech[:-1])) + "#" + func_sig), "ok"
    except Exception:
        c = strip_noop_rules(code)
        nh = DM._norm_hash(c)
        return (("N:" + nh) if nh else
                ("C:" + hashlib.sha1(c.encode("utf-8", "ignore")).hexdigest())), "fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="game_synth/dataset",
                    help="dataset dir with games/*.txt to audit for true distinctness")
    args = ap.parse_args()
    gdir = _REPO / args.audit / "games"
    files = sorted(gdir.glob("*.txt"))
    DM._worker_init()
    keys = {}
    status = {}
    for i, f in enumerate(files):
        code = f.read_text(errors="ignore")
        k, st = token_key(code)
        keys.setdefault(k, []).append(f.name)
        status[st] = status.get(st, 0) + 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} | distinct so far {len(keys)}", flush=True)
    dups = {k: v for k, v in keys.items() if len(v) > 1}
    print(f"\nfiles: {len(files)} | TRULY DISTINCT (token): {len(keys)}")
    print(f"collision groups: {len(dups)} | files in collisions: {sum(len(v) for v in dups.values())}")
    print(f"parse status: {status}")
    parsed = sum(v for k, v in status.items() if k == "ok")
    print(f"parsed-by-lark: {parsed}/{len(files)} "
          f"({'token-deduped' if parsed else 'fallback'} the rest by norm/content hash)")


if __name__ == "__main__":
    main()
