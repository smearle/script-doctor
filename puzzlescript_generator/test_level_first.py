"""Tests for level_first.py.

For each (game, level):
- text: to_standard(level_first(c, level)) equals render() of the relabelled game, and
  prompt_of() cuts at MECHANICS;
- engine: ps_check.js accepts the standard text, and ps_equiv.js does not find it different
  from render(c, [level]) under the level-first renaming (a checker timeout, the engine's
  warning cap or a nondeterministic original leaves a level undecided; these are listed);
- determinism: identical texts under three PYTHONHASHSEED values.
Malformed variants of one document must raise FormatError; formatting variants (blank
lines, flag, object and legend order, spacing) must give the same standard text.

Games: three built-in sources covering properties, aggregates, background and player as
properties and as synonyms, synonym chains, objects absent from the level and late, +,
again and random rules; with --extract (a prepare_*.py extract.jsonl), also up to
--per-feature corpus games showing each feature (stage-2-verified ones, when equiv.jsonl
is beside the extract).

    python test_level_first.py --engine-dir DIR [--extract FILE] [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from canonicalize import CanonError, from_engine, render
from check_games import check_texts
from level_first import FormatError, level_first, prompt_of, relabel, to_standard
from prepare_level_first import equiv_outcome

GAMES = {
    "push": """title push and sparks
noaction
OBJECTS
Background
black
Wall
grey
Player
white
Crate
orange
Target
red
CrateOnTarget
yellow
Spark
blue

LEGEND
. = Background
# = Wall
P = Player
* = Crate
O = Target
Pushable = Crate or CrateOnTarget
Heavy = Pushable or Crate
Q = Player and Target

SOUNDS
COLLISIONLAYERS
Background
Target
Spark
Player, Wall, Crate, CrateOnTarget

RULES
[ > Player | Pushable ] -> [ > Player | > Pushable ]
+ [ > Pushable | Pushable ] -> [ > Pushable | > Pushable ]
late [ Crate Target ] -> [ CrateOnTarget Target Spark ]
late [ CrateOnTarget no Target ] -> [ Crate ]
[ Spark ] -> [ ] again
[ > Heavy | Wall ] -> [ Heavy | Wall ]

WINCONDITIONS
all Target on CrateOnTarget

LEVELS
########
#.P*..O#
#..*.O.#
#.Q....#
########

########
#P.*.O.#
########
""",
    "properties": """title background and player as properties
run_rules_on_level_start
OBJECTS
Grass
green
Sand
yellow
PlayerL
white
PlayerR
white
Rock
grey
Gem
blue

LEGEND
Background = Grass or Sand
Player = PlayerL or PlayerR
. = Grass
, = Sand
L = PlayerL and Grass
R = PlayerR and Sand
k = Rock and Grass
g = Gem and Sand
Solid = Rock

SOUNDS
COLLISIONLAYERS
Background
Gem
Player, Rock

RULES
[ left PlayerR ] -> [ left PlayerL ]
[ right PlayerL ] -> [ right PlayerR ]
[ > Player | Solid ] -> [ > Player | > Solid ]
late [ Player Gem ] -> [ Player ]

WINCONDITIONS
no Gem

LEVELS
.,.,.,
.L.k,g
,,..g.

,,R,
,k.g
""",
    "synonyms": """title synonym chains and aggregates
OBJECTS
Floor
darkgrey
Carpet
red
Hero
white
Box
orange
Coin
yellow
Spawner
purple
Ghost
lightblue

LEGEND
BgAll = Floor or Carpet
Background = BgAll
Player = Hero
Mover = Player or Box
Pile = Box and Coin
. = Floor
- = Carpet
H = Hero and Floor
b = Box and Carpet
p = Pile and Floor
s = Spawner and Floor

SOUNDS
COLLISIONLAYERS
Background
Coin
Spawner
Mover, Ghost

RULES
[ > Mover | Box ] -> [ > Mover | > Box ]
random [ Spawner no Coin ] -> [ Spawner Coin ]
late [ Hero Coin ] -> [ Hero ]
late [ Pile ] -> [ Box ]
[ > Hero | Ghost ] -> [ Hero | Ghost ]

WINCONDITIONS
no Coin
some Hero

LEVELS
..-.-..
.H.b.s.
-.p..--
""",
}

FEATURES = {
    "property": lambda c: any(op == "or" for _, op, _ in c.legend),
    "aggregate": lambda c: any(op == "and" for _, op, _ in c.legend),
    "repeated legend member": lambda c: any(len(set(ms)) < len(ms) for _, _, ms in c.legend),
    "background in legend": lambda c: any(n == "background" for n, _, _ in c.legend),
    "player in legend": lambda c: any(n == "player" for n, _, _ in c.legend),
    "object absent from a level": lambda c: any(
        set(c.objects) - frozenset().union(*(cell for row in lv for cell in row)) for lv in c.levels),
    "late rule": lambda c: any(r[0] == "late" for r in c.rules),
    "+ rule": lambda c: any(r[0] == "+" for r in c.rules),
    "again": lambda c: any("again" in r for r in c.rules),
    "random": lambda c: any("random" in r for r in c.rules),
}


def check_text(c, lv):
    text, rename = level_first(c, lv)
    prompt = prompt_of(text)
    assert text.startswith(prompt) and prompt.endswith("\nMECHANICS\n"), "prompt_of"
    assert "\n\n\n" not in text and text.endswith("\n") and not text.endswith("\n\n"), "layout"
    in_level = sorted({rename[o] for row in lv for cell in row for o in cell} - {"background", "player"},
                      key=lambda o: int(o[1:]))
    assert in_level == [f"o{k}" for k in range(1, len(in_level) + 1)], "level objects are o1..oK"
    first = [o for line in prompt.split("\n") if re.match(r"^\S = ", line)
             for o in re.findall(r"\bo\d+\b", line)]
    assert sorted(set(first), key=first.index) == in_level, "numbered by first appearance"
    std = to_standard(text)
    want = render(relabel(c, rename, lv))
    assert std == want, f"to_standard differs from render:\n{std}\n---\n{want}"
    return text, rename, std


def mutations(text):
    """(name, malformed variant) pairs for a document with prompt definitions, a mechanics
    legend entry, l1 and object o1."""
    lines = text.split("\n")
    grid0 = lines[2]
    i_wc = text.index("WINCONDITIONS")
    yield "truncated", text[:i_wc]
    yield "no delimiter", text.replace("\nMECHANICS\n", "\n")
    yield "SOUNDS section", text.replace("\nCOLLISIONLAYERS\n", "\nSOUNDS\n\nCOLLISIONLAYERS\n")
    yield "sections swapped", text.replace("\nRULES\n", "\nTMP\n").replace("\nWINCONDITIONS\n", "\nRULES\n") \
        .replace("\nTMP\n", "\nWINCONDITIONS\n")
    yield "text before LEVEL", "title x\n" + text
    yield "wrong colour", text.replace("\no1\n#e6194b\n", "\no1\n#3cb44b\n")
    yield "object without colour", text.replace("\no1\n#e6194b\n", "\no1\n")
    yield "duplicate object", text.replace("\no1\n#e6194b\n", "\no1\n#e6194b\n\no1\n#e6194b\n")
    yield "bad object name", text.replace("\no1\n#e6194b\n", "\nwall\n#e6194b\n")
    yield "unknown prelude flag", text.replace("\nMECHANICS\n", "\nMECHANICS\n\nagain_interval 0.1\n")
    yield "undefined legend name", text.replace("\nl1 = ", "\nl1 = o99 or ")
    yield "legend cycle", text.replace("\nCOLLISIONLAYERS\n", "\nl8 = l9\nl9 = l8\n\nCOLLISIONLAYERS\n")
    yield "self reference", text.replace("\nCOLLISIONLAYERS\n", "\nl8 = l8\n\nCOLLISIONLAYERS\n")
    yield "mixed and/or", text.replace("\nCOLLISIONLAYERS\n", "\nl8 = o1 or o2 and o3\n\nCOLLISIONLAYERS\n")
    yield "dangling op", text.replace("\nCOLLISIONLAYERS\n", "\nl8 = o1 or\n\nCOLLISIONLAYERS\n")
    yield "player in mechanics", text.replace("\nCOLLISIONLAYERS\n", "\nplayer = o1\n\nCOLLISIONLAYERS\n")
    yield "glyph in mechanics", text.replace("\nCOLLISIONLAYERS\n", "\nz = o1\n\nCOLLISIONLAYERS\n")
    yield "l-name defined twice", text.replace("\nCOLLISIONLAYERS\n", "\nl1 = o1\n\nCOLLISIONLAYERS\n")
    yield "unused prompt definition", text.replace("\n\nMECHANICS\n", "\nl8 = o1\n\nMECHANICS\n")
    yield "prompt uses mechanics name", text.replace("\nl3 = o1 or o2\nbackground = l3\n", "\nbackground = l1\n")
    yield "background undefined", text.replace("\nbackground = l3\n", "\n")
    yield "undefined grid glyph", text.replace(f"\n{grid0}\n", f"\nz{grid0[1:]}\n", 1)
    yield "ragged grid", text.replace(f"\n{grid0}\n", f"\n{grid0[:-1]}\n", 1)
    yield "glyph with or", text.replace("\na = o1\n", "\na = o1 or o2\n")
    yield "glyph names a legend name", text.replace("\na = o1\n", "\na = l1\n")
    upper = grid0.replace("a", "A")
    yield "glyph not in alphabet", text.replace("\na = o1\n", "\nA = o1\n").replace(f"\n{grid0}\n", f"\n{upper}\n")
    yield "glyph defined twice", text.replace("\na = o1\n", "\na = o1\na = o2\n")
    yield "glyph lists an object twice", text.replace("\na = o1\n", "\na = o1 and o1\n")
    yield "bad layer", text.replace("\nCOLLISIONLAYERS\n\n", "\nCOLLISIONLAYERS\n\no1,, o2\n")
    yield "three LEVEL blocks", text.replace("\n\na = o1\n", "\n\nb = o2\n\na = o1\n")


def variants(text):
    """Formatting variants of a document with no prelude, >= 2 objects and l1 depending on
    nothing defined after it: each must give the same standard text once the listed flags
    are added in canonical order."""
    yield "no final newline", text[:-1], text
    yield "extra blank lines", text.replace("\n\n", "\n\n\n"), text
    yield "spacing in rules and wins", text.replace(" -> ", "   ->  ").replace("\nno o6", "\nno   o6"), text
    blocks = text.split("\n\nOBJECTS\n\n")[1].split("\n\nLEGEND\n")[0].split("\n\n")
    yield "objects reversed", text.replace("\n\n".join(blocks), "\n\n".join(blocks[::-1])), text
    yield "flags reversed", text.replace("\nMECHANICS\n\n", "\nMECHANICS\n\nnoundo\nrun_rules_on_level_start\n\n"), \
        text.replace("\nMECHANICS\n\n", "\nMECHANICS\n\nrun_rules_on_level_start\nnoundo\n\n")
    yield "legend reordered", text.replace("\nLEGEND\n\nl1 = o3 or o4\n", "\nLEGEND\n\nl8 = l1\nl1 = o3 or o4\n"), \
        text.replace("\nLEGEND\n\nl1 = o3 or o4\n", "\nLEGEND\n\nl1 = o3 or o4\nl8 = l1\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--extract", type=Path, default=None, help="extract.jsonl of a prepare_*.py run")
    ap.add_argument("--per-feature", type=int, default=4)
    ap.add_argument("--levels-per-game", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stall-s", type=float, default=120.0)
    args = ap.parse_args()

    recs = check_texts([{"id": k, "text": v} for k, v in GAMES.items()], args.engine_dir, workers=args.workers,
                       script="ps_extract.js")
    assert all(r.get("ok") for r in recs), [r["id"] for r in recs if not r.get("ok")]
    games = {r["id"]: from_engine(r) for r in recs}
    coverage = {f: [g for g, c in games.items() if test(c)] for f, test in FEATURES.items()}
    if args.extract:
        equiv = args.extract.with_name("equiv.jsonl")  # stage-2 verdicts, when beside the extract
        verified = ({q["id"] for q in map(json.loads, open(equiv)) if equiv_outcome(q) == "equivalent"}
                    if equiv.exists() else None)
        for line in open(args.extract):
            e = json.loads(line)
            if not e.get("ok") or (verified is not None and e["id"] not in verified):
                continue
            try:
                c = from_engine(e)
            except CanonError:
                continue
            for f, test in FEATURES.items():
                if len(coverage[f]) < 3 + args.per_feature and test(c):
                    games.setdefault(e["id"], c)
                    coverage[f].append(e["id"])
    print("games per feature:", {f: len(v) for f, v in coverage.items()})
    assert all(coverage.values()), coverage

    # text round trip
    pairs = [(g, k) for g, c in games.items() for k in range(min(len(c.levels), args.levels_per_game))]
    docs = {}
    for g, k in pairs:
        docs[g, k] = check_text(games[g], games[g].levels[k])
    print(f"text round trip: {len(pairs)} levels of {len(games)} games ok")

    # malformed input raises; formatting variants do not change the standard text
    text = docs["synonyms", 0][0]
    assert "\nbackground = l3\n" in text and "\nl1 = o3 or o4\n" in text and "\na = o1\n" in text
    for name, bad in mutations(text):
        try:
            to_standard(bad)
        except FormatError as ex:
            print(f"  rejects {name}: {ex}")
            continue
        raise AssertionError(f"accepted malformed input: {name}")
    for name, variant, same in variants(text):
        assert variant != same and to_standard(variant) == to_standard(same), name
    print("malformed input and formatting variants ok")

    # determinism across hash seeds
    probe = Path(os.environ.get("TMPDIR", "/tmp")) / f"lf-determinism-{os.getpid()}.jsonl"
    with probe.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    code = ("import json, sys, hashlib\nfrom canonicalize import from_engine\n"
            "from level_first import level_first, to_standard\nh = hashlib.sha256()\n"
            "for line in open(sys.argv[1]):\n    c = from_engine(json.loads(line))\n"
            "    for lv in c.levels:\n        t, rn = level_first(c, lv)\n"
            "        h.update((t + to_standard(t) + json.dumps(rn, sort_keys=True)).encode())\nprint(h.hexdigest())")
    here = Path(__file__).resolve().parent
    digests = {subprocess.run([sys.executable, "-c", code, str(probe)], cwd=here, capture_output=True, text=True,
                              env={**os.environ, "PYTHONHASHSEED": s}, check=True).stdout for s in ("1", "2", "3")}
    probe.unlink()
    assert len(digests) == 1, digests
    print("deterministic across PYTHONHASHSEED 1, 2, 3")

    # engine: the standard text compiles, and is equivalent to render(c, [level]). A checker
    # timeout (e.g. an endless `again` loop under random play), the engine's warning cap or a
    # nondeterministic original leaves a level undecided; only real differences fail.
    ids = [f"{g}:{k}" for g, k in pairs]
    chk = check_texts([{"id": i, "text": docs[p][2]} for i, p in zip(ids, pairs)], args.engine_dir,
                      workers=args.workers, stall_s=args.stall_s)
    eq = check_texts([{"id": i, "orig": render(games[g], [games[g].levels[k]]), "canon": docs[g, k][2],
                       "map": docs[g, k][1]} for i, (g, k) in zip(ids, pairs)], args.engine_dir,
                     workers=args.workers, stall_s=args.stall_s, script="ps_equiv.js")
    verdicts = [equiv_outcome(q) for q in eq]
    failed = [(i, k.get("errors"), v) for i, k, v in zip(ids, chk, verdicts)
              if not k.get("ok") or v in ("canonical does not compile", "initial cells differ", "dynamics differ")]
    print("engine:", sum(bool(k.get("ok")) for k in chk), "of", len(chk), "compile and are playable; equivalence",
          dict(Counter(verdicts)), [(i, v) for i, v in zip(ids, verdicts) if v != "equivalent"])
    assert not failed, failed
    print("all tests passed:", hashlib.sha256("".join(d[0] for d in docs.values()).encode()).hexdigest()[:12])


if __name__ == "__main__":
    main()
