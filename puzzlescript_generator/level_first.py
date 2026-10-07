"""Level-first canonical PuzzleScript: one level, then the mechanics written for it.

A level-first document is a canonical game (canonicalize.py) restricted to ONE of its
levels and written level first, so a language model can be prompted with a level and write
the mechanics. Objects are numbered level-first, so the prompt depends on the level alone,
except for the last tie-break below and the legend numbers in a background or player
definition:
- objects present in the level become o1..oK by first appearance, scanning cells
  row-major; objects that first appear in the same cell are ordered by their occupancy
  signature in the level (the sorted tuple of cell indices where they appear), then by
  their canonical number;
- objects absent from the level follow as o(K+1).., in canonical order;
- `background` and `player` keep their names, legend names keep their canonical numbers
  l1.., and an object's colour is a function of its number, as in canonicalize.render.

Layout: the prompt holds everything the level fixes, then comes the delimiter line
MECHANICS, then the rest, in canonical order and style:

    LEVEL
    <blank>
    grid rows          glyphs by first appearance, "." = background alone
    <blank>
    glyph lines        the objects in each distinct cell (background and player first)
    definitions        background's and player's, when they are legend entries, with the
                       legend names they use, transitively (canonical legend order)
    <blank>
    MECHANICS
    <blank>
    prelude flags, then a blank line (when there are any)
    OBJECTS, LEGEND, COLLISIONLAYERS, RULES, WINCONDITIONS, each header followed by a blank
    line: every object with its colour (also those absent from the level), the remaining
    legend entries, layers, rules and win conditions. No SOUNDS or LEVELS section.

Example (block pushing; wall o1, crate o2 and target o3 are in the level, the crate on a
target o4 and the spark o5 are not; with `player = o3 or o6`, say, the player's definition
would follow the glyph lines):

    LEVEL

    aaaaaaaa
    a.bc..da
    a..c.d.a
    a.e....a
    aaaaaaaa

    a = background and o1
    . = background
    b = background and player
    c = background and o2
    d = background and o3
    e = background and player and o3

    MECHANICS

    noaction

    OBJECTS

    background
    #202020

    player
    #ffffff

    o1
    #e6194b
    ...
    o5
    #f58231

    LEGEND

    l1 = o2 or o4

    COLLISIONLAYERS

    background
    o3
    o5
    player, o1, o2, o4

    RULES

    [ > player | l1 ] -> [ > player | > l1 ]
    + [ > l1 | l1 ] -> [ > l1 | > l1 ]
    late [ o2 o3 ] -> [ o4 o3 o5 ]
    late [ o4 no o3 ] -> [ o2 ]
    [ o5 ] -> [ ] again

    WINCONDITIONS

    all o3 on o4

`to_standard(text)` turns a level-first text, model samples included, into valid
PuzzleScript with that one level, in render's style and section order (OBJECTS, LEGEND,
SOUNDS, COLLISIONLAYERS, RULES, WINCONDITIONS, LEVELS): the legend in canonical emit
order (background and player, then l1, l2, ..., each after the names it uses), objects and
prelude flags in canonical order, glyphs re-assigned by first appearance. Blank lines,
spacing and the order of objects, flags and legend entries are formatting; anything else
off-format raises FormatError and nothing is guessed: missing, extra or reordered sections,
names outside the scheme, a colour that is not its object's, an undefined or cyclic legend
name, a definition in the wrong part, an undefined glyph, a ragged grid. Rules, layers and
win conditions are passed through for the engine to judge. `prompt_of(text)` is the prefix
up to and including the MECHANICS line, for sampling mechanics for a given level; with BPE
that only merges within a line, its tokens are a prefix of the document's tokens.

    from level_first import level_first, prompt_of, to_standard
    text, rename = level_first(c, c.levels[0])   # rename: canonical -> level-first object names
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from canonicalize import (BACKGROUND_COLOR, GLYPHS, KEEP_PRELUDE, PALETTE, PLAYER_COLOR, SPECIAL, Canon,
                          CanonError, _num, _rule_text, render)

DELIM = "MECHANICS"
HEADERS = ("LEVEL", DELIM, "OBJECTS", "LEGEND", "COLLISIONLAYERS", "RULES", "WINCONDITIONS")
FOREIGN_HEADERS = ("SOUNDS", "LEVELS")  # PuzzleScript sections a level-first text never has
OBJECT_RE = re.compile(r"background|player|o[1-9]\d*")
PROMPT_NAME_RE = re.compile(r"background|player|l[1-9]\d*")
MECH_NAME_RE = re.compile(r"l[1-9]\d*")


class FormatError(ValueError):
    """The text is not a well-formed level-first document."""


def _obj_key(o):
    # background, player, then by number, as in render()
    return (o not in SPECIAL, SPECIAL.index(o) if o in SPECIAL else _num(o))


def _color(o):
    return BACKGROUND_COLOR if o == "background" else PLAYER_COLOR if o == "player" else \
        PALETTE[(_num(o) - 1) % len(PALETTE)]


def _legend_line(name, op, members):
    return f"{name} = " + (f" {op} ".join(members) if op != "=" else members[0])


def _glyphs(level) -> dict:
    """Cell -> glyph by first appearance, "." for background alone (render's assignment)."""
    glyph, gi = {}, 0
    for row in level:
        for cell in row:
            if cell in glyph:
                continue
            if cell == frozenset(["background"]):
                glyph[cell] = "."
            elif gi < len(GLYPHS):
                glyph[cell] = GLYPHS[gi]
                gi += 1
            else:
                raise CanonError("too many distinct cells")
    return glyph


def _glyph_line(ch, cell):
    return f"{ch} = " + " and ".join(sorted(cell, key=_obj_key))


def _closure(legend) -> set:
    """Names of the legend entries that define background and player, and of the legend
    names those use, transitively."""
    defs = {name: members for name, _, members in legend}
    out, todo = set(), [n for n in SPECIAL if n in defs]
    while todo:
        n = todo.pop()
        if n not in out:
            out.add(n)
            todo += [m for m in defs[n] if m in defs]
    return out


def level_rename(c: Canon, level) -> dict:
    """Canonical object name -> level-first object name, for every object of c."""
    cells = [cell for row in level for cell in row]
    where = defaultdict(list)  # object -> occupancy signature (cell indices, ascending)
    for i, cell in enumerate(cells):
        for o in cell:
            where[o].append(i)
    order, seen = [], set(SPECIAL)
    for cell in cells:
        new = sorted((o for o in cell if o not in seen), key=lambda o: (where[o], _num(o)))
        order += new
        seen.update(new)
    order += [o for o in c.objects if o not in seen]
    rename = {o: o for o in c.objects if o in SPECIAL}
    rename.update((o, f"o{k}") for k, o in enumerate(order, 1))
    return rename


def relabel(c: Canon, rename: dict, level) -> Canon:
    """c with its objects renamed and `level` (canonical names) as its only level. Legend
    names, and the order of legend entries, layers, rules and win conditions, are kept."""
    def rn(t):  # in canonical rules only names can look like object names
        return rename.get(t, t)

    return Canon(prelude=c.prelude, objects=sorted(map(rn, c.objects), key=_obj_key),
                 legend=[(name, op, [rn(m) for m in members]) for name, op, members in c.legend],
                 layers=[[rn(o) for o in layer] for layer in c.layers],
                 rules=[[rn(t) for t in r] for r in c.rules], wins=[[rn(t) for t in w] for w in c.wins],
                 levels=[[[frozenset(map(rn, cell)) for cell in row] for row in level]],
                 rename={k: rn(v) for k, v in c.rename.items()})


def level_first(c: Canon, level) -> tuple:
    """(level-first text of c restricted to `level`, rename: canonical -> level-first
    object names). `level` is one of c's levels, in canonical names."""
    rename = level_rename(c, level)
    r = relabel(c, rename, level)
    lv = r.levels[0]
    glyph = _glyphs(lv)
    pre = _closure(r.legend)  # a prefix of the canonical legend order
    lines = ["LEVEL", ""] + ["".join(glyph[cell] for cell in row) for row in lv] + [""]
    lines += [_glyph_line(ch, cell) for cell, ch in glyph.items()]
    lines += [_legend_line(*e) for e in r.legend if e[0] in pre]
    lines += ["", DELIM, ""]
    if r.prelude:
        lines += [f"{k} {v}".strip() for k, v in r.prelude] + [""]
    lines += ["OBJECTS", ""]
    for o in r.objects:
        lines += [o, _color(o), ""]
    for header, body in (("LEGEND", [_legend_line(*e) for e in r.legend if e[0] not in pre]),
                         ("COLLISIONLAYERS", [", ".join(layer) for layer in r.layers]),
                         ("RULES", [_rule_text(t) for t in r.rules]),
                         ("WINCONDITIONS", [" ".join(w) for w in r.wins])):
        lines += [header, ""] + body + ([""] if body else [])
    return "\n".join(lines).rstrip("\n") + "\n", rename


def prompt_of(text: str) -> str:
    """The prompt of a level-first text: everything up to and including the MECHANICS line."""
    lines = text.split("\n")
    if DELIM not in lines:
        raise FormatError(f"no {DELIM} line")
    return "\n".join(lines[:lines.index(DELIM) + 1]) + "\n"


def _blocks(lines):
    """Runs of non-blank lines."""
    out, cur = [], []
    for line in lines + [""]:
        if line.strip():
            cur.append(line)
        elif cur:
            out.append(cur)
            cur = []
    return out


def _nonblank(lines):
    return [line for line in lines if line.strip()]


def _legend_entry(line):
    """(name, op, members) of "a = b", "a = b and c ..." or "a = b or c ..."; members may
    repeat (the engine only warns, and its flattened properties keep repeats)."""
    t = line.split()
    ops = set(t[3::2])
    if len(t) < 3 or len(t) % 2 == 0 or t[1] != "=" or not ops <= {"and", "or"} or len(ops) > 1:
        raise FormatError(f"bad legend line {line.strip()!r}")
    return t[0], ops.pop() if ops else "=", t[2::2]


def _legend_order(entries) -> list:
    """Legend entries in canonical emit order (canonicalize.from_engine): background and
    player (in text order), then l1, l2, ..., each after the legend names it uses."""
    defs = {e[0]: e for e in entries}
    out, done = [], set()
    for root in sorted(defs, key=lambda n: (n not in SPECIAL, _num(n))):
        stack = [] if root in done else [(root, iter(defs[root][2]))]
        while stack:  # depth-first, each entry after its members
            name, members = stack[-1]
            dep = next((m for m in members if m in defs and m not in done), None)
            if dep is None:
                stack.pop()
                done.add(name)
                out.append(defs[name])
            elif any(dep == n for n, _ in stack):
                raise FormatError(f"legend cycle through {dep!r}")
            else:
                stack.append((dep, iter(defs[dep][2])))
    return out


def parse(text: str) -> Canon:
    """The canonical game of a level-first text (level-first names, one level). Raises
    FormatError when the text is malformed."""
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    marks = [i for i, line in enumerate(lines) if line in HEADERS + FOREIGN_HEADERS]
    found = [lines[i] for i in marks]
    if found != list(HEADERS):
        raise FormatError(f"section lines {found}, expected {list(HEADERS)}")
    if _nonblank(lines[:marks[0]]):
        raise FormatError("text before the LEVEL line")
    body = {h: lines[a + 1:b] for h, a, b in zip(HEADERS, marks, marks[1:] + [len(lines)])}

    blocks = _blocks(body["LEVEL"])
    if len(blocks) != 2:
        raise FormatError(f"LEVEL has {len(blocks)} blocks of lines; expected the grid, a blank line, its legend")
    grid, prompt_legend = blocks
    if len({len(row) for row in grid}) != 1:
        raise FormatError("grid rows differ in length")
    glyphs, pre_defs = {}, []
    for line in prompt_legend:
        name, op, members = _legend_entry(line)
        if len(name) == 1:
            if name not in "." + GLYPHS:
                raise FormatError(f"{name!r} is not a level glyph")
            if op == "or":
                raise FormatError(f"glyph {name!r} is defined with 'or'")
            if name in glyphs:
                raise FormatError(f"glyph {name!r} is defined twice")
            if len(set(members)) < len(members):
                raise FormatError(f"glyph {name!r} lists an object twice")
            glyphs[name] = members
        elif PROMPT_NAME_RE.fullmatch(name):
            pre_defs.append((name, op, members))
        else:
            raise FormatError(f"{name!r} cannot be defined in the prompt")

    prelude = {}
    for line in _nonblank(body[DELIM]):
        key, *value = line.split()
        if key not in KEEP_PRELUDE:
            raise FormatError(f"prelude flag {key!r} is not a canonical one")
        if key in prelude:
            raise FormatError(f"prelude flag {key!r} is given twice")
        prelude[key] = " ".join(value)

    objects = []
    for block in _blocks(body["OBJECTS"]):
        if len(block) != 2:
            raise FormatError(f"object entry {block[0].strip()!r}: expected a name line and a colour line")
        name, color = (line.strip() for line in block)
        if not OBJECT_RE.fullmatch(name):
            raise FormatError(f"{name!r} is not an object name")
        if color != _color(name):
            raise FormatError(f"object {name} has colour {color!r}, not {_color(name)}")
        objects.append(name)

    mech_defs = []
    for line in _nonblank(body["LEGEND"]):
        name, op, members = _legend_entry(line)
        if not MECH_NAME_RE.fullmatch(name):
            raise FormatError(f"{name!r} is defined in the mechanics legend (glyphs and the background and "
                              f"player definitions belong in the prompt)")
        mech_defs.append((name, op, members))

    names = objects + [name for name, _, _ in pre_defs + mech_defs]
    twice = sorted(n for n, k in Counter(names).items() if k > 1)
    if twice:
        raise FormatError(f"defined more than once: {twice}")
    for n in SPECIAL:
        if n not in names:
            raise FormatError(f"{n} is not defined")
    defined, objset = set(names), set(objects)
    for name, _, members in pre_defs + mech_defs:
        for m in members:
            if m not in defined:
                raise FormatError(f"legend entry {name} uses {m!r}, which is not defined")
    pre = _closure(pre_defs)
    for name, _, members in pre_defs:
        if name not in pre:
            raise FormatError(f"the prompt defines {name}, which background and player do not use")
        for m in members:
            if m in defined - objset and m not in pre:
                raise FormatError(f"prompt definition {name} uses {m}, which only the mechanics define")
    for ch, members in glyphs.items():
        for m in members:
            if m not in objset:
                raise FormatError(f"glyph {ch!r} lists {m!r}, which is not an object")
    for row in grid:
        for ch in row:
            if ch not in glyphs:
                raise FormatError(f"grid glyph {ch!r} is not defined")

    layers = []
    for line in _nonblank(body["COLLISIONLAYERS"]):
        layer = [n.strip() for n in line.split(",")]
        if not all(len(n.split()) == 1 for n in layer):
            raise FormatError(f"bad collision layer {line.strip()!r}")
        layers.append(layer)
    return Canon(prelude=[(k, prelude[k]) for k in KEEP_PRELUDE if k in prelude],
                 objects=sorted(objects, key=_obj_key), legend=_legend_order(pre_defs + mech_defs),
                 layers=layers, rules=[[line] for line in _nonblank(body["RULES"])],
                 wins=[line.split() for line in _nonblank(body["WINCONDITIONS"])],
                 levels=[[[frozenset(glyphs[ch]) for ch in row] for row in grid]], rename={})


def to_standard(text: str) -> str:
    """Valid PuzzleScript in canonical style for a level-first text (one level); raises
    FormatError when the text is malformed."""
    return render(parse(text))
