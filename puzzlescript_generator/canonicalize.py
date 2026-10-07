"""Canonical PuzzleScript: keep the dynamics, drop names and aesthetics.

The input is the reference engine's own parse of a game (ps_extract.js), so the objects,
legend, collision layers and initial levels are exactly what the engine compiled. A
canonical game is still valid PuzzleScript:
- prelude: only flags that change the dynamics, the usable inputs or the view are kept;
- names: `background` and `player` keep their required names; other objects become
  o1, o2, ... and other legend names l1, l2, ..., numbered by first use (rules, then win
  conditions, then collision layers); each object gets one colour by number, no sprite;
- collision layers list objects in the engine's order (an object listed twice keeps its
  last place, as in the compiler), so the relative order of object ids, which the
  compiler assigns in layer order, is unchanged;
- comments, sounds, sfx and message commands (and rules left with nothing to do),
  message levels and metadata such as title and author are dropped;
- rule, layer, win-condition and level order are kept;
- levels are the compiler's initial cells (ragged rows padded, background filled), with
  glyphs assigned in order of first appearance, so glyph choice carries no information.

`mechanics_key` hashes everything but the levels; `level_key` hashes one level's cells.
ps_equiv.js checks that a canonical game behaves like its original.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

KEEP_PRELUDE = ("run_rules_on_level_start", "require_player_movement", "noaction", "noundo", "norestart",
                "realtime_interval", "throttle_movement", "flickscreen", "zoomscreen")
SPECIAL = ("background", "player")
# Inside a rule's cells these take precedence over names (the engine's reg_directions_only).
CELL_KEYWORDS = {">", "<", "^", "v", "up", "down", "left", "right", "moving", "stationary", "no", "randomdir",
                 "random", "horizontal", "vertical", "orthogonal", "perpendicular", "parallel", "action"}
SFX = {f"sfx{i}" for i in range(11)}
# Level glyphs: no "." (reserved for plain background) and nothing that is PuzzleScript
# syntax or a keyword ("v" is a direction; brackets, pipes, arrows, commas, quotes).
GLYPHS = ("abcdefghijklmnopqrstuwxyz0123456789#@*%&$!?~:;/\\{}_"
          + "".join(chr(c) for c in range(0x3b1, 0x3ca))     # greek lowercase
          + "".join(chr(c) for c in range(0xe0, 0xff) if c != 0xf7)  # latin-1 lowercase
          + "".join(chr(c) for c in range(0x430, 0x450)))     # cyrillic lowercase
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
           "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
           "#808000", "#ffd8b1", "#000075", "#808080", "#a9a9a9", "#ff7f50", "#6495ed", "#dc143c"]
BACKGROUND_COLOR, PLAYER_COLOR = "#202020", "#ffffff"


class CanonError(ValueError):
    pass


def tokenize_rule(line: str, names) -> list:
    """Tokenize a rule line as the engine does (compiler.js processRuleString) and return
    [(token, is_name)]. Before the first "[" come modifiers ("+" only as the first token);
    in cells, direction words win over names, and names win over commands. Sound commands
    and a message (with the rest of the line, its text) are dropped."""
    line = line.replace("[", " [ ").replace("]", " ] ").replace("|", " | ").replace("->", " -> ").strip()
    if line.startswith("+"):
        line = "+ " + line[1:]
    out, prefix, rhs = [], True, False
    for t in line.split():
        if prefix and t != "[":
            out.append((t, False))
            continue
        prefix = False
        if t == "->":
            rhs = True
        elif t not in ("[", "]", "|", "...") and t not in CELL_KEYWORDS and t in names:
            out.append((t, True))
            continue
        elif rhs and t == "message":
            break
        elif rhs and t in SFX:
            continue
        out.append((t, False))
    return out


def functional(toks: list) -> bool:
    """False for a rule left with nothing to do (it only showed a message or played a sound)."""
    return "->" not in toks or toks.index("->") < len(toks) - 1


@dataclass
class Canon:
    prelude: list        # [(key, value)]
    objects: list        # canonical object names, emission order
    legend: list         # [(name, op, [names])], op in = and or; glyphs excluded
    layers: list         # [[canonical object names]], engine order
    rules: list          # [[tokens]]
    wins: list           # [[tokens]]
    levels: list         # [[[frozenset of canonical objects per cell] per row] per level]
    rename: dict         # original name -> canonical name


def _prelude(meta: dict) -> list:
    out = []
    for k in KEEP_PRELUDE:
        if k not in meta:
            continue
        v = meta[k]
        if isinstance(v, list):
            v = "x".join(str(int(x)) for x in v)
        elif v is True or str(v).lower() == "true":
            v = ""
        out.append((k, str(v)))
    return out


def from_engine(e: dict) -> Canon:
    """Canonicalize one ok record of ps_extract.js."""
    objects = list(e["objects"])
    objset = set(objects)
    defs = {}
    for name, target in (s[:2] for s in e["synonyms"]):
        defs.setdefault(name, ("=", [target]))
    for name, *members in e["aggregates"]:
        defs.setdefault(name, ("and", members))
    for name, *members in e["properties"]:
        defs.setdefault(name, ("or", members))
    names_all = objset | set(defs)

    # Rules. Dropping a rule that heads a group (the following lines start with "+") makes
    # the next kept "+" rule the head, so groups keep their members.
    rules, head_dropped = [], False
    for line in e["rules"]:
        toks = tokenize_rule(line, names_all)
        if not toks:
            continue
        if not functional([t for t, _ in toks]):
            head_dropped = head_dropped or toks[0][0] != "+"
            continue
        if toks[0][0] == "+" and head_dropped:
            toks = toks[1:]
        head_dropped = False
        rules.append(toks)
    # win conditions: quantifier, name, and optionally "on" and a second name
    wins = [[(t, j in (1, 3) and t in names_all) for j, t in enumerate(w)] for w in e["wins"]]

    # Collision layers: an object listed more than once (repeated, or reached again
    # through a property) ends up at its last position, as in the compiler (which leaves
    # the earlier ids unused), so the relative order of object ids is unchanged.
    last = {o: (li, pos) for li, layer in enumerate(e["layers"]) for pos, o in enumerate(layer)}
    layers = [kept for li, layer in enumerate(e["layers"])
              if (kept := [o for pos, o in enumerate(layer) if last[o] == (li, pos)])]

    # Canonical numbering by first use: rules, win conditions, collision layers (which hold
    # every object), then legend names reached through other legend names, then the
    # required names. Glyph-only legend names are not kept: levels are re-encoded below.
    order = []
    for toks in rules + wins:
        order += [t for t, is_name in toks if is_name]
    for layer in layers:
        order += layer
    order += [n for n in SPECIAL if n in names_all]
    i = 0
    while i < len(order):
        if order[i] in defs and order[i] not in objset:
            order += [n for n in defs[order[i]][1] if n not in order]
        i += 1
    rename, n_obj, n_leg = {}, 0, 0
    for name in dict.fromkeys(order):
        if name in SPECIAL:
            rename[name] = name
        elif name in objset:
            n_obj += 1
            rename[name] = f"o{n_obj}"
        else:
            n_leg += 1
            rename[name] = f"l{n_leg}"
    missing = objset - set(rename)
    if missing:
        raise CanonError(f"objects outside every collision layer: {sorted(missing)[:3]}")

    entries = {}
    for name, (op, members) in defs.items():
        if name in rename and name not in objset:
            if any(m not in rename for m in members):
                raise CanonError(f"legend {name!r} refers to an undefined name")
            entries[rename[name]] = (op, [rename[m] for m in members])
    canon_layers = []
    for layer in layers:
        names = [rename[o] for o in layer]
        if "background" in names and len(names) > 1:
            # a layer line naming `background` must hold nothing else; reach the layer's
            # objects through a property instead, as the original must have
            n_leg += 1
            entries[f"l{n_leg}"] = ("or", names)
            names = [f"l{n_leg}"]
        canon_layers.append(names)
    legend, done = [], set()

    def emit(name, stack=()):
        if name in done or name not in entries:
            return
        if name in stack:
            raise CanonError("legend cycle")
        for dep in entries[name][1]:
            emit(dep, stack + (name,))
        done.add(name)
        legend.append((name, *entries[name]))

    for name in sorted(entries, key=lambda n: (n not in SPECIAL, _num(n))):
        emit(name)

    levels = []
    for lv in e["levels"]:
        sets = [frozenset(rename[objects[k]] for k in ids) for ids in lv["sets"]]
        if any(not s for s in sets):
            raise CanonError("empty cell in a compiled level")
        w = lv["w"]
        levels.append([[sets[k] for k in lv["grid"][r * w:(r + 1) * w]] for r in range(lv["h"])])

    def sub(toks):
        return [rename[t] if is_name else t for t, is_name in toks]

    canon_objects = sorted({rename[o] for o in objects},
                           key=lambda o: (o not in SPECIAL, SPECIAL.index(o) if o in SPECIAL else _num(o)))
    return Canon(prelude=_prelude(e.get("metadata") or {}), objects=canon_objects, legend=legend,
                 layers=canon_layers,
                 rules=[sub(r) for r in rules], wins=[sub(w) for w in wins],
                 levels=levels, rename=rename)


def _num(name):
    m = re.search(r"\d+$", name)
    return int(m.group()) if m else -1


def _rule_text(toks):
    return re.sub(r"\s+", " ", " ".join(toks)).strip()


def mechanics_text(c: Canon) -> str:
    out = [f"{k} {v}".strip() for k, v in c.prelude]
    out.append("\nOBJECTS\n")
    for o in c.objects:
        color = BACKGROUND_COLOR if o == "background" else PLAYER_COLOR if o == "player" else \
            PALETTE[(_num(o) - 1) % len(PALETTE)]
        out.append(f"{o}\n{color}\n")
    out.append("LEGEND\n")
    for name, op, names in c.legend:
        out.append(f"{name} = " + f" {op} ".join(names) if op != "=" else f"{name} = {names[0]}")
    out.append("\nSOUNDS\n")  # the engine requires every section, in order
    out.append("COLLISIONLAYERS\n")
    out += [", ".join(l) for l in c.layers]
    out.append("\nRULES\n")
    out += [_rule_text(r) for r in c.rules]
    out.append("\nWINCONDITIONS\n")
    out += [" ".join(w) for w in c.wins]
    return "\n".join(out) + "\n"


def render(c: Canon, levels=None) -> str:
    """Full canonical source; level glyphs assigned by first appearance in `levels`."""
    levels = c.levels if levels is None else levels
    glyph, gi = {}, 0
    for lv in levels:
        for row in lv:
            for cell in row:
                if cell not in glyph:
                    if cell == frozenset(["background"]):
                        glyph[cell] = "."
                    else:
                        if gi >= len(GLYPHS):
                            raise CanonError("too many distinct cells")
                        glyph[cell] = GLYPHS[gi]
                        gi += 1
    mech = mechanics_text(c)
    glines = [f"{ch} = " + " and ".join(sorted(cell, key=lambda o: (o not in SPECIAL, SPECIAL.index(o)
                                                                    if o in SPECIAL else _num(o))))
              for cell, ch in glyph.items()]
    mech = mech.replace("\nSOUNDS\n", "\n" + "\n".join(glines) + "\n\nSOUNDS\n", 1)
    lv_text = "\n\n".join("\n".join("".join(glyph[cell] for cell in row) for row in lv) for lv in levels)
    return mech + "\nLEVELS\n\n" + lv_text + "\n"


def mechanics_key(c: Canon) -> str:
    return hashlib.sha1(mechanics_text(c).encode()).hexdigest()[:20]


def level_key(level) -> str:
    s = "\n".join("|".join(",".join(sorted(cell)) for cell in row) for row in level)
    return hashlib.sha1(s.encode()).hexdigest()[:20]
