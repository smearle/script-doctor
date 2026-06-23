"""Tokenize an Autumn DSL program into a domain-general integer sequence.

Parallel to the PuzzleScript tokenizer (`nca_wm/tokenize_game.py`): turn a game
*program* into a flat, name-invariant token sequence suitable for a transformer
"rule encoder" (the Perceiver-style `RuleSlotEncoder` / FiLM `GameSpecEncoder`).

Autumn programs are S-expressions (`.sexp`) implementing a functional-reactive
language: object declarations, temporal state (`initnext` / `prev`), event
handlers (`on`), lambdas (`-->`), and list combinators. Unlike PuzzleScript's
spatial rewrite rules, the structure here is a genuine AST, so we linearize the
tree directly with explicit `OPEN`/`CLOSE` nesting tokens plus typed leaf tokens.

Name-invariance (same trick as the PS tokenizer's `ch0`/`g0` scheme): user
identifiers carry no information in their spelling, so they are replaced by
first-seen indices within three per-program palettes —
  * object-type names      -> OBJ0, OBJ1, ...   (heads of `(object NAME ...)`)
  * other identifiers       -> VAR0, VAR1, ...   (vars, fn names, params, fields)
  * non-color string atoms  -> STR0, STR1, ...
and color string literals -> COLOR0, COLOR1, ... (per-program palette order).
Reserved words, operators, native builtins, stdlib functions, and types are a
closed set drawn from the interpreter source (TokenType.hpp, Interpreter.cpp's
`define(...)` table, and autumnstdlib/stdlib.sexp), so they get dedicated tokens.

Entry point:
    from nca_wm.autumn.tokenize_program import tokenize_program, VOCAB, STOI
    tokens, info = tokenize_program(open("gameOfLife.sexp").read())
    # tokens: list[int]; info: TokenizeInfo with the discovered palettes.

The output is framework-agnostic (plain ints), so it can feed either the PyTorch
Autumn encoder or, once Autumn is ported, the shared JAX/Flax dual encoder. The
index-family layout below is deliberately disjoint from the PS vocab so a future
*union* vocab can concatenate the two without colliding IDs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Closed-set symbol tables (authoritative — from the interpreter source).
# ---------------------------------------------------------------------------

# Special / structural tokens.
_SPECIAL = ["PAD", "OPEN", "CLOSE", "SEP", "UNK"]

# Keywords & special forms (lexer keywords + the symbol heads that act as forms).
# Maps the source atom -> token name.
_FORMS = {
    "program": "KW_PROGRAM",
    "object": "KW_OBJECT",
    "on": "KW_ON",
    "if": "KW_IF",
    "then": "KW_THEN",
    "else": "KW_ELSE",
    "let": "KW_LET",
    "fn": "KW_FN",
    "fun": "KW_FN",
    "initnext": "KW_INITNEXT",
    "true": "KW_TRUE",
    "false": "KW_FALSE",
    "nil": "KW_NIL",
    "list": "KW_LIST",
    "-->": "KW_LAMBDA",
    "=": "KW_ASSIGN",
    ":": "KW_TYPEDECL",
    "..": "KW_GET",
}

# Operators (several source spellings collapse to one semantic token).
_OPS = {
    "==": "OP_EQ", "!=": "OP_NE",
    "<": "OP_LT", "<=": "OP_LE", ">": "OP_GT", ">=": "OP_GE",
    "+": "OP_ADD", "-": "OP_SUB", "*": "OP_MUL", "/": "OP_DIV", "%": "OP_MOD",
    "!": "OP_NOT",
    "and": "OP_AND", "&": "OP_AND", "&&": "OP_AND",
    "or": "OP_OR", "|": "OP_OR", "||": "OP_OR",
}

# Built-in types / constructors (globals->defineType in Interpreter.cpp).
_TYPES = [
    "Bool", "String", "Number", "Int", "Position", "Cell", "RenderedElem",
    "List",
]

# Magic globals.
_GLOBALS = {
    "GRID_SIZE": "GLOB_GRID_SIZE",
    "FRAME_RATE": "GLOB_FRAME_RATE",
    "background": "GLOB_BACKGROUND",
    "click": "GLOB_CLICK",
}

# Structural fields that are universal (Position / Cell / object metadata),
# distinguished from user-declared object fields so the encoder sees them.
_STRUCT_FIELDS = {"origin": "FLD_ORIGIN", "x": "FLD_X", "y": "FLD_Y"}

# Native builtins registered in C++ (Interpreter.cpp `define(...)` + stdlib/*.cpp).
_NATIVE = [
    "map", "concat", "filter", "foldl", "length", "head", "at", "tail",
    "renderAll", "defined", "arrayEqual", "sqrt", "prev", "isList",
    "uniformChoice", "adjPositions", "addObj", "isFreePos", "range", "print",
    "removeObj", "updateObj", "randomPositions", "allPositions", "clicked",
    "left", "right", "up", "down", "isWithinBounds", "allObjs", "rotate",
    "any", "isOutsideBounds",
]

# Stdlib functions defined in autumnstdlib/stdlib.sexp (the Autumn-level library).
_STDLIB = [
    "abs", "adj", "adjacentElem", "adjacentObjs", "adjacentObjsDiag",
    "adjacentPoss", "adjacentPossDiag", "adjacentTwoObjs", "adjacentTwoObjsDiag",
    "allCheckedPos", "closest", "closestHole", "closestPos", "delta",
    "deltaElem", "deltaObj", "deltaPos", "dir", "displacement", "holes", "in",
    "intersects", "intersectsElems", "intersectsPosElems", "intersectsPosPoss",
    "isFree", "isFreeExcept", "isFreePosExceptObj", "isFreeRangeExceptObj",
    "max", "min", "move", "moveCanCollision", "moveDown", "moveDownNoCollision",
    "moveDownPos", "moveLeft", "moveLeftNoCollision", "moveLeftPos",
    "moveNoCollision", "movePos", "moveRight", "moveRightNoCollision",
    "moveRightPos", "moveUp", "moveUpNoCollision", "moveUpPos", "movedObj",
    "nextLiquid", "nextLiquidClosestHole", "nextLiquidMoveClosestHole",
    "nextRowPos", "nextSolid", "objClicked", "rect", "rectPos", "renderValue",
    "rotateNoCollision", "sign", "sign_x", "sign_y", "sqdist", "sum",
    "unitVector", "unitVectorObjPos", "unitVectorSinglePos", "vcat", "wbound",
    "xmax", "xmin", "ymax", "ymin",
]

_BUILTINS = {name: f"BLT_{name}" for name in dict.fromkeys(_NATIVE + _STDLIB)}
_TYPE_TOKS = {name: f"TY_{name}" for name in _TYPES}

# Numeric literal buckets: small ints get dedicated tokens; rest -> NUM_BIG.
_NUM_LO, _NUM_HI = -4, 64

def _num_tok(v: int) -> str:
    return f"NUM_m{-v}" if v < 0 else f"NUM_{v}"

_NUM_TOKS = [_num_tok(v) for v in range(_NUM_LO, _NUM_HI + 1)] + ["NUM_BIG"]

# Name-invariant index families. Caps chosen to cover the AutumnBench corpus
# with headroom; out-of-range falls back to UNK.
MAX_OBJ, MAX_VAR, MAX_COLOR, MAX_STR = 64, 160, 32, 32
_OBJ_TOKS = [f"OBJ{i}" for i in range(MAX_OBJ)]
_VAR_TOKS = [f"VAR{i}" for i in range(MAX_VAR)]
_COLOR_TOKS = [f"COLOR{i}" for i in range(MAX_COLOR)]
_STR_TOKS = [f"STR{i}" for i in range(MAX_STR)]

# ---------------------------------------------------------------------------
# Assemble the vocabulary (order fixes the integer IDs — append only).
# ---------------------------------------------------------------------------
VOCAB: list[str] = (
    _SPECIAL
    + list(_FORMS.values())
    + list(dict.fromkeys(_OPS.values()))
    + list(_TYPE_TOKS.values())
    + list(_GLOBALS.values())
    + list(_STRUCT_FIELDS.values())
    + list(_BUILTINS.values())
    + _NUM_TOKS
    + _OBJ_TOKS + _VAR_TOKS + _COLOR_TOKS + _STR_TOKS
)
STOI = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

PAD, OPEN, CLOSE, SEP, UNK = (STOI[t] for t in ("PAD", "OPEN", "CLOSE", "SEP", "UNK"))

# A pragmatic set of CSS/Autumn color names (distinguishes COLOR vs STR atoms).
KNOWN_COLORS = {
    "black", "white", "gray", "grey", "silver", "red", "green", "blue",
    "yellow", "gold", "orange", "purple", "pink", "brown", "tan", "cyan",
    "magenta", "maroon", "navy", "teal", "olive", "lime", "aqua", "coral",
    "salmon", "khaki", "violet", "indigo", "crimson", "darkblue", "darkgreen",
    "darkorange", "darkred", "lightblue", "lightgreen", "lightpink",
    "lightgray", "lightgrey", "skyblue", "steelblue", "mediumpurple",
    "orangered", "goldenrod", "sandybrown", "darkgray", "darkgrey", "beige",
    "turquoise", "lavender", "plum", "chocolate", "tomato", "firebrick",
    "forestgreen", "seagreen", "royalblue", "slateblue", "hotpink", "deeppink",
    "transparent",
}


# ---------------------------------------------------------------------------
# S-expression reader.
# ---------------------------------------------------------------------------
# Atom representation: a bare atom is a `str`; a string literal is ("str", val).
_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\(|\)|[^()\s]+')


def _strip_comments(text: str) -> str:
    return re.sub(r";[^\n]*", "", text)


def parse_sexp(text: str):
    """Parse Autumn source into a nested list. Returns the top-level program node
    (a list). Raises ValueError on unbalanced parens."""
    toks = _TOKEN_RE.findall(_strip_comments(text))
    pos = 0

    def read():
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("unexpected EOF in sexp")
        t = toks[pos]
        pos += 1
        if t == "(":
            lst = []
            while pos < len(toks) and toks[pos] != ")":
                lst.append(read())
            if pos >= len(toks):
                raise ValueError("unbalanced '(' in sexp")
            pos += 1  # consume ')'
            return lst
        if t == ")":
            raise ValueError("unexpected ')' in sexp")
        if t.startswith('"'):
            return ("str", t[1:-1])
        return t

    node = read()
    return node


# ---------------------------------------------------------------------------
# Tokenizer.
# ---------------------------------------------------------------------------
@dataclass
class TokenizeInfo:
    objs: dict = field(default_factory=dict)     # name -> OBJ index
    vars: dict = field(default_factory=dict)     # name -> VAR index
    colors: dict = field(default_factory=dict)   # color string -> COLOR index
    strs: dict = field(default_factory=dict)     # other string -> STR index
    n_unk: int = 0
    overflow: list = field(default_factory=list)  # families that hit their cap

    @property
    def n_tokens(self) -> int:  # filled in by tokenize_program
        return self._n_tokens

    _n_tokens: int = 0


_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


def _prescan_names(node):
    """Pre-scan the AST for: (a) object-type names declared via
    `(object NAME ...)`, and (b) the set of all bare identifier atoms.

    (a) lets a constructor used before its declaration still map to OBJ.
    (b) lets quoted strings that name a field/var (Autumn passes field keys as
    strings, e.g. `(updateObj obj "living" v)` / `(prev "particles")`) unify
    with the bare identifier of the same name instead of fragmenting into a
    separate STR token — preserving name-invariance.
    """
    obj_names, idents = set(), set()

    def is_bare_ident(a: str) -> bool:
        return not (
            a in _FORMS or a in _OPS or a in _GLOBALS or a in _STRUCT_FIELDS
            or a in _TYPE_TOKS or a in _BUILTINS
            or _INT_RE.match(a) or _FLOAT_RE.match(a)
        )

    def walk(n):
        if isinstance(n, list):
            if len(n) >= 2 and n[0] == "object" and isinstance(n[1], str):
                obj_names.add(n[1])
            for c in n:
                walk(c)
        elif isinstance(n, str) and is_bare_ident(n):
            idents.add(n)

    walk(node)
    return obj_names, idents


def tokenize_program(text: str, color_order=None, max_len: int | None = None):
    """Tokenize Autumn source -> (tokens: list[int], info: TokenizeInfo).

    If `max_len` is given, the sequence is truncated to that length (no padding;
    the caller pads/batches). Object-type names are resolved against a pre-scan
    so a constructor used before its `object` declaration still maps to OBJ.

    `color_order`: optional sequence (or name->index dict) fixing the COLOR
    palette so token `COLORi` aligns with state channel `i`. Pass the game's
    collected npz palette here so a color referenced in a rule points at the
    same multihot channel the state uses (mirrors the PuzzleScript ch0/ch1
    token↔channel correspondence). Any name in `color_order` is treated as a
    color even if it's not in KNOWN_COLORS. When None, colors fall back to
    first-seen per-program indexing.
    """
    node = parse_sexp(text)
    obj_names, ident_names = _prescan_names(node)
    info = TokenizeInfo()
    fixed_colors = None
    if color_order is not None:
        fixed_colors = (color_order if isinstance(color_order, dict)
                        else {c: i for i, c in enumerate(color_order)})
        info.colors = dict(fixed_colors)  # report the aligned palette
    out: list[int] = []

    def family_id(table: dict, key, toks: list[str], fam_name: str) -> int:
        idx = table.get(key)
        if idx is None:
            idx = len(table)
            table[key] = idx
        if idx >= len(toks):
            if fam_name not in info.overflow:
                info.overflow.append(fam_name)
            info.n_unk += 1
            return UNK
        return STOI[toks[idx]]

    def emit_atom(a: str):
        # 1. operators / forms / builtins / types / globals / struct fields.
        for tbl in (_FORMS, _OPS):
            if a in tbl:
                out.append(STOI[tbl[a]]); return
        if a in _GLOBALS:
            out.append(STOI[_GLOBALS[a]]); return
        if a in _STRUCT_FIELDS:
            out.append(STOI[_STRUCT_FIELDS[a]]); return
        if a in _TYPE_TOKS:
            out.append(STOI[_TYPE_TOKS[a]]); return
        if a in _BUILTINS:
            out.append(STOI[_BUILTINS[a]]); return
        # 2. numeric literals.
        if _INT_RE.match(a):
            v = int(a)
            tok = _num_tok(v) if _NUM_LO <= v <= _NUM_HI else "NUM_BIG"
            out.append(STOI.get(tok, STOI["NUM_BIG"])); return
        if _FLOAT_RE.match(a):
            out.append(STOI["NUM_BIG"]); return
        # 3. name-invariant identifiers.
        if a in obj_names:
            out.append(family_id(info.objs, a, _OBJ_TOKS, "OBJ")); return
        out.append(family_id(info.vars, a, _VAR_TOKS, "VAR"))

    def emit_color_fixed(s: str):
        idx = fixed_colors[s]
        if idx >= len(_COLOR_TOKS):
            if "COLOR" not in info.overflow:
                info.overflow.append("COLOR")
            info.n_unk += 1
            out.append(UNK)
        else:
            out.append(STOI[_COLOR_TOKS[idx]])

    def emit_string(s: str):
        # Colors render to the grid -> color palette. A string that names a
        # declared object / bare identifier (Autumn's stringly-typed field &
        # var keys) unifies with that identifier's family. Everything else
        # (mode flags like "none"/"up"/"active") is a genuine string literal.
        if fixed_colors is not None and s in fixed_colors:
            emit_color_fixed(s)
        elif s in KNOWN_COLORS:
            out.append(family_id(info.colors, s, _COLOR_TOKS, "COLOR"))
        elif s in obj_names:
            out.append(family_id(info.objs, s, _OBJ_TOKS, "OBJ"))
        elif s in ident_names:
            out.append(family_id(info.vars, s, _VAR_TOKS, "VAR"))
        else:
            out.append(family_id(info.strs, s, _STR_TOKS, "STR"))

    def walk(n):
        if isinstance(n, list):
            out.append(OPEN)
            for c in n:
                walk(c)
            out.append(CLOSE)
        elif isinstance(n, tuple) and n[0] == "str":
            emit_string(n[1])
        else:
            emit_atom(n)

    walk(node)
    if max_len is not None and len(out) > max_len:
        out = out[:max_len]
    info._n_tokens = len(out)
    return out, info


# ---------------------------------------------------------------------------
# Corpus validation / stats (also the smoke test).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import glob
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--programs",
        default="/home/jupyter-smearle/mara/MARA/resources/autumnbench/programs",
        help="dir of .sexp programs",
    )
    ap.add_argument("--show", default=None, help="print token stream for this game")
    args = ap.parse_args()

    print(f"VOCAB_SIZE = {VOCAB_SIZE}")
    files = sorted(
        f for f in glob.glob(os.path.join(args.programs, "*.sexp"))
        if "wrong_program" not in f
    )
    lens, overflows, n_unk_total, fails = [], {}, 0, []
    used = set()
    for f in files:
        name = os.path.basename(f)[:-5]
        try:
            toks, info = tokenize_program(open(f).read())
        except Exception as e:  # noqa: BLE001
            fails.append((name, repr(e)))
            continue
        lens.append((name, len(toks), len(info.objs), len(info.vars),
                     len(info.colors)))
        used.update(toks)
        n_unk_total += info.n_unk
        for fam in info.overflow:
            overflows[fam] = overflows.get(fam, 0) + 1
        if args.show == name:
            print(f"\n--- {name}: {len(toks)} tokens ---")
            print(" ".join(VOCAB[t] for t in toks[:400]))

    lens.sort(key=lambda r: -r[1])
    print(f"\nparsed {len(lens)}/{len(files)} programs; {len(fails)} failed")
    if fails:
        for n, e in fails:
            print(f"  FAIL {n}: {e}")
    if lens:
        seq = [r[1] for r in lens]
        print(f"seq len: min={min(seq)} med={sorted(seq)[len(seq)//2]} "
              f"max={max(seq)} mean={sum(seq)//len(seq)}")
        print("longest 5:", [(r[0], r[1]) for r in lens[:5]])
        print("max #objs:", max(r[2] for r in lens),
              " max #vars:", max(r[3] for r in lens),
              " max #colors:", max(r[4] for r in lens))
    print(f"vocab coverage: {len(used)}/{VOCAB_SIZE} tokens used")
    print(f"total UNK emissions: {n_unk_total}")
    print(f"family overflows (hit cap): {overflows or 'none'}")
