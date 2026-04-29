"""Detokenize a token sequence (from tokenize_game) back to PuzzleScript source.

This is the inverse of `tokenize_game.tokenize_game`. The mapping is mostly
deterministic, with the following caveats — token vocab is intentionally
abstract and lossy, so a perfect round-trip on the original .txt is *not*
expected:

  * Object names are not preserved (vocab uses generic CH<i>). We emit
    `Obj<i>` for each channel index seen in the stream.
  * Legend key strings (`@`, `#`, etc.) are not preserved. We assign generic
    single-letter keys `a, b, c, ...` to each object, and `A, B, C, ...` to
    each OR/AND group; this lets us emit a syntactically valid LEVELS
    section if `generate_placeholder_level=True`.
  * Title / author / homepage are not in the vocab. We emit a synthetic
    title and omit author.
  * SOUNDS, collision flags, and any rule constructs outside the vocab are
    dropped — the SOUNDS section is emitted empty.
  * LEVELS are not in the vocab. We emit either an empty LEVELS section or
    a tiny 3x3 placeholder that uses the first object as the background.

Round-trip guarantee: for any game G whose tree was produced by the
puzzlescript_jax lark parser, `tokenize_game(G)` -> `detokenize` ->
re-parse -> re-tokenize should produce the *same* token sequence
(modulo the lossy fields above), so long as the original game uses only
constructs covered by the token vocab.
"""
from __future__ import annotations

from nca_wm.tokenize_game import (
    INV_VOCAB, VOCAB,
    color_q_to_rgb, COLOR_Q_TOTAL,
)


# Inverse-of-MOD_MAP: token name -> rule-cell modifier symbol/word.
_INV_MOD = {
    "MOD_PUSH":       ">",
    "MOD_UP":         "^",
    "MOD_DOWN":       "v",
    "MOD_LEFT":       "<",
    "MOD_NO":         "no",
    "MOD_RANDOM":     "random",
    "MOD_RANDOMDIR":  "randomdir",
    "MOD_STATIONARY": "stationary",
    # The following tokens double as rule-level prefixes (when emitted
    # *before* `LHS`) and as within-cell modifiers (when emitted *inside*
    # a kernel — `[ perpendicular Player | ... ]`). Listing them here
    # makes them re-emit cleanly as cell modifiers; the prefix path above
    # handles the other case based on token position.
    "DIR_HORIZONTAL":    "horizontal",
    "DIR_VERTICAL":      "vertical",
    "DIR_PERPENDICULAR": "perpendicular",
    "DIR_PARALLEL":      "parallel",
}

# Rule-level prefixes (apply to whole rule).
_INV_PREFIX = {
    "DIR_UP":            "up",
    "DIR_DOWN":          "down",
    "DIR_LEFT":          "left",
    "DIR_RIGHT":         "right",
    "DIR_HORIZONTAL":    "horizontal",
    "DIR_VERTICAL":      "vertical",
    "DIR_PERPENDICULAR": "perpendicular",
    "DIR_PARALLEL":      "parallel",
    "LATE":              "late",
    "RANDOM":            "random",
    "RIGID":             "rigid",
}

# Win-condition quantifiers.
_INV_WC = {
    "WC_ALL":  "all",
    "WC_SOME": "some",
    "WC_NO":   "no",
    "WC_ANY":  "any",
}

# Rule commands.
_INV_CMD = {
    "CMD_AGAIN":      "again",
    "CMD_CANCEL":     "cancel",
    "CMD_CHECKPOINT": "checkpoint",
    "CMD_RESTART":    "restart",
    "CMD_WIN":        "win",
}


def _ch_idx(name: str) -> int | None:
    if name.startswith("CH"):
        try:
            return int(name[2:])
        except ValueError:
            return None
    return None


def _g_idx(name: str) -> int | None:
    if name.startswith("G") and len(name) > 1:
        try:
            return int(name[1:])
        except ValueError:
            return None
    return None


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _obj_name(ch: int) -> str:
    return f"Obj{ch}"


def _grp_name(g: int) -> str:
    return f"Grp{g}"


# Single-char legend keys, distinct per object/group.
# Objects get 'a'..'z' (CH0..CH25), then 'A'..'Z' (CH26..CH51), digits...
# Groups use a parallel alphabet starting at the *end* of the object range
# to minimize collision risk.
def _legend_key_for_obj(ch: int) -> str:
    pool = "abcdefghijklmnopqrstuvwxyz0123456789"
    return pool[ch % len(pool)]


def _legend_key_for_grp(g: int) -> str:
    pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return pool[g % len(pool)]


def _split_sections(tokens: list[int]) -> list[list[int]]:
    """Split a token sequence on SEP into per-section chunks."""
    sep = VOCAB["SEP"]
    out: list[list[int]] = []
    cur: list[int] = []
    for t in tokens:
        if t == sep:
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def _classify_section(toks: list[int]) -> str:
    """Identify which section type a SEP-bounded chunk represents.

    LOOP_START / LOOP_END are bare tokens that aren't SEP-bounded, so they
    naturally cluster at the head of whichever chunk they precede. Skip
    past them and classify by the first content token.
    """
    if not toks:
        return "empty"
    j = 0
    while j < len(toks) and INV_VOCAB.get(toks[j], "") in {"LOOP_START",
                                                            "LOOP_END"}:
        j += 1
    if j == len(toks):
        return "loop_marker"
    head_name = INV_VOCAB.get(toks[j], "")
    if head_name in {"PRE_NOACTION", "PRE_REQUIRE_PLAYER_MOVEMENT",
                     "PRE_RUN_RULES_ON_LEVEL_START"}:
        return "prelude"
    if head_name == "OBJ_START":
        return "sprites"
    if head_name == "LAYER":
        return "layer"
    if head_name in {"GROUP_OR", "GROUP_AND"}:
        return "group"
    if head_name == "RULE":
        return "rule"
    if head_name == "WIN":
        return "win"
    return "unknown"


# ---------------------------------------------------------------------------
# Per-section parsers — each returns a structured intermediate dict.
# ---------------------------------------------------------------------------

def _parse_prelude(toks: list[int]) -> dict:
    flags = {INV_VOCAB.get(t) for t in toks}
    return {
        "noaction":                    "PRE_NOACTION" in flags,
        "require_player_movement":     "PRE_REQUIRE_PLAYER_MOVEMENT" in flags,
        "run_rules_on_level_start":    "PRE_RUN_RULES_ON_LEVEL_START" in flags,
    }


def _parse_sprites(toks: list[int]) -> list[dict]:
    """Parse a sprites section into a list of {"ch": int, "palette": [...], "sprite": [[...]]} dicts."""
    objs: list[dict] = []
    i = 0
    while i < len(toks):
        if INV_VOCAB.get(toks[i]) != "OBJ_START":
            i += 1
            continue
        i += 1
        # Channel index
        ch = None
        if i < len(toks):
            ch = _ch_idx(INV_VOCAB.get(toks[i], ""))
            if ch is not None:
                i += 1
        # Palette
        palette: list[str | None] = []
        if i < len(toks) and INV_VOCAB.get(toks[i]) == "PALETTE_START":
            i += 1
            while i < len(toks):
                name = INV_VOCAB.get(toks[i], "")
                if name == "SPRITE_START" or name == "OBJ_END":
                    break
                if name.startswith("COLOR_Q"):
                    bucket = int(name[len("COLOR_Q"):])
                    palette.append(_rgb_to_hex(color_q_to_rgb(bucket)))
                elif name == "PIX_TRANSPARENT":
                    palette.append(None)  # transparent palette entry
                i += 1
        # Sprite grid
        sprite: list[list[str]] = []
        cur_row: list[str] = []
        if i < len(toks) and INV_VOCAB.get(toks[i]) == "SPRITE_START":
            i += 1
            while i < len(toks):
                name = INV_VOCAB.get(toks[i], "")
                if name == "OBJ_END":
                    break
                if name == "PIXEL_ROW_SEP":
                    sprite.append(cur_row)
                    cur_row = []
                elif name == "PIX_TRANSPARENT":
                    cur_row.append(".")
                elif name.startswith("PIX") and name[3:].isdigit():
                    cur_row.append(name[3:])
                # ignore any stray tokens
                i += 1
            if cur_row:
                sprite.append(cur_row)
        # Skip OBJ_END
        if i < len(toks) and INV_VOCAB.get(toks[i]) == "OBJ_END":
            i += 1
        if ch is not None:
            objs.append({"ch": ch, "palette": palette, "sprite": sprite})
    return objs


def _parse_layer(toks: list[int]) -> list[tuple[str, int]]:
    """Return [(kind, idx)] where kind is 'ch' or 'g'."""
    out: list[tuple[str, int]] = []
    for t in toks[1:]:  # skip the LAYER head token
        name = INV_VOCAB.get(t, "")
        ch = _ch_idx(name)
        if ch is not None:
            out.append(("ch", ch))
            continue
        g = _g_idx(name)
        if g is not None:
            out.append(("g", g))
    return out


def _parse_group(toks: list[int]) -> dict | None:
    head = INV_VOCAB.get(toks[0], "")
    op = "or" if head == "GROUP_OR" else ("and" if head == "GROUP_AND" else None)
    if op is None or len(toks) < 2:
        return None
    g = _g_idx(INV_VOCAB.get(toks[1], ""))
    if g is None:
        return None
    members: list[tuple[str, int]] = []
    for t in toks[2:]:
        name = INV_VOCAB.get(t, "")
        ch = _ch_idx(name)
        if ch is not None:
            members.append(("ch", ch))
            continue
        gi = _g_idx(name)
        if gi is not None:
            members.append(("g", gi))
    return {"op": op, "g": g, "members": members}


def _parse_rule(toks: list[int]) -> dict | None:
    """Parse one RULE-headed section into dict(prefixes, lhs_kernels, rhs_kernels, command).

    `lhs_kernels` / `rhs_kernels` are lists of kernels, each kernel being a
    list of cells, each cell a list of token-strings. Multi-kernel rules
    emit multiple kernels per side (split on KERNEL_SEP); single-kernel
    rules emit one kernel on each side.

    Tolerates leading LOOP_START / LOOP_END markers (they're tracked via
    the second-pass loop-bracketing scanner in `detokenize`).
    """
    if not toks:
        return None
    i = 0
    while i < len(toks) and INV_VOCAB.get(toks[i], "") in {"LOOP_START",
                                                            "LOOP_END"}:
        i += 1
    if i >= len(toks) or INV_VOCAB.get(toks[i]) != "RULE":
        return None
    i += 1
    prefixes: list[str] = []
    while i < len(toks):
        name = INV_VOCAB.get(toks[i], "")
        if name in _INV_PREFIX:
            prefixes.append(_INV_PREFIX[name])
            i += 1
        else:
            break
    # LHS marker
    if i < len(toks) and INV_VOCAB.get(toks[i]) == "LHS":
        i += 1
    lhs_kernels, i = _parse_kernels_until(toks, i, stop={"RHS"})
    if i < len(toks) and INV_VOCAB.get(toks[i]) == "RHS":
        i += 1
    rhs_kernels, i = _parse_kernels_until(
        toks, i, stop=set(_INV_CMD.keys()))
    command = None
    if i < len(toks):
        name = INV_VOCAB.get(toks[i], "")
        if name in _INV_CMD:
            command = _INV_CMD[name]
    return {"prefixes": prefixes, "lhs": lhs_kernels,
            "rhs": rhs_kernels, "command": command}


def _parse_kernels_until(toks: list[int], i: int, stop: set[str]):
    """Parse one rule side as a list of kernels (each kernel = list of cells).

    KERNEL_SEP starts a new kernel; CELL_SEP starts a new cell within the
    current kernel. Stops when a token whose name is in `stop` is reached.
    Returns (kernels, new_i).

    Empty kernels explicitly bounded by KERNEL_SEP are preserved (so
    `[A] [ ]` round-trips through tokenize_game). A fully-empty stream
    (no content tokens, no KERNEL_SEP) collapses to `[]`.
    """
    kernels: list[list[list[str]]] = []
    cur: list[list[str]] | None = None  # current kernel; lazily allocated

    def _ensure_cur():
        nonlocal cur
        if cur is None:
            cur = [[]]
        return cur

    while i < len(toks):
        name = INV_VOCAB.get(toks[i], "")
        if name in stop:
            break
        if name == "KERNEL_SEP":
            # Close current kernel (allocate empty if we never started one)
            # and explicitly start a new one.
            kernels.append(cur if cur is not None else [[]])
            cur = [[]]
            i += 1
            continue
        if name == "CELL_SEP":
            _ensure_cur().append([])
            i += 1
            continue
        if name == "ELLIPSIS":
            _ensure_cur()[-1].append("...")
        elif name in _INV_MOD:
            _ensure_cur()[-1].append(_INV_MOD[name])
        else:
            ch = _ch_idx(name)
            if ch is not None:
                _ensure_cur()[-1].append(_obj_name(ch))
            else:
                g = _g_idx(name)
                if g is not None:
                    _ensure_cur()[-1].append(_grp_name(g))
        i += 1

    if cur is not None:
        kernels.append(cur)
    return kernels, i


def _parse_win(toks: list[int]) -> dict | None:
    if not toks or INV_VOCAB.get(toks[0]) != "WIN":
        return None
    i = 1
    quant = "all"
    if i < len(toks):
        name = INV_VOCAB.get(toks[i], "")
        if name in _INV_WC:
            quant = _INV_WC[name]
            i += 1
    src = None
    trg = None
    while i < len(toks):
        name = INV_VOCAB.get(toks[i], "")
        if name == "WC_ON":
            i += 1
            continue
        ch = _ch_idx(name)
        if ch is not None:
            tag = _obj_name(ch)
        else:
            g = _g_idx(name)
            tag = _grp_name(g) if g is not None else None
        if tag is None:
            i += 1
            continue
        if src is None:
            src = tag
        else:
            trg = tag
        i += 1
    return {"quant": quant, "src": src, "trg": trg}


# ---------------------------------------------------------------------------
# Source emission
# ---------------------------------------------------------------------------

def _emit_objects(sprites: list[dict], all_obj_chs: list[int]) -> str:
    """Emit OBJECTS section. Includes any object referenced anywhere, even if
    we have no sprite block for it (uses a stub palette+grid)."""
    lines: list[str] = []
    by_ch = {o["ch"]: o for o in sprites}
    for ch in all_obj_chs:
        name = _obj_name(ch)
        key = _legend_key_for_obj(ch)
        lines.append(f"{name} {key}")
        obj = by_ch.get(ch)
        if obj is None or not obj["palette"]:
            # Stub: opaque mid-gray palette, blank 5x5 transparent sprite.
            lines.append("#808080")
            for _ in range(5):
                lines.append(".....")
        else:
            # Hex palette (skip None entries which represent transparent palette
            # slots — shouldn't normally appear, but be robust).
            palette_strs = [c for c in obj["palette"] if c is not None]
            if not palette_strs:
                palette_strs = ["#808080"]
            lines.append(" ".join(palette_strs))
            grid = obj["sprite"]
            # Pad/clip to 5x5 so the lark grammar accepts it.
            grid = [list(row) for row in grid][:5]
            while len(grid) < 5:
                grid.append(["."] * 5)
            for r in range(5):
                row = grid[r][:5]
                while len(row) < 5:
                    row.append(".")
                # Replace any digit beyond palette length with '.' so the
                # PuzzleScript renderer doesn't blow up.
                clean: list[str] = []
                for cell in row:
                    if cell == ".":
                        clean.append(".")
                    elif cell.isdigit() and 0 <= int(cell) < len(palette_strs):
                        clean.append(cell)
                    else:
                        clean.append(".")
                lines.append("".join(clean))
        lines.append("")
    return "\n".join(lines)


def _emit_legend(all_obj_chs: list[int], groups: list[dict]) -> str:
    lines: list[str] = []
    # Object aliases — purely for level use; rules use the long name directly.
    for ch in all_obj_chs:
        lines.append(f"{_legend_key_for_obj(ch)} = {_obj_name(ch)}")
    # OR/AND groups
    for grp in groups:
        members = []
        for kind, idx in grp["members"]:
            members.append(_obj_name(idx) if kind == "ch" else _grp_name(idx))
        if not members:
            continue
        # Group symbol (single char) maps to first member; the GroupName itself
        # is then assigned via member-list using OR/AND.
        # PuzzleScript syntax: `Name = a or b or c`
        lines.append(
            f"{_grp_name(grp['g'])} = " +
            f" {grp['op']} ".join(members)
        )
    return "\n".join(lines)


def _emit_collision_layers(layers: list[list[tuple[str, int]]]) -> str:
    out: list[str] = []
    for layer in layers:
        names = []
        for kind, idx in layer:
            names.append(_obj_name(idx) if kind == "ch" else _grp_name(idx))
        if names:
            out.append(", ".join(names))
    return "\n".join(out)


def _emit_rule_kernel(cells: list[list[str]]) -> str:
    """Render [ cell1 | cell2 | ... ] from a list-of-cell-token-lists."""
    if not cells:
        return "[ ]"
    parts = [" ".join(c) if c else "" for c in cells]
    return "[ " + " | ".join(parts) + " ]"


def _emit_rule_side(kernels: list[list[list[str]]]) -> str:
    """Render `[ ... ] [ ... ] ...` for a side made of one or more kernels."""
    if not kernels:
        return "[ ]"
    return " ".join(_emit_rule_kernel(k) for k in kernels)


def _emit_rules(rules: list[dict],
                loop_starts: set[int], loop_ends: set[int]) -> str:
    out: list[str] = []
    for i, r in enumerate(rules):
        if i in loop_starts:
            out.append("startloop")
        prefix_str = " ".join(r["prefixes"])
        lhs = _emit_rule_side(r["lhs"])
        rhs = _emit_rule_side(r["rhs"])
        line = (f"{prefix_str} " if prefix_str else "") + f"{lhs} -> {rhs}"
        if r["command"]:
            line += f" {r['command']}"
        out.append(line)
        if i in loop_ends:
            out.append("endloop")
    return "\n".join(out)


def _emit_winconditions(wins: list[dict]) -> str:
    out: list[str] = []
    for w in wins:
        if not w or w.get("src") is None:
            continue
        s = f"{w['quant']} {w['src']}"
        if w.get("trg"):
            s += f" on {w['trg']}"
        out.append(s)
    return "\n".join(out)


def _emit_levels_placeholder(all_obj_chs: list[int]) -> str:
    """Tiny 3x3 level using the first object as background, second (if any) as
    a single sentinel. Just so the LEVELS section is non-empty for engines
    that require it; the puzzlescript_jax grammar accepts empty too."""
    if not all_obj_chs:
        return ""
    bg = _legend_key_for_obj(all_obj_chs[0])
    rows = [bg * 3 for _ in range(3)]
    if len(all_obj_chs) >= 2:
        sentinel = _legend_key_for_obj(all_obj_chs[1])
        rows[1] = bg + sentinel + bg
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detokenize(
    token_ids: list[int],
    title: str = "Decoded Game",
    generate_placeholder_level: bool = True,
) -> str:
    """Convert a token-id sequence into a PuzzleScript source string.

    The output is intended to be parseable by the puzzlescript_jax lark
    grammar; it is not guaranteed to be playable (e.g. levels are
    synthetic, sprite palettes are quantized).
    """
    # Strip BOS/PAD if present.
    pad = VOCAB["PAD"]
    while token_ids and token_ids[0] == pad:
        token_ids = token_ids[1:]
    while token_ids and token_ids[-1] == pad:
        token_ids = token_ids[:-1]

    chunks = _split_sections(token_ids)

    prelude = {"noaction": False,
               "require_player_movement": False,
               "run_rules_on_level_start": False}
    sprites: list[dict] = []
    layers: list[list[tuple[str, int]]] = []
    groups: list[dict] = []
    rules_flat: list[dict] = []
    loop_starts: set[int] = set()
    loop_ends: set[int] = set()
    wins: list[dict] = []

    # First pass for prelude / sprites / layers / groups / wins.
    rule_chunks: list[list[int]] = []
    in_loop_depth = 0
    for ch in chunks:
        kind = _classify_section(ch)
        if kind == "prelude":
            prelude.update(_parse_prelude(ch))
        elif kind == "sprites":
            sprites.extend(_parse_sprites(ch))
        elif kind == "layer":
            layers.append(_parse_layer(ch))
        elif kind == "group":
            g = _parse_group(ch)
            if g is not None:
                groups.append(g)
        elif kind == "rule":
            # Detect leading loop markers (LOOP_START / LOOP_END can appear
            # adjacent to a rule chunk, but the tokenizer emits them as their
            # own tokens not separated by SEP — so they'd land at the head of
            # this chunk if SEP doesn't precede them. Be tolerant.).
            rule_chunks.append(ch)
        elif kind == "win":
            w = _parse_win(ch)
            if w is not None:
                wins.append(w)
        # 'loop_marker' / 'unknown' / 'empty' are handled below.

    # Second pass for loop markers — they may appear as bare tokens between
    # rule SEP boundaries. Scan the raw token stream to assign loop start/end
    # to surrounding rule indices.
    rule_idx = 0
    in_loop = False
    loop_open_at: int | None = None
    for t in token_ids:
        name = INV_VOCAB.get(t, "")
        if name == "LOOP_START":
            in_loop = True
            loop_open_at = rule_idx
        elif name == "LOOP_END":
            in_loop = False
            if loop_open_at is not None and rule_idx > loop_open_at:
                loop_starts.add(loop_open_at)
                loop_ends.add(rule_idx - 1)
            loop_open_at = None
        elif name == "RULE":
            rule_idx += 1

    # Parse rules into structured dicts.
    rules_struct: list[dict] = []
    for rc in rule_chunks:
        r = _parse_rule(rc)
        if r is not None:
            rules_struct.append(r)

    # Collect ALL object channels referenced anywhere so the OBJECTS section
    # is consistent with whatever the rules / layers / wincons reference.
    all_chs: set[int] = set(o["ch"] for o in sprites)
    for layer in layers:
        for kind, idx in layer:
            if kind == "ch":
                all_chs.add(idx)
    for grp in groups:
        for kind, idx in grp["members"]:
            if kind == "ch":
                all_chs.add(idx)
    def _walk_kernels(kernels):
        for kernel in kernels:
            for cell in kernel:
                for tok in cell:
                    if tok.startswith("Obj"):
                        try:
                            all_chs.add(int(tok[3:]))
                        except ValueError:
                            pass
    for r in rules_struct:
        _walk_kernels(r["lhs"])
        _walk_kernels(r["rhs"])
    for w in wins:
        for tag in (w["src"], w["trg"]):
            if tag and tag.startswith("Obj"):
                try:
                    all_chs.add(int(tag[3:]))
                except ValueError:
                    pass
    obj_chs = sorted(all_chs)

    # ---- Emit final source string ---------------------------------------
    parts: list[str] = []
    # Prelude
    parts.append(f"title {title}")
    parts.append("author NCA-WM")
    parts.append("homepage script-doctor")
    if prelude["noaction"]:
        parts.append("noaction")
    if prelude["require_player_movement"]:
        parts.append("require_player_movement")
    if prelude["run_rules_on_level_start"]:
        parts.append("run_rules_on_level_start")
    parts.append("")

    parts.append("========")
    parts.append("OBJECTS")
    parts.append("========")
    parts.append("")
    parts.append(_emit_objects(sprites, obj_chs))

    parts.append("========")
    parts.append("LEGEND")
    parts.append("========")
    parts.append("")
    parts.append(_emit_legend(obj_chs, groups))
    parts.append("")

    parts.append("========")
    parts.append("SOUNDS")
    parts.append("========")
    parts.append("")

    parts.append("================")
    parts.append("COLLISIONLAYERS")
    parts.append("================")
    parts.append("")
    coll = _emit_collision_layers(layers)
    if not coll:
        # Grammar requires at least one layer. Put every object in one layer.
        coll = ", ".join(_obj_name(c) for c in obj_chs) if obj_chs else "Obj0"
    parts.append(coll)
    parts.append("")

    parts.append("======")
    parts.append("RULES")
    parts.append("======")
    parts.append("")
    parts.append(_emit_rules(rules_struct, loop_starts, loop_ends))
    parts.append("")

    parts.append("==============")
    parts.append("WINCONDITIONS")
    parts.append("==============")
    parts.append("")
    parts.append(_emit_winconditions(wins))
    parts.append("")

    parts.append("======")
    parts.append("LEVELS")
    parts.append("======")
    parts.append("")
    if generate_placeholder_level:
        parts.append(_emit_levels_placeholder(obj_chs))
    parts.append("")

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# CLI smoke test: round-trip an existing checkpoint's training games.
# ---------------------------------------------------------------------------

def _main():
    import argparse
    import os
    import pickle
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    p = argparse.ArgumentParser()
    p.add_argument("--load", required=True,
                   help="A run dir (with game_infos.pkl). Round-trips each "
                        "training game's tokens through detokenize.")
    p.add_argument("--out_dir", default=None,
                   help="Where to write detokenized .ps.txt files (default: "
                        "<load>/detokenized).")
    p.add_argument("--parse_check", action="store_true",
                   help="Re-parse each emitted source with the lark parser.")
    args = p.parse_args()

    with open(os.path.join(args.load, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)

    out_dir = args.out_dir or os.path.join(args.load, "detokenized")
    os.makedirs(out_dir, exist_ok=True)

    n_ok = 0
    n_fail = 0
    for info in game_infos:
        name = info["name"]
        tokens = info.get("token_ids", [])
        if not tokens:
            continue
        src = detokenize(tokens, title=name)
        path = os.path.join(out_dir, f"{name}.ps.txt")
        with open(path, "w") as f:
            f.write(src)
        if args.parse_check:
            try:
                from puzzlescript_jax.utils import init_ps_lark_parser
                from puzzlescript_jax.preprocessing import (
                    preprocess_ps, StripPuzzleScript)
                from puzzlescript_jax.gen_tree import GenPSTree
                if not hasattr(_main, "_parser"):
                    _main._parser = init_ps_lark_parser()
                pt = _main._parser.parse(preprocess_ps(src))
                mt = StripPuzzleScript().transform(pt)
                tree2 = GenPSTree().transform(mt)
                print(f"  {name:30s} -> {path}  [parse OK; "
                      f"objects={len(tree2.objects)} rules={len(tree2.rules)}]")
                n_ok += 1
            except Exception as e:
                print(f"  {name:30s} -> {path}  [parse FAIL: {type(e).__name__}: {str(e)[:120]}]")
                n_fail += 1
        else:
            print(f"  {name:30s} -> {path}")

    if args.parse_check:
        print(f"\nParse check: {n_ok} ok, {n_fail} fail")


if __name__ == "__main__":
    _main()
