"""PuzzleScript color palettes — single source of truth, loaded from
``PuzzleScript/src/js/colors.js`` (the upstream JS implementation).

Each PuzzleScript game implicitly uses the ``arnecolors`` palette unless
it declares a different one via the ``color_palette`` prelude directive
(e.g. ``color_palette amstrad`` or ``color_palette 9``). Named colors in
an OBJECTS block resolve against the game's palette.

Usage::

    from puzzlescript_jax.colors import palette_for, resolve_color_to_rgb

    pal = palette_for("arnecolors")       # dict[name → '#rrggbb']
    rgb = resolve_color_to_rgb("red", palette_name="arnecolors")
"""
from __future__ import annotations

import re
from pathlib import Path


_COLORS_JS = (Path(__file__).resolve().parent.parent
              / "PuzzleScript" / "src" / "js" / "colors.js")


def _parse_colors_js(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Parse colors.js and return (palettes, aliases).

    palettes: {palette_name -> {color_name -> '#rrggbb'}}
    aliases:  {str_digit -> palette_name}   (e.g. "9" → "amstrad")
    """
    text = path.read_text()

    # --- Parse the numeric-alias table ---
    aliases: dict[str, str] = {}
    m = re.search(r"colorPalettesAliases\s*=\s*\{([^}]*)\}", text, re.S)
    if m:
        for num, name in re.findall(r"(\d+)\s*:\s*\"([^\"]+)\"", m.group(1)):
            aliases[num] = name

    # --- Parse the main colorPalettes dict ---
    # Find "colorPalettes = {" and its matching closing brace.
    start = text.find("colorPalettes = {")
    if start < 0:
        raise RuntimeError(f"colorPalettes block not found in {path}")
    brace_open = text.find("{", start)
    depth = 0
    end = None
    for i in range(brace_open, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise RuntimeError("unbalanced braces in colorPalettes")

    body = text[brace_open + 1 : end]

    # Split into "name : { ... }," palette blocks. Each block starts with an
    # identifier followed by ":" and "{", ends at the matching "}".
    palettes: dict[str, dict[str, str]] = {}
    i = 0
    while i < len(body):
        m = re.search(r"(\w+)\s*:\s*\{", body[i:])
        if not m:
            break
        pal_name = m.group(1)
        block_start = i + m.end()   # position just after the '{'
        depth = 1
        j = block_start
        while j < len(body) and depth > 0:
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            j += 1
        palette_body = body[block_start : j - 1]
        entries: dict[str, str] = {}
        for nm, hx in re.findall(
            r"(\w+)\s*:\s*[\"']([^\"']+)[\"']", palette_body
        ):
            entries[nm.lower()] = hx
        palettes[pal_name.lower()] = entries
        i = j

    if not palettes:
        raise RuntimeError(f"no palettes parsed from {path}")

    return palettes, aliases


# Load once at import time. Module-level cache.
_PALETTES, _PALETTE_ALIASES = _parse_colors_js(_COLORS_JS)

# Default palette per PuzzleScript/src/js/compiler.js ("arnecolors").
DEFAULT_PALETTE = "arnecolors"


def palette_names() -> list[str]:
    """Return all known palette names."""
    return list(_PALETTES.keys())


def palette_for(name: str | None) -> dict[str, str]:
    """Return the palette dict for a name or numeric alias.

    Falls back to ``arnecolors`` when name is None, empty, or unknown.
    """
    if not name:
        return _PALETTES[DEFAULT_PALETTE]
    key = str(name).strip().lower()
    if key in _PALETTE_ALIASES:
        key = _PALETTE_ALIASES[key]
    return _PALETTES.get(key, _PALETTES[DEFAULT_PALETTE])


def resolve_color_to_rgb(
    s: str, palette_name: str | None = None
) -> tuple[int, int, int] | None:
    """Resolve a color string (name or hex) to (r, g, b) in 0..255.

    Returns None for 'transparent' or unresolvable values.
    """
    if s is None:
        return None
    s = s.strip().lower()
    if s == "transparent":
        return None
    pal = palette_for(palette_name)
    if s in pal:
        s = pal[s]
    if s.startswith("#"):
        hx = s[1:]
        if len(hx) == 3:
            hx = "".join(c * 2 for c in hx)
        if len(hx) >= 6:
            try:
                return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
            except ValueError:
                return None
    return None
