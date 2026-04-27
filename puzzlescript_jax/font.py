"""PuzzleScript's in-game bitmap font (5×12 glyphs).

Parsed from ``PuzzleScript/src/js/font.js`` at import time. Each entry maps
a single character to a (12, 5) uint8 array (1 = pixel on, 0 = pixel off).

Usage::

    from puzzlescript_jax.font import draw_text
    draw_text(image, "hello", x=2, y=2, color=(255, 255, 255))
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_FONT_JS = (Path(__file__).resolve().parent.parent
            / "PuzzleScript" / "src" / "js" / "font.js")

GLYPH_H = 12
GLYPH_W = 5


def _parse_font_js(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text()
    # Each entry looks like:   '<char>': `\n00000\n01110\n...\n` ,
    # We match the key (either '<x>' or "<x>") followed by a backtick-quoted
    # multi-line bitmap body.
    pattern = re.compile(
        r"['\"]((?:\\.|[^'\"])+)['\"]\s*:\s*`([^`]+)`", re.S
    )
    glyphs: dict[str, np.ndarray] = {}
    for m in pattern.finditer(text):
        raw_key = m.group(1)
        # Unescape common JS escapes
        key = (raw_key
               .replace(r"\\", "\\")
               .replace(r"\'", "'")
               .replace(r"\"", "\"")
               .replace(r"\n", "\n"))
        if len(key) != 1:
            continue
        body = m.group(2).strip("\n")
        rows = [r for r in body.split("\n") if r]
        if not rows:
            continue
        width = max(len(r) for r in rows)
        height = len(rows)
        g = np.zeros((height, width), dtype=np.uint8)
        for y, row in enumerate(rows):
            for x, c in enumerate(row):
                if c == "1":
                    g[y, x] = 1
        glyphs[key] = g
    return glyphs


GLYPHS: dict[str, np.ndarray] = _parse_font_js(_FONT_JS)


def _trim_glyphs(glyphs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Crop the blank top rows from every glyph to the *minimum* rows that
    any one glyph actually uses, leaving the glyphs bottom-aligned and
    compact. Skips the literal space glyph (' ').

    Reduces the natural 12-tall bitmap to ~8 rows, making the font closer
    to 1-tile-tall at native PS scale.
    """
    out: dict[str, np.ndarray] = {}
    # Figure out minimum top-padding across all non-empty glyphs
    min_top = 12
    for ch, g in glyphs.items():
        if ch == " " or not g.any():
            continue
        row_sums = g.sum(axis=1)
        t = int(np.argmax(row_sums > 0))
        min_top = min(min_top, t)
    if min_top < 1:
        return glyphs
    for ch, g in glyphs.items():
        out[ch] = g[min_top:]
    return out


# Compact variant used by small in-training banners.
GLYPHS_COMPACT: dict[str, np.ndarray] = _trim_glyphs(GLYPHS)
GLYPH_H_COMPACT = max((g.shape[0] for g in GLYPHS_COMPACT.values()), default=GLYPH_H)


def draw_text(
    image: np.ndarray, text: str, *,
    x: int = 0, y: int = 0,
    color: tuple[int, int, int] = (255, 255, 255),
    char_spacing: int = 1,
    compact: bool = False,
) -> np.ndarray:
    """Draw ``text`` into a uint8 RGB ``image`` using the PuzzleScript font.

    Modifies ``image`` in-place and returns it. Characters missing from the
    font (including most punctuation — the font covers a-z, 0-9, and a few
    symbols) render as empty boxes.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image must be HxWx3 uint8")
    H, W = image.shape[:2]
    color_arr = np.array(color, dtype=np.uint8)
    glyph_table = GLYPHS_COMPACT if compact else GLYPHS
    cx = x
    for ch in text:
        glyph = glyph_table.get(ch)
        if glyph is None:
            glyph = glyph_table.get(ch.lower())
        if glyph is None:
            # Unknown char — skip its width of space
            cx += GLYPH_W + char_spacing
            continue
        gh, gw = glyph.shape
        # Clip to image bounds
        x0, y0 = cx, y
        x1, y1 = min(W, x0 + gw), min(H, y0 + gh)
        if x0 >= W or y0 >= H:
            break
        gs = glyph[: y1 - y0, : x1 - x0].astype(bool)
        if gs.any():
            image[y0:y1, x0:x1][gs] = color_arr
        cx += gw + char_spacing
    return image


def text_width(text: str, char_spacing: int = 1) -> int:
    return sum(GLYPH_W + char_spacing for _ in text) - char_spacing
