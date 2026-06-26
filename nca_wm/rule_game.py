"""Stitch a Rule list into a complete PuzzleScript .txt file.

Boilerplate is templated off ``custom_games/varislide.txt``: solid-color
sprite palette, all interacting objects share the player collision layer
(so they block each other, matching the engine semantics rules typically
assume), no sounds, no win conditions (the rule_gp prototype is a dynamics
test bed, not a goal-completion test).
"""
from __future__ import annotations

from nca_wm.rule_gp import Rule, WinCondition


DEFAULT_OBJECTS = ["ObjA", "ObjB", "ObjC"]
_OBJECT_COLORS = {
    "ObjA": "red",
    "ObjB": "green",
    "ObjC": "yellow",
    "ObjD": "purple",
}
# Single-char level pixel for each object name. The JS engine breaks
# (TypeError in serializeCompiledState) when an object's full name is a
# single char *and* the LEGEND aliases it to itself; >=3 such objects
# trigger the failure. Multi-char object names + distinct single-char
# legend aliases is the canonical PuzzleScript pattern.
_OBJECT_PIXEL = {
    "ObjA": "A",
    "ObjB": "B",
    "ObjC": "C",
    "ObjD": "D",
}

DEFAULT_LEVEL = (
    "########\n"
    "#......#\n"
    "#..A..B#\n"
    "#......#\n"
    "#..P...#\n"
    "#.C....#\n"
    "#......#\n"
    "########\n"
)


def random_level_text(rng, *, w: int = 8, h: int = 8,
                      n_a: int | None = None, n_b: int | None = None,
                      n_c: int | None = None,
                      wall_density: float = 0.0,
                      objects_pixels: dict[str, str] | None = None) -> str:
    """Build a random level: wall border, 1 player, scattered typed objects.

    Counts default to a small uniform draw (1–3 of each); optional internal
    wall placement controlled by ``wall_density`` (0 disables).
    """
    pixels = list((objects_pixels or _OBJECT_PIXEL).values())
    if n_a is None:
        n_a = rng.randint(1, 3)
    if n_b is None:
        n_b = rng.randint(1, 3)
    if n_c is None:
        n_c = rng.randint(1, 3)
    grid = [["#" if r == 0 or r == h - 1 or c == 0 or c == w - 1 else "."
             for c in range(w)] for r in range(h)]
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(interior)
    cursor = 0
    if wall_density > 0:
        n_walls = int(wall_density * len(interior))
        for _ in range(n_walls):
            if cursor >= len(interior):
                break
            r, c = interior[cursor]; cursor += 1
            grid[r][c] = "#"
    placements = [("P", 1)] + [(p, n) for p, n in
                               zip(("A", "B", "C"), (n_a, n_b, n_c))]
    for sym, count in placements:
        for _ in range(count):
            if cursor >= len(interior):
                break
            r, c = interior[cursor]; cursor += 1
            grid[r][c] = sym
    return "\n".join("".join(row) for row in grid) + "\n"


def _objects_section(objects: list[str]) -> str:
    blocks = [
        "Background\nwhite\n",
        "Wall\ndarkgray\n",
        "Player\nblue\n.000.\n.000.\n00000\n.000.\n.0.0.\n",
    ]
    for obj in objects:
        color = _OBJECT_COLORS.get(obj, "orange")
        blocks.append(f"{obj}\n{color}\n00000\n00000\n00000\n00000\n00000\n")
    return "\n".join(blocks)


def _legend_section(objects: list[str]) -> str:
    lines = [
        ". = Background",
        "# = Wall",
        "P = Player",
    ]
    for obj in objects:
        pixel = _OBJECT_PIXEL.get(obj, obj[-1])
        lines.append(f"{pixel} = {obj}")
    return "\n".join(lines) + "\n"


def _collisionlayers_section(objects: list[str], layers: list[list[str]] | None = None) -> str:
    if layers:
        return "Background\n" + "\n".join(", ".join(L) for L in layers) + "\n"
    interactives = ["Player", "Wall"] + list(objects)
    return "Background\n" + ", ".join(interactives) + "\n"


def _rules_section(rules: list[Rule]) -> str:
    return "\n".join(r.unparse() for r in rules) + "\n"


def _winconditions_section(wins: list[WinCondition]) -> str:
    if not wins:
        return "(no win)\n"
    return "\n".join(w.unparse() for w in wins) + "\n"


def assemble_game(
    rules: list[Rule],
    *,
    title: str = "rule_gp_game",
    objects: list[str] | None = None,
    level: str | None = None,
    levels: list[str] | None = None,
    wins: list[WinCondition] | None = None,
    layers: list[list[str]] | None = None,
) -> str:
    """Assemble a complete PuzzleScript game string.

    Pass ``levels`` (list of grid strings) to embed multiple levels — they
    will appear as separate playable levels in the LEVELS section, each
    addressable via ``cpp_engine.load_level(i)``. ``level`` (singular) is a
    shorthand for ``levels=[level]``.
    """
    objs = objects or DEFAULT_OBJECTS
    if levels is None:
        levels = [level] if level is not None else [DEFAULT_LEVEL]
    lvl = "\n\n".join(l.rstrip() + "\n" for l in levels)
    win_block = _winconditions_section(wins or [])
    return (
        f"title {title}\n"
        f"author rule_gp\n"
        f"\n"
        f"========\n"
        f"OBJECTS\n"
        f"========\n\n"
        f"{_objects_section(objs)}\n"
        f"=======\n"
        f"LEGEND\n"
        f"=======\n\n"
        f"{_legend_section(objs)}\n"
        f"=======\n"
        f"SOUNDS\n"
        f"=======\n\n"
        f"================\n"
        f"COLLISIONLAYERS\n"
        f"================\n\n"
        f"{_collisionlayers_section(objs, layers)}\n"
        f"======\n"
        f"RULES\n"
        f"======\n\n"
        f"{_rules_section(rules)}\n"
        f"==============\n"
        f"WINCONDITIONS\n"
        f"==============\n\n"
        f"{win_block}\n"
        f"=======\n"
        f"LEVELS\n"
        f"=======\n\n"
        f"{lvl}\n"
    )
