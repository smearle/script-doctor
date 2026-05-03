"""Stitch a Rule list into a complete PuzzleScript .txt file.

Boilerplate is templated off ``custom_games/varislide.txt``: solid-color
sprite palette, all interacting objects share the player collision layer
(so they block each other, matching the engine semantics rules typically
assume), no sounds, no win conditions (the rule_gp prototype is a dynamics
test bed, not a goal-completion test).
"""
from __future__ import annotations

from nca_wm.rule_gp import Rule


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


def _collisionlayers_section(objects: list[str]) -> str:
    interactives = ["Player", "Wall"] + list(objects)
    return "Background\n" + ", ".join(interactives) + "\n"


def _rules_section(rules: list[Rule]) -> str:
    return "\n".join(r.unparse() for r in rules) + "\n"


def assemble_game(
    rules: list[Rule],
    *,
    title: str = "rule_gp_game",
    objects: list[str] | None = None,
    level: str | None = None,
) -> str:
    objs = objects or DEFAULT_OBJECTS
    lvl = level or DEFAULT_LEVEL
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
        f"{_collisionlayers_section(objs)}\n"
        f"======\n"
        f"RULES\n"
        f"======\n\n"
        f"{_rules_section(rules)}\n"
        f"==============\n"
        f"WINCONDITIONS\n"
        f"==============\n\n"
        f"=======\n"
        f"LEVELS\n"
        f"=======\n\n"
        f"{lvl}\n"
    )
