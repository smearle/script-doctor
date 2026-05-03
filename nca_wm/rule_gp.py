"""Rule-grammar GP — AST-style rule representation, mutators, and enumeration.

See ``RULE_GP_DESIGN.md``. This module knows nothing about PuzzleScript file
assembly or the engine; it only models a *rule* as a small dataclass tree
and supports deterministic unparsing back to PuzzleScript text.

Hand-written dataclasses (not lark trees) so mutators stay readable and the
unparser is a few lines instead of a tree visitor.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field


VALID_OBJECT_MODIFIERS = {"", ">", "<", "^", "v", "no", "random", "randomDir"}
VALID_RULE_PREFIXES = {"", "late", "random", "horizontal", "vertical",
                       "left", "right", "up", "down"}
VALID_COMMANDS = {"again", "cancel", "checkpoint", "restart", "win"}


@dataclass
class CellContent:
    obj: str
    modifier: str = ""

    def unparse(self) -> str:
        return f"{self.modifier} {self.obj}".strip() if self.modifier else self.obj


@dataclass
class Cell:
    contents: list[CellContent] = field(default_factory=list)

    def unparse(self) -> str:
        return " ".join(c.unparse() for c in self.contents)


@dataclass
class RulePart:
    cells: list[Cell] = field(default_factory=list)

    def unparse(self) -> str:
        return "[ " + " | ".join(c.unparse() for c in self.cells) + " ]"


@dataclass
class Rule:
    prefixes: list[str] = field(default_factory=list)
    lhs: list[RulePart] = field(default_factory=list)
    rhs_parts: list[RulePart] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)

    def unparse(self) -> str:
        toks: list[str] = list(self.prefixes)
        toks.extend(p.unparse() for p in self.lhs)
        toks.append("->")
        toks.extend(p.unparse() for p in self.rhs_parts)
        toks.extend(self.commands)
        return " ".join(toks)


# ---------------------------------------------------------------------------
# Mutators — each returns a fresh deep-copied Rule.
# ---------------------------------------------------------------------------

def set_object_modifier(rule: Rule, target_obj: str, modifier: str,
                        side: str = "lhs") -> Rule:
    """Set ``modifier`` on the first occurrence of ``target_obj`` in side."""
    if modifier not in VALID_OBJECT_MODIFIERS:
        raise ValueError(f"unknown object modifier: {modifier!r}")
    out = copy.deepcopy(rule)
    parts = out.lhs if side == "lhs" else out.rhs_parts
    for part in parts:
        for cell in part.cells:
            for cc in cell.contents:
                if cc.obj == target_obj:
                    cc.modifier = modifier
                    return out
    return out


def truncate_cells(rule: Rule, n: int) -> Rule:
    """Keep only the first ``n`` cells in every LHS/RHS part."""
    out = copy.deepcopy(rule)
    for part in out.lhs:
        part.cells = part.cells[:n]
    for part in out.rhs_parts:
        part.cells = part.cells[:n]
    return out


def set_command(rule: Rule, command: str) -> Rule:
    """Replace trailing commands with ``[command]`` (or none if empty)."""
    if command and command not in VALID_COMMANDS:
        raise ValueError(f"unknown rule command: {command!r}")
    out = copy.deepcopy(rule)
    out.commands = [command] if command else []
    return out


def set_prefix(rule: Rule, prefix: str) -> Rule:
    """Replace prefix list with ``[prefix]`` (or none)."""
    if prefix and prefix not in VALID_RULE_PREFIXES:
        raise ValueError(f"unknown rule prefix: {prefix!r}")
    out = copy.deepcopy(rule)
    out.prefixes = [prefix] if prefix else []
    return out


# ---------------------------------------------------------------------------
# Base templates and the smoketest enumeration.
# ---------------------------------------------------------------------------

def base_two_cell_rule() -> Rule:
    """``[ Player | ObjA ] -> [ Player | ObjB ]`` — convert ObjA to ObjB."""
    return Rule(
        lhs=[RulePart(cells=[
            Cell([CellContent("Player")]),
            Cell([CellContent("ObjA")]),
        ])],
        rhs_parts=[RulePart(cells=[
            Cell([CellContent("Player")]),
            Cell([CellContent("ObjB")]),
        ])],
    )


def enumerate_smoketest_rulesets() -> list[tuple[dict, list[Rule]]]:
    """Cross-product of three axes from RULE_GP_DESIGN.md step (3).

    Returns list of ``(axes_dict, rules)`` pairs. Each ruleset is a list of
    Rule (currently always a single rule).

    Axes: object modifier on the second LHS cell ∈ {none, >, no},
    trailing command ∈ {none, again}, LHS cell count ∈ {1, 2}. 12 cells.
    """
    modifiers = ["", ">", "no"]
    commands = ["", "again"]
    cell_counts = [1, 2]
    out = []
    for mod, cmd, n_cells in itertools.product(modifiers, commands, cell_counts):
        rule = base_two_cell_rule()
        if n_cells == 2:
            # Only meaningful to set a modifier on the second-cell object.
            rule = set_object_modifier(rule, "ObjA", mod, side="lhs")
        rule = truncate_cells(rule, n_cells)
        rule = set_command(rule, cmd)
        out.append(({"modifier": mod, "command": cmd, "lhs_cells": n_cells},
                    [rule]))
    return out


def axis_label(axes: dict) -> str:
    """Compact human-readable label for an axes dict."""
    mod = axes["modifier"] or "_"
    cmd = axes["command"] or "_"
    return f"mod={mod:<3s} cmd={cmd:<5s} cells={axes['lhs_cells']}"
