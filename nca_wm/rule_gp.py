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


@dataclass
class WinCondition:
    """``WINCONDITIONS`` line: ``{quantifier} {obj} [on {on_obj}]``."""
    quantifier: str          # "no" | "some" | "all"
    obj: str
    on_obj: str = ""

    def unparse(self) -> str:
        if self.on_obj:
            return f"{self.quantifier} {self.obj} on {self.on_obj}"
        return f"{self.quantifier} {self.obj}"


@dataclass
class Ruleset:
    """Bundle of rules plus win conditions — one such bundle defines a game."""
    rules: list[Rule] = field(default_factory=list)
    wins: list[WinCondition] = field(default_factory=list)


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


def base_push_rule(obj: str = "ObjA") -> Rule:
    """``[ > Player | obj ] -> [ > Player | > obj ]`` — sokoban push."""
    return Rule(
        lhs=[RulePart(cells=[
            Cell([CellContent("Player", modifier=">")]),
            Cell([CellContent(obj)]),
        ])],
        rhs_parts=[RulePart(cells=[
            Cell([CellContent("Player", modifier=">")]),
            Cell([CellContent(obj, modifier=">")]),
        ])],
    )


def base_delete_rule(obj: str = "ObjA") -> Rule:
    """``[ Player | obj ] -> [ Player | ]`` — pickup/destroy on contact."""
    return Rule(
        lhs=[RulePart(cells=[
            Cell([CellContent("Player")]),
            Cell([CellContent(obj)]),
        ])],
        rhs_parts=[RulePart(cells=[
            Cell([CellContent("Player")]),
            Cell([]),  # empty cell on RHS deletes whatever was there
        ])],
    )


BASE_RULE_FACTORIES = {
    "convert_a_to_b": lambda: base_two_cell_rule(),
    "push_a":         lambda: base_push_rule("ObjA"),
    "push_b":         lambda: base_push_rule("ObjB"),
    "delete_a":       lambda: base_delete_rule("ObjA"),
    "delete_b":       lambda: base_delete_rule("ObjB"),
}


# ---------------------------------------------------------------------------
# Random ruleset mutation for the evolutionary loop.
# ---------------------------------------------------------------------------

def random_mutate_rule(rule: Rule, rng) -> Rule:
    """Apply one random mutation to a single rule."""
    op = rng.choice([
        "set_modifier_lhs", "set_modifier_rhs", "toggle_again",
        "toggle_prefix", "swap_obj_in_rhs",
    ])
    if op == "set_modifier_lhs":
        target = rng.choice(["Player", "ObjA", "ObjB", "ObjC"])
        mod = rng.choice(list(VALID_OBJECT_MODIFIERS))
        return set_object_modifier(rule, target, mod, side="lhs")
    if op == "set_modifier_rhs":
        target = rng.choice(["Player", "ObjA", "ObjB", "ObjC"])
        mod = rng.choice(list(VALID_OBJECT_MODIFIERS))
        return set_object_modifier(rule, target, mod, side="rhs")
    if op == "toggle_again":
        return set_command(rule, "" if "again" in rule.commands else "again")
    if op == "toggle_prefix":
        if rule.prefixes:
            return set_prefix(rule, "")
        return set_prefix(rule, rng.choice(["late", "right", "left", "up", "down"]))
    if op == "swap_obj_in_rhs":
        # Replace one RHS object with a random different one.
        import copy
        out = copy.deepcopy(rule)
        for part in out.rhs_parts:
            for cell in part.cells:
                for cc in cell.contents:
                    if cc.obj in {"ObjA", "ObjB", "ObjC"}:
                        cc.obj = rng.choice([o for o in ("ObjA", "ObjB", "ObjC")
                                             if o != cc.obj])
                        return out
        return out
    return rule


VALID_WIN_QUANTIFIERS = ("no", "some", "all")
DEFAULT_WIN_OBJECTS = ("ObjA", "ObjB", "ObjC")


def mutate_wins(wins: list[WinCondition], rng, *,
                objects=DEFAULT_WIN_OBJECTS,
                max_wins: int = 2) -> list[WinCondition]:
    """Apply one random mutation to a win-condition list."""
    wins = list(wins)
    op = rng.choices(
        ["add", "drop", "change_quantifier", "change_obj", "toggle_on"],
        weights=[0.45, 0.15, 0.15, 0.15, 0.10],
        k=1,
    )[0]
    if op == "drop" and wins:
        del wins[rng.randrange(len(wins))]
        return wins
    if op == "add" and len(wins) < max_wins:
        q = rng.choice(["no", "some", "all"])
        obj = rng.choice(list(objects))
        on_obj = ""
        if q == "all":
            on_obj = rng.choice([o for o in objects if o != obj])
        wins.append(WinCondition(q, obj, on_obj))
        return wins
    if not wins:
        return wins
    i = rng.randrange(len(wins))
    cur = wins[i]
    if op == "change_quantifier":
        new_q = rng.choice([q for q in ("no", "some", "all") if q != cur.quantifier])
        new_on = cur.on_obj
        if new_q == "all" and not new_on:
            new_on = rng.choice([o for o in objects if o != cur.obj])
        elif new_q != "all":
            new_on = ""
        wins[i] = WinCondition(new_q, cur.obj, new_on)
    elif op == "change_obj":
        wins[i] = WinCondition(
            cur.quantifier,
            rng.choice([o for o in objects if o != cur.obj]),
            cur.on_obj,
        )
    elif op == "toggle_on":
        if cur.on_obj:
            wins[i] = WinCondition(cur.quantifier, cur.obj, "")
        else:
            wins[i] = WinCondition(
                cur.quantifier, cur.obj,
                rng.choice([o for o in objects if o != cur.obj]),
            )
    return wins


def mutate_ruleset(rules: list[Rule], rng, *,
                   max_rules: int = 4) -> list[Rule]:
    """Apply one random ruleset-level mutation."""
    rules = list(rules)
    op = rng.choices(
        ["mutate_one", "add", "drop"],
        weights=[0.6, 0.3, 0.1],
        k=1,
    )[0]
    if op == "drop" and len(rules) > 1:
        del rules[rng.randrange(len(rules))]
        return rules
    if op == "add" and len(rules) < max_rules:
        factory = rng.choice(list(BASE_RULE_FACTORIES.values()))
        rules.append(factory())
        return rules
    if not rules:
        # Fallback: ensure at least one rule.
        factory = rng.choice(list(BASE_RULE_FACTORIES.values()))
        rules.append(factory())
        return rules
    i = rng.randrange(len(rules))
    rules[i] = random_mutate_rule(rules[i], rng)
    return rules


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
