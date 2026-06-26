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


VALID_OBJECT_MODIFIERS = {"", ">", "<", "^", "v", "no",
                          "moving", "stationary", "action",
                          "up", "down", "left", "right",
                          "random", "randomDir"}

# Stochastic constructs (random / randomDir / the `random` rule prefix) make
# transitions irreducibly unpredictable -- a cheap way to inflate world-model
# loss without adding any learnable mechanics. ALLOW_RANDOM (set by the GP
# driver) gates them out.
ALLOW_RANDOM = True
STOCHASTIC_MODIFIERS = {"random", "randomDir"}


def object_modifier_pool():
    """Sampleable object modifiers (deterministic order), minus stochastic ones
    when ALLOW_RANDOM is False."""
    pool = VALID_OBJECT_MODIFIERS if ALLOW_RANDOM else (VALID_OBJECT_MODIFIERS - STOCHASTIC_MODIFIERS)
    return sorted(pool)
VALID_RULE_PREFIXES = {"", "late", "random", "horizontal", "vertical",
                       "left", "right", "up", "down"}
VALID_COMMANDS = {"again", "cancel", "checkpoint", "restart", "win"}


def rule_prefix_pool():
    """Sampleable rule prefixes, minus the stochastic `random` prefix when
    ALLOW_RANDOM is False."""
    pool = VALID_RULE_PREFIXES if ALLOW_RANDOM else (VALID_RULE_PREFIXES - {"random"})
    return sorted(pool)


@dataclass
class CellContent:
    obj: str
    modifier: str = ""

    def unparse(self) -> str:
        return f"{self.modifier} {self.obj}".strip() if self.modifier else self.obj


@dataclass
class Cell:
    contents: list[CellContent] = field(default_factory=list)
    is_ellipsis: bool = False        # `...` spacer cell (matches arbitrary distance)

    def unparse(self) -> str:
        if self.is_ellipsis:
            return "..."
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
class RuleGroup:
    """A ``startloop`` ... ``endloop`` block: its rules are re-applied to a
    fixpoint each turn (chain reactions / propagation). Lives in a ruleset's
    rule list alongside plain Rules; unparses to the loop block."""
    rules: list[Rule] = field(default_factory=list)

    def unparse(self) -> str:
        body = "\n".join(r.unparse() for r in self.rules)
        return f"startloop\n{body}\nendloop"


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


def base_multi_bracket_rule(a: str = "ObjA", b: str = "ObjB", c: str = "ObjC") -> Rule:
    """``[ A ] [ Player | B ] -> [ A ] [ Player | C ]`` — a multi-bracket rule:
    fire only when ``A`` exists *somewhere*, then convert an adjacent ``B``.
    Needs grid-wide context (the WM's global_pool) to model."""
    return Rule(
        lhs=[RulePart(cells=[Cell([CellContent(a)])]),
             RulePart(cells=[Cell([CellContent("Player")]), Cell([CellContent(b)])])],
        rhs_parts=[RulePart(cells=[Cell([CellContent(a)])]),
                   RulePart(cells=[Cell([CellContent("Player")]), Cell([CellContent(c)])])],
    )


def base_ellipsis_rule(a: str = "Player", b: str = "ObjA") -> Rule:
    """``[ A | ... | B ] -> [ A | ... | B ]`` skeleton (no-op until mutated) —
    an arbitrary-distance pattern along a line. Needs axis context to model."""
    return Rule(
        lhs=[RulePart(cells=[Cell([CellContent(a)]), Cell(is_ellipsis=True),
                             Cell([CellContent(b)])])],
        rhs_parts=[RulePart(cells=[Cell([CellContent(a)]), Cell(is_ellipsis=True),
                                   Cell([CellContent(b)])])],
    )


BASE_RULE_FACTORIES = {
    "convert_a_to_b": lambda: base_two_cell_rule(),
    "push_a":         lambda: base_push_rule("ObjA"),
    "push_b":         lambda: base_push_rule("ObjB"),
    "delete_a":       lambda: base_delete_rule("ObjA"),
    "delete_b":       lambda: base_delete_rule("ObjB"),
    "multi_bracket":  lambda: base_multi_bracket_rule(),
    "ellipsis":       lambda: base_ellipsis_rule(),
}


_OBJ_CHOICES = ("Player", "ObjA", "ObjB", "ObjC")


def add_bracket(rule: Rule, rng) -> Rule:
    """Append a single-cell condition bracket (present on both sides) -> a
    multi-bracket rule that fires only when the extra object exists somewhere."""
    out = copy.deepcopy(rule)
    obj = rng.choice(_OBJ_CHOICES)
    out.lhs.append(RulePart(cells=[Cell([CellContent(obj)])]))
    out.rhs_parts.append(RulePart(cells=[Cell([CellContent(obj)])]))
    return out


def add_cell(rule: Rule, rng) -> Rule:
    """Extend one LHS bracket (and its aligned RHS) by one cell."""
    out = copy.deepcopy(rule)
    cand = [i for i in range(len(out.lhs)) if not any(c.is_ellipsis for c in out.lhs[i].cells)]
    if not cand:
        return out
    i = rng.choice(cand)
    obj = rng.choice(_OBJ_CHOICES)
    out.lhs[i].cells.append(Cell([CellContent(obj)]))
    if i < len(out.rhs_parts) and len(out.rhs_parts[i].cells) == len(out.lhs[i].cells) - 1:
        out.rhs_parts[i].cells.append(Cell([CellContent(obj)]))
    return out


def add_ellipsis(rule: Rule, rng) -> Rule:
    """Insert a ``...`` spacer between two cells of an aligned LHS/RHS bracket
    pair (an arbitrary-distance pattern)."""
    out = copy.deepcopy(rule)
    cand = [i for i in range(min(len(out.lhs), len(out.rhs_parts)))
            if len(out.lhs[i].cells) == len(out.rhs_parts[i].cells) >= 2
            and not any(c.is_ellipsis for c in out.lhs[i].cells)]
    if not cand:
        return out
    i = rng.choice(cand)
    pos = rng.randint(1, len(out.lhs[i].cells) - 1)
    out.lhs[i].cells.insert(pos, Cell(is_ellipsis=True))
    out.rhs_parts[i].cells.insert(pos, Cell(is_ellipsis=True))
    return out


# ---------------------------------------------------------------------------
# Random ruleset mutation for the evolutionary loop.
# ---------------------------------------------------------------------------

def random_mutate_rule(rule, rng):
    """Apply one random mutation to a single rule (or recurse into a RuleGroup)."""
    if isinstance(rule, RuleGroup):
        out = copy.deepcopy(rule)
        if out.rules:
            i = rng.randrange(len(out.rules))
            out.rules[i] = random_mutate_rule(out.rules[i], rng)
        return out
    op = rng.choice([
        "set_modifier_lhs", "set_modifier_rhs", "toggle_again",
        "toggle_prefix", "swap_obj_in_rhs",
        "add_bracket", "add_cell", "add_ellipsis",
    ])
    if op == "add_bracket":
        return add_bracket(rule, rng)
    if op == "add_cell":
        return add_cell(rule, rng)
    if op == "add_ellipsis":
        return add_ellipsis(rule, rng)
    if op == "set_modifier_lhs":
        target = rng.choice(["Player", "ObjA", "ObjB", "ObjC"])
        mod = rng.choice(object_modifier_pool())
        return set_object_modifier(rule, target, mod, side="lhs")
    if op == "set_modifier_rhs":
        target = rng.choice(["Player", "ObjA", "ObjB", "ObjC"])
        mod = rng.choice(object_modifier_pool())
        return set_object_modifier(rule, target, mod, side="rhs")
    if op == "toggle_again":
        return set_command(rule, "" if "again" in rule.commands else "again")
    if op == "toggle_prefix":
        if rule.prefixes:
            return set_prefix(rule, "")
        return set_prefix(rule, rng.choice([p for p in rule_prefix_pool() if p]))
    if op == "swap_obj_in_rhs":
        # Replace one RHS object with a random different one.
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
    """Apply one random ruleset-level mutation. ``rules`` may contain plain
    Rules and RuleGroups (startloop/endloop blocks)."""
    rules = list(rules)
    op = rng.choices(
        ["mutate_one", "add", "drop", "wrap_loop", "unwrap_loop"],
        weights=[0.5, 0.25, 0.1, 0.1, 0.05],
        k=1,
    )[0]
    if op == "drop" and len(rules) > 1:
        del rules[rng.randrange(len(rules))]
        return rules
    if op == "add" and len(rules) < max_rules:
        factory = rng.choice(list(BASE_RULE_FACTORIES.values()))
        rules.append(factory())
        return rules
    if op == "wrap_loop":
        idxs = [i for i, r in enumerate(rules) if isinstance(r, Rule)]
        if idxs:
            i = rng.choice(idxs)
            j = min(i + rng.randint(0, 1), len(rules) - 1)
            seg = rules[i:j + 1]
            if seg and all(isinstance(r, Rule) for r in seg):
                rules[i:j + 1] = [RuleGroup(rules=copy.deepcopy(seg))]
        return rules
    if op == "unwrap_loop":
        gidxs = [i for i, r in enumerate(rules) if isinstance(r, RuleGroup)]
        if gidxs:
            i = rng.choice(gidxs)
            rules[i:i + 1] = rules[i].rules
        return rules
    if not rules:
        # Fallback: ensure at least one rule.
        factory = rng.choice(list(BASE_RULE_FACTORIES.values()))
        rules.append(factory())
        return rules
    i = rng.randrange(len(rules))
    rules[i] = random_mutate_rule(rules[i], rng)
    return rules


def sample_layers(rng, objs=("ObjA", "ObjB", "ObjC")):
    """Pick a collision-layer partition. Player+Wall always share the first
    interactive layer (so walls keep blocking the player); the remaining objects
    are split into one or more further layers. Objects on DIFFERENT layers may
    occupy the same cell (overlay/floor mechanics); same-layer objects block.
    Returns list[list[str]] (each a layer) or None for the all-in-one default."""
    objs = list(objs)
    variants = [
        None,                                                  # all interactives one layer
        [["Player", "Wall"]] + [[o] for o in objs],           # every obj overlayable
        [["Player", "Wall", objs[0]]] + [[o] for o in objs[1:]],
        [["Player", "Wall"] + objs[:1], objs[1:]] if len(objs) > 1 else None,
    ]
    variants = [v for v in variants if v is not None] + [None]
    return rng.choice(variants)


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
