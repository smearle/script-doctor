"""Minimal S-expression span parser for Autumn (.sexp) programs.

We do not need a full semantic parser -- only enough structure to (a) locate the
top-level forms inside ``(program ...)`` and (b) hand the ELM mutation operator a
*mutable target span*: a whole ``(on ...)`` handler or the NEXT-clause of an
``(initnext init next)`` global definition.  Everything else (object/type decls,
GRID_SIZE/background, and the INIT-clause of every initnext) is frozen, which is
how we keep the initial level layout fixed while only the dynamics evolve.

Each node carries the half-open char offsets ``[start, end)`` into the original
source so a mutation can be spliced back by string slicing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    start: int                 # offset of '(' in source
    end: int                   # offset just past matching ')'
    children: list["Node"] = field(default_factory=list)
    # head atom (first token) if this list begins with one, else None
    head: str | None = None

    def text(self, src: str) -> str:
        return src[self.start:self.end]


def parse(src: str) -> Node:
    """Parse a program string into a single root Node (the outer ``(program ...)``)."""
    roots = _parse_forms(src)
    if not roots:
        raise ValueError("no top-level S-expression found")
    return roots[0]


def _parse_forms(src: str) -> list[Node]:
    """Parse all top-level parenthesized forms, tracking string literals."""
    stack: list[Node] = []
    roots: list[Node] = []
    i, n = 0, len(src)
    in_str = False
    pending_head: list[bool] = []  # whether current node still needs its head atom
    while i < n:
        c = src[i]
        if in_str:
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == '(':
            node = Node(start=i, end=-1)
            if stack:
                stack[-1].children.append(node)
            stack.append(node)
            pending_head.append(True)
            i += 1
            continue
        if c == ')':
            node = stack.pop()
            node.end = i + 1
            pending_head.pop()
            if not stack:
                roots.append(node)
            i += 1
            continue
        if c.isspace():
            i += 1
            continue
        # an atom token: read to next whitespace/paren/quote
        j = i
        while j < n and not src[j].isspace() and src[j] not in '()"':
            j += 1
        atom = src[i:j]
        if stack and pending_head[-1]:
            stack[-1].head = atom
            pending_head[-1] = False
        i = j
    if stack:
        raise ValueError("unbalanced parentheses")
    return roots


# ---- mutable-target extraction --------------------------------------------

# top-level heads whose forms are entirely frozen (layout / declarations)
_FROZEN_HEADS = {"object", ":"}
# global assignments to these names are frozen (board geometry / cosmetics)
_FROZEN_GLOBALS = {"GRID_SIZE", "background"}


@dataclass
class Target:
    """A mutable span within the program."""
    start: int
    end: int
    kind: str          # 'on' or 'next'
    label: str         # human-readable handle for logging
    index: int         # ordinal among targets

    def text(self, src: str) -> str:
        return src[self.start:self.end]


def mutable_targets(src: str) -> list[Target]:
    """Return the spans the mutation operator is allowed to replace.

    - every ``(on cond body)`` handler  -> the whole form
    - every ``(= name (initnext init next))`` -> just the NEXT span
    Frozen: object/type decls, GRID_SIZE/background, and all INIT clauses.
    """
    root = parse(src)
    assert root.head == "program", f"expected (program ...), got {root.head}"
    targets: list[Target] = []
    for form in root.children:
        if form.head in _FROZEN_HEADS:
            continue
        if form.head == "on":
            targets.append(Target(form.start, form.end, "on",
                                  _on_label(src, form), len(targets)))
        elif form.head == "=":
            # (= name expr); children: ['='atom-head, name-node?, expr-node]
            # name is an atom (no Node); locate the value node = last child list
            name = _second_atom(src, form)
            if name in _FROZEN_GLOBALS:
                continue
            init_next = _find_initnext(form)
            if init_next is not None:
                elems = _elements(src, init_next)   # [head, init, next]
                if len(elems) >= 3:
                    ns, ne = elems[2]               # NEXT clause span
                    targets.append(Target(ns, ne, "next",
                                          f"next:{name}", len(targets)))
    return targets


def _elements(src: str, node: Node) -> list[tuple[int, int]]:
    """Ordered (start,end) spans of top-level elements inside ``node`` --
    both atoms and sub-lists -- so we can index initnext's [head, init, next]
    regardless of whether init/next are atoms or lists."""
    inner_start, inner_end = node.start + 1, node.end - 1
    spans: list[tuple[int, int]] = []
    i, depth, in_str = inner_start, 0, False
    tok_start = -1
    while i < inner_end:
        c = src[i]
        if in_str:
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            if depth == 0 and tok_start < 0:
                tok_start = i
            in_str = True
            i += 1
            continue
        if c == '(':
            if depth == 0:
                spans.append((i, _match_paren(src, i)))
                i = spans[-1][1]
                continue
            depth += 1
            i += 1
            continue
        if c == ')':
            depth -= 1
            i += 1
            continue
        if c.isspace():
            if depth == 0 and tok_start >= 0:
                spans.append((tok_start, i))
                tok_start = -1
            i += 1
            continue
        if depth == 0 and tok_start < 0:
            tok_start = i
        i += 1
    if depth == 0 and tok_start >= 0:
        spans.append((tok_start, inner_end))
    return spans


def _match_paren(src: str, open_idx: int) -> int:
    """Return offset just past the ')' matching the '(' at open_idx."""
    depth, in_str, i = 0, False, open_idx
    while i < len(src):
        c = src[i]
        if in_str:
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced")


def _second_atom(src: str, form: Node) -> str | None:
    """The name token in (= name ...): first atom after the '=' head."""
    inner = src[form.start + 1:form.end - 1]
    toks = inner.replace("(", " ( ").split()
    # head is toks[0] == '=', name is toks[1] if it's a plain atom
    if len(toks) >= 2 and toks[0] == "=":
        cand = toks[1]
        if cand not in ("(",):
            return cand
    return None


def _find_initnext(form: Node) -> Node | None:
    for ch in form.children:
        if ch.head == "initnext":
            return ch
    return None


def _on_label(src: str, form: Node) -> str:
    txt = form.text(src)
    one = " ".join(txt.split())
    return "on:" + (one[:60] + "...") if len(one) > 63 else "on:" + one[3:].strip()[:60]


def splice(src: str, target: Target, replacement: str) -> str:
    """Replace target span with replacement, returning the new program string."""
    return src[:target.start] + replacement + src[target.end:]


if __name__ == "__main__":
    import sys
    prog = open(sys.argv[1]).read()
    tgts = mutable_targets(prog)
    print(f"{len(tgts)} mutable targets:")
    for t in tgts:
        snippet = " ".join(t.text(prog).split())
        print(f"  [{t.index}] {t.kind:5s} {t.label}")
        print(f"        {snippet[:100]}")
