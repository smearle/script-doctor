"""Constrained decoding for rules-only sampling (level_eval.py --prompt-until RULES).

The prompt fixes a level's objects, legend and collision layers; the model writes the RULES
and WINCONDITIONS sections. A RulesChecker follows the text the model writes and says whether a
candidate token can extend it. Two modes:
- "names": every word in the rules and win conditions is a name the prompt defines or a
  PuzzleScript keyword, and the sections end as the level-first format does (a blank line, the
  WINCONDITIONS header, win conditions, then the end of the document);
- "full": also the rule and win-condition syntax, and the checks of the reference engine
  (PuzzleScript dfdeabcd, compiler.js) that a line can be held to while it is written:
  rule prefixes, one direction per object and no movements in late rules, no object twice in
  a cell, ellipses only as whole inner cells matched on both sides, as many bracketed
  patterns and cells on the right as on the left (or commands alone), commands after the
  arrow, no two objects of one collision layer in a cell, no "no" before an "and" aggregate,
  no right-hand cell that excludes an object it also sets, and right-hand properties the engine
  can infer (in the same left-hand cell, or exactly once on the left). Two legacy forms the
  engine accepts with a warning stay allowed: a direction between brackets, and a command
  inside a right-hand cell.
With deterministic=True the words random and randomdir are not allowed either.
Words are checked in canonical spelling: lower case, separated by spaces.

Sampling (generate_constrained) is exact for the masked distribution: a row draws a token from
its temperature-scaled distribution; a token that cannot extend the text is removed and the
row draws again from the rest, which samples the distribution renormalised over the allowed
tokens. After MAX_TRIES removals the allowed set is computed over the whole vocabulary; a row
with no allowed token stops (dead end). Rows record rejections, fallbacks and dead ends.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import torch

HEADERS = ("OBJECTS", "LEGEND", "COLLISIONLAYERS", "RULES", "WINCONDITIONS")
PREFIX_DIRS = frozenset({"up", "down", "left", "right", "horizontal", "vertical", "orthogonal"})
MODIFIERS = frozenset({">", "<", "^", "v", "up", "down", "left", "right", "moving", "stationary", "no",
                       "randomdir", "random", "horizontal", "vertical", "orthogonal", "perpendicular",
                       "parallel", "action"})
LATE_MODIFIERS = frozenset({"no", "random", "randomdir"})
COMMANDS = frozenset({"again", "cancel", "checkpoint", "restart", "win"})
QUANTIFIERS = frozenset({"all", "any", "no", "some"})
LOOP_WORDS = frozenset({"startloop", "endloop"})
SYMBOLS = frozenset({"[", "]", "|", "->", "...", "+"})
KEYWORDS = PREFIX_DIRS | MODIFIERS | COMMANDS | QUANTIFIERS | LOOP_WORDS | {"late", "rigid", "random", "on"}
STOCHASTIC = frozenset({"random", "randomdir"})
ELLIPSIS = (("...", "..."),)
MAX_TRIES = 64


# ---------------------------------------------------------------- prompt inventory

@dataclass(frozen=True)
class Inventory:
    names: frozenset        # names a rule or win condition may use: objects and legend names
    layer_of: dict          # name -> its collision layer, if it has a single one (object or property)
    properties: frozenset   # "or" legend names with more than one object (the engine's propertiesDict)
    aggregates: frozenset   # "and" legend names
    objects_of: dict        # name -> the objects it stands for
    legend: dict            # legend name -> ("and" | "or", members)


def inventory(prompt: str) -> Inventory:
    """Objects, legend and layers of a level-first rules-only prompt (level_first.py layout)."""
    objects, legend, layer_lines, section = [], {}, [], None
    for raw in prompt.split("\n"):
        s = raw.strip()
        if s in HEADERS:
            section = s
        elif not s:
            continue
        elif " = " in s:
            name, rhs = (p.strip().lower() for p in s.split("=", 1))
            if len(name) > 1:  # single characters are level glyphs
                words = rhs.split()
                legend[name] = ("and" if "and" in words else "or", tuple(w for w in words if w not in ("and", "or")))
        elif section == "OBJECTS" and not s.startswith("#"):
            objects.append(s.lower())
        elif section == "COLLISIONLAYERS":
            layer_lines.append(s.lower().replace(",", " ").split())
    objs = frozenset(objects)

    def expand(name, seen=()):
        if name in objs:
            return {name}
        ent = legend.get(name)
        if ent is None or name in seen:
            return set()
        out = set()
        for m in ent[1]:
            out |= expand(m, seen + (name,))
        return out

    layer = {}
    for i, words in enumerate(layer_lines):
        for w in words:
            for o in expand(w):
                layer.setdefault(o, i)
    names = objs | frozenset(legend)
    layer_of = {}
    for n in names:
        ls = {layer.get(o) for o in expand(n)}
        if len(ls) == 1 and None not in ls:
            layer_of[n] = ls.pop()
    props = frozenset(n for n, (op, ms) in legend.items() if op == "or" and len(expand(n)) > 1)
    aggs = frozenset(n for n, (op, ms) in legend.items() if op == "and")
    return Inventory(names, layer_of, props, aggs, {n: frozenset(expand(n)) for n in names}, legend)


# ---------------------------------------------------------------- rule lines

def lex_spans(line: str):
    """Words of a rule line as the engine splits them, each with its start index."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
        elif c in "[]|":
            out.append((i, c))
            i += 1
        elif line.startswith("->", i):
            out.append((i, "->"))
            i += 2
        else:
            j = i
            while j < n and not line[j].isspace() and line[j] not in "[]|" and not line.startswith("->", j):
                j += 1
            out.append((i, line[i:j]))
            i = j
    return out


def lex(line: str):
    """Words of a rule line as the engine splits them; the last word may still be growing."""
    toks = [w for _, w in lex_spans(line)]
    growing = bool(toks) and not line[-1].isspace() and toks[-1] not in ("[", "]", "|", "->")
    return toks, growing


@dataclass(frozen=True)
class RuleState:
    phase: str = "start"      # start, prefix, lhs, rhs, done (after startloop / endloop)
    in_cell: bool = False
    plus: bool = False
    late: bool = False
    rigid: bool = False
    random: bool = False
    lhs: tuple = ()           # rows of cells; a cell is a tuple of (modifier, name)
    rhs: tuple = ()
    row: tuple = ()
    cell: tuple = ()
    mod: str | None = None
    commands: int = 0
    loop_word: str | None = None


@dataclass(frozen=True)
class LineContext:
    inv: Inventory
    mode: str
    deterministic: bool
    first_rule: bool
    loop_depth: int


def word_ok(st: RuleState, w: str, cx: LineContext) -> bool:
    """May word w come next in a rule line in state st?"""
    if cx.deterministic and w in STOCHASTIC:
        return False
    if cx.mode == "names":
        return w in cx.inv.names or w in KEYWORDS or w in SYMBOLS
    ph = st.phase
    if ph == "done":
        return False
    if ph in ("start", "prefix"):
        if w == "[" or w in PREFIX_DIRS:
            return True
        if w == "late":
            return not st.rigid
        if w == "rigid":
            return not st.late
        if w == "random":
            return not (st.random or st.plus)
        if ph == "start":
            if w == "+":
                return not cx.first_rule
            if w == "startloop":
                return cx.loop_depth == 0
            if w == "endloop":
                return cx.loop_depth > 0
        return False
    rhs = ph == "rhs"
    if not st.in_cell:
        if w in PREFIX_DIRS:
            return True  # legacy: ignored by the engine with a warning
        if not rhs:
            return w == "[" or (w == "->" and bool(st.lhs))
        if w == "[":
            return st.commands == 0 and len(st.rhs) < len(st.lhs)
        return w in COMMANDS and len(st.rhs) in (0, len(st.lhs))
    k = len(st.row)
    lhs_row = st.lhs[len(st.rhs)] if rhs else None
    lhs_cell = lhs_row[k] if rhs else None
    if st.cell == ELLIPSIS:
        return w == "|" and (not rhs or k + 1 < len(lhs_row))
    if rhs and lhs_cell == ELLIPSIS:
        return w == "..." and not st.cell
    if rhs and w in COMMANDS and st.mod is None:
        return True  # legacy: a command inside a cell, accepted with a warning
    if w in ("|", "]"):
        if st.mod is not None:
            return False
        if not rhs:
            return True
        return k + 1 < len(lhs_row) if w == "|" else k + 1 == len(lhs_row)
    if w == "...":
        if rhs or st.cell or st.mod is not None or k == 0 or st.row[-1] == ELLIPSIS:
            return False
        return sum(1 for c in st.row if c == ELLIPSIS) < 2
    if w in MODIFIERS and w not in cx.inv.names:
        if st.mod is not None or (st.late and w not in LATE_MODIFIERS):
            return False
        return rhs or w != "random"
    if w not in cx.inv.names:
        return False
    inv = cx.inv
    if any(n == w for (_, n) in st.cell) and not (st.mod == "no" and ("no", w) in st.cell):
        return False
    positive = st.mod not in ("no", "random")
    if st.mod == "no" and w in inv.aggregates:
        return False
    if rhs:  # a right-hand cell may not exclude an object it also sets (on the left it is a warning)
        concrete = lambda n: inv.objects_of[n] == frozenset((n,))
        if st.mod == "no":
            if any(m != "no" and concrete(n) and n in inv.objects_of[w] for (m, n) in st.cell):
                return False
        elif concrete(w) and any(m == "no" and w in inv.objects_of[n] for (m, n) in st.cell):
            return False
    if positive:
        lay = inv.layer_of.get(w)
        if lay is not None and any(m not in ("no", "random") and inv.layer_of.get(n) == lay for (m, n) in st.cell):
            return False
    if rhs and positive:
        if w in cx.inv.properties:
            same = any(n == w and m not in ("no", "random") for (m, n) in lhs_cell)
            count = sum(1 for row in st.lhs for c in row for (m, n) in c if n == w and m not in ("no", "random"))
            if not same and count != 1:
                return False
    return True


def feed(st: RuleState, w: str, cx: LineContext) -> RuleState:
    """The state after word w, which word_ok allowed."""
    if cx.mode == "names":
        return st
    if st.phase in ("start", "prefix"):
        if w in LOOP_WORDS:
            return replace(st, phase="done", loop_word=w)
        if w == "[":
            return replace(st, phase="lhs", in_cell=True, row=(), cell=(), mod=None)
        if w == "+":
            return replace(st, phase="prefix", plus=True)
        if w in ("late", "rigid", "random"):
            return replace(st, phase="prefix", **{w: True})
        return replace(st, phase="prefix")
    if not st.in_cell:
        if w in PREFIX_DIRS:
            return st
        if w == "->":
            return replace(st, phase="rhs")
        if w == "[":
            return replace(st, in_cell=True, row=(), cell=(), mod=None)
        return replace(st, commands=st.commands + 1)
    if st.phase == "rhs" and w in COMMANDS and st.mod is None:
        return replace(st, commands=st.commands + 1)
    if w in ("|", "]"):
        row = st.row + (st.cell,)
        if w == "|":
            return replace(st, row=row, cell=())
        side = "lhs" if st.phase == "lhs" else "rhs"
        return replace(st, **{side: getattr(st, side) + (row,)}, row=(), cell=(), in_cell=False)
    if w == "...":
        return replace(st, cell=ELLIPSIS)
    if st.mod is None and w in MODIFIERS and w not in cx.inv.names:
        return replace(st, mod=w)
    return replace(st, cell=st.cell + ((st.mod or "", w),), mod=None)


def line_complete(st: RuleState, cx: LineContext, last) -> bool:
    """May a rule line end here? `last` is its last word (None for none)."""
    if cx.mode == "names":
        return last is None or last not in ("[", "|", "->")
    return st.phase == "done" or (st.phase == "rhs" and not st.in_cell and
                                  (len(st.rhs) == len(st.lhs) or (not st.rhs and st.commands > 0)))


def words_of(cx: LineContext):
    return cx.inv.names | KEYWORDS | SYMBOLS


def win_ok(i: int, w: str, inv: Inventory, mode: str) -> bool:
    if mode == "names":
        return w in inv.names or w in KEYWORDS
    return w in (QUANTIFIERS, inv.names, {"on"}, inv.names)[i] if i < 4 else False


# ---------------------------------------------------------------- whole completion

@dataclass(frozen=True)
class CheckState:
    section: str = "rules"   # rules, wins
    line: str = ""
    prev_blank: bool = False
    rules: int = 0
    loop_depth: int = 0


class RulesChecker:
    """Follows the completion of one rules-only prompt; see the module docstring."""

    def __init__(self, prompt: str, mode: str, token_texts: list, eos_id: int, deterministic: bool = False):
        if mode not in ("names", "full"):
            raise ValueError(mode)
        if not prompt.endswith("RULES\n"):
            raise ValueError("a rules-only prompt ends with its RULES header line")
        self.inv, self.mode, self.texts, self.eos = inventory(prompt), mode, token_texts, eos_id
        self.deterministic = deterministic
        self.state = CheckState()
        self._pre, self._next = None, {}

    def _context(self, st: CheckState) -> LineContext:
        return LineContext(self.inv, self.mode, self.deterministic, st.rules == 0, st.loop_depth)

    def _prefix(self, line: str, cx: LineContext):
        """(start, n, last, state) for the committed part of a rule line. Its last word starts at `start`
        and may still change when text is appended (a growing word, or "-" that becomes "->"); the n
        words before it cannot, `last` is the final one of those (None for none) and `state` the rule
        state after them (None if one is not allowed). All candidate tokens of a sampling step extend
        the same committed line, and the next step's line extends it, so one entry is kept and updated
        from the previous one: a candidate costs only its own text, not the whole line."""
        ctx = (cx.first_rule, cx.loop_depth)
        pre = self._pre
        if pre is not None and pre[0] == ctx and pre[1] == line:
            return pre[2]
        if pre is not None and pre[0] == ctx and line.startswith(pre[1]):
            start, n, last, rs = pre[2]
        else:
            start, n, last, rs = 0, 0, None, RuleState()
        spans = lex_spans(line[start:])
        if spans and not line[-1].isspace():
            fixed, start = spans[:-1], start + spans[-1][0]
        else:
            fixed, start = spans, len(line)
        for _, w in fixed:
            if rs is not None:
                rs = feed(rs, w, cx) if word_ok(rs, w, cx) else None
        if fixed:
            n, last = n + len(fixed), fixed[-1][1]
        val = (start, n, last, rs)
        self._pre, self._next = (ctx, line, val), {}
        return val

    def _word_prefixes(self, rs: RuleState, cx: LineContext):
        """Every prefix of every word that may come next in state rs (kept for the current line)."""
        out = self._next.get(rs)
        if out is None:
            out = {w[:k] for w in words_of(cx) if word_ok(rs, w, cx) for k in range(1, len(w) + 1)}
            self._next[rs] = out
        return out

    def rule_line(self, committed: str, text: str, st: CheckState, complete: bool):
        """The rule state after `text` extends the committed part of a rule line, or None."""
        cx = self._context(st)
        start, n, last, rs = self._prefix(committed, cx)
        if rs is None:
            return None
        line = committed + text
        tail = [w for _, w in lex_spans(line[start:])]
        if tail:
            n, last = n + len(tail), tail[-1]
        growing = n > 0 and not line[-1].isspace() and last not in ("[", "]", "|", "->")
        for w in (tail[:-1] if growing and not complete else tail):
            if not word_ok(rs, w, cx):
                return None
            rs = feed(rs, w, cx)
        if growing and not complete:
            return rs if last in self._word_prefixes(rs, cx) else None
        return rs if not complete or line_complete(rs, cx, last if n else None) else None

    def win_line(self, line: str, complete: bool) -> bool:
        toks = line.split()
        growing = bool(toks) and not line[-1].isspace()
        for i, w in enumerate(toks):
            if i == len(toks) - 1 and growing and not complete:
                pool = self.inv.names | KEYWORDS if self.mode == "names" else \
                    (QUANTIFIERS, self.inv.names, {"on"}, self.inv.names)[i] if i < 4 else ()
                return any(a.startswith(w) and win_ok(i, a, self.inv, self.mode) for a in pool)
            if not win_ok(i, w, self.inv, self.mode):
                return False
        return not complete or self.mode == "names" or len(toks) in (2, 4)

    def advance(self, st: CheckState, text: str):
        """State after appending `text` (one token's text), or None if it is not allowed."""
        if "\n" in text[:-1]:
            return None
        complete = text.endswith("\n")
        line = st.line + (text[:-1] if complete else text)
        s = line.strip()
        if not s:
            return replace(st, line="", prev_blank=True) if complete else replace(st, line=line)
        if st.section == "rules":
            if s[0].isupper():  # only the next section's header starts with a capital
                if line != s or not st.prev_blank or st.loop_depth or not "WINCONDITIONS".startswith(s):
                    return None
                if not complete:
                    return replace(st, line=line)
                return CheckState("wins", "", False, st.rules, 0) if s == "WINCONDITIONS" else None
            r = self.rule_line(st.line, text[:-1] if complete else text, st, complete)
            if r is None:
                return None
            if not complete:
                return replace(st, line=line)
            depth = st.loop_depth + {"startloop": 1, "endloop": -1}.get(r.loop_word, 0)
            return CheckState("rules", "", False, st.rules + (r.loop_word is None), depth)
        if not self.win_line(line, complete):
            return None
        return replace(st, line="", prev_blank=False) if complete else replace(st, line=line)

    def try_token(self, token: int):
        if token == self.eos:
            return self.state if self.state.section == "wins" and self.state.line == "" else None
        text = self.texts[token] if token < len(self.texts) else None
        return None if text is None else self.advance(self.state, text)

    def allowed(self):
        return [t for t in range(len(self.texts)) if self.try_token(t) is not None] + \
            ([self.eos] if self.try_token(self.eos) is not None else [])


def check_text(prompt: str, completion: str, mode: str, deterministic: bool = False):
    """Feed a completion line by line (and at every character within a line); return
    (accepted, index of the first rejected character or None)."""
    ck = RulesChecker(prompt, mode, [], -1, deterministic)
    st = ck.state
    for i, ch in enumerate(completion):
        new = ck.advance(st, ch)
        if new is None:
            return False, i
        st = new
    return st.section == "wins" and st.line == "", None


# ---------------------------------------------------------------- tokens and sampling

def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_texts(tokenizer_json) -> list:
    """Each token's text (byte-level BPE); None for special tokens and partial UTF-8."""
    tok = json.load(open(tokenizer_json))
    vocab = tok["model"]["vocab"]
    dec, out = _byte_decoder(), [None] * (max(vocab.values()) + 1)
    special = {a["id"] for a in tok.get("added_tokens", [])}
    for s, i in vocab.items():
        if i not in special:
            try:
                out[i] = bytes(dec[ch] for ch in s).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                out[i] = None
    return out


@torch.no_grad()
def generate_constrained(model, prompt: torch.Tensor, max_new: int, eos_id: int, checkers: list,
                         temperature: float = 1.0, generator: torch.Generator | None = None):
    """GPT.generate with each row held to its RulesChecker. Returns the new tokens per row (without
    the eos), per-row eos flags and per-row stats (rejections, fallbacks, dead_end)."""
    B, T0 = prompt.shape
    total = min(model.cfg.max_seq_len, T0 + max_new)
    dtype = torch.bfloat16 if prompt.is_cuda else next(model.parameters()).dtype
    caches = model.new_caches(B, total, dtype, prompt.device)
    out, done, hit_eos = [[] for _ in range(B)], [False] * B, [False] * B
    stats = [dict(rejections=0, fallbacks=0, dead_end=False) for _ in range(B)]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=prompt.is_cuda):
        logits = model(prompt, caches, 0)[:, -1]
        pos = T0
        while pos < total:
            probs = (logits.float() / max(temperature, 1e-6)).softmax(-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1).tolist()
            for b in range(B):
                if done[b]:
                    continue
                ck, t = checkers[b], nxt[b]
                new, row, tries = ck.try_token(t), None, 0
                while new is None:
                    stats[b]["rejections"] += 1
                    row = probs[b].clone() if row is None else row
                    row[t] = 0
                    tries += 1
                    if tries == MAX_TRIES:
                        stats[b]["fallbacks"] += 1
                        mask = torch.zeros_like(row)
                        mask[ck.allowed()] = 1
                        row *= mask
                    if float(row.sum()) <= 0:
                        stats[b]["dead_end"] = True
                        break
                    t = int(torch.multinomial(row, 1, generator=generator))
                    new = ck.try_token(t)
                nxt[b] = t
                if new is None:
                    done[b] = True
                    continue
                ck.state = new
                if t == eos_id:
                    done[b] = hit_eos[b] = True
                else:
                    out[b].append(t)
            if all(done) or pos + 1 >= total:
                break
            logits = model(torch.tensor([[t] for t in nxt], dtype=torch.long, device=prompt.device), caches, pos)[:, -1]
            pos += 1
    return out, hit_eos, stats
