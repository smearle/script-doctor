"""LLM mutation operator for Autumn ELM.

Given a program and one mutable target span (a whole ``(on ...)`` handler or an
``initnext`` NEXT-clause), prompt an LLM to (1) name a mutation kind and (2)
rewrite ONLY that span, using the 15 authored AutumnBench environments as a
mutation palette. Backend is pluggable: local vLLM (OpenAI-compatible, free) by
default, or the Anthropic API (paid, opt-in).
"""

from __future__ import annotations

import ast
import glob
import os
import re
from dataclasses import dataclass

from sexp import Target, splice

_PALETTE_DIR = ("/home/jupyter-smearle/mara/MARA/domains/autumn/pythonic_autumn/"
                "src/MARA/domains/autumn/pythonic_autumn/data")


def load_palette() -> str:
    blocks = []
    for f in sorted(glob.glob(os.path.join(_PALETTE_DIR, "*.py"))):
        name = os.path.basename(f)[:-3]
        if name == "__init__":
            continue
        ds = ast.get_docstring(ast.parse(open(f).read())) or ""
        # drop the boilerplate "translated from X.sexp" first line
        body = "\n".join(l for l in ds.splitlines()
                         if "translated from" not in l).strip()
        if body:
            blocks.append(f"### {name}\n{body}")
    return "\n\n".join(blocks)


_SYS = """\
You mutate programs written in Autumn, a Lisp-like reactive DSL for grid-world \
games. A program declares objects, global variables defined with \
(initnext INIT NEXT) (INIT sets the first frame, NEXT computes each subsequent \
frame from (prev ...)), and event handlers (on COND BODY) that fire on player \
actions (left/right/up/down, (clicked), click position) or world conditions.

You are given ONE marked span to rewrite. Produce a DIFFERENT, still-VALID \
mechanic by editing only that span. Keep it type-correct and self-contained: use \
only objects, globals, and stdlib functions already referenced in the program \
(e.g. moveLeft, moveUp, moveDownNoCollision, intersects, prev, updateObj, \
addObj, removeObj, ..). Do NOT change the initial level layout, GRID_SIZE, or \
object definitions. Keep the same outer form: rewrite an (on ...) span as an \
(on ...) form, and a NEXT span as a single expression.

Respond in EXACTLY this format and nothing else:
KIND: <2-6 word name of the mutation>
CODE:
```
<the single replacement S-expression>
```"""


@dataclass
class Mutation:
    ok: bool
    kind: str = ""
    replacement: str = ""
    raw: str = ""
    error: str = ""
    in_tok: int = 0
    out_tok: int = 0


# input/output USD per million tokens, by served model
PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


# ---- backends --------------------------------------------------------------

def _call_vllm(system: str, user: str, model: str, base_url: str,
               max_tokens: int, temperature: float):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    u = resp.usage
    return resp.choices[0].message.content or "", (u.prompt_tokens, u.completion_tokens)


def _call_anthropic(system: str, user: str, model: str,
                    max_tokens: int, temperature: float):
    import anthropic
    key = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, (resp.usage.input_tokens, resp.usage.output_tokens)


# ---- parsing ---------------------------------------------------------------

def _balanced(s: str) -> bool:
    depth, in_str = 0, False
    for c in s:
        if in_str:
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _extract_sexpr(text: str) -> str:
    """Pull the first balanced top-level S-expression (or bare atom) from text."""
    i = text.find("(")
    if i < 0:
        return text.strip().split()[0] if text.strip() else ""
    depth, in_str = 0, False
    for j in range(i, len(text)):
        c = text[j]
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
                return text[i:j + 1]
    return ""


def _parse_response(raw: str) -> Mutation:
    km = re.search(r"KIND:\s*(.+)", raw)
    kind = km.group(1).strip() if km else "unnamed"
    # prefer a fenced block after CODE:
    code_part = raw.split("CODE:", 1)[1] if "CODE:" in raw else raw
    fence = re.search(r"```[a-zA-Z]*\n?(.*?)```", code_part, re.DOTALL)
    body = fence.group(1) if fence else code_part
    repl = _extract_sexpr(body).strip()
    if not repl:
        return Mutation(ok=False, raw=raw, error="no S-expression in response")
    if not _balanced(repl):
        return Mutation(ok=False, raw=raw, error="unbalanced replacement")
    return Mutation(ok=True, kind=kind[:60], replacement=repl, raw=raw)


# ---- public API ------------------------------------------------------------

def propose(prog: str, target: Target, palette: str,
            backend: str = "vllm", model: str | None = None,
            base_url: str = "http://localhost:8011/v1",
            max_tokens: int = 1024, temperature: float = 0.9) -> Mutation:
    span = target.text(prog)
    user = (
        f"Other AutumnBench environments, for mechanic inspiration:\n\n{palette}\n\n"
        f"=== FULL PROGRAM ===\n{prog}\n\n"
        f"=== SPAN TO REWRITE (target {target.index}, kind={target.kind}) ===\n"
        f"{span}\n\n"
        "Rewrite ONLY this span into a different valid mechanic."
    )
    if backend == "vllm":
        model = model or "qwen3-4b"
        raw, (in_tok, out_tok) = _call_vllm(_SYS, user, model, base_url, max_tokens, temperature)
    elif backend == "anthropic":
        model = model or "claude-sonnet-4-6"
        raw, (in_tok, out_tok) = _call_anthropic(_SYS, user, model, max_tokens, temperature)
    else:
        raise ValueError(f"unknown backend {backend}")
    mut = _parse_response(raw)
    mut.in_tok, mut.out_tok = in_tok, out_tok
    return mut


def apply(prog: str, target: Target, mut: Mutation) -> str:
    return splice(prog, target, mut.replacement)


if __name__ == "__main__":
    import sys
    from sexp import mutable_targets
    prog = open(sys.argv[1]).read()
    tgts = mutable_targets(prog)
    pal = load_palette()
    t = tgts[int(sys.argv[2]) if len(sys.argv) > 2 else 8]
    print(f"TARGET [{t.index}] {t.kind} {t.label}")
    print("ORIG:", " ".join(t.text(prog).split())[:120])
    m = propose(prog, t, pal)
    print("KIND:", m.kind, "| ok:", m.ok, m.error)
    print("REPL:", " ".join(m.replacement.split())[:200])
