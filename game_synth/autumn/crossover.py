"""LLM-mediated crossover for multi-game Autumn ELM.

Authored Autumn programs have largely disjoint object vocabularies (Mario uses
Mario/Coin/Enemy; sand uses Button/Sand/Water), so blind AST-splice crossover
across two games produces undefined-reference errors. Instead we ask the LLM to
recombine two parents into one valid child, adapting object/type references.

Two modes:
  - "inject": take a distinctive mechanic from donor A and integrate it into
    recipient B, keeping B's objects and initial layout.
  - "levelswap": keep recipient B's dynamics (rules) but stage them in donor A's
    setting/initial layout, re-skinning objects as needed.

Returns a COMPLETE program; the caller validates it via rollout (subprocess).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from mutate import _balanced, _call_anthropic, _call_vllm

# object names declared in a program, for choosing compatible pairs / prompts
def object_vocab(src: str) -> set[str]:
    return set(re.findall(r"\(object\s+([A-Za-z_]\w*)", src))


_SYS_X = """\
You recombine two grid-world games written in Autumn, a Lisp-like reactive DSL, \
into ONE new valid game. A program declares objects, global variables defined \
with (initnext INIT NEXT), and event handlers (on COND BODY).

Output a COMPLETE, self-contained, type-correct Autumn program that runs without \
error. Use only objects, globals, and stdlib functions you actually declare or \
that already exist (moveLeft/Right/Up/Down, moveDownNoCollision, intersects, \
prev, updateObj, addObj, removeObj, .., Position, Cell). Every object referenced \
in a rule must have an (object ...) declaration and an init. Keep GRID_SIZE and \
object cell-definitions internally consistent.

Respond in EXACTLY this format:
KIND: <2-6 word name of the resulting mechanic blend>
CODE:
```
<the complete (program ...) S-expression>
```"""

_MODE_INSTR = {
    "inject": ("Take the single most distinctive mechanic from GAME A (the donor) "
               "and integrate it into GAME B (the recipient). Keep GAME B's "
               "objects and initial layout; adapt the donor mechanic to act on "
               "GAME B's objects. Output the full modified GAME B."),
    "levelswap": ("Produce a game that runs GAME B's dynamics/rules but staged in "
                  "GAME A's setting: use GAME A's objects and initial layout, and "
                  "re-skin GAME B's rules to drive GAME A's objects. Output one "
                  "complete program."),
}


@dataclass
class CrossMut:
    ok: bool
    program: str = ""
    kind: str = ""
    raw: str = ""
    error: str = ""
    in_tok: int = 0
    out_tok: int = 0


def _extract_program(text: str) -> str:
    """Pull the first balanced (program ...) form from the response."""
    i = text.find("(program")
    if i < 0:
        i = text.find("(")
    if i < 0:
        return ""
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


def propose_crossover(prog_a: str, prog_b: str, name_a: str, name_b: str,
                      mode: str = "inject", backend: str = "vllm",
                      model: str | None = None,
                      base_url: str = "http://localhost:8011/v1",
                      max_tokens: int = 3072, temperature: float = 0.9) -> CrossMut:
    user = (
        f"=== GAME A ({name_a}) ===\n{prog_a}\n\n"
        f"=== GAME B ({name_b}) ===\n{prog_b}\n\n"
        f"{_MODE_INSTR[mode]}"
    )
    if backend == "vllm":
        model = model or "qwen3-4b"
        raw, (it, ot) = _call_vllm(_SYS_X, user, model, base_url, max_tokens, temperature)
    elif backend == "anthropic":
        model = model or "claude-sonnet-4-6"
        raw, (it, ot) = _call_anthropic(_SYS_X, user, model, max_tokens, temperature)
    else:
        raise ValueError(f"unknown backend {backend}")

    km = re.search(r"KIND:\s*(.+)", raw)
    kind = km.group(1).strip()[:60] if km else "crossover"
    body = raw.split("CODE:", 1)[1] if "CODE:" in raw else raw
    fence = re.search(r"```[a-zA-Z]*\n?(.*?)```", body, re.DOTALL)
    prog = _extract_program(fence.group(1) if fence else body).strip()
    if not prog:
        return CrossMut(ok=False, raw=raw, error="no program in response", in_tok=it, out_tok=ot)
    if not _balanced(prog):
        return CrossMut(ok=False, raw=raw, error="unbalanced program", in_tok=it, out_tok=ot)
    return CrossMut(ok=True, program=prog, kind=kind, raw=raw, in_tok=it, out_tok=ot)


if __name__ == "__main__":
    import sys
    TESTS = "/home/jupyter-smearle/mara/MARA/domains/autumnbench/Autumn.wasm/tests"
    a, b = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "inject"
    backend = sys.argv[4] if len(sys.argv) > 4 else "vllm"
    pa = open(f"{TESTS}/{a}.sexp").read()
    pb = open(f"{TESTS}/{b}.sexp").read()
    m = propose_crossover(pa, pb, a, b, mode=mode, backend=backend)
    print(f"mode={mode} backend={backend} ok={m.ok} kind={m.kind!r} err={m.error} tok={m.in_tok}/{m.out_tok}")
    if m.ok:
        import rollout
        sig = rollout.evaluate_multi(m.program, (0, 1, 2), 120)
        print("child valid?", sig.ok, sig.error or f"covered={sig.covered} states={sig.n_states_max} dead={sig.dead}")
