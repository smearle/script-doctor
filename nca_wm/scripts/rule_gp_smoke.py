"""Smoketest the rule-GP curriculum prototype.

Enumerates the 12 rulesets from ``rule_gp.enumerate_smoketest_rulesets``,
materializes each into ``custom_games/`` via the ``rule_game`` assembler,
compiles and BFS-explores each, and applies the transition-diff no-op
filter against an empty-rules baseline (described in
``RULE_GP_DESIGN.md``).

Reports survival rate, no-op rejection rate, and per-axis no-op rates.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp._puzzlescript_cpp import Engine, collect_transitions_bfs
from puzzlescript_cpp import CppPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_jax.globals import CUSTOM_GAMES_DIR

from nca_wm.rule_gp import enumerate_smoketest_rulesets, axis_label
from nca_wm.rule_game import assemble_game


def _materialize(name: str, rules) -> Path:
    text = assemble_game(rules, title=name)
    p = Path(CUSTOM_GAMES_DIR) / f"{name}.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _compile_and_bfs(parser, name: str, *, max_iters: int = 2000,
                     timeout_ms: int = 2000):
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(parser, name)
    eng = Engine()
    if not eng.load_from_json(json_str):
        raise RuntimeError("Engine.load_from_json returned false")
    eng.load_level(0)
    return collect_transitions_bfs(eng, max_iters, timeout_ms)


def _transition_set(result) -> set:
    """Frozenset of (state, action, next_state) tuples for diffing."""
    states = list(result.states)
    actions = list(result.actions)
    next_states = list(result.next_states)
    return {
        (tuple(int(x) for x in s), int(a), tuple(int(x) for x in ns))
        for s, a, ns in zip(states, actions, next_states)
    }


def _slug(label: str) -> str:
    return (label.replace(" ", "_")
                 .replace("=", "-")
                 .replace(",", ""))


def main() -> None:
    parser = init_ps_lark_parser()

    print("[rule_gp_smoke] Materializing baseline (empty rule set)...")
    base_name = "rule_gp_smoke_baseline"
    _materialize(base_name, [])
    base = _compile_and_bfs(parser, base_name)
    base_set = _transition_set(base)
    print(f"[rule_gp_smoke]   baseline: iterations={int(base.iterations)}, "
          f"transitions={len(base_set)}")

    rulesets = enumerate_smoketest_rulesets()
    print(f"[rule_gp_smoke] Enumerated {len(rulesets)} rulesets across axes "
          f"{{modifier, command, lhs_cells}}.\n")

    rows = []
    for axes, rules in rulesets:
        label = axis_label(axes)
        name = f"rule_gp_smoke_{_slug(label)}"
        _materialize(name, rules)
        try:
            r = _compile_and_bfs(parser, name)
            ts = _transition_set(r)
            no_op = (ts == base_set)
            rows.append({
                "axes": axes, "label": label, "compiled": True,
                "iterations": int(r.iterations),
                "n_transitions": len(ts),
                "diff_size": len(ts.symmetric_difference(base_set)),
                "no_op": no_op,
                "rule_text": rules[0].unparse(),
            })
        except Exception as e:
            rows.append({
                "axes": axes, "label": label, "compiled": False,
                "error": str(e)[:120],
                "rule_text": rules[0].unparse(),
            })

    # Headline table.
    print(f"{'label':<28s}  cmpl  iters  trans  diff  no-op  rule")
    print("-" * 110)
    for r in rows:
        if r["compiled"]:
            no_op_tag = "YES" if r["no_op"] else "no"
            print(f"{r['label']:<28s}  OK    "
                  f"{r['iterations']:5d}  {r['n_transitions']:5d}  "
                  f"{r['diff_size']:4d}  {no_op_tag:<5s}  {r['rule_text']}")
        else:
            print(f"{r['label']:<28s}  FAIL                              "
                  f"       {r['rule_text']}    [{r['error']}]")

    # Survival + no-op summary.
    n_total = len(rows)
    n_compiled = sum(1 for r in rows if r["compiled"])
    n_no_op = sum(1 for r in rows if r["compiled"] and r["no_op"])
    n_survivors = n_compiled - n_no_op
    print()
    print(f"[rule_gp_smoke] compile rate: {n_compiled}/{n_total}")
    print(f"[rule_gp_smoke] no-op rate (of compiled): {n_no_op}/{n_compiled}")
    print(f"[rule_gp_smoke] survivors (compiled and not no-op): "
          f"{n_survivors}/{n_total}")

    # Per-axis no-op rate (counts only compiled rules).
    by_axis: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if not r["compiled"]:
            continue
        for axis_name, val in r["axes"].items():
            by_axis[axis_name][val][1] += 1
            if r["no_op"]:
                by_axis[axis_name][val][0] += 1
    print("\n[rule_gp_smoke] per-axis no-op rate (of compiled rulesets):")
    for axis_name in sorted(by_axis.keys()):
        for val in sorted(by_axis[axis_name].keys(), key=lambda x: str(x)):
            no_ops, total = by_axis[axis_name][val]
            print(f"  {axis_name:<10s} = {val!r:<8s}  {no_ops}/{total}")


if __name__ == "__main__":
    main()
