"""Validate the haoo' data-source mechanism end-to-end on real assembled games.

The whole active-learning scheme (Notion doc) hinges on being able to draw two
i.i.d. next observations o, o' from the *same* pre-action snapshot under the same
hidden world theta. We get that from the C++ engine via:

    backup_level()              # snapshot at state-after-history, pre-action
    seed_rng(s0); step(a) -> o
    restore_level(); seed_rng(s1); step(a) -> o'

This script confirms the open risk: that the JS->C++ compile path *preserves*
`random` rules, so a stochastic game actually diverges (o != o' for some seeds)
while a deterministic game does not (o == o' for all seeds).

Run: .venv/bin/python -m nca_wm.active_learning.check_double_step
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repo-root on path (mirrors train.py).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import rule_gp, rule_game
from nca_wm import game_curriculum as gc


def _compile(name: str, code: str) -> str:
    """Materialize generated game text and return serialized engine JSON."""
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser

    gc._materialize_game(name, code)
    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    return backend.compile_and_serialize(parser, name)


def _step_once(engine, action: int, seed: str, max_again: int = 50) -> np.ndarray:
    """Seed, apply one action (resolving `again` loops), return 2D obs grid."""
    engine.seed_rng(seed)
    engine.process_input(action)
    n = 0
    while engine.is_againing() and n < max_again:
        engine.process_input(-1)
        n += 1
    return np.asarray(engine.get_objects_2d(), dtype=np.int32)


def _double_step(json_str: str, level_i: int, action: int, seeds) -> list[np.ndarray]:
    """Draw one next-obs per seed from the same pre-action snapshot."""
    from puzzlescript_cpp._puzzlescript_cpp import Engine

    eng = Engine()
    eng.load_from_json(json_str)
    eng.load_level(level_i)
    bak = eng.backup_level()  # snapshot pre-action
    outs = []
    for s in seeds:
        eng.restore_level(bak)
        outs.append(_step_once(eng, action, s))
    return outs


def _report(label: str, outs: list[np.ndarray]) -> bool:
    base = outs[0]
    diverged = any(not np.array_equal(base, o) for o in outs[1:])
    uniq = {o.tobytes() for o in outs}
    print(f"  [{label}] {len(uniq)} distinct outcome(s) over {len(outs)} seeds "
          f"-> {'DIVERGES (stochastic)' if diverged else 'identical (deterministic)'}")
    return diverged


def main() -> None:
    seeds = [str(i) for i in range(8)]
    action = 4  # SPACE/action in the cpp convention; any turn-advancing input

    # Deterministic world: convert every ObjA next to the player into ObjB.
    det_rules = [rule_gp.base_two_cell_rule()]
    # A level with several ObjA so a `random` rule has multiple matches to pick from.
    level = (
        "########\n"
        "#......#\n"
        "#.AAAA.#\n"
        "#.P....#\n"
        "#......#\n"
        "########\n"
    )
    det_code = rule_game.assemble_game(det_rules, title="det_world", level=level)

    # Stochastic world: `random [ ObjA ] -> [ ObjB ]` flips ONE random ObjA per turn.
    rand_rule = rule_gp.set_prefix(
        rule_gp.Rule(
            lhs=[rule_gp.RulePart(cells=[rule_gp.Cell([rule_gp.CellContent("ObjA")])])],
            rhs_parts=[rule_gp.RulePart(cells=[rule_gp.Cell([rule_gp.CellContent("ObjB")])])],
        ),
        "random",
    )
    rand_code = rule_game.assemble_game([rand_rule], title="rand_world", level=level)

    print("Deterministic game source:\n" + det_code)
    print("RULES (random):", rand_rule.unparse())

    gc._set_materialize_dir(_REPO / "nca_wm" / "active_learning" / "_check_games")

    print("\nDeterministic world:")
    det_json = _compile("al_check_det", det_code)
    det_ok = not _report("det", _double_step(det_json, 0, action, seeds))

    print("\nStochastic world:")
    rand_json = _compile("al_check_rand", rand_code)
    rand_ok = _report("rand", _double_step(rand_json, 0, action, seeds))

    print("\nVERDICT:",
          "PASS — det identical, random diverges" if (det_ok and rand_ok)
          else "FAIL — see above (random preservation / seeding issue)")


if __name__ == "__main__":
    main()
