"""Gate: validate the sokoban push-mechanic variants behave as specified.

Authors each variant's push rule(s), compiles through the real JS->C++ engine,
scripts a single RIGHT push of a box with open space, and asserts the one-push
signature. The two real authoring risks live here: the `slide` again-loop must
terminate moving the box to the first obstacle, and `chaos` (randomDir) must
actually randomize (validated by a double-step like check_double_step.py).

Box == ObjA. Player/Wall/Background come from the rule_game boilerplate (single
collision layer is fine here — `target` / multi-layer is only needed for the
full family later).

Run: .venv/bin/python -m nca_wm.active_learning.check_sokoban_variants
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from nca_wm import rule_game


class _Raw:
    """Minimal rule wrapper exposing .unparse() for rule_game.assemble_game."""
    def __init__(self, s: str):
        self.s = s
    def unparse(self) -> str:
        return self.s


# Each variant = list of raw PuzzleScript push rules over Player + ObjA (box).
VARIANTS = {
    "classic": ["[ > Player | ObjA ] -> [ > Player | > ObjA ]"],
    "inert": [],  # no push rule -> box blocks player, nothing moves
    "slide": [
        "[ > Player | ObjA ] -> [ > Player | > ObjA ]",
        "[ > ObjA | no Wall no ObjA ] -> [ | > ObjA ] again",
    ],
    "chaos": ["[ > Player | ObjA ] -> [ Player | randomDir ObjA ]"],
    "swap": ["[ > Player | ObjA ] -> [ ObjA | Player ]"],
}

# 5x5 open room; player just left of a box, with space all around the box.
LEVEL = (
    ".....\n"
    ".....\n"
    ".PA..\n"
    ".....\n"
    ".....\n"
)
RIGHT = 3  # engine input id


def _compile(name: str, rules) -> str:
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser
    code = rule_game.assemble_game(rules, title=name, objects=["ObjA"], level=LEVEL)
    gc._set_materialize_dir(_REPO / "nca_wm" / "active_learning" / "_sok_games")
    gc._materialize_game(f"sok_{name}", code)
    return CppPuzzleScriptBackend().compile_and_serialize(init_ps_lark_parser(), f"sok_{name}")


def _engine(json_str: str):
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    e = Engine(); e.load_from_json(json_str); e.load_level(0)
    return e


def _objA_id(engine) -> int:
    return list(engine.get_id_dict()).index("obja")


def _box_cells(engine, oid: int):
    a = np.asarray(engine.get_objects_2d())[:, :, 0]  # (W,H), bitmask
    W, H = a.shape
    return sorted((x, y) for x in range(W) for y in range(H) if int(a[x, y]) & (1 << oid))


def _player_cells(engine):
    a = np.asarray(engine.get_objects_2d())[:, :, 0]
    pid = list(engine.get_id_dict()).index("player")
    W, H = a.shape
    return sorted((x, y) for x in range(W) for y in range(H) if int(a[x, y]) & (1 << pid))


def _push(engine, max_again: int = 50):
    engine.process_input(RIGHT)
    n = 0
    while engine.is_againing() and n < max_again:
        engine.process_input(-1); n += 1


def main() -> None:
    print(f"{'variant':>8} | {'box dx,dy':>9} | {'player':>8} | {'note'}")
    results = {}
    for name, rules in VARIANTS.items():
        js = _compile(name, [_Raw(s) for s in rules])
        e = _engine(js); oid = _objA_id(e)
        box0 = _box_cells(e, oid)[0]; p0 = _player_cells(e)[0]
        _push(e)
        box1 = _box_cells(e, oid)[0]; p1 = _player_cells(e)[0]
        dx, dy = box1[0] - box0[0], box1[1] - box0[1]
        note = ""
        if name == "chaos":
            # double-step: same pre-push snapshot, different seeds -> divergence?
            e2 = _engine(js); bak = e2.backup_level(); outs = set()
            for s in range(10):
                e2.restore_level(bak); e2.seed_rng(str(s)); _push(e2)
                outs.add(tuple(_box_cells(e2, oid)))
            note = f"{len(outs)} distinct box outcomes / 10 seeds"
        print(f"{name:>8} | {dx:+d},{dy:+d}     | {p0}->{p1} | {note}")
        results[name] = (dx, dy, box1, p1, note)

    # Assertions on the expected one-push signatures.
    ok = True
    checks = {
        "classic": results["classic"][:2] == (1, 0),
        "inert": results["inert"][:2] == (0, 0),
        "slide": results["slide"][0] >= 2,                 # slid >=2 cells
        "swap": results["swap"][:2] == (-1, 0),            # box ended behind player
        "chaos": int(results["chaos"][4].split()[0]) >= 2,  # >=2 distinct outcomes
    }
    print("\nSignature checks:")
    for k, v in checks.items():
        print(f"  {k:>8}: {'PASS' if v else 'FAIL'}")
        ok = ok and v
    print("\nVERDICT:", "ALL PASS" if ok else "FAIL — see above")


if __name__ == "__main__":
    main()
