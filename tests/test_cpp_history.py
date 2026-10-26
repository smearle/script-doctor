"""Native history/reference parity and complete-state continuation tests.

Run directly to avoid importing the training/JAX stack:
  python tests/test_cpp_history.py --extension path/to/_puzzlescript_cpp...so
Use --expect-off for a default build.
"""
import argparse
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_extension(path):
    spec = importlib.util.spec_from_file_location("_puzzlescript_cpp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def game(rules="", metadata="", level=None):
    objects = [
        ("Background", "black"), ("Player", "white"), ("Wall", "grey"),
        ("Crate", "orange"), ("Target", "yellow"), ("Pulse", "red"),
        ("Trail", "green"), ("Done", "blue"), ("Red", "red"), ("Blue", "blue"),
    ]
    return (
        "title Native history test\n" + metadata + "\nOBJECTS\n\n"
        + "\n\n".join(name + "\n" + color + "\n" + "00000\n" * 5
                     for name, color in objects)
        + "\nLEGEND\n. = Background\n# = Wall\nP = Player\nC = Crate\nT = Target\n"
          "Color = Red or Blue\n\nSOUNDS\n\nCOLLISIONLAYERS\n"
          "Background\nTarget\nPlayer, Wall, Crate\nPulse\nTrail\nDone\nRed, Blue\n\n"
          "RULES\n[ > Player | Crate ] -> [ > Player | > Crate ]\n"
        + rules + "\n\nWINCONDITIONS\nAll Crate on Target\n\nLEVELS\n"
        + (level or "########\n#P.CT..#\n#......#\n########") + "\n"
    )


def fixtures():
    rng = random.Random(410)
    mixed = [rng.choice([0, 1, 2, 3, 4, -1, "undo", "restart"]) for _ in range(96)]
    cases = [
        ("push", "", "", [3, 3, "undo", "undo", 3, "restart", "undo"] + mixed),
        ("metadata", "", "noundo\nnorestart",
         [3, "undo", "restart", "force_undo", 3, "force_restart"] + mixed),
        ("cancel", "[action Player] -> cancel", "", [3, 4, "undo"] + mixed),
        ("require_movement", "[action Player] -> [Player Red]", "require_player_movement", mixed),
        ("checkpoint", "late [Player Target] -> checkpoint", "",
         [2, 3, 3, 3, 0, 3, "restart", "undo", "undo", "restart"] + mixed),
        ("restart_command", "[action Player] -> restart", "",
         [3, 4, "undo", "undo"] + mixed),
        ("random", "[action Player] -> [Player random Color]", "",
         [4, "undo", 4, "restart", "undo", 4] + mixed),
        ("again", "[action Player] -> [Player Pulse]\n[Trail] -> []\nlate [Pulse] -> [Trail] again", "",
         [4, -1, "undo", 4, "restart", "undo"] + mixed),
        ("startup", "[Player no Done] -> [Player Done Pulse]\n[Trail] -> []\nlate [Pulse] -> [Trail] again",
         "run_rules_on_level_start", [-1, "undo", 3, "restart", -1, "undo"] + mixed),
        ("random_probe", "[action Player] -> [Player Pulse]\n[Trail] -> [random Color]\nlate [Pulse] -> [Trail] again", "",
         [4, -1, "undo", 4, "restart"] + mixed),
        ("startup_win", "[Player no Done] -> [Player Done] win", "run_rules_on_level_start",
         [3, "restart", "undo"] + mixed),
        ("again_no_change", "[action Player] -> [Player Red] again", "",
         [4, -1, "undo", 4, "restart"] + mixed),
        ("random_again", "[action Player] -> [Player Pulse random Color]\n[Trail] -> []\nlate [Pulse] -> [Trail] again", "",
         [4, -1, "undo", 4, "restart"] + mixed),
    ]
    result = [{"name": name, "text": game(rules, meta), "seed": "history-parity",
               "actions": actions} for name, rules, meta, actions in cases]
    # Native JS switches to diff-compressed history above 1024 cells.
    level = "\n".join(["#" * 33, "#P" + "." * 29 + "C#"]
                       + ["#" + "." * 31 + "#"] * 30 + ["#" * 33])
    result.append({"name": "large_history", "text": game(level=level),
                   "seed": "history-parity",
                   "actions": [3] * 8 + [-1, "undo", "restart", "undo_distinct"] + mixed})
    return result


def step(e, a):
    if a == "undo":
        e.undo()
    elif a == "force_undo":
        e.undo(True)
    elif a == "undo_distinct":
        e.undo(False, True)
    elif a == "restart":
        e.restart_game()
    elif a == "force_restart":
        e.restart_game(True)
    else:
        e.process_input(a)


def snapshot(e, history=True):
    value = {"objects": e.get_objects().tolist(), "winning": e.is_winning(),
             "againing": e.is_againing()}
    if history:
        value["depth"] = e.undo_depth
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--expect-off", action="store_true")
    parser.add_argument("--resume-child", action="store_true")
    args = parser.parse_args()
    native = load_extension(args.extension.resolve())
    if args.resume_child:
        payload = json.load(sys.stdin)
        e = native.Engine()
        e.load_from_json(json.loads(payload["state"])["compiled_game"])
        e.restore_state(payload["state"])
        for a in payload["actions"]:
            step(e, a)
        print(e.serialize_state())
        return

    assert native.HISTORY_ENABLED == (not args.expect_off)
    fs = fixtures()
    if args.expect_off:
        assert not hasattr(native.Engine, "undo")
        assert not hasattr(native.Engine, "serialize_state")
        # The same move/tick path, including 5 == -1, stays available by default.
        fs = [dict(f, actions=[a for a in f["actions"] if isinstance(a, int)])
              for f in fs if f["name"] not in ("checkpoint", "restart_command")]
    proc = subprocess.run(["node", str(ROOT / "tests/cpp_history_reference.cjs")],
                          input=json.dumps(fs), text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    refs = json.loads(proc.stdout)
    comparisons = 0
    for fixture, ref in zip(fs, refs):
        e = native.Engine()
        assert e.load_from_json(ref["compiled"])
        e.load_level(0, fixture["seed"])
        e.set_track_rules_fired(True)
        actual = [snapshot(e, not args.expect_off)]
        saved = []
        for a in fixture["actions"]:
            if not args.expect_off:
                saved.append(e.serialize_state())
            step(e, a)
            actual.append(snapshot(e, not args.expect_off))
        expected = ref["trace"]
        if args.expect_off:
            expected = [{k: v for k, v in s.items() if k != "depth"} for s in expected]
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert a == b, (fixture["name"], i, a, b)
            comparisons += 1
        if args.expect_off:
            e.load_level(0, "noop")
            before = e.backup_level()
            e.process_input(5)
            a = snapshot(e, False)
            e.restore_level(before)
            e.seed_rng("noop")
            e.process_input(-1)
            assert snapshot(e, False) == a
            continue

        # Restore at every action boundary, including pending AGAIN and after
        # checkpoint/Undo/Restart, then replay the suffix with exact RNG history.
        final = e.serialize_state()
        for i, state in enumerate(saved):
            restored = native.Engine()
            restored.load_from_json(ref["compiled"])
            restored.restore_state(state)
            assert restored.serialize_state() == state
            for a in fixture["actions"][i:]:
                step(restored, a)
            assert restored.serialize_state() == final, (fixture["name"], i, "resume")

        # Representative mid-episode snapshot also survives a fresh interpreter.
        cut = 3
        child = subprocess.run(
            [sys.executable, __file__, "--extension", str(args.extension.resolve()), "--resume-child"],
            input=json.dumps({"state": saved[cut], "actions": fixture["actions"][cut:]}),
            capture_output=True, text=True, check=True)
        assert json.loads(child.stdout) == json.loads(final)

        # Bad restore is atomic; wrong game, shape, RNG and versions are rejected.
        for modify in (
            lambda x: x.update(version=99),
            lambda x: x.update(compiled_game="{}"),
            lambda x: x["board"].update(dat=[]),
            lambda x: x["rng"].update(s=[0] * 256),
            lambda x: x["rng"].update(i=1.5),
            lambda x: x.update(level_index=2**32),
            lambda x: x["restart_target"].update(width=999),
            lambda x: x["telemetry"].update(current=[]),
        ):
            bad = json.loads(final)
            modify(bad)
            try:
                e.restore_state(json.dumps(bad))
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("Accepted invalid snapshot")
            assert e.serialize_state() == final
        e.load_level(0, "fresh")
        assert e.undo_depth == 0
        assert not e.undo()
    if not args.expect_off:
        ref = refs[6]  # stochastic action-color fixture
        e = native.Engine()
        e.load_from_json(ref["compiled"])
        e.load_level(0, "undo-rng")
        e.process_input(4)
        rng_after = json.loads(e.serialize_state())["rng"]
        e.undo()
        assert json.loads(e.serialize_state())["rng"] == rng_after
        e.restart_game()
        assert json.loads(e.serialize_state())["rng"] == rng_after
        # Restoring a snapshot can select a different level of the same game.
        compiled = json.loads(ref["compiled"])
        second = dict(compiled["levels"][0])
        second["index"] = 1
        compiled["levels"].append(second)
        e.load_from_json(json.dumps(compiled))
        e.load_level(1, "level-two")
        e.process_input(3)
        saved = e.serialize_state()
        e.load_level(0, "level-zero")
        e.restore_state(saved)
        assert e.serialize_state() == saved
        # Recompiling must clear history and resize enabled rule telemetry.
        e.load_from_json(refs[0]["compiled"])
        e.set_track_rules_fired(True)
        e.load_level(0, "recompiled")
        assert e.undo_depth == 0
    print(json.dumps({"history_enabled": native.HISTORY_ENABLED,
                      "fixtures": len(fs), "reference_states": comparisons,
                      "resume_suffixes": sum(len(f["actions"]) for f in fs)
                      if not args.expect_off else 0}))


if __name__ == "__main__":
    main()
