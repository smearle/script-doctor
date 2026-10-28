"""Deterministic execution limits, atomic failures, and matching-order parity.

Build the extension, initialize PuzzleScript, then run with --extension PATH.
Both history-enabled and history-free builds are supported.
"""
import argparse
import json
import subprocess
from pathlib import Path
from test_cpp_history import fixtures, game, load_extension, snapshot, step

ROOT = Path(__file__).resolve().parents[1]


def compile_games(games):
    p = subprocess.run(["node", str(ROOT / "tests/compile_budget_fixtures.cjs")],
                       input=json.dumps(games), text=True, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    m = load_extension(args.extension)
    assert m.DEFAULT_EXECUTION_BUDGET == 100_000_000
    ordinary = fixtures()
    extra = [
        ("tuple_order", "right [action Player] [Crate] [Crate] -> [Player] [Player] [Player]"),
        ("random_tuples", "random [action Player] [Crate] -> [Player Red] [Crate]"),
        ("random_groups", "random [action Player] [Crate] -> [Player Red] [Crate]\n"
                          "+ [action Player] [Target] -> [Player Blue] [Target]"),
        ("ellipsis", "right [action Player | ... | Crate] -> [Player Red | ... | Crate]"),
        ("two_ellipsis", "right [action Player | ... | Crate | ... | Target] -> "
                          "[Player Red | ... | Crate | ... | Target]"),
    ]
    actions = [4, 3, 4, 2, 4, 1, 0, -1] * 12
    for name, rules in extra:
        ordinary.append(dict(name=name, text=game(rules, level=
                            "##########\n#P.C.C.T.#\n#........#\n##########"),
                             seed="budget-parity", actions=actions))
    if not m.HISTORY_ENABLED:
        ordinary = [dict(f, actions=[a for a in f["actions"] if isinstance(a, int)])
                    for f in ordinary if f["name"] not in ("checkpoint", "restart_command")]
    p = subprocess.run(["node", str(ROOT / "tests/cpp_history_reference.cjs")],
                       input=json.dumps(ordinary), text=True, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr
    refs = json.loads(p.stdout)
    checks = 0
    max_work = 0
    for fixture, ref in zip(ordinary, refs):
        e = m.Engine()
        e.load_from_json(ref["compiled"])
        e.load_level(0, fixture["seed"])
        for i, action in enumerate(fixture["actions"]):
            before = e.serialize_state() if m.HISTORY_ENABLED else None
            step(e, action)
            after = e.serialize_state() if m.HISTORY_ENABLED else None
            work = e.execution_work
            max_work = max(max_work, work)
            expected = ref["trace"][i+1]
            if not m.HISTORY_ENABLED:
                expected = {k:v for k,v in expected.items() if k != "depth"}
            assert snapshot(e, m.HISTORY_ENABLED) == expected, (fixture["name"], i)
            # Boundary equality, under-budget atomic rollback, and replay after
            # rejection. Include state-changing random operations and nested
            # startup/AGAIN/Restart calls, not merely no-op transitions.
            if m.HISTORY_ENABLED and work > 1 and action not in ("undo", "force_undo", "undo_distinct"):
                e.restore_state(before)
                e.execution_budget = work
                step(e, action)
                actual = json.loads(e.serialize_state())
                target = json.loads(after)
                actual["execution_budget"] = target["execution_budget"]
                assert actual == target, (fixture["name"], i, "exact limit")
                e.restore_state(before)
                e.execution_budget = work - 1
                limited = e.serialize_state()
                try:
                    step(e, action)
                except m.ExecutionBudgetExceeded:
                    pass
                else:
                    raise AssertionError((fixture["name"], i, "failed to enforce limit"))
                assert e.serialize_state() == limited, (fixture["name"], i, "rollback")
                e.restore_state(after)
                checks += 1

    explosion = game("right " + "[Player] " * 89 + "-> " + "[Player] " * 89,
                     level="#####\n#PP.#\n#####")
    compiled = compile_games([dict(name="2**89", text=explosion)])[0]["compiled"]
    e = m.Engine(); e.load_from_json(compiled); e.load_level(0, "explosion")
    for budget in (100_000_000, 1_000_000_000, 2**63 - 1):
        e.execution_budget = budget
        before = e.serialize_state() if m.HISTORY_ENABLED else snapshot(e, False)
        for repeat in range(2):
            try:
                e.process_input(4)
            except m.ExecutionBudgetExceeded as exc:
                assert "tuple bindings" in str(exc), str(exc)
            else:
                raise AssertionError("Explosive Cartesian product was accepted")
            after = e.serialize_state() if m.HISTORY_ENABLED else snapshot(e, False)
            assert before == after

    finite = game("right " + "[Player] " * 16 + "-> " + "[Player] " * 16,
                  level="#####\n#PP.#\n#####")
    medium = compile_games([dict(name="finite-product", text=finite)])[0]["compiled"]
    e = m.Engine(); e.load_from_json(medium); e.load_level(0, "finite")
    e.execution_budget = 1000
    try:
        e.process_input(4)
    except m.ExecutionBudgetExceeded:
        pass
    else:
        raise AssertionError("Small override did not reject a large product")
    e.execution_budget = m.DEFAULT_EXECUTION_BUDGET
    e.process_input(4)
    assert 1_000_000 < e.execution_work < e.execution_budget

    # Invalid limits do not change the configuration.
    for invalid in (0, -1, 2**63, 2**64-1):
        old = e.execution_budget
        try:
            e.execution_budget = invalid
        except (ValueError, TypeError, OverflowError):
            pass
        else:
            raise AssertionError("Accepted invalid budget")
        assert e.execution_budget == old

    # Exceptions in batch workers are rethrown on the caller, including reset.
    for startup in (False, True):
        value = json.loads(compiled)
        if startup:
            value["metadata"]["run_rules_on_level_start"] = True
        batch = m.BatchedEngine(4)
        batch.set_num_threads(2)
        batch.load_from_json(json.dumps(value))
        batch.set_levels([0]*4)
        if not startup:
            batch.reset_all()
        try:
            batch.reset_all() if startup else batch.step([4]*4)
        except m.ExecutionBudgetExceeded:
            pass
        else:
            raise AssertionError("Batch execution did not propagate resource failure")

    if m.HISTORY_ENABLED:
        e = m.Engine(); e.load_from_json(refs[0]["compiled"]); e.load_level(0, "v1")
        state = json.loads(e.serialize_state())
        e.execution_budget = 123456
        old = dict(state); old["version"] = 1; old.pop("execution_budget")
        e.restore_state(json.dumps(old))
        assert e.execution_budget == m.DEFAULT_EXECUTION_BUDGET
        state["execution_budget"] = 987654321
        e.restore_state(json.dumps(state))
        assert e.execution_budget == 987654321

    print(json.dumps(dict(history=m.HISTORY_ENABLED, parity_games=len(ordinary),
                          atomic_boundary_checks=checks, max_ordinary_work=max_work,
                          default_budget=m.DEFAULT_EXECUTION_BUDGET,
                          explosion_rejected=True, batch_exceptions=True)))


if __name__ == "__main__":
    main()
