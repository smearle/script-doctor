#!/usr/bin/env python
"""Validate custom action sequences against both NodeJS and JAX PuzzleScript backends.

Quick-start for creating test cases:
    1. Create a file: custom_action_sequences/GAME_NAME/level_N_actions_M
    2. Put one action per line: up, left, down, right, action (or 0-4)
    3. Run: python validate_actions.py
    4. Or for one game: python validate_actions.py --game GAME_NAME

File format (one action per line, # comments allowed):
    up
    right
    down
    action

Or equivalently with JS engine action indices:
    0
    3
    2
    4

Numeric mapping: 0=up, 1=left, 2=down, 3=right, 4=action, 5=undo, 6=restart

Comment directives:
    # xfail: expected mismatch reason
"""

import argparse
import json
import os
import re
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from lark import Lark

from backends import NodeJSPuzzleScriptBackend
from puzzlescript_jax.env import PJParams, PuzzleJaxEnv
from puzzlescript_jax.env_utils import multihot_to_desc
from puzzlescript_jax.globals import JS_TO_JAX_ACTIONS, LARK_SYNTAX_PATH
from puzzlescript_jax.preprocessing import PJParseErrors, get_tree_from_txt
from puzzlescript_nodejs.utils import replay_actions_js
from validate_sols_jax import multihot_level_from_js_state

ACTION_SEQUENCES_DIR = "custom_action_sequences"
VALIDATED_ACTION_SEQUENCES_DIR = os.path.join("data", "jax_validated_action_sequences")

JS_ACTION_NAMES = {
    "up": 0,
    "left": 1,
    "down": 2,
    "right": 3,
    "action": 4,
    "undo": 5,
    "restart": 6,
}

JS_ACTION_ID_TO_NAME = {v: k for k, v in JS_ACTION_NAMES.items()}


def get_action_file_metadata(path):
    """Return extracted metadata from an action file."""
    with open(path, "r", encoding="utf-8") as f:
        comments = []
        xfail_reason = None
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("#"):
                comment = line[1:].strip()
                comments.append(comment)
                if xfail_reason is None and comment.lower().startswith("xfail:"):
                    xfail_reason = comment.split(":", 1)[1].strip() or "expected mismatch"
        return {
            "comments": comments,
            "xfail_reason": xfail_reason,
        }


def get_action_file_comments(path):
    """Return cleaned comment lines from an action file."""
    return get_action_file_metadata(path)["comments"]


def get_action_file_xfail_reason(path):
    """Return expected-failure reason, if declared in comments."""
    return get_action_file_metadata(path)["xfail_reason"]


def parse_action_file(path):
    """Parse an action file. One action per line: integer (0-6) or name.

    Returns a list of JS action indices (ints).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    content_lines = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        content_lines.append(line)

    text = "\n".join(content_lines).strip()

    # Support JSON format: [0, 1, 2, 3]
    if text.startswith("["):
        return json.loads(text)

    actions = []
    for line in content_lines:
        token = line.lower()
        if token in JS_ACTION_NAMES:
            actions.append(JS_ACTION_NAMES[token])
        else:
            actions.append(int(token))
    return actions


def discover_test_cases(base_dir, game_filter=None):
    """Find all test cases under base_dir.

    Returns list of (game_name, level_i, seq_id, file_path).
    """
    cases = []
    if not os.path.isdir(base_dir):
        return cases
    for game_name in sorted(os.listdir(base_dir)):
        game_dir = os.path.join(base_dir, game_name)
        if not os.path.isdir(game_dir):
            continue
        if game_filter and game_name != game_filter:
            continue
        for fname in sorted(os.listdir(game_dir)):
            m = re.match(r"level_(\d+)_actions_(\d+)", fname)
            if not m:
                continue
            level_i = int(m.group(1))
            seq_id = int(m.group(2))
            cases.append((game_name, level_i, seq_id, os.path.join(game_dir, fname)))
    return cases


def format_jax_state(state, env):
    """Human-readable JAX state description."""
    desc = multihot_to_desc(
        np.asarray(state.multihot_level),
        env.objs_to_idxs,
        env.n_objs,
        env.obj_idxs_to_force_idxs,
    )
    return (
        f"  win={bool(state.win)}, heuristic={int(state.heuristic)}, "
        f"score={int(state.score)}, step={int(state.step_i)}\n{desc}"
    )


def format_actions(js_actions):
    """Pretty-print action list."""
    names = [JS_ACTION_ID_TO_NAME.get(a, str(a)) for a in js_actions]
    return "[" + ", ".join(names) + "]"


def write_case_result(output_dir, case_result):
    """Persist a single validation result to disk."""
    game_dir = os.path.join(output_dir, case_result["game"])
    os.makedirs(game_dir, exist_ok=True)
    base_name = f"level_{case_result['level']}_actions_{case_result['seq_id']}"
    output_path = os.path.join(game_dir, f"{base_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(case_result, f, indent=2)


def write_summary(output_dir, summary):
    """Persist the run summary to disk."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def compare_states_at_step(
    step_i,
    js_state,
    js_score,
    js_winning,
    jax_state,
    env,
    obj_list,
):
    """Compare JS and JAX states at a single step.

    Returns (ok: bool, error_msg: str|None).
    """
    # Convert JS state to multihot for comparison.
    multihot_js = multihot_level_from_js_state(
        js_state, obj_list, target_obj_names=env.atomic_obj_names,
    )
    multihot_jax = np.asarray(jax_state.multihot_level)

    errors = []

    # Check win.
    jax_win = bool(jax_state.win)
    if js_winning != jax_win:
        errors.append(f"win mismatch: JS={js_winning}, JAX={jax_win}")

    # Check score / heuristic.
    jax_heuristic = int(jax_state.heuristic)
    if -js_score != jax_heuristic:
        errors.append(f"score mismatch: JS score={js_score} (neg={-js_score}), JAX heuristic={jax_heuristic}")

    # Check level state.
    if multihot_js.shape != multihot_jax.shape:
        errors.append(
            f"shape mismatch: JS={multihot_js.shape}, JAX={multihot_jax.shape}"
        )
    elif not np.array_equal(multihot_js, multihot_jax):
        diff_channels = np.where(np.any(multihot_js != multihot_jax, axis=(1, 2)))[0]
        obj_names = list(env.objs_to_idxs.keys())
        diff_names = [obj_names[c] if c < len(obj_names) else f"ch{c}" for c in diff_channels]
        errors.append(f"level state mismatch on channels: {diff_names}")
        for c in diff_channels:
            name = obj_names[c] if c < len(obj_names) else f"ch{c}"
            js_pos = list(zip(*np.where(multihot_js[c])))
            jax_pos = list(zip(*np.where(multihot_jax[c])))
            errors.append(f"  {name}: JS={js_pos} JAX={jax_pos}")

    if errors:
        return False, "; ".join(errors)
    return True, None


def run_test_case(game_name, level_i, js_actions, backend, parser, compare_per_step=True):
    """Run a single test case through both backends and compare.

    Returns (success: bool, message: str).
    """
    engine = backend.engine
    solver = backend.solver

    # --- Compile game for JS ---
    try:
        game_text = backend.compile_game(parser, game_name)
    except Exception:
        return False, f"JS compile failed:\n{traceback.format_exc()}"

    # --- Load game for JAX ---
    tree, status, err_msg = get_tree_from_txt(
        parser, game_name, test_env_init=False, timeout=60 * 5,
    )
    if status != PJParseErrors.SUCCESS:
        return False, f"JAX parse failed ({status}): {err_msg}"

    try:
        env = PuzzleJaxEnv(tree, debug=False, print_score=False, level_i=level_i)
    except Exception:
        return False, f"JAX env init failed:\n{traceback.format_exc()}"

    level = env.get_level(level_i)
    if level is None:
        return False, f"Level {level_i} not found in game"
    params = PJParams(level=level, level_i=level_i)

    # --- Replay in JS ---
    try:
        js_scores, js_states, js_winning = replay_actions_js(
            engine, solver, js_actions, game_text, level_i,
            stop_on_win=False, return_winning=True,
        )
    except Exception:
        return False, f"JS replay failed:\n{traceback.format_exc()}"

    # Get obj list from JS engine for state conversion.
    obj_list = list(engine.getState().idDict)

    # --- Replay in JAX ---
    # JAX env doesn't support undo (5) or restart (6). Reject if present.
    unsupported = [a for a in js_actions if a > 4]
    if unsupported:
        return False, f"Unsupported actions for JAX comparison: {unsupported} (undo=5, restart=6 not supported)"

    key = jax.random.PRNGKey(0)
    jax_actions = [JS_TO_JAX_ACTIONS[a] for a in js_actions]
    jax_actions_arr = jnp.array([int(a) for a in jax_actions], dtype=jnp.int32)

    try:
        _, init_state = env.reset(key, params)

        def step_fn(state, action):
            _, new_state, reward, done, info = env.step_env(key, state, action, params)
            return new_state, new_state

        if len(jax_actions_arr) > 0:
            final_state, jax_states = jax.lax.scan(step_fn, init_state, jax_actions_arr)
            # Prepend initial state.
            jax_states = jax.tree.map(
                lambda init, rest: jnp.concatenate([init[None], rest]),
                init_state, jax_states,
            )
        else:
            final_state = init_state
            jax_states = jax.tree.map(lambda x: x[None], init_state)
    except Exception:
        return False, f"JAX replay failed:\n{traceback.format_exc()}"

    # --- Compare states ---
    # JS states includes init + one per action (but may be shorter if stop_on_win).
    n_steps = min(len(js_states), jax_states.multihot_level.shape[0])

    errors = []
    for step in range(n_steps):
        js_state_step = js_states[step]
        jax_state_step = jax.tree.map(lambda x: x[step], jax_states)
        ok, err = compare_states_at_step(
            step, js_state_step, js_scores[step], js_winning[step],
            jax_state_step, env, obj_list,
        )
        if not ok:
            label = "init" if step == 0 else f"step {step}"
            errors.append(f"[{label}] {err}")
            if not compare_per_step:
                break

    jax.clear_caches()

    if errors:
        detail = "\n  ".join(errors)
        return False, f"{len(errors)} step(s) with mismatches:\n  {detail}"

    return True, "all steps match"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", type=str, default=None, help="Only run tests for this game")
    ap.add_argument("--dir", type=str, default=ACTION_SEQUENCES_DIR, help="Action sequences directory")
    ap.add_argument(
        "--output-dir",
        type=str,
        default=VALIDATED_ACTION_SEQUENCES_DIR,
        help="Directory for persisted validation results",
    )
    ap.add_argument("--first-error-only", action="store_true", help="Stop at first mismatch per test case")
    ap.add_argument("--fail-fast", action="store_true", help="Stop after first failing test case")
    args = ap.parse_args()

    cases = discover_test_cases(args.dir, game_filter=args.game)
    if not cases:
        print(f"No test cases found in {args.dir}/")
        if args.game:
            print(f"  (filtered to game={args.game})")
        print(f"\nTo create a test case, add a file like:")
        print(f"  {args.dir}/Sokoban_Basic/level_0_actions_0")
        print(f"with one action per line (up/left/down/right/action or 0-4).")
        sys.exit(0)

    # Initialize backends once.
    print("Initializing backends...")
    backend = NodeJSPuzzleScriptBackend()
    with open(LARK_SYNTAX_PATH, "r", encoding="utf-8") as f:
        grammar = f.read()
    parser = Lark(grammar, start="ps_game", maybe_placeholders=False)

    n_pass = 0
    n_fail = 0
    n_error = 0
    n_xfail = 0
    n_xpass = 0
    failures = []
    case_results = []

    os.makedirs(args.output_dir, exist_ok=True)

    for game_name, level_i, seq_id, fpath in cases:
        label = f"{game_name}/level_{level_i}_actions_{seq_id}"
        case_result = {
            "game": game_name,
            "level": level_i,
            "seq_id": seq_id,
            "source_path": fpath,
        }
        xfail_reason = get_action_file_xfail_reason(fpath)
        if xfail_reason:
            case_result["xfail_reason"] = xfail_reason
        try:
            js_actions = parse_action_file(fpath)
            case_result["actions"] = js_actions
            case_result["action_names"] = [JS_ACTION_ID_TO_NAME.get(a, str(a)) for a in js_actions]
        except Exception as e:
            print(f"  ERROR {label}: cannot parse action file: {e}")
            n_error += 1
            case_result["status"] = "error"
            case_result["message"] = f"cannot parse action file: {e}"
            case_results.append(case_result)
            write_case_result(args.output_dir, case_result)
            continue

        print(f"  RUN  {label}  actions={format_actions(js_actions)}")

        try:
            ok, msg = run_test_case(
                game_name, level_i, js_actions, backend, parser,
                compare_per_step=not args.first_error_only,
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            ok = False
            msg = f"unexpected error:\n{traceback.format_exc()}"

        if ok and not xfail_reason:
            print(f"  PASS {label}: {msg}")
            n_pass += 1
            case_result["status"] = "pass"
        elif ok and xfail_reason:
            print(f"  XPASS {label}: {xfail_reason}")
            n_xpass += 1
            n_fail += 1
            failures.append(label)
            case_result["status"] = "xpass"
        elif xfail_reason:
            print(f"  XFAIL {label}: {xfail_reason}")
            n_xfail += 1
            case_result["status"] = "xfail"
        else:
            print(f"  FAIL {label}: {msg}")
            n_fail += 1
            failures.append(label)
            case_result["status"] = "fail"
        case_result["message"] = msg
        case_results.append(case_result)
        write_case_result(args.output_dir, case_result)

        if not ok and args.fail_fast:
            break

    print(f"\n{'='*60}")
    print(f"Results: {n_pass} passed, {n_fail} failed, {n_error} errors, {n_xfail} xfailed, {n_xpass} xpassed")
    summary = {
        "output_dir": args.output_dir,
        "source_dir": args.dir,
        "game_filter": args.game,
        "stats": {
            "passed": n_pass,
            "failed": n_fail,
            "errors": n_error,
            "xfailed": n_xfail,
            "xpassed": n_xpass,
            "total": len(case_results),
        },
        "failures": failures,
        "cases": case_results,
    }
    write_summary(args.output_dir, summary)
    print(f"Saved validation results to {args.output_dir}")
    if failures:
        print("Failed tests:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    main()
