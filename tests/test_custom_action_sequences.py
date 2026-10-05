import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path

import pytest

from backends import NodeJSPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_jax.env import PuzzleJaxEnv
from puzzlescript_jax.env_switch import PuzzleJaxEnvSwitch
from puzzlejax.validate_actions import (
    ACTION_SEQUENCES_DIR,
    discover_test_cases,
    get_action_file_comments,
    get_action_file_xfail_reason,
    parse_action_file,
    run_test_case,
)


def _case_label(game_name, level_i, seq_id):
    return f"{game_name}/level_{level_i}_actions_{seq_id}"


def _case_id(case):
    game_name, level_i, seq_id, path = case
    comments = get_action_file_comments(path)
    if comments:
        return f"{_case_label(game_name, level_i, seq_id)} [{comments[0]}]"
    return _case_label(game_name, level_i, seq_id)


ACTION_SEQUENCE_CASES = discover_test_cases(ACTION_SEQUENCES_DIR)


@pytest.fixture(scope="session")
def parser():
    return init_ps_lark_parser()


@pytest.fixture(scope="session")
def backend():
    backend = NodeJSPuzzleScriptBackend()
    yield backend
    backend.unload_game()


@pytest.mark.parametrize(
    ("game_name", "level_i", "seq_id", "path", "env_cls"),
    [
        pytest.param(*case, cls, id=f"{_case_id(case)}-{name}")
        for case in ACTION_SEQUENCE_CASES
        for name, cls in (
            # Keep the full standard suite and focused switch regressions.
            # Large switch replays can take tens of minutes;
            # validate_actions --backend switch opts into that full run.
            [("standard", PuzzleJaxEnv), ("switch", PuzzleJaxEnvSwitch)]
            if case[0].startswith("test_") or case[0] == "sokoban_basic"
            else [("standard", PuzzleJaxEnv)]
        )
    ],
)
def test_custom_action_sequences_match_js(game_name, level_i, seq_id, path, backend, parser, env_cls):
    label = _case_label(game_name, level_i, seq_id)
    comments = get_action_file_comments(path)
    xfail_reason = get_action_file_xfail_reason(path)
    details = f"\nNotes: {' | '.join(comments)}" if comments else ""

    assert Path(path).is_file(), label

    js_actions = parse_action_file(path)
    ok, message = run_test_case(game_name, level_i, js_actions, backend, parser, env_cls=env_cls)

    if xfail_reason:
        if ok:
            pytest.fail(f"{label} unexpectedly passed; remove xfail marker. Notes: {xfail_reason}{details}")
        pytest.xfail(f"{label}: {xfail_reason}\nObserved mismatch: {message}{details}")

    assert ok, f"{label} failed: {message}{details}"
