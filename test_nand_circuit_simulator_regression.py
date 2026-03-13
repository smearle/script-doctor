import json
from pathlib import Path

import jax
import numpy as np
import pytest

from backends.nodejs import NodeJSPuzzleScriptBackend
from puzzlescript_jax.env import PJParams, PuzzleJaxEnv
from puzzlescript_jax.preprocessing import PJParseErrors, get_tree_from_txt
from puzzlescript_jax.utils import init_ps_lark_parser, level_to_int_arr, to_binary_vectors
from puzzlescript_nodejs.utils import replay_actions_js


GAME_NAME = "NAND_Circuit_Simulator"
JS_TO_JAX_ACTIONS = [3, 0, 1, 2, 4]


def _js_level_to_target_multihot(level, obj_names, target_obj_names):
    level_arr = level_to_int_arr(level, len(obj_names))
    multihot = to_binary_vectors(level_arr.T, len(obj_names))
    multihot = np.transpose(multihot, (2, 0, 1))[::-1]

    obj_names_to_idx = {str(name).lower(): i for i, name in enumerate(obj_names)}
    target = np.zeros((len(target_obj_names), *multihot.shape[1:]), dtype=bool)
    for target_idx, obj_name in enumerate(target_obj_names):
        obj_idx = obj_names_to_idx.get(str(obj_name).lower())
        if obj_idx is not None:
            target[target_idx] = multihot[obj_idx]
    return target


@pytest.mark.parametrize("level_i", [1, 4])
def test_nand_circuit_simulator_matches_js_terminal_state(level_i):
    parser = init_ps_lark_parser()
    tree, success, err_msg = get_tree_from_txt(parser, GAME_NAME, test_env_init=False)
    assert success == PJParseErrors.SUCCESS, err_msg

    env = PuzzleJaxEnv(tree, jit=True, debug=False, print_score=False, level_i=level_i)
    params = PJParams(level=env.get_level(level_i), level_i=level_i)

    sol_path = Path("data/js_sols") / GAME_NAME / f"bfs_100000-steps_level-{level_i}.json"

    with sol_path.open("r", encoding="utf-8") as f:
        sol = json.load(f)

    backend = NodeJSPuzzleScriptBackend()
    try:
        game_text = backend.compile_game(parser, GAME_NAME)
        _, js_states = replay_actions_js(
            backend.engine,
            backend.solver,
            sol["actions"],
            game_text,
            level_i,
        )
    finally:
        backend.unload_game()

    expected = _js_level_to_target_multihot(
        js_states[-1],
        sol["objs"],
        env.atomic_obj_names,
    )

    _, state = env.reset(jax.random.PRNGKey(0), params)
    for action in sol["actions"]:
        _, state, _, _, _ = env.step_env(
            jax.random.PRNGKey(0),
            state,
            JS_TO_JAX_ACTIONS[action],
            params,
        )

    assert np.array_equal(np.asarray(state.multihot_level), expected)
