"""Binary worker IPC preserves batch observations and episode bookkeeping."""
from pathlib import Path

import numpy as np
import pytest

from puzzlescript_nodejs.rl_env import _NodeJSBatchedController


@pytest.mark.parametrize("auto_reset", [True, False])
def test_serializers_agree_across_truncation_and_partial_reset(auto_reset):
    game = Path("custom_games/test_push_chain.txt").read_text()
    game = game.rsplit("LEVELS", 1)[0] + "LEVELS\n=======\n\n######\n#P*O.#\n######\n"
    controllers = []
    try:
        for serialization, reuse_score in (("json", False), ("json", True), ("advanced", True)):
            controllers.append(_NodeJSBatchedController(
                game_text=game, level_i=0, n_envs=3, max_episode_steps=3,
                auto_reset=auto_reset, ipc_serialization=serialization, reuse_score=reuse_score))
        actions = np.random.default_rng(17).integers(0, 5, (16, 3))
        actions[0] = [3, 2, 4]  # First environment wins; other environments continue.
        saw_win = saw_truncation = False
        for i, action in enumerate(actions):
            if i == 7:
                expected = controllers[0].reset([1])
                for controller in controllers[1:]:
                    np.testing.assert_array_equal(expected, controller.reset([1]))
            a, *others = [controller.step(action) for controller in controllers]
            saw_win |= bool(a[4]["won"].any())
            saw_truncation |= bool(a[3].any())
            for b in others:
                for left, right in zip(a[:4], b[:4]):
                    np.testing.assert_array_equal(left, right)
                assert a[4].keys() == b[4].keys()
                for key in a[4]:
                    np.testing.assert_array_equal(a[4][key], b[4][key])
        assert saw_win and saw_truncation
    finally:
        for controller in controllers:
            controller.close()
