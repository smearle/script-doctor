"""Test the gym-style C++ PuzzleScript environment interfaces."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puzzlejax.globals import SIMPLIFIED_GAMES_DIR, JS_SOLS_DIR
from javascript import require


def get_json_str(game="sokoban_basic"):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(root_dir, "puzzlescript_nodejs", "puzzlescript", "engine.js")
    js_engine = require(engine_path)

    game_path = os.path.join(SIMPLIFIED_GAMES_DIR, f"{game}_simplified.txt")
    with open(game_path, "r") as f:
        game_text = f.read()
    js_engine.compile(["restart"], game_text)
    return str(js_engine.serializeCompiledStateJSON()), game


def test_single_env():
    json_str, game = get_json_str()
    from puzzlescript_cpp import CppPuzzleScriptEnv

    env = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=100)
    print(f"Single env:")
    print(f"  obs_shape: {env.observation_shape}")
    print(f"  num_actions: {env.num_actions}")
    print(f"  num_levels: {env.num_levels}")

    obs, info = env.reset()
    print(f"  reset obs shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == env.observation_shape
    assert obs.dtype == np.uint8
    assert obs.sum() > 0, "obs should have some objects"

    # Take a step
    obs2, reward, done, truncated, info = env.step(0)  # up
    print(f"  step: reward={reward:.3f}, done={done}, truncated={truncated}")
    assert obs2.shape == env.observation_shape

    # Replay a known solution
    sol_dir = os.path.join(JS_SOLS_DIR, game)
    sol_files = sorted(glob.glob(os.path.join(sol_dir, "*level-0.json")))
    with open(sol_files[0]) as f:
        sol = json.load(f)
    actions = sol.get("actions", [])

    obs, _ = env.reset()
    for i, a in enumerate(actions):
        obs, reward, done, truncated, info = env.step(a)
        if done:
            print(f"  Won at step {i+1}! reward={reward:.3f}")
            assert info["won"]
            break
    else:
        raise AssertionError(f"Did not win after {len(actions)} steps")

    print("  PASSED\n")


def test_batched_env():
    json_str, game = get_json_str()
    from puzzlescript_cpp import CppBatchedPuzzleScriptEnv

    batch_size = 8
    benv = CppBatchedPuzzleScriptEnv(
        json_str, batch_size=batch_size,
        level_indices=[0] * batch_size, max_episode_steps=200
    )
    print(f"Batched env (batch_size={batch_size}):")
    print(f"  obs_shape: {benv.observation_shape}")
    print(f"  num_actions: {benv.num_actions}")

    obs = benv.reset()
    print(f"  reset obs shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape[0] == batch_size
    assert obs.dtype == np.uint8
    assert obs.sum() > 0

    # All envs should have identical initial obs
    for i in range(1, batch_size):
        assert np.array_equal(obs[0], obs[i]), f"env 0 != env {i}"
    print("  All envs identical after reset: OK")

    # Take a step (different actions)
    actions = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)
    obs, rewards, dones, truncated, infos = benv.step(actions)
    print(f"  step obs shape: {obs.shape}")
    print(f"  rewards: {rewards}")
    print(f"  dones: {dones}")
    assert obs.shape[0] == batch_size
    assert rewards.shape == (batch_size,)
    assert dones.shape == (batch_size,)

    # Replay solution on all envs
    sol_dir = os.path.join(JS_SOLS_DIR, game)
    sol_files = sorted(glob.glob(os.path.join(sol_dir, "*level-0.json")))
    with open(sol_files[0]) as f:
        sol = json.load(f)
    action_list = sol.get("actions", [])

    obs = benv.reset()
    won_step = None
    for i, a in enumerate(action_list):
        actions_batch = np.full(batch_size, a, dtype=np.int32)
        obs, rewards, dones, truncated, infos = benv.step(actions_batch)
        if dones.any():
            won_envs = np.where(dones)[0]
            print(f"  Envs {won_envs.tolist()} won at step {i+1}")
            won_step = i + 1
            # After auto-reset, envs should have fresh obs
            break

    assert won_step is not None, f"No env won after {len(action_list)} steps"

    # Test partial reset
    obs_before = benv.reset()
    # Step once to change state
    benv.step(np.zeros(batch_size, dtype=np.int32))
    # Reset only envs 0, 2, 4
    obs_after = benv.reset(env_indices=[0, 2, 4])
    assert np.array_equal(obs_after[0], obs_before[0]), "env 0 should be reset"
    assert np.array_equal(obs_after[2], obs_before[2]), "env 2 should be reset"
    print("  Partial reset: OK")

    print("  PASSED\n")


def test_obs_consistency():
    """Verify batched obs matches single-env obs."""
    json_str, _ = get_json_str()
    from puzzlescript_cpp import CppPuzzleScriptEnv, CppBatchedPuzzleScriptEnv

    env = CppPuzzleScriptEnv(json_str, level_i=0)
    benv = CppBatchedPuzzleScriptEnv(json_str, batch_size=4, level_indices=[0]*4)

    obs_single, _ = env.reset()
    obs_batch = benv.reset()

    assert np.array_equal(obs_single, obs_batch[0]), \
        "Single env obs should match batched env obs[0]"
    print("Obs consistency: PASSED\n")


if __name__ == "__main__":
    test_single_env()
    test_batched_env()
    test_obs_consistency()
    print("All gym interface tests passed!")
