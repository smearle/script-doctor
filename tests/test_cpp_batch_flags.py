"""Parallel result flags must have independent storage for each environment."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from puzzlescript_cpp import CppBatchedPuzzleScriptEnv


@pytest.fixture(scope="module")
def one_push_game():
    root = Path(__file__).resolve().parents[1]
    source = (root / "custom_games/test_push_chain.txt").read_text()
    source = source.rsplit("LEVELS", 1)[0] + "LEVELS\n=======\n\n######\n#P*O.#\n######\n"
    script = "const fs=require('fs'),e=require(process.argv[1]);e.compile(['loadLevel',0],fs.readFileSync(0,'utf8'));console.log(e.serializeCompiledStateJSON());"
    result = subprocess.run(["node", "-e", script, str(root / "puzzlescript_nodejs/puzzlescript/engine.js")],
                            input=source, text=True, capture_output=True, check=True)
    compiled = result.stdout.splitlines()[-1]
    json.loads(compiled)
    return compiled


@pytest.mark.parametrize("batch", [32, 65, 128])
def test_parallel_mixed_wins_and_auto_reset_match_serial(one_push_game, batch):
    serial, parallel = [CppBatchedPuzzleScriptEnv(one_push_game, batch, level_indices=[0] * batch,
                                                num_threads=threads, max_episode_steps=10000)
                        for threads in (1, 8)]
    rng = np.random.default_rng(7)
    for _ in range(128):
        serial.reset()
        parallel.reset()
        actions = np.where(rng.integers(0, 2, batch), 3, 4).astype(np.int32)
        left, right = serial.step(actions), parallel.step(actions)
        np.testing.assert_array_equal(left[2], actions == 3)
        for a, b in zip(left[:4], right[:4]):
            np.testing.assert_array_equal(a, b)
        for key in left[4]:
            np.testing.assert_array_equal(left[4][key], right[4][key])
