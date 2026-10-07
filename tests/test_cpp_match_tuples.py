"""Compare overlapping and Cartesian rule matches with the original JS engine."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from puzzlescript_cpp import _cpp


def game_text(rule, board):
    return f"""title Tuple order regression
text_color #aabbcc
background_color #112233

OBJECTS

Background
black

A
red

B
blue

C
green

Player
white

LEGEND
. = Background
p = Player

SOUNDS

COLLISIONLAYERS
Background
A, B, C
Player

RULES
{rule}

WINCONDITIONS

LEVELS
{board}
"""


@pytest.mark.parametrize("rule,board", [
    ("right [ A | A ] -> [ B | B ]", "paaaa"),
    ("right [ A | A ] [ A ] -> [ B | B ] [ C ]", "paa.aaa"),
    ("right [ A ] [ A | A ] -> [ C ] [ B | B ]", "paa.aaa"),
    ("right [ A ] [ B ] [ C ] -> [ B ] [ C ] [ A ]", "paabbcc"),
    ("right [ A | ... | B ] [ C ] -> [ B | ... | C ] [ A ]", "paa.bbcc"),
    ("random [ A ] -> [ B ]", "paaaa"),
    ("random [ A ] [ B ] -> [ C ] [ A ]", "paaa.bbb"),
])
def test_streamed_matches_preserve_order_and_recheck_stale_matches(rule, board):
    root = Path(__file__).resolve().parents[1]
    script = """
const fs=require('fs'), e=require(process.argv[1]);
e.compile(['loadLevel',0],fs.readFileSync(0,'utf8'),'tuple-order');
const compiled=e.serializeCompiledStateJSON();
const states=[Array.from(e.backupLevel().dat)];
for(let i=0;i<4;i++){e.processInput(-1);states.push(Array.from(e.backupLevel().dat));}
console.log(JSON.stringify({compiled,states}));
"""
    result = subprocess.run(["node", "-e", script, str(root / "puzzlescript_nodejs/puzzlescript/engine.js")],
                            input=game_text(rule, board), text=True, capture_output=True, check=True, timeout=30)
    expected = json.loads(result.stdout.splitlines()[-1])
    engine = _cpp.Engine()
    assert engine.load_from_json(expected["compiled"])
    engine.load_level(0, "tuple-order")
    np.testing.assert_array_equal(engine.get_objects(), expected["states"][0])
    for state in expected["states"][1:]:
        engine.process_input(-1)
        np.testing.assert_array_equal(engine.get_objects(), state)
