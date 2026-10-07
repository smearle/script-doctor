import json
from pathlib import Path
from backends import NodeJSPuzzleScriptBackend
from puzzlejax.validate_actions import run_test_case
from puzzlescript_jax.utils import init_ps_lark_parser
ok, message = run_test_case('overlap_layer_clear', 0, [4], NodeJSPuzzleScriptBackend(), init_ps_lark_parser())
result = {'passed': ok, 'message': message, 'actions': [4]}
print(json.dumps(result), flush=True)
Path('results/overlap-layer-clear.json').write_text(json.dumps(result, indent=2) + '\n')
