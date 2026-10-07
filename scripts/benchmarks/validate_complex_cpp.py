"""Replay the expanded game's raw C++ boards/scores/wins against original JS."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.benchmarks.benchmark_paper_throughput import BENCHMARK_GAMES
from scripts.benchmarks.complex_games import COMPLEX_GAMES
from scripts.benchmarks.profile_rand_nodejs import _load_nodejs_native_game_text

NODE = r'''
const fs=require('fs'), e=require(process.argv[1]), solver=require(process.argv[2]);
const input=JSON.parse(fs.readFileSync(0,'utf8'));
e.compile(['loadLevel',0],input.source,'complex-validation');
solver.precalcDistances(e);
const compiled=e.serializeCompiledStateJSON();
const take=()=>({board:Array.from(e.backupLevel().dat),score:solver.getScore(e),won:!!e.getWinning()});
const states=[take()];
for(const action of input.actions){
 e.processInput(action);
 let again=0;
 while(e.getAgaining() && again<50){e.processInput(-1);again++;}
 states.push(take());
 // Compare through the first terminal state; avoid post-win UI transitions.
 if(e.getWinning())break;
}
console.log(JSON.stringify({compiled,states}));
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--games', nargs='+', default=BENCHMARK_GAMES + COMPLEX_GAMES)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    name = 'puzzlescript_cpp._puzzlescript_cpp'
    spec = importlib.util.spec_from_file_location(name, args.library)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    root = Path(__file__).resolve().parents[2]
    actions = list(range(5)) * 8 + np.random.default_rng(42).integers(0, 5, 88).tolist()
    result = {'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(), 'actions': actions,
              'protocol': 'level0; direct inputs with up to50 AGAIN ticks; raw board, native score and win; stop after first win',
              'source_hashes': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                                ['PuzzleScript/src/js/engine.js', 'PuzzleScript/src/js/compiler.js',
                                 'puzzlescript_nodejs/puzzlescript/engine.js', 'puzzlescript_nodejs/puzzlescript/solver.js',
                                 str(Path(__file__).relative_to(root))]}, 'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for game in args.games:
        row = {'game': game, 'passed': False}
        try:
            source = _load_nodejs_native_game_text(game)
            process = subprocess.run(['node', '-e', NODE, str(root / 'puzzlescript_nodejs/puzzlescript/engine.js'),
                                      str(root / 'puzzlescript_nodejs/puzzlescript/solver.js')],
                                     input=json.dumps({'source': source, 'actions': actions}), text=True,
                                     capture_output=True, check=True, timeout=180)
            expected = json.loads(process.stdout.splitlines()[-1])
            row['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
            row['compiled_sha256'] = hashlib.sha256(expected['compiled'].encode()).hexdigest()
            engine = module.Engine(); assert engine.load_from_json(expected['compiled'])
            engine.load_level(0, 'complex-validation')
            errors = []
            for step, state in enumerate(expected['states']):
                if step:
                    engine.process_input(actions[step - 1])
                    again = 0
                    while engine.is_againing() and again < 50:
                        engine.process_input(-1); again += 1
                fields = []
                if not np.array_equal(engine.get_objects(), state['board']): fields.append('board')
                if engine.get_score() != state['score']: fields.append('score')
                if engine.is_winning() != state['won']: fields.append('win')
                if fields: errors.append({'step': step, 'fields': fields})
            row.update(passed=not errors, compared_states=len(expected['states']), mismatches=errors)
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        result['results'].append(row)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    raise SystemExit(0 if all(r['passed'] for r in result['results']) else 1)


if __name__ == '__main__':
    main()
