"""Test a single game: C++ vs JS. Called as subprocess with game name argument.

Exit codes: 0=all OK, 1=failure, 2=error, 3=no solutions
Output format: JSON dict with success/fail/error counts and details.
"""
import json
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from javascript import require

JS_SOLS_DIR = 'data/js_sols'
SIMPLIFIED_GAMES_DIR = 'data/simplified_games'

MAX_AGAIN = 50
MAX_LEVELS = 3

def main(game_name):
    engine_path = os.path.join(os.path.dirname(__file__), 'puzzlescript_nodejs', 'puzzlescript', 'engine.js')
    js = require(engine_path)
    import puzzlescript_cpp._puzzlescript_cpp as ps

    game_path = os.path.join(SIMPLIFIED_GAMES_DIR, f'{game_name}_simplified.txt')
    if not os.path.isfile(game_path):
        print(json.dumps({"error": "no_simplified_file"}))
        sys.exit(2)
    
    with open(game_path) as f:
        game_text = f.read()
    
    try:
        js.compile(['restart'], game_text)
        json_str = str(js.serializeCompiledStateJSON())
    except Exception as e:
        print(json.dumps({"error": f"compile: {e}"}))
        sys.exit(2)
    
    engine = ps.Engine()
    if not engine.load_from_json(json_str):
        print(json.dumps({"error": "cpp_load_failed"}))
        sys.exit(2)
    
    sol_dir = os.path.join(JS_SOLS_DIR, game_name)
    sol_files = sorted(glob.glob(os.path.join(sol_dir, 'bfs_*level-*.json')))
    
    results = {"successes": 0, "failures": 0, "details": []}
    levels_tested = set()
    
    for sol_path in sol_files:
        if len(levels_tested) >= MAX_LEVELS:
            break
        
        basename = os.path.basename(sol_path)
        level_i = int(basename.split('level-')[-1].split('.')[0])
        if level_i in levels_tested:
            continue
        levels_tested.add(level_i)
        
        try:
            with open(sol_path) as f:
                sol = json.load(f)
            if not sol.get('won', False):
                continue
            actions = sol.get('actions', [])
            
            # JS replay
            js.compile(['loadLevel', level_i], game_text)
            for a in actions:
                js.processInput(a)
                ag = 0
                while bool(js.getAgaining()) and ag < MAX_AGAIN:
                    js.processInput(-1)
                    ag += 1
            js_final = list(js.getLevel()['objects'])
            js_won = bool(js.getWinning())
            
            # C++ replay
            engine.load_from_json(json_str)
            engine.load_level(level_i)
            for a in actions:
                engine.process_input(a)
                ag = 0
                while engine.is_againing() and ag < MAX_AGAIN:
                    engine.process_input(-1)
                    ag += 1
            cpp_final = list(engine.get_objects())
            cpp_won = engine.is_winning()
            
            if cpp_final == js_final and cpp_won == js_won:
                results["successes"] += 1
            else:
                results["failures"] += 1
                detail = {"level": level_i, "actions": len(actions)}
                if cpp_final != js_final:
                    n_diffs = sum(1 for a, b in zip(cpp_final, js_final) if a != b)
                    detail["state_diffs"] = n_diffs
                if cpp_won != js_won:
                    detail["win_mismatch"] = f"JS={js_won} C++={cpp_won}"
                results["details"].append(detail)
        except Exception as e:
            results["failures"] += 1
            results["details"].append({"level": level_i, "error": str(e)})
    
    print(json.dumps(results))
    sys.exit(1 if results["failures"] > 0 else 0)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_cpp_single.py <game_name>")
        sys.exit(2)
    main(sys.argv[1])
