"""Debug a specific game: find first step where C++ diverges from JS."""
import json
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from javascript import require

engine_path = os.path.join(os.path.dirname(__file__), 'puzzlescript_nodejs', 'puzzlescript', 'engine.js')
js = require(engine_path)

import puzzlescript_cpp._puzzlescript_cpp as ps

JS_SOLS_DIR = 'data/js_sols'
SIMPLIFIED_GAMES_DIR = 'data/simplified_games'
MAX_AGAIN = 50

def debug_game(game_name, level_i=0):
    game_path = os.path.join(SIMPLIFIED_GAMES_DIR, f'{game_name}_simplified.txt')
    with open(game_path) as f:
        game_text = f.read()
    
    # Compile and serialize
    js.compile(['restart'], game_text)
    json_str = str(js.serializeCompiledStateJSON())
    data = json.loads(json_str)
    
    print(f"Game: {game_name}")
    print(f"Objects: {data['objectCount']}, Layers: {data['layerCount']}")
    print(f"STRIDE_OBJ: {data['STRIDE_OBJ']}, STRIDE_MOV: {data['STRIDE_MOV']}")
    print(f"Rules: {len(data['rules'])} groups, Late: {len(data['lateRules'])} groups")
    print(f"idDict: {data['idDict']}")
    
    # Find solution
    sol_dir = os.path.join(JS_SOLS_DIR, game_name)
    sol_files = sorted(glob.glob(os.path.join(sol_dir, f'bfs_*level-{level_i}.json')))
    if not sol_files:
        print(f"No solution for level {level_i}")
        return
    
    with open(sol_files[0]) as f:
        sol = json.load(f)
    actions = sol.get('actions', [])
    print(f"Solution: {len(actions)} actions: {actions}")
    
    # Setup both engines
    js.compile(['loadLevel', level_i], game_text)
    
    engine = ps.Engine()
    engine.load_from_json(json_str)
    engine.load_level(level_i)
    
    # Compare initial state
    js_objs = list(js.getLevel()['objects'])
    cpp_objs = list(engine.get_objects())
    
    if js_objs != cpp_objs:
        print("\nINITIAL STATE MISMATCH!")
        return
    print(f"\nInitial state: OK (size={len(js_objs)})")
    
    # Step through actions
    for step, action in enumerate(actions):
        # JS step
        js.processInput(action)
        ag = 0
        while bool(js.getAgaining()) and ag < MAX_AGAIN:
            js.processInput(-1)
            ag += 1
        js_objs = list(js.getLevel()['objects'])
        js_won = bool(js.getWinning())
        
        # C++ step
        engine.process_input(action)
        ag = 0
        while engine.is_againing() and ag < MAX_AGAIN:
            engine.process_input(-1)
            ag += 1
        cpp_objs = list(engine.get_objects())
        cpp_won = engine.is_winning()
        
        if js_objs != cpp_objs or js_won != cpp_won:
            diffs = [(i, cpp_objs[i], js_objs[i]) 
                     for i in range(min(len(cpp_objs), len(js_objs)))
                     if cpp_objs[i] != js_objs[i]]
            print(f"\nDIVERGENCE at step {step} (action={action})!")
            print(f"  JS won: {js_won}, C++ won: {cpp_won}")
            print(f"  {len(diffs)} cell diffs:")
            
            width = engine.get_width()
            stride = data['STRIDE_OBJ']
            for idx, cv, jv in diffs[:20]:
                cell = idx // stride
                word = idx % stride
                row = cell // width
                col = cell % width
                # Show bit differences
                diff_bits = cv ^ jv
                print(f"    cell ({row},{col}) word {word}: C++={cv:#010x} JS={jv:#010x} diff={diff_bits:#010x}")
                # Decode which objects differ
                for bit in range(min(32, len(data['idDict']) - word * 32)):
                    if diff_bits & (1 << bit):
                        obj_id = word * 32 + bit
                        obj_name = data['idDict'][obj_id] if obj_id < len(data['idDict']) else f"obj{obj_id}"
                        in_cpp = "present" if cv & (1 << bit) else "absent"
                        in_js = "present" if jv & (1 << bit) else "absent"
                        print(f"      {obj_name}: C++={in_cpp}, JS={in_js}")
            return
        
        status = f"step {step}: action={action} ok"
        if js_won:
            status += " WON!"
        sys.stdout.write(f"\r  {status}")
        sys.stdout.flush()
    
    print(f"\n\nAll {len(actions)} steps match! Final won: JS={js_won}, C++={cpp_won}")

if __name__ == '__main__':
    game = sys.argv[1] if len(sys.argv) > 1 else 'sokoban_basic'
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    debug_game(game, level)
