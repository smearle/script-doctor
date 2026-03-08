"""Quick end-to-end test: JS compile -> serialize -> C++ load -> step -> compare."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from javascript import require

# Load JS engine
engine_path = os.path.join(os.path.dirname(__file__), 'puzzlescript_nodejs', 'puzzlescript', 'engine.js')
js = require(engine_path)

# Load sokoban_basic simplified
game_path = 'data/simplified_games/sokoban_basic_simplified.txt'
with open(game_path, 'r') as f:
    game_text = f.read()

print("1. Compiling game via JS...")
js.compile(['restart'], game_text)
print("   OK")

print("2. Serializing compiled state...")
json_str = str(js.serializeCompiledStateJSON())
print(f"   JSON size: {len(json_str)} bytes")

# Save for debugging
with open('/tmp/sokoban_basic_compiled.json', 'w') as f:
    f.write(json_str)

# Quick sanity check of the serialized data
data = json.loads(json_str)
print(f"   objectCount: {data['objectCount']}")
print(f"   layerCount: {data['layerCount']}")
print(f"   STRIDE_OBJ: {data['STRIDE_OBJ']}")
print(f"   STRIDE_MOV: {data['STRIDE_MOV']}")
print(f"   num levels: {len(data['levels'])}")
print(f"   num rules: {len(data['rules'])}")
print(f"   num lateRules: {len(data['lateRules'])}")
print(f"   num winconditions: {len(data['winconditions'])}")
print(f"   idDict: {data.get('idDict', 'N/A')}")

# Print first level info
for lv in data['levels']:
    if lv['type'] == 'level':
        print(f"   Level {lv['index']}: {lv['width']}x{lv['height']}, objects len: {len(lv['objects'])}")

print("\n3. Loading into C++ engine...")
import puzzlescript_cpp._puzzlescript_cpp as ps
engine = ps.Engine()
ok = engine.load_from_json(json_str)
print(f"   load_from_json returned: {ok}")
if not ok:
    print("   FAILED to load. Exiting.")
    sys.exit(1)

print(f"   num_levels: {engine.get_num_levels()}")
print(f"   object_count: {engine.get_object_count()}")

print("\n4. Loading level 0...")
engine.load_level(0)
print(f"   width: {engine.get_width()}, height: {engine.get_height()}")
cpp_objects_init = list(engine.get_objects())
print(f"   objects len: {len(cpp_objects_init)}")

# Compare initial objects with JS
print("\n5. Comparing initial state with JS...")
js.compile(['loadLevel', 0], game_text)
js_level = js.getLevel()
js_objects_init = list(js_level['objects'])
print(f"   JS objects len: {len(js_objects_init)}")

if cpp_objects_init == js_objects_init:
    print("   MATCH! Initial states are identical.")
else:
    print("   MISMATCH! Initial states differ.")
    # Find first difference
    for i in range(min(len(cpp_objects_init), len(js_objects_init))):
        if cpp_objects_init[i] != js_objects_init[i]:
            print(f"   First diff at index {i}: CPP={cpp_objects_init[i]} JS={js_objects_init[i]}")
            break
    if len(cpp_objects_init) != len(js_objects_init):
        print(f"   Length mismatch: CPP={len(cpp_objects_init)} JS={len(js_objects_init)}")

# Now test one step
print("\n6. Testing processInput(2) [down]...")
engine.process_input(2)
cpp_objects_after = list(engine.get_objects())
cpp_winning = engine.is_winning()
cpp_againing = engine.is_againing()
print(f"   winning: {cpp_winning}, againing: {cpp_againing}")

# Do the same in JS using processInput directly
js.compile(['loadLevel', 0], game_text)
js.processInput(2)
js_level_after = js.getLevel()
js_objects_after = list(js_level_after['objects'])
js_winning = bool(js.getWinning())

if cpp_objects_after == js_objects_after:
    print("   MATCH! States after one step are identical.")
else:
    print("   MISMATCH! States after one step differ.")
    diffs = [(i, cpp_objects_after[i], js_objects_after[i]) 
             for i in range(min(len(cpp_objects_after), len(js_objects_after)))
             if cpp_objects_after[i] != js_objects_after[i]]
    print(f"   Number of differences: {len(diffs)}")
    for idx, c, j in diffs[:5]:
        print(f"     index {idx}: CPP={c} JS={j}")

# Try replaying the full solution
print("\n7. Full solution replay...")
with open('data/js_sols/sokoban_basic/bfs_5000-steps_level-0.json', 'r') as f:
    sol = json.load(f)
actions = sol['actions']
print(f"   Solution has {len(actions)} actions")

# JS replay
js.compile(['loadLevel', 0], game_text)
for a in actions:
    js.processInput(a)
    while bool(js.getAgaining()):
        js.processInput(-1)
js_final = list(js.getLevel()['objects'])
js_won = bool(js.getWinning())
print(f"   JS final: won={js_won}")

# C++ replay
engine.load_level(0)
for a in actions:
    engine.process_input(a)
    while engine.is_againing():
        engine.process_input(-1)
cpp_final = list(engine.get_objects())
cpp_won = engine.is_winning()
print(f"   C++ final: won={cpp_won}")

if js_final == cpp_final:
    print("   MATCH! Final states are identical after full replay.")
else:
    diffs = [(i, cpp_final[i], js_final[i])
             for i in range(min(len(cpp_final), len(js_final)))
             if cpp_final[i] != js_final[i]]
    print(f"   MISMATCH! {len(diffs)} differences in final state.")
    for idx, c, j in diffs[:10]:
        print(f"     index {idx}: CPP={c} JS={j}")

print(f"\n   Win match: {'YES' if js_won == cpp_won else 'NO'}")
print("\nDone!")
