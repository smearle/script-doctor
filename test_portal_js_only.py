"""Quick test: verify JS behavior for `.PI..PO...` with 2 right moves.
Traces object and movement state at each phase of step 2."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
from backends import NodeJSPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser


def print_state(engine, label):
    """Print objects and movements at each cell."""
    bak = engine.backupLevel()
    dat = list(bak['dat'])
    width = int(bak['width'])
    height = int(bak['height'])
    print(f"  {label}:")
    for col in range(width):
        cell_val = dat[col]
        if cell_val != 1:
            print(f"    pos {col}: objects=0b{cell_val:016b} ({cell_val})")
    # Print movements
    movs = list(engine.getMovements()) if hasattr(engine, 'getMovements') and engine.getMovements else None
    if movs:
        print(f"    movements: {movs}")


def main():
    backend = NodeJSPuzzleScriptBackend()
    engine = backend.engine
    solver = backend.solver
    parser = init_ps_lark_parser()
    game = "test_gdd301_lab"
    game_text = backend.compile_game(parser, game)
    level_i = 0

    # Load level
    engine.compile(['loadLevel', level_i], game_text)
    print("Initial state:")
    print_state(engine, "after load")

    # Step 1: right
    engine.processInput(3)  # 3 = right
    print("\nAfter step 1 (right):")
    print_state(engine, "after processInput")

    # Step 2: right
    engine.processInput(3)
    print("\nAfter step 2 (right):")
    print_state(engine, "after processInput")

    # Also try: reset and do just 1 right move to see intermediate state
    engine.compile(['loadLevel', level_i], game_text)
    engine.processInput(3)
    bak1 = engine.backupLevel()

    # Now do a second right - but this time, let's check if the portal rule even fires
    engine.processInput(3)
    bak2 = engine.backupLevel()

    print(f"\nDirect comparison:")
    print(f"  After 1 right: {list(bak1['dat'])}")
    print(f"  After 2 right: {list(bak2['dat'])}")

    # Decode the bitmasks
    # bit 0 = Background, bit 7 = Player (128), bit 11 = Portal1 (2048), bit 12 = PortalEnd (4096)
    for step, dat in [("1 right", bak1['dat']), ("2 right", bak2['dat'])]:
        print(f"\n  After {step}:")
        for col in range(int(bak1['width'])):
            val = int(dat[col])
            objs = []
            if val & 1: objs.append('bg')
            if val & 128: objs.append('PLAYER')
            if val & 2048: objs.append('Portal1')
            if val & 4096: objs.append('PortalEnd')
            other = val & ~(1 | 128 | 2048 | 4096)
            if other: objs.append(f'other({other})')
            print(f"    pos {col}: {', '.join(objs)}")


if __name__ == "__main__":
    main()
