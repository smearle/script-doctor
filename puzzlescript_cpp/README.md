# Native player history and runtime snapshots

The C++ engine can retain PuzzleScript's native Undo history and Restart
checkpoint. This feature is **disabled at compile time by default**.

```sh
PUZZLESCRIPT_CPP_HISTORY=1 python setup_cpp.py build_ext --inplace

# Return to the default history-free build.
PUZZLESCRIPT_CPP_HISTORY=0 python setup_cpp.py build_ext --inplace
```

On PowerShell, set `$env:PUZZLESCRIPT_CPP_HISTORY = "1"` before building.
The build detects changes to this option and rebuilds cached objects. For direct
C++ compilation, define `PUZZLESCRIPT_CPP_HISTORY=1` consistently for every
translation unit; the option changes `Engine`'s layout.

## Python API

`puzzlescript_cpp.HISTORY_ENABLED` (also on the native extension) reports the
build capability. These methods are available on the native `Engine`
and public `CppPuzzleScriptEngine` wrapper only in a history-enabled build:

- `undo(force=False, ignore_duplicates=False)`: native Undo; returns whether a
  saved board was restored. Respects `noundo` unless forced.
- `restart_game(force=False)`: native player Restart. Respects `norestart`
  unless forced; restores the latest checkpoint, runs startup rules when
  configured, and is itself undoable.
- `undo_depth`: number of retained Undo entries.
- `serialize_state()`: versioned JSON containing the current board, RNG,
  win/AGAIN flags, complete Undo history, restart checkpoint, compiled game
  identity, and rule-firing telemetry.
- `restore_state(snapshot)`: validates and restores that runtime after loading
  the same compiled game. Invalid snapshots leave the live engine unchanged.

```python
from puzzlescript_cpp import HISTORY_ENABLED, CppPuzzleScriptEngine
assert HISTORY_ENABLED, "Rebuild with PUZZLESCRIPT_CPP_HISTORY=1"

engine = CppPuzzleScriptEngine()
engine.load_from_json(compiled_game_json)
engine.load_level(0)
engine.process_input(3)
engine.undo()
engine.restart_game()
saved = engine.serialize_state()

other = CppPuzzleScriptEngine()
other.load_from_json(compiled_game_json)
other.restore_state(saved)
```

Undo and Restart do **not** rewind the RNG, matching PuzzleScript.
`restore_state` does restore it, so resuming a saved runtime is distinct from a
player pressing Undo. Automatic ticks belong to the preceding player turn.
Callers still settle pending AGAIN ticks with `process_input(-1)` when needed.

The existing `restart()` remains a level reload. `backup_level()` /
`restore_level()` remain board-only helpers for existing search callers; they
are not complete runtime snapshots. Input IDs are unchanged: 0-4 are player
forces, and -1 / 5 are no-force ticks. Undo and Restart are separate methods;
a learner wrapper chooses its own action IDs. This change does not silently add
actions to existing gym wrappers or alter running experiments.

## Cost and scope

The default build contains no Undo stack, restart-board storage, compiled-game
snapshot identity, or history bookkeeping. Enabled builds retain a full board
per changed player input, plus a restart board: memory grows with episode
history, approximately `4 * width * height * stride_obj` bytes per entry plus
container overhead. Unlike the JavaScript UI, these boards are not diff
compressed. No automatic truncation changes native Undo behavior.

Snapshot generation is explicit and costs time/space proportional to the game,
current state and retained history. There is no serialization on ordinary steps.
Snapshots include the exact compiled game JSON for identity checking and
portability. They describe the native simulation, not renderer/UI state,
learner memory, replay, or optimizers; preserve those separately when resuming
training. In-level message popups remain omitted by this headless engine.

The implementation also fixes native command lifecycle behavior needed by
history: startup/AGAIN scheduling, movement clearing after board restoration,
and latched wins. These correctness fixes apply to both builds, independently
of history support.

## Verification

Initialize the pinned PuzzleScript submodule (`git submodule update --init
PuzzleScript`), build the desired variant, and run:

```sh
python tests/test_cpp_history.py --extension path/to/_puzzlescript_cpp.so
python tests/test_cpp_history.py --extension path/to/default/_puzzlescript_cpp.so --expect-off
```

The test compares native JavaScript traces for pushing, metadata, cancellation,
checkpoints, Restart commands, random rules, startup rules and AGAIN; it also
checks complete restoration at every action boundary, fresh-process continuation,
and rejection of invalid snapshots. Run the two builds in separate processes.
