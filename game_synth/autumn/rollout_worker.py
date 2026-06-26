"""Child process: load an Autumn program, run a fixed-seed probe rollout, emit a
behavioral signature as JSON on stdout.

Run in isolation because the C++ interpreter *segfaults* (core dumps) on many
malformed programs rather than raising -- so this must be a throwaway process
whose death the parent can detect via a non-zero exit code.

Usage: rollout_worker.py <prog_file> <seed> <n_steps>
Emits:  {"ok": true, "n_on": int, "covered": [int...], "traj": "<sha1>",
         "n_states": int, "dead": bool}
   or:  {"ok": false, "error": "..."}
"""

import hashlib
import json
import random
import sys

from MARA.autumn_cpp.interpreter_module import Interpreter
from MARA.autumn_cpp.autumnstdlib import autumnstdlib


def _canon(render_json: str) -> str:
    """Canonicalize a render_all() result so equal boards hash equally."""
    try:
        d = json.loads(render_json)
    except Exception:
        return render_json
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            cells = []
            for o in v:
                pos = o.get("position", {}) if isinstance(o, dict) else {}
                cells.append((pos.get("x"), pos.get("y"), o.get("color")))
            out[k] = sorted(cells, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))
        else:
            out[k] = v
    return json.dumps(out, sort_keys=True)


def main():
    prog_file, seed, n_steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    prog = open(prog_file).read()
    itp = Interpreter()
    itp.set_verbose(False)
    itp.run_script(prog, autumnstdlib, "", seed)

    n_on = itp.get_on_clause_count()
    # grid size: read from the first render
    first = _canon(itp.render_all())
    try:
        grid = json.loads(itp.render_all()).get("GRID_SIZE", 16)
    except Exception:
        grid = 16
    grid = int(grid) if grid else 16

    rng = random.Random(seed)
    hashes = [first]
    for _ in range(n_steps):
        a = rng.random()
        if a < 0.20:
            itp.left()
        elif a < 0.40:
            itp.right()
        elif a < 0.55:
            itp.up()
        elif a < 0.70:
            itp.down()
        elif a < 0.88:
            itp.click(rng.randint(0, grid - 1), rng.randint(0, grid - 1))
        else:
            itp.step()
        hashes.append(_canon(itp.render_all()))

    covered = sorted(itp.get_covered_on_clause_indices())
    traj = hashlib.sha1("\n".join(hashes).encode()).hexdigest()
    # "dead": board never changes across the whole probe
    dead = len(set(hashes)) <= 1
    print(json.dumps({
        "ok": True, "n_on": n_on, "covered": covered,
        "traj": traj, "n_states": len(set(hashes)), "dead": dead,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}))
