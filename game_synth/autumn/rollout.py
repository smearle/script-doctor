"""Parent-side wrapper: validate + behaviorally fingerprint an Autumn program by
running rollout_worker.py in a throwaway subprocess (segfault/timeout isolation).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

MARA_ROOT = "/home/jupyter-smearle/mara/MARA"
VENV_PY = "/home/jupyter-smearle/script-doctor/.venv/bin/python3"
_WORKER = os.path.join(os.path.dirname(__file__), "rollout_worker.py")


@dataclass
class Signature:
    ok: bool
    error: str | None = None
    n_on: int = 0
    covered: tuple[int, ...] = ()
    traj: str = ""
    n_states: int = 0
    dead: bool = False

    def key(self) -> tuple:
        """Behavioral identity for dedup: trajectory hash + covered-handler set."""
        return (self.traj, self.covered)


def evaluate(prog: str, seed: int = 0, n_steps: int = 120,
             timeout: float = 60.0) -> Signature:
    """Run the probe in a subprocess. Returns Signature(ok=False, error=...) on
    compile error, runtime exception, segfault (core dump), or timeout."""
    env = dict(os.environ)
    env["PYTHONPATH"] = MARA_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.NamedTemporaryFile("w", suffix=".sexp", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        proc = subprocess.run(
            [VENV_PY, _WORKER, path, str(seed), str(n_steps)],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        os.unlink(path)
        return Signature(ok=False, error="timeout")
    os.unlink(path)
    if proc.returncode != 0:
        # non-zero exit with no JSON => segfault / core dump
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        return Signature(ok=False, error=f"crash(rc={proc.returncode}): {tail[0][:120]}")
    # last stdout line is the JSON payload (interpreter may print to stdout too)
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.strip().startswith("{")), "")
    if not line:
        return Signature(ok=False, error="no-output")
    d = json.loads(line)
    if not d.get("ok"):
        return Signature(ok=False, error=d.get("error", "unknown"))
    return Signature(ok=True, n_on=d["n_on"], covered=tuple(d["covered"]),
                     traj=d["traj"], n_states=d["n_states"], dead=d["dead"])


@dataclass
class MultiSig:
    """Behavioral fingerprint across several random-probe seeds. Novelty under
    'distinct rollouts under random play' = any seed's trajectory differs."""
    ok: bool
    error: str | None = None
    sigs: tuple = ()           # per-seed Signature
    covered: tuple = ()        # union of covered handlers across seeds
    n_states_max: int = 0
    dead: bool = False

    def key(self) -> tuple:
        return tuple(s.traj for s in self.sigs)


def evaluate_multi(prog: str, seeds=(0, 1, 2), n_steps: int = 120,
                   timeout: float = 60.0) -> MultiSig:
    sigs = []
    covered = set()
    for sd in seeds:
        s = evaluate(prog, sd, n_steps, timeout)
        if not s.ok:                       # any seed failing => invalid program
            return MultiSig(ok=False, error=s.error)
        sigs.append(s)
        covered |= set(s.covered)
    dead = all(s.dead for s in sigs)
    return MultiSig(ok=True, sigs=tuple(sigs), covered=tuple(sorted(covered)),
                    n_states_max=max(s.n_states for s in sigs), dead=dead)


if __name__ == "__main__":
    import sys
    prog = open(sys.argv[1]).read()
    sig = evaluate(prog, seed=0, n_steps=120)
    print(sig)
