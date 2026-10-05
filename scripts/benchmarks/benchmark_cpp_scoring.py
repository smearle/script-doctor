"""Paired full-output C++ rollouts using isolated extension builds.

Each build runs in a persistent subprocess to avoid pybind type-registration
conflicts. Initialization, reset, compilation, IPC and correctness hashing are
outside the timed region. Both builds receive the same seeded actions.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import numpy as np


def worker(library):
    name = "puzzlescript_cpp._puzzlescript_cpp"
    spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from puzzlescript_cpp import CppBatchedPuzzleScriptEnv
    env = actions = None
    for line in sys.stdin:
        request = json.loads(line)
        if request["command"] == "init":
            env = CppBatchedPuzzleScriptEnv(request["compiled"], request["batch"],
                                          level_indices=[0] * request["batch"],
                                          num_threads=request["threads"], max_episode_steps=100)
            actions = np.random.default_rng(request["seed"]).integers(
                0, 5, (request["steps"], request["batch"]), dtype=np.int32)
            result = {"ready": True}
        elif request["command"] == "run":
            env.reset()
            digest = hashlib.sha256() if request.get("check") else None
            trace = []
            start = time.perf_counter()
            for action in actions:
                output = env.step(action)
                if digest is not None:
                    step_trace = {}
                    fields = dict(zip(("obs", "reward", "done", "truncated"), output[:4]))
                    fields.update({f"info/{k}": v for k, v in output[4].items()})
                    for key, value in sorted(fields.items()):
                        array = np.asarray(value)
                        digest.update(str((array.shape, array.dtype.str)).encode())
                        digest.update(array.tobytes())
                        step_trace[key] = hashlib.sha256(array.tobytes()).hexdigest()
                        if array.dtype == bool:
                            step_trace[key + "/bits"] = np.packbits(array).tobytes().hex()
                    trace.append(step_trace)
            result = {"seconds": time.perf_counter() - start,
                      "sha256": digest.hexdigest() if digest is not None else None,
                      "trace": trace}
        else:
            raise ValueError(request)
        print(json.dumps(result), flush=True)
        # Park all OpenMP threads while the other build is being measured.
        os.kill(os.getpid(), signal.SIGSTOP)


def request(process, payload):
    os.kill(process.pid, signal.SIGCONT)
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"C++ benchmark worker exited: {process.poll()}")
    _, status = os.waitpid(process.pid, os.WUNTRACED)
    if not os.WIFSTOPPED(status):
        raise RuntimeError(f"Worker failed to park: {status}")
    return json.loads(line)


def compile_game(game):
    from scripts.benchmarks.profile_rand_nodejs import _load_nodejs_native_game_text
    source = _load_nodejs_native_game_text(game)
    engine = Path(__file__).resolve().parents[2] / "puzzlescript_nodejs/puzzlescript/engine.js"
    script = "const fs=require('fs'), e=require(process.argv[1]); e.compile(['loadLevel',0],fs.readFileSync(0,'utf8')); console.log(e.serializeCompiledStateJSON());"
    result = subprocess.run(["node", "-e", script, str(engine)], input=source, capture_output=True,
                            text=True, check=True)
    compiled = result.stdout.splitlines()[-1]
    json.loads(compiled)
    return compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--games", nargs="+", default=["sokoban_basic", "blocks", "Zen_Puzzle_Garden", "kettle"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 32, 256])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1042])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not all([args.baseline, args.candidate, args.output]):
        parser.error("--baseline, --candidate and --output are required")
    if min([*args.batches, *args.threads, args.steps, args.trials]) < 1:
        parser.error("positive batch, thread, step and trial counts required")
    processes = []
    root = Path(__file__).resolve().parents[2]
    tracked_sources = [Path(__file__), root / "puzzlescript_cpp/__init__.py",
                       root / "scripts/benchmarks/profile_rand_nodejs.py"]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in tracked_sources}
    result = {"host": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "source_hashes": source_hashes,
              "idle_workers": "SIGSTOP between requests; only the measured build runs",
              "thread_environment": {key: os.environ.get(key) for key in
                                     ("OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES", "OPENBLAS_NUM_THREADS")},
              "trials": args.trials, "seeds": args.seeds, "steps": args.steps,
              "library_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in [("baseline", args.baseline), ("candidate", args.candidate)]},
              "timing": "full Python wrapper outputs; alternating warmed calls; reset/actions/IPC excluded",
              "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for library in (args.baseline, args.candidate):
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "scripts.benchmarks.benchmark_cpp_scoring", "--worker", str(library.resolve())],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True))
        for game in args.games:
            compiled = compile_game(game)
            for batch in args.batches:
                for threads in args.threads:
                    if threads > batch:
                        continue
                    for seed in args.seeds:
                        for process in processes:
                            request(process, {"command": "init", "compiled": compiled, "batch": batch,
                                              "threads": threads, "steps": args.steps, "seed": seed})
                        checks = [request(p, {"command": "run", "check": True}) for p in processes]
                        if checks[0]["sha256"] != checks[1]["sha256"]:
                            for step, (left, right) in enumerate(zip(checks[0]["trace"], checks[1]["trace"])):
                                if left != right:
                                    differing = {k: [left[k], right[k]] for k in left if left[k] != right[k]}
                                    raise AssertionError(f"Full outputs differ for {game}/{batch}/{threads}/{seed}, "
                                                         f"step {step}: {differing}")
                        for _ in range(2):
                            for process in processes:
                                request(process, {"command": "run"})
                        samples = [[], []]
                        for trial in range(args.trials):
                            for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
                                samples[index].append(request(processes[index], {"command": "run"})["seconds"])
                        fps = [float(batch * args.steps / np.median(s)) for s in samples]
                        row = {"game": game, "batch": batch, "threads": threads, "seed": seed,
                               "compiled_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
                               "output_sha256": checks[0]["sha256"], "samples_s": samples,
                               "baseline_fps": fps[0], "candidate_fps": fps[1], "speedup": fps[1]/fps[0]}
                        for path in tracked_sources:
                            if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[str(path.relative_to(root))]:
                                raise RuntimeError(f"Source changed during measurement: {path}")
                        result["results"].append(row)
                        temporary = args.output.with_suffix(".json.tmp")
                        temporary.write_text(json.dumps(result, indent=2) + "\n")
                        temporary.replace(args.output)
                        print(json.dumps(row), flush=True)
    finally:
        for process in processes:
            process.stdin.close()
            if process.poll() is None:
                os.kill(process.pid, signal.SIGCONT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
